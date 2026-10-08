"""L2 weight prefetch during the decode all-gathers. ``DSV41_L2PF_MB=0`` disables it.

The all-gather is a network wait: the GPU is busy, DRAM is idle. Reading the next layer's weights on
a side stream then leaves them in L2 for the kernels that follow, out of bandwidth nothing else is
using. Loads only, a private sink, no value changes -- exact.

The budget must fit the window (tools/bench_l2_prefetch.py): more than ``window x DRAM rate`` and the
join delays the next read instead of helping. RoCE shrinks the window to ~15 us, so ~2-3 MB is the
budget there; on NCCL's ~45 us, ~8 MB. Weights are read, never written, so this is safe under CUDA
graph capture (the fork/join must be inside the graph, before the window and after it).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

MB = int(os.environ.get("DSV41_L2PF_MB", "2"))
MODE = os.environ.get("DSV41_L2PF_MODE", "touch")
PACE_GBPS = int(os.environ.get("DSV41_L2PF_PACE_GBPS", "0"))
if MODE not in ("touch", "bulk") or PACE_GBPS < 0:
    raise ValueError("L2 prefetch mode must be touch/bulk and pace >= 0")
VERSION = 3  # Raw-byte loads: FP8 storage must not enter Triton's masked-load casts.
_SINK = None


@triton.jit
def _touch(P, SINK, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(P + i, mask=i < N, other=0, eviction_policy="evict_last")
    tl.store(SINK + pid, tl.sum(v.to(tl.float32), 0))


@triton.jit
def _bulk_prefetch(P, N: tl.constexpr, PACE: tl.constexpr, PIECE: tl.constexpr):
    # A cache hint only; no weight writes and no requirement that prefetch finish
    # before the consumer. Two CTAs issue 32 KiB pieces, optionally rate-limited.
    pid = tl.program_id(0)
    start = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", constraints="=l",
                                     args=[], dtype=tl.uint64, is_pure=False, pack=1)
    for piece in range(pid, tl.cdiv(N, PIECE), 2):
        if PACE > 0:
            due = start + (piece * PIECE // PACE).to(tl.uint64)
            now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", constraints="=l",
                                           args=[], dtype=tl.uint64, is_pure=False, pack=1)
            while now < due:
                tl.inline_asm_elementwise("nanosleep.u32 32; mov.u32 $0, 0;", constraints="=r",
                                          args=[], dtype=tl.int32, is_pure=False, pack=1)
                now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", constraints="=l",
                                               args=[], dtype=tl.uint64, is_pure=False, pack=1)
        count = tl.minimum(PIECE, N - piece * PIECE).to(tl.int32)
        tl.inline_asm_elementwise(
            "{ .reg .pred p; .reg .u32 t; mov.u32 t, %tid.x; setp.eq.u32 p,t,0; "
            "@p cp.async.bulk.prefetch.L2.global [$1], $2; mov.u32 $0,0; }",
            constraints="=r,l,r", args=[P + piece * PIECE, count], dtype=tl.int32,
            is_pure=False, pack=1)


def enabled() -> bool:
    return MB > 0


def budget_bytes() -> int:
    return MB * 1024 * 1024


def raw(weight):
    """The storage tensor behind an FP8Weight / FP4Weight / TP wrapper, or None."""
    if torch.is_tensor(weight):
        return weight
    for attr in ("w", "local"):
        inner = getattr(weight, attr, None)
        if inner is not None:
            got = raw(inner)
            if got is not None:
                return got
    return None


def touch(stream, tensors, budget: int, *, mode=None, pace=None):
    """Read up to `budget` bytes of `tensors` on `stream` with evict_last, into a private sink."""
    global _SINK
    mode = MODE if mode is None else mode
    pace = PACE_GBPS if pace is None else pace
    if _SINK is None:
        _SINK = torch.zeros(8192, dtype=torch.float32, device="cuda")
    left = budget
    with torch.cuda.stream(stream):
        for t in tensors:
            flat = raw(t)
            if flat is None or flat.numel() == 0 or left <= 0:
                continue
            # Prefetch storage, not numeric values. A typed FP8 masked load tries to cast
            # `other=0` from int32 to fp8e4nv and fails during the first decode capture
            # when DSV41_DENSE_FP4=off. This view aliases the same bytes without conversion.
            flat = flat.reshape(-1).view(torch.uint8)
            take = min(flat.numel(), left)
            left -= take
            blocks = min(triton.cdiv(take, 4096), _SINK.numel())
            if mode == "bulk" and flat.data_ptr() % 16 == 0 and take % 16 == 0:
                _bulk_prefetch[(2,)](flat, take, pace, 32768, num_warps=1)
            else:
                _touch[(blocks,)](flat[:take], _SINK[:blocks], take, BLOCK=4096, num_warps=4)
    return budget - left
