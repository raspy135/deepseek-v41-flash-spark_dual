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
VERSION = 2  # Raw-byte loads: FP8 storage must not enter Triton's masked-load casts.
_SINK = None


@triton.jit
def _touch(P, SINK, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(P + i, mask=i < N, other=0, eviction_policy="evict_last")
    tl.store(SINK + pid, tl.sum(v.to(tl.float32), 0))


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


def touch(stream, tensors, budget: int):
    """Read up to `budget` bytes of `tensors` on `stream` with evict_last, into a private sink."""
    global _SINK
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
            _touch[(blocks,)](flat[:take], _SINK[:blocks], take, BLOCK=4096, num_warps=4)
    return budget - left
