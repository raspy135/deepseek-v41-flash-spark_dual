"""JIT loader + orchestration for :mod:`exl3_moe_cuda.cu` (the EXL3 decode kernels).

Same shape as :mod:`fp4_moe_cuda`: nvcc builds the .cu once into a small C-ABI .so and ctypes
passes tensor pointers and the current stream, so launches stay CUDA-graph capturable and there is
no libtorch ABI dependency.  ``moe_forward`` runs the five kernels over an ``exl3_moe.Exl3Arena``.

v1 scope: TP-world-1 and 3-bit slots only.  TF's tile stride is ``4*K2`` words, so a 2-bit expert
must live in a 2-bit pool, not padded into a 3-bit slot -- per-bit-width pools are the next step.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import torch

SOURCE = Path(__file__).with_name("exl3_moe_cuda.cu")

# (nt, warps, pf) tile setting; K must divide 16*warps and N by 16*nt. See grouped_launch.
NT, WARPS, PF = 8, 4, 1
CB_MUL1 = 2


def source_digest() -> str:
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def _library():
    nvcc = os.environ.get("NVCC", os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc"))
    version = subprocess.check_output([nvcc, "--version"])
    key = hashlib.sha256(SOURCE.read_bytes() + version + platform.machine().encode()
                         + str((NT, WARPS, PF)).encode()).hexdigest()[:24]
    folder = Path(os.environ.get("TRITON_CACHE_DIR", "/tmp/dsv41-native")) / "exl3-moe-cuda"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = folder / f"exl3-moe-{key}.so"
    with (folder / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            with tempfile.TemporaryDirectory(prefix="build-", dir=folder) as tmp:
                built = Path(tmp) / "exl3-moe.so"
                subprocess.run([
                    nvcc, "-std=c++17", "-O3", "-DNDEBUG",
                    "-Xcompiler=-fPIC", "-shared",
                    "-gencode", "arch=compute_121a,code=sm_121a",
                    str(SOURCE), "-o", str(built),
                ], check=True)
                os.replace(built, target)
    lib = ctypes.CDLL(str(target))
    ptr, i64, i32, f32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_float
    lib.exl3m_group.argtypes = [ptr, ptr, ptr, ptr, i32, i32, i32, i32, ptr]
    lib.exl3m_rot_in.argtypes = [ptr, i32, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, ptr]
    lib.exl3m_grouped.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr,
                                  i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ptr]
    lib.exl3m_gateup.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, f32, i32, ptr]
    lib.exl3m_down_combine.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, ptr]
    return lib


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = _library()
    return _LIB


def _stream():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def _ptrs(arena):
    """Per-slot trellis pointers (int64 [slots]) and K2 (int32 [slots]); built once per arena."""
    cached = getattr(arena, "_cuda_ptrs", None)
    if cached is not None:
        return cached
    if arena.tp_world not in (1, 2):
        raise NotImplementedError("exl3_moe_cuda supports tp_world=1 or 2")
    dev = arena.t1.device
    s = arena.slots
    out = {}
    for name, tens in (("t1p", arena.t1), ("t3p", arena.t3), ("t2p", arena.t2)):
        stride = tens[0].numel() * tens.element_size()
        base = tens.data_ptr()
        out[name] = torch.tensor([base + i * stride for i in range(s)], dtype=torch.int64, device=dev)
    out["k2"] = arena.bits_gpu   # device tensor, updated by load_slot (widths change on swaps)
    arena._cuda_ptrs = out
    return out


def warm(arena) -> None:
    """Build the JIT library and the per-slot pointer tables at boot, not inside a capture."""
    _lib()
    _ptrs(arena)


@torch.no_grad()
def _run_pipeline(x, slots, weights, arena, uids, ucount, members, maxm, nexp_max, swiglu_limit,
                  nt=NT, warps=WARPS, pf=PF):
    """The five kernels over a prebuilt grouping. Shared by decode (P<=64) and prefill (P>64).

    `pick` is the arena slot per pair, in [T, K] order; uids/members say which pairs belong to which
    slot.  Under TP-output each rank owns half the intermediate columns and half the hidden outputs;
    the intermediate is all-gathered before down and the per-token totals after, where fp4_moe does.
    """
    T, K = slots.shape
    P = T * K
    dim, inter, down_n = arena.shapes["suh1"][0], arena.inter, arena.down_n
    world = arena.tp_world
    dev = x.device
    p = _ptrs(arena)
    group = None
    if world > 1:
        from engine.collective_rails import group as _g
        group = _g()

    xh0 = torch.empty((P, dim), dtype=torch.float16, device=dev)
    xh1 = torch.empty((P, dim), dtype=torch.float16, device=dev)
    z = torch.empty((2, P, inter), dtype=torch.float32, device=dev)
    xd = torch.empty((P, inter), dtype=torch.float16, device=dev)
    zd = torch.empty((1, P, down_n), dtype=torch.float32, device=dev)
    y = torch.empty((P, down_n), dtype=torch.float32, device=dev)
    local = torch.empty((T, down_n), dtype=torch.float32, device=dev)
    pick = slots.reshape(-1).to(torch.int32).contiguous()

    lib = _lib()
    st = _stream()
    lib.exl3m_rot_in(ctypes.c_void_p(x.data_ptr()), int(x.stride(0)), ctypes.c_void_p(pick.data_ptr()),
                     ctypes.c_void_p(arena.suh1.data_ptr()), ctypes.c_void_p(arena.suh3.data_ptr()),
                     ctypes.c_void_p(xh0.data_ptr()), ctypes.c_void_p(xh1.data_ptr()),
                     int(T), int(dim), int(K), int(arena.slots), 1, st)
    suh2_stride = int(arena.shapes["suh2"][0])
    suh2_base = arena.suh2.data_ptr() + arena.tp_rank * inter * 2
    lib.exl3m_grouped(ctypes.c_void_p(xh0.data_ptr()), ctypes.c_void_p(xh1.data_ptr()),
                      ctypes.c_void_p(p["t1p"].data_ptr()), ctypes.c_void_p(p["t3p"].data_ptr()),
                      ctypes.c_void_p(p["k2"].data_ptr()), ctypes.c_void_p(p["k2"].data_ptr()),
                      ctypes.c_void_p(uids.data_ptr()), ctypes.c_void_p(ucount.data_ptr()),
                      ctypes.c_void_p(members.data_ptr()), ctypes.c_void_p(z.data_ptr()),
                      int(dim), int(inter), int(P), 1, int(maxm), int(K), int(nexp_max), 2, nt, warps, pf, 2, 10,
                      CB_MUL1, st)
    lib.exl3m_gateup(ctypes.c_void_p(z.data_ptr()), ctypes.c_void_p(pick.data_ptr()),
                     ctypes.c_void_p(arena.svh1.data_ptr()), ctypes.c_void_p(arena.svh3.data_ptr()),
                     ctypes.c_void_p(suh2_base), ctypes.c_void_p(xd.data_ptr()),
                     int(T), int(P), int(inter), 1, int(K), int(arena.slots), suh2_stride,
                     float(swiglu_limit), 1, st)
    if world > 1:
        gathered = torch.empty((world * P, inter), dtype=torch.float16, device=dev)
        torch.distributed.all_gather_into_tensor(gathered, xd, group=group)
        xd = gathered.view(world, P, inter).transpose(0, 1).reshape(P, world * inter)
    down_k = world * inter
    lib.exl3m_grouped(ctypes.c_void_p(xd.data_ptr()), ctypes.c_void_p(xd.data_ptr()),
                      ctypes.c_void_p(p["t2p"].data_ptr()), ctypes.c_void_p(p["t2p"].data_ptr()),
                      ctypes.c_void_p(p["k2"].data_ptr()), ctypes.c_void_p(p["k2"].data_ptr()),
                      ctypes.c_void_p(uids.data_ptr()), ctypes.c_void_p(ucount.data_ptr()),
                      ctypes.c_void_p(members.data_ptr()), ctypes.c_void_p(zd.data_ptr()),
                      int(down_k), int(down_n), int(P), 1, int(maxm), int(K), int(nexp_max), 1, nt, warps, pf, 2, 10,
                      CB_MUL1, st)
    lib.exl3m_down_combine(ctypes.c_void_p(zd.data_ptr()), ctypes.c_void_p(pick.data_ptr()),
                           ctypes.c_void_p(arena.svh2.data_ptr()), ctypes.c_void_p(y.data_ptr()),
                           ctypes.c_void_p(weights.reshape(-1).float().contiguous().data_ptr()),
                           ctypes.c_void_p(local.data_ptr()),
                           int(T), int(P), int(down_n), 1, int(K), int(arena.slots), st)
    if world == 1:
        return local
    out = torch.empty((world * T, down_n), dtype=torch.float32, device=dev)
    torch.distributed.all_gather_into_tensor(out, local, group=group)
    return out.view(world, T, down_n).transpose(0, 1).reshape(T, dim)


@torch.no_grad()
def moe_forward(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena,
                swiglu_limit: float = 10.0) -> torch.Tensor:
    """Decode-sized (P<=64): group the picks with the O(P^2) kernel, then the pipeline."""
    T, K = slots.shape
    P = T * K
    if P > 64:
        raise ValueError("exl3_moe_cuda.moe_forward is decode-shaped (P<=64); use moe_forward_prefill")
    uids = torch.empty((P,), dtype=torch.int32, device=slots.device)
    ucount = torch.empty((1,), dtype=torch.int32, device=slots.device)
    members = torch.empty((P, 16), dtype=torch.int32, device=slots.device)
    s = torch.cuda.current_stream()
    lib = _lib()
    lib.exl3m_group(ctypes.c_void_p(slots.reshape(-1).to(torch.int32).contiguous().data_ptr()),
                    ctypes.c_void_p(uids.data_ptr()), ctypes.c_void_p(ucount.data_ptr()),
                    ctypes.c_void_p(members.data_ptr()), int(T), int(K), int(arena.slots), 16, _stream())
    del s
    return _run_pipeline(x, slots, weights, arena, uids, ucount, members, 16, P, swiglu_limit,
                         nt=8, warps=4, pf=1)


@torch.no_grad()
def moe_forward_prefill(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena,
                        swiglu_limit: float = 10.0, block_m: int = 64) -> torch.Tensor:
    """Prefill: group with the engine's build_routing (BM blocks, one expert each), then the pipeline.

    build_routing's pair ids are flat (row*K + k); the grouped kernel decodes members as
    row*32 + k, so encode them and leave -1 padding negative (the kernel skips it).
    """
    import fp4_moe as F4
    T, K = slots.shape
    # Compact the slot ids BEFORE routing, exactly as fp4_moe does: build_routing's cost tracks
    # n_slots, not the experts a chunk touches, and at a 13.5k-slot arena that is the dominant
    # prefill cost (~1.5 s a layer here against <1 ms compacted).  Pair ids stay original.
    valid = slots >= 0
    uniq, inv = torch.unique(slots[valid], return_inverse=True)
    compact = torch.full_like(slots, -1, dtype=torch.int32)
    compact[valid] = inv.to(torch.int32)
    block_slot, block_pair, NB = F4.build_routing(compact, int(uniq.numel()), block_m)
    block_slot = torch.where(block_slot >= 0, uniq.to(torch.int32)[block_slot.clamp_min(0)], block_slot)
    pair = block_pair.view(NB, block_m)
    members = torch.where(pair >= 0, (pair // K) * 32 + (pair % K), pair).to(torch.int32).contiguous()
    uids = block_slot.to(torch.int32).contiguous()
    ucount = torch.full((1,), NB, dtype=torch.int32, device=slots.device)
    # pf=2 is the prefill winner (58 vs 86 ms at T=2048); decode keeps the sweep-tuned pf=1.
    return _run_pipeline(x, slots, weights, arena, uids, ucount, members, block_m, NB, swiglu_limit,
                         nt=8, warps=4, pf=2)
