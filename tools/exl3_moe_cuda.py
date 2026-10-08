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
    lib.exl3m_gateup.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i32, i32, i32, i32, f32, i32, ptr]
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
    out["k2"] = torch.tensor([2 * int(b) for b in arena.bits], dtype=torch.int32, device=dev)
    arena._cuda_ptrs = out
    return out


def warm(arena) -> None:
    """Build the JIT library and the per-slot pointer tables at boot, not inside a capture."""
    _lib()
    _ptrs(arena)


@torch.no_grad()
def moe_forward(x: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor, arena,
                swiglu_limit: float = 10.0) -> torch.Tensor:
    """x bf16 [T, DIM], slot ids int32 [T, 6], weights fp32 [T, 6] -> fp32 [T, DIM].

    Every routed slot must be loaded (all-resident / LUT path).  Under TP-output each rank owns half
    the intermediate columns and half the hidden outputs; the intermediate is all-gathered before the
    down projection and the per-token totals after it, at the same points fp4_moe does.

    Decode-shaped only: TF's group_kernel caps at 16 members a slot, so P must be <= 16 per slot.
    """
    T, K = slots.shape
    P = T * K
    dim, inter, down_n = arena.shapes["suh1"][0], arena.inter, arena.down_n
    world = arena.tp_world
    assert x.shape == (T, dim) and x.dtype == torch.bfloat16
    dev = x.device
    p = _ptrs(arena)
    if slots.numel() > 64:
        raise ValueError("exl3_moe_cuda is decode-shaped (P<=64); use the prefill kernel")
    # no torch.unique/.tolist() or any device->host sync here: this runs inside graph capture.

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
    uids = torch.empty((P,), dtype=torch.int32, device=dev)
    ucount = torch.empty((1,), dtype=torch.int32, device=dev)
    members = torch.empty((P, 16), dtype=torch.int32, device=dev)
    pick = slots.reshape(-1).to(torch.int32).contiguous()

    lib = _lib()
    st = _stream()
    lib.exl3m_group(ctypes.c_void_p(pick.data_ptr()), ctypes.c_void_p(uids.data_ptr()),
                    ctypes.c_void_p(ucount.data_ptr()), ctypes.c_void_p(members.data_ptr()),
                    int(T), int(K), int(arena.slots), 16, st)
    lib.exl3m_rot_in(ctypes.c_void_p(x.data_ptr()), int(x.stride(0)), ctypes.c_void_p(pick.data_ptr()),
                     ctypes.c_void_p(arena.suh1.data_ptr()), ctypes.c_void_p(arena.suh3.data_ptr()),
                     ctypes.c_void_p(xh0.data_ptr()), ctypes.c_void_p(xh1.data_ptr()),
                     int(T), int(dim), int(K), int(arena.slots), 1, st)
    lib.exl3m_grouped(ctypes.c_void_p(xh0.data_ptr()), ctypes.c_void_p(xh1.data_ptr()),
                      ctypes.c_void_p(p["t1p"].data_ptr()), ctypes.c_void_p(p["t3p"].data_ptr()),
                      ctypes.c_void_p(p["k2"].data_ptr()), ctypes.c_void_p(p["k2"].data_ptr()),
                      ctypes.c_void_p(uids.data_ptr()), ctypes.c_void_p(ucount.data_ptr()),
                      ctypes.c_void_p(members.data_ptr()), ctypes.c_void_p(z.data_ptr()),
                      int(dim), int(inter), int(P), 1, 16, int(K), int(P), 2, NT, WARPS, PF, 2, 10, CB_MUL1, st)
    lib.exl3m_gateup(ctypes.c_void_p(z.data_ptr()), ctypes.c_void_p(pick.data_ptr()),
                     ctypes.c_void_p(arena.svh1.data_ptr()), ctypes.c_void_p(arena.svh3.data_ptr()),
                     ctypes.c_void_p(arena.suh2.data_ptr()), ctypes.c_void_p(xd.data_ptr()),
                     int(T), int(P), int(inter), 1, int(K), int(arena.slots), float(swiglu_limit), 1, st)
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
                      int(down_k), int(down_n), int(P), 1, 16, int(K), int(P), 1, NT, WARPS, PF, 2, 10, CB_MUL1, st)
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
