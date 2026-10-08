"""Experimental checkpoint-BF16 router GEMM, with FP32 logits/accumulation.

The checkpoint gate and decode activations already contain BF16 values. This
does not quantize them, but tensor-core/reduction order differs from padded
cuBLAS FP32 GEMM. It remains opt-in: the short whole-model screen changed text
and did not show a code gain. Never narrow an actually FP32 gate through it.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _project(X, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             XM: tl.constexpr, WN: tl.constexpr, SPLIT: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    part = tl.program_id(1)
    m = tl.arange(0, 16)
    k = tl.arange(0, BK)
    chunk: tl.constexpr = triton.cdiv(K, BK * SPLIT) * BK
    acc = tl.zeros((16, BN), tl.float32)
    for start in range(part * chunk, tl.minimum((part + 1) * chunk, K), BK):
        xx = tl.load(X + m[:, None] * XM + start + k[None, :],
                     (m[:, None] < M) & (start + k[None, :] < K), 0)
        ww = tl.load(W + n[None, :] * WN + start + k[:, None],
                     (n[None, :] < N) & (start + k[:, None] < K), 0)
        acc = tl.dot(xx, ww, acc)
    tl.store(P + (part * M + m[:, None]) * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _reduce(P, Y, COUNT: tl.constexpr, SPLIT: tl.constexpr, B: tl.constexpr = 256):
    i = tl.program_id(0) * B + tl.arange(0, B)
    acc = tl.load(P + i, i < COUNT, 0)
    for part in range(1, SPLIT):
        acc = acc + tl.load(P + part * COUNT + i, i < COUNT, 0)
    tl.store(Y + i, acc, i < COUNT)


def project(x, w, *, split=4, bn=32, bk=128):
    """[1..16,K] BF16 @ [N,K] BF16 -> FP32; fixed row-independent math."""
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise ValueError("requires original BF16 values, not a narrowed FP32 gate")
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError("incompatible router matrices")
    if not 0 < x.shape[0] <= 16 or x.stride(1) != 1 or w.stride(1) != 1:
        raise ValueError("requires unit-stride decode rows")
    if split not in (1, 2, 4, 8) or bn not in (32, 64) or bk not in (64, 128):
        raise ValueError("unsupported experimental tile")
    m, k = x.shape
    n = w.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    parts = out if split == 1 else torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    _project[(triton.cdiv(n, bn), split)](x, w, parts, m, n, k,
        x.stride(0), w.stride(0), SPLIT=split, BN=bn, BK=bk, num_warps=4, num_stages=2)
    if split > 1:
        _reduce[(triton.cdiv(m * n, 256),)](parts, out, m * n, SPLIT=split,
                                         num_warps=4, enable_fp_fusion=False)
    return out
