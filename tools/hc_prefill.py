"""
hc_prefill.py -- the Hyper-Connection mix front for prefill-sized row counts, in one pass.

Model._hc_mixes computes  mixes = (x @ hc_fn^T) * rsqrt(mean(x^2) + eps)  over the hc-wide residual
x [T, 4 * 5120]. In torch that is x.float() (a [T, 20480] FP32 copy: 168 MB at a 2,048-row
chunk), a cuBLAS FP32 SIMT GEMM with N = 24 (1.6 ms -- the shape gives cuBLAS nothing to tile),
then square and mean over the copy again: ~4.4 ms a call, two calls a layer, ~0.9 s of an 8K
prefill. Here one program owns BLOCK_M rows: it reads x once in BF16, widens it exactly to FP32,
and accumulates both the projection (3-pass TF32 emulation of FP32 on the tensor cores, a partial
per K tile summed tile by tile) and the sum of squares, K in a fixed order. Every row's arithmetic
depends only on that row, so a row gets the same bits whatever chunk it is in (the chunk
invariance mm() pads decode for). 0.64 vs 5.4 ms at a 2,048-row chunk.

Not bit-identical to the torch path: cuBLAS and torch.mean sum in their own orders, ~1e-7
relative here (6.6e-7 vs FP64, torch 5.2e-7). A numerics switch (DSV41_HC_PREFILL_FUSED), in the boot guard.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_front_kernel(X, W, OUT, T, N, K, stride_x, stride_w, stride_o, eps,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rm = rows < T
    cols = tl.arange(0, BLOCK_N)
    cm = cols < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    ss = tl.zeros((BLOCK_M,), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        km = kk < K
        x = tl.load(X + rows[:, None].to(tl.int64) * stride_x + kk[None, :],
                    mask=rm[:, None] & km[None, :], other=0.0).to(tl.float32)
        w = tl.load(W + cols[:, None] * stride_w + kk[None, :], mask=cm[:, None] & km[None, :], other=0.0)
        # tf32x3 (the decode HC kernel's DSV41_HC_PREC): 6.6e-7 max relative error against FP64,
        # torch/cuBLAS 5.2e-7. "ieee" measured 5e-6 here, and slower (1.5 vs 0.64 ms).
        acc += tl.dot(x, tl.trans(w), input_precision="tf32x3", out_dtype=tl.float32)
        ss += tl.sum(x * x, 1)
    r = 1.0 / tl.sqrt_rn(ss / K + eps)
    out = acc * r[:, None]
    tl.store(OUT + rows[:, None] * stride_o + cols[None, :], out, mask=rm[:, None] & cm[None, :])


def hc_front(x: torch.Tensor, w: torch.Tensor, eps: float, block_m: int = 64, block_k: int = 64,
             num_warps: int = 4) -> torch.Tensor:
    """x bf16 [T, K] (any row stride, unit column stride); w fp32 [N, K] -> fp32 [T, N]."""
    T, K = x.shape
    N = w.shape[0]
    assert x.dtype == torch.bfloat16 and w.dtype == torch.float32 and w.shape[1] == K
    assert x.stride(1) == 1 and w.stride(1) == 1 and N <= 64
    out = torch.empty(T, N, dtype=torch.float32, device=x.device)
    if T:
        _hc_front_kernel[(triton.cdiv(T, block_m),)](
            x, w, out, T, N, K, x.stride(0), w.stride(0), out.stride(0), eps,
            BLOCK_M=block_m, BLOCK_N=max(16, triton.next_power_of_2(N)), BLOCK_K=block_k,
            num_warps=num_warps)
    return out


def hc_front_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf @ w.t()) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
