"""
indexer_prefill.py -- the CSA indexer's score matrix for prefill in one kernel.

Model._indexer builds score[t, n] = sum_h relu(bf16(q[t, h] . k[n])) * w[t, h] (stored BF16) as a
Python loop over 64-query x 512-key tiles: an einsum to a [64, 32, 512] BF16 tensor, then .float(),
relu_, the weight multiply, the head sum and the BF16 store -- six kernels and ~20 MB of traffic
per tile. The work is [T x context] per indexer layer, so it grows with the square of the prompt:
~0.3 s of an 8K prefill and several seconds of a 32K one.

Here a program owns a BM x BN tile of the score matrix in registers and walks the heads: one
tensor-core dot per head (BF16 inputs, FP32 accumulation, rounded to BF16 as the einsum's output
was), relu, times the head weight, added into the FP32 tile; one BF16 store at the end. The heads
are summed in order 0..H-1, which torch's reduction does not promise, so scores can differ in the
last FP32 bit before the BF16 rounding -- a ranking-only numerics change (top-k near-ties), behind
DSV41_INDEX_FUSED in the boot guard. Rows are independent, so the chunk invariance holds.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _index_score_kernel(Q, K, W, OUT, T, N, stride_qt, stride_qh, stride_k, stride_wt, stride_o,
                        H: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid_n = tl.program_id(0)   # fast axis: programs sharing a query block run together (q from L2)
    pid_m = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    rm = rows < T
    cols = pid_n * BN + tl.arange(0, BN)
    cm = cols < N
    d = tl.arange(0, D)
    k = tl.load(K + cols[:, None].to(tl.int64) * stride_k + d[None, :], mask=cm[:, None], other=0.0)
    acc = tl.zeros((BM, BN), tl.float32)
    for h in range(H):
        q = tl.load(Q + rows[:, None].to(tl.int64) * stride_qt + h * stride_qh + d[None, :],
                    mask=rm[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
        w = tl.load(W + rows.to(tl.int64) * stride_wt + h, mask=rm, other=0.0)
        acc += tl.maximum(s, 0.0) * w[:, None]
    tl.store(OUT + rows[:, None].to(tl.int64) * stride_o + cols[None, :], acc.to(OUT.dtype.element_ty),
             mask=rm[:, None] & cm[None, :])


def index_scores(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, out_dtype=torch.bfloat16,
                 bm: int = 64, bn: int = 64, num_warps: int = 4, num_stages: int = 2) -> torch.Tensor:
    """q bf16 [T, H, D]; k bf16 [N, D]; w fp32 [T, H] -> score [T, N] (out_dtype)."""
    T, H, D = q.shape
    N = k.shape[0]
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and w.dtype == torch.float32
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and k.shape[1] == D and w.shape == (T, H)
    out = torch.empty(T, N, dtype=out_dtype, device=q.device)
    if T and N:
        _index_score_kernel[(triton.cdiv(N, bn), triton.cdiv(T, bm))](
            q, k, w, out, T, N, q.stride(0), q.stride(1), k.stride(0), w.stride(0), out.stride(0),
            H=H, D=D, BM=bm, BN=bn, num_warps=num_warps, num_stages=num_stages)
    return out


def index_scores_ref(q, k, w, out_dtype=torch.bfloat16, B=64, NB=512):
    """Model._indexer's tiled torch loop (monolithic path), for tests."""
    T = q.shape[0]
    n_pad = k.shape[0]
    score = torch.empty(T, n_pad, dtype=out_dtype, device=q.device)
    for i in range(0, T, B):
        j = min(i + B, T)
        qt, wt = q[i:j], w[i:j]
        if j - i < B:
            qt = torch.cat([qt, qt.new_zeros(B - (j - i), *qt.shape[1:])])
            wt = torch.cat([wt, wt.new_zeros(B - (j - i), wt.size(1))])
        for jb in range(0, n_pad, NB):
            sc = torch.einsum("thd,nd->thn", qt, k[jb:jb + NB])
            sc = sc.float().relu_() * wt[:, :, None]
            score[i:j, jb:jb + NB] = sc.sum(dim=1)[:j - i].to(score.dtype)
    return score
