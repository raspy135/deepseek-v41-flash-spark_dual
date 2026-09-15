"""Fused Hyper-Connection stream ops (Triton): hc_post and hc_pre+rmsnorm.

The hyper-connections carry the residual as `hc` parallel streams, so every op around them touches
hc x the data of an ordinary residual ([T, 4, 5120] is 84 MB at a 2,048-token chunk). The torch
versions in tools/v41_ref.py are written for clarity and materialise large fp32 temporaries:

    hc_pre :  sum(pre_mix.unsqueeze(-1) * x.float(), dim=1)
              -> x.float() is 168 MB, the product another 168 MB, for a 42 MB result
    hc_post:  einsum("sij,sid->sjd", comb.float(), residual.float()) + post * x.float()
              -> ~700 MB of traffic to read 126 MB and write 84 MB

Measured on the GPU timeline, the four hyper-connection sites of a layer are 33 % of prefill --
more than the MoE kernel -- while doing almost no arithmetic. These kernels do the same arithmetic
in one pass: bf16 loads, fp32 accumulation, bf16 stores, nothing materialised in between.

Arithmetic is matched to the reference exactly, including that hc_post sums over the FIRST index of
comb (`y[j] = post[j]*x + sum_i comb[i,j]*residual[i]`) -- v41_ref records that transposing it
leaves the model coherent but measurably worse, so it is not a free choice.

Both kernels are one program per (token, d-block): a row's result cannot depend on how many rows
are in the call, which is the chunk-invariance property v41_ref.tiled_rows exists to give the torch
path. Run `python engine/hc_ops.py` to check both against the reference.
"""
from __future__ import annotations

import os
import sys

import torch
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))


@triton.jit
def _hc_post_kernel(X, RES, POST, COMB, OUT, D, HC: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    db = tl.program_id(1) * BD + tl.arange(0, BD)
    m = db < D
    x = tl.load(X + t * D + db, mask=m, other=0.0).to(tl.float32)          # [BD]
    j = tl.arange(0, HC)
    post = tl.load(POST + t * HC + j).to(tl.float32)                        # [HC]
    # y[j] = post[j]*x + sum_i comb[i,j]*res[i]
    # Match v41_ref.hc_post's two-stage expression: einsum builds the residual mix first,
    # then the projected branch is added.  Starting with post*x reassociates five FP32 terms;
    # after the BF16 store that is enough to flip downstream router ties.
    acc = tl.zeros([HC, BD], dtype=tl.float32)
    for i in range(HC):
        res_i = tl.load(RES + (t * HC + i) * D + db, mask=m, other=0.0).to(tl.float32)
        c_ij = tl.load(COMB + (t * HC + i) * HC + j).to(tl.float32)         # [HC] over j
        acc += c_ij[:, None] * res_i[None, :]
    acc += post[:, None] * x[None, :]
    tl.store(OUT + (t * HC + j[:, None]) * D + db[None, :], acc.to(tl.bfloat16),
             mask=m[None, :])


def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
    """x [T, D] -> out [T, HC, D]; residual [T, HC, D], post [T, HC], comb [T, HC, HC]."""
    T, HC, D = residual.shape
    out = torch.empty(T, HC, D, dtype=torch.bfloat16, device=x.device)
    BD = 1024
    _hc_post_kernel[(T, triton.cdiv(D, BD))](
        x.contiguous(), residual.contiguous(), post.contiguous().float(), comb.contiguous().float(),
        out, D, HC=HC, BD=BD, num_warps=4)
    return out


@triton.jit
def _hc_pre_norm_kernel(X, PRE, W, SCRATCH, OUT, D, EPS, HC: tl.constexpr, BD: tl.constexpr):
    """One program per token: weighted sum over the hc streams, then rmsnorm, in one pass."""
    t = tl.program_id(0)
    j = tl.arange(0, HC)
    pre = tl.load(PRE + t * HC + j).to(tl.float32)
    ssq = tl.zeros([], dtype=tl.float32)
    for d0 in range(0, D, BD):
        db = d0 + tl.arange(0, BD)
        m = db < D
        acc = tl.zeros([BD], dtype=tl.float32)
        for i in range(HC):
            acc += tl.load(X + (t * HC + i) * D + db, mask=m, other=0.0).to(tl.float32) * tl.sum(
                tl.where(j == i, pre, 0.0))
        # hc_pre returns BF16, then rmsnorm promotes that rounded tensor back to FP32.  Keeping
        # this intermediate in FP32 is not an accuracy improvement: it changes the checkpoint's
        # calibrated computation and was the main fused-vs-reference discrepancy.
        acc = acc.to(tl.bfloat16).to(tl.float32)
        ssq += tl.sum(tl.where(m, acc * acc, 0.0))
        tl.store(SCRATCH + t * D + db, acc, mask=m)      # unnormalised, fp32 scratch
    rs = 1.0 / tl.sqrt(ssq / D + EPS)
    for d0 in range(0, D, BD):
        db = d0 + tl.arange(0, BD)
        m = db < D
        y = tl.load(SCRATCH + t * D + db, mask=m, other=0.0) * rs
        y = y * tl.load(W + db, mask=m, other=0.0).to(tl.float32)
        tl.store(OUT + t * D + db, y, mask=m)


def hc_pre_rmsnorm(x: torch.Tensor, pre_mix: torch.Tensor, w: torch.Tensor, eps: float,
                   *, direct_store: bool | None = None):
    """hc_pre(x, pre_mix) followed by rmsnorm(., w, eps). x [T, HC, D] -> [T, D] bf16."""
    T, HC, D = x.shape
    scratch = torch.empty(T, D, dtype=torch.float32, device=x.device)
    # Avoid the final FP32 write/read/cast on prefill-sized calls. The FP32 scratch,
    # BF16 hc_pre boundary, and reduction order are unchanged. Measured bit-identical
    # to the old path; small decode batches retain their original allocation/launch path.
    if direct_store is None:
        direct_store = T >= 512
    out = torch.empty(T, D, dtype=torch.bfloat16, device=x.device) if direct_store else scratch
    BD = 1024
    _hc_pre_norm_kernel[(T,)](x.contiguous(), pre_mix.contiguous().float(), w.contiguous().float(),
                              scratch, out, D, eps, HC=HC, BD=BD, num_warps=8)
    return out.to(torch.bfloat16)


if __name__ == "__main__":
    import v41_ref as R
    dev = "cuda"
    torch.manual_seed(0)
    T, HC, D = 512, 4, 5120
    x = torch.randn(T, HC, D, dtype=torch.bfloat16, device=dev)
    y = torch.randn(T, D, dtype=torch.bfloat16, device=dev)
    pre = torch.rand(T, HC, device=dev)
    post = torch.rand(T, HC, device=dev)
    comb = torch.rand(T, HC, HC, device=dev)
    w = torch.randn(D, dtype=torch.bfloat16, device=dev)

    a = R.hc_post(y, x, post, comb)
    b = hc_post(y, x, post, comb)
    print("hc_post   max rel", ((a.float() - b.float()).abs().max()
                                / a.float().abs().max()).item())
    a2 = R.rmsnorm(R.hc_pre(x, pre), w, 1e-6)
    b2 = hc_pre_rmsnorm(x, pre, w, 1e-6)
    print("hc_pre+rn max rel", ((a2.float() - b2.float()).abs().max()
                                / a2.float().abs().max()).item())
