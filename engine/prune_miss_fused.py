"""One-launch DSV41_PRUNE_MISS accounting for decode-sized router blocks.

`Model._record_prune_miss` spells the bookkeeping as ~20 small torch kernels -- a second topk and
its sort, float64 casts, three scatter_adds, a mask gather, sums and in-place adds. The decode
graphs replay it once per layer, so a verify step pays ~800 launches for it (2026-09-16 trace,
~1.5 ms of a ~129 ms iteration). This kernel does the same accounting in one launch per layer.

It touches only the demand database; routing, logits and tokens are unchanged by construction.
Rank 0 plans swaps from its own database and broadcasts the plan (V41Engine.plan_swaps path), so
a small accumulation-order difference between the two spellings cannot split the pair; it is in
the boot guard anyway.

Differences from the torch spelling, all confined to the database:
* One program reduces the block's picks per expert and adds that total once. The torch
  scatter_add adds one pick at a time in atomic order. Integer-valued counts are identical; a
  decayed (non-integer) count or the float64 score mass can differ in the last ulp.
* Ties: lowest expert index wins. torch.topk's tie order is unspecified. The logits are
  continuous fp32 plus a bias, so a tie at the k-th place is not expected in practice.
* NaN logits are not selected (torch.topk ranks NaN highest). A NaN router logit is already a
  broken model; the database is not where that should surface.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# One CTA holds the whole block: fine for a verify block, wrong for a 2k-row prefill chunk.
MAX_ROWS = 16
NUM_WARPS = 4


@triton.jit
def _prune_miss_kernel(LOGITS, SCORES, KEEP, COUNTS, MASS, PHASE, MISS_TOT, MISS_PHASE,
                       T, E, stride_l, stride_s,
                       K: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_E: tl.constexpr):
    # All rows at once as one [BLOCK_T, BLOCK_E] tile: K row-wise argmax passes, not T * K.
    rows = tl.arange(0, BLOCK_T)
    offs = tl.arange(0, BLOCK_E)
    rvalid = rows < T
    valid = offs < E
    m2 = rvalid[:, None] & valid[None, :]
    lg = tl.load(LOGITS + rows[:, None] * stride_l + offs[None, :], mask=m2, other=float("-inf"))
    sc = tl.load(SCORES + rows[:, None] * stride_s + offs[None, :], mask=m2, other=0.0)
    picked = tl.zeros([BLOCK_T, BLOCK_E], dtype=tl.int1)
    for _ in tl.static_range(K):
        top = tl.max(lg, axis=1)
        pick = tl.min(tl.where(lg == top[:, None], offs[None, :], BLOCK_E), axis=1)
        hit = (offs[None, :] == pick[:, None]) & rvalid[:, None]
        picked = picked | hit
        lg = tl.where(hit, float("-inf"), lg)
    cnt = tl.sum(picked.to(tl.float64), axis=0)
    mass = tl.sum(tl.where(picked, sc.to(tl.float64), 0.0), axis=0)
    keep = tl.load(KEEP + offs, mask=valid, other=1)
    tl.store(COUNTS + offs, tl.load(COUNTS + offs, mask=valid, other=0.0) + cnt, mask=valid)
    tl.store(MASS + offs, tl.load(MASS + offs, mask=valid, other=0.0) + mass, mask=valid)
    tl.store(PHASE + offs, tl.load(PHASE + offs, mask=valid, other=0.0) + cnt, mask=valid)
    missed = tl.sum(tl.where(keep == 0, cnt, 0.0), axis=0)
    slots = tl.full([], T * K, tl.float64)   # T == 1 arrives as a specialized Python int
    tl.store(MISS_TOT, tl.load(MISS_TOT) + missed)
    tl.store(MISS_TOT + 1, tl.load(MISS_TOT + 1) + slots)
    tl.store(MISS_PHASE, tl.load(MISS_PHASE) + missed)
    tl.store(MISS_PHASE + 1, tl.load(MISS_PHASE + 1) + slots)


def supported(logits: torch.Tensor, scores: torch.Tensor, keep_mask: torch.Tensor) -> bool:
    return (logits.is_cuda and logits.dim() == 2 and 0 < logits.size(0) <= MAX_ROWS
            and logits.dtype == torch.float32 and scores.dtype == torch.float32
            and logits.stride(-1) == 1 and scores.stride(-1) == 1
            and scores.shape == logits.shape
            and keep_mask.dtype == torch.bool and keep_mask.is_contiguous()
            and keep_mask.numel() == logits.size(1))


def record(logits, scores, keep_mask, k: int, counts, mass, phase, miss_tot, miss_phase):
    """Accumulate one block: `counts`/`mass`/`phase` are this layer's [E] float64 rows,
    `miss_tot`/`miss_phase` its [2] float64 (missed, slots) rows. All updated in place."""
    T, E = logits.shape
    for t in (counts, mass, phase, miss_tot, miss_phase):
        assert t.dtype == torch.float64 and t.is_contiguous() and t.device == logits.device
    assert counts.numel() == mass.numel() == phase.numel() == E and 0 < k <= E
    _prune_miss_kernel[(1,)](logits, scores, keep_mask.view(torch.uint8),
                             counts, mass, phase, miss_tot, miss_phase,
                             T, E, logits.stride(0), scores.stride(0),
                             K=k, BLOCK_T=max(2, triton.next_power_of_2(T)),
                             BLOCK_E=triton.next_power_of_2(E), num_warps=NUM_WARPS)
