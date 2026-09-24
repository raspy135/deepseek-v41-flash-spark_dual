"""Batched sampled speculative verification, with one compact result readback.

The acceptance ratio and positive-residual distribution match the sequential verifier.
Random consumption does not: every call draws B acceptance uniforms and one categorical
uniform, including unused suffix positions. This is an opt-in sampling implementation,
not seed-for-seed equivalence with torch.multinomial/the original sequential loop.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sample_probs_batch(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Apply the existing temperature/nucleus rule independently to each vocabulary row."""
    if logits.ndim != 2 or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("sampled verification requires 2D logits, temperature > 0, 0 < top_p <= 1")
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1:
        values, ids = probs.sort(dim=-1, descending=True)
        keep = values.cumsum(dim=-1) - values < top_p
        values = torch.where(keep, values, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, ids, values)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def verify_probabilities(probs: torch.Tensor, q: torch.Tensor, drafts: torch.Tensor,
                         uniforms: torch.Tensor, stop_ids=()) -> torch.Tensor:
    """Return device [accepted_count, bonus_or_minus_one, *drafts], without scalar readbacks.

    probs is [B+1,V], q [B,V], drafts [B], uniforms [B+1] in [0,1). Inputs must be
    finite normalized probability rows and valid vocabulary IDs. Fixed uniforms make the
    stochastic decision testable independently of RNG consumption. Works on CPU or CUDA.
    """
    b = drafts.numel()
    if (b < 1 or drafts.ndim != 1 or probs.ndim != 2 or q.shape != (b, probs.shape[1])
            or probs.shape[0] != b + 1 or uniforms.shape != (b + 1,)):
        raise ValueError("expected probabilities [B+1,V], draft probabilities [B,V], IDs [B], uniforms [B+1]")
    rows = torch.arange(b, device=drafts.device)
    proposed = drafts.unsqueeze(-1)
    pd = probs[:-1].gather(1, proposed).squeeze(-1)
    qd = q.gather(1, proposed).squeeze(-1)
    accept = uniforms[:b] < (pd / qd.clamp_min(1e-20)).clamp(max=1)
    first_reject = torch.where(accept, b, rows).amin()
    is_stop = torch.zeros_like(drafts, dtype=torch.bool)
    for token in sorted(stop_ids):
        is_stop |= drafts == token
    first_stop = torch.where(is_stop & (rows < first_reject), rows, b).amin()
    stopped = first_stop < b
    accepted = torch.where(stopped, first_stop + 1, first_reject)

    # index_select keeps a GPU index on-device; scalar tensor indexing can synchronize.
    target = probs.index_select(0, first_reject.reshape(1)).squeeze(0)
    draft = q.index_select(0, first_reject.clamp_max(b - 1).reshape(1)).squeeze(0)
    residual = (target - draft).clamp_min(0)
    weights = torch.where((first_reject < b) & (residual.sum() > 0), residual, target)
    cdf = weights.cumsum(0)
    total = cdf[-1]
    # Multiplication can round U * total up to total. Keep the draw strictly inside the
    # positive mass so searchsorted cannot land on trailing zero-probability vocabulary IDs.
    threshold = torch.minimum(uniforms[-1] * total, torch.nextafter(total, torch.zeros_like(total)))
    bonus = torch.searchsorted(cdf, threshold, right=True).clamp_max(probs.shape[1] - 1)
    bonus = torch.where(stopped, -1, bonus)
    return torch.cat((accepted.reshape(1), bonus.reshape(1), drafts)).to(torch.int64)


def verify_sampled(logits, q, drafts, temperature, top_p, stop_ids=(), *, generator=None):
    probs = sample_probs_batch(logits, temperature, top_p)
    uniforms = torch.rand(drafts.numel() + 1, device=logits.device, generator=generator)
    result = verify_probabilities(probs, q, drafts, uniforms, stop_ids).cpu().tolist()
    accepted, bonus = result[:2]
    return accepted, result[2:2 + accepted], None if bonus < 0 else bonus
