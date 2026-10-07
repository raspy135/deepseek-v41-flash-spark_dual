"""Bounded prefill rescue by predicted contribution, not routing frequency.

Calibrated unweighted output norms are a proxy for residual impact, not a
measurement of downstream answer loss. Unknown experts cannot qualify. Rank 0
plans, and every prefill layer broadcasts even an empty plan. Decode stays on
the resident graph; this deliberately does not implement decode streaming.
"""
import hashlib
import math
import os
from pathlib import Path

import numpy as np


def select(wanted, weights, keep, norms, samples, *, threshold, cap, min_samples=3):
    """Rank misses by their largest predicted share in ANY token, preserving rare impact.

    Unknown outputs use the calibrated layer median in the denominator, but never
    qualify for rescue themselves. The estimate is therefore uncertain on sparse
    calibration; the returned plan must be judged by held-out answer quality.
    """
    wanted = np.asarray(wanted, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    keep = np.asarray(keep, dtype=bool)
    norms, samples = np.asarray(norms), np.asarray(samples)
    if wanted.shape != weights.shape or wanted.ndim != 2:
        raise ValueError('wanted and weights must be matching [tokens, topk] arrays')
    if cap <= 0:
        return []
    known = (samples >= min_samples) & (norms > 0)
    if not known.any():
        return []
    estimate = np.where(known, norms, np.median(norms[known]))
    impact = weights * estimate[wanted]
    share = impact / np.maximum(impact.sum(axis=1, keepdims=True), 1e-20)
    eligible = (~keep[wanted]) & known[wanted] & (share >= threshold)
    priorities = np.zeros(len(keep), dtype=np.float64)
    np.maximum.at(priorities, wanted[eligible], share[eligible])
    # Stable ID tie break: neither request length nor occurrence count breaks ties.
    order = sorted(np.flatnonzero(priorities > 0), key=lambda e: (-priorities[e], int(e)))
    return [int(e) for e in order[:cap]]


class CriticalStream:
    def __init__(self, n_layers, n_experts, transient_slots):
        self.enabled = os.environ.get('DSV41_CRITICAL_PREFILL', '0') == '1'
        self.norms = self.samples = None
        self.digest = None
        self.threshold = float(os.environ.get('DSV41_CRITICAL_SHARE', '.25'))
        self.per_layer = int(os.environ.get('DSV41_CRITICAL_PER_LAYER', '1'))
        self.budget = int(os.environ.get('DSV41_CRITICAL_BUDGET', '8'))
        if not math.isfinite(self.threshold) or not 0 < self.threshold <= 1:
            raise ValueError('DSV41_CRITICAL_SHARE must be in (0, 1]')
        if not 0 < self.per_layer <= transient_slots or self.budget < 0:
            raise ValueError('critical load cap must fit the transient ring; budget must be nonnegative')
        if self.enabled:
            path = os.environ.get('DSV41_CRITICAL_PROFILE', '')
            if not path:
                raise ValueError('critical streaming requires a calibrated DSV41_CRITICAL_PROFILE')
            data = Path(path).read_bytes()
            self.digest = hashlib.sha256(data).hexdigest()
            with np.load(path, allow_pickle=False) as d:
                self.norms = d['norms'].astype(np.float64)
                self.samples = d['samples'].astype(np.float64)
            shape = (n_layers, n_experts)
            if (self.norms.shape != shape or self.samples.shape != shape or
                    not np.isfinite(self.norms).all() or not np.isfinite(self.samples).all() or
                    (self.norms < 0).any() or (self.samples < 0).any()):
                raise ValueError('invalid critical profile: expected finite nonnegative [layers, experts]')
        self.reset()

    def reset(self):
        self.remaining = self.budget
        self.rescued = 0
        self.layers = []

    def boot_fields(self):
        return dict(critical_prefill=self.enabled, critical_profile_sha256=self.digest,
                    critical_share=self.threshold, critical_per_layer=self.per_layer,
                    critical_budget=self.budget, critical_policy_version=1)

    def plan(self, layer, logits, scores, keep, k, ep):
        packet = None
        if ep.rank == 0:
            try:
                picks = []
                if self.remaining and (self.samples[layer] >= 3).any():
                    ids = logits.topk(k, dim=-1).indices
                    raw = scores.gather(1, ids)
                    picks = select(ids.cpu().numpy(), raw.cpu().numpy(), keep.cpu().numpy(),
                        self.norms[layer], self.samples[layer], threshold=self.threshold,
                        cap=min(self.per_layer, self.remaining))
                packet = dict(picks=picks)
            except Exception as exc:
                packet = dict(error=type(exc).__name__)
        # Both ranks reach this for EVERY backbone prefill layer, including no-data/no-budget.
        packet = ep.broadcast_obj(packet)
        if 'error' in packet:
            raise RuntimeError('critical streaming planner failed: ' + packet['error'])
        picks = packet['picks']
        self.remaining -= len(picks)
        self.rescued += len(picks)
        if picks:
            self.layers.append(dict(layer=int(layer), experts=picks))
        return picks

    def report(self):
        return dict(enabled=self.enabled, rescued_layer_experts=self.rescued,
                    budget=self.budget, remaining=self.remaining, layers=self.layers,
                    scope='prefill', importance='calibrated output-norm proxy')
