"""Topic-specific resident selection; no changes to expert weights or inference kernels.

maxmin_counts is ported from 0xBakeer/deepseek-v41-flash-spark at
45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f (MIT; see LICENSE).
Admission priorities must never be blended with observed router-score histories.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SOURCES = ('counts', 'saliency')
RANKS = ('sum', 'max', 'maxmin')
VERSION = 1


def load_histograms(path, source, topics=(), n_layers=40, n_experts=384):
    """Read a complete selected family; never substitute frequency for saliency."""
    if source not in SOURCES:
        raise ValueError(f'unknown expert histogram source {source!r}')
    raw = Path(path).read_bytes()
    pl = json.loads(raw)['per_layer']
    if set(pl) != {str(l) for l in range(n_layers)}:
        raise ValueError(f'expert profile requires exactly layers 0..{n_layers-1}')
    available = sorted(k[len('counts_'):] for k in pl['0'] if k.startswith('counts_'))
    chosen = tuple(dict.fromkeys(topics)) if topics else tuple(available)
    if not chosen:
        raise ValueError('expert profile has no topic histograms')
    per = {}
    for topic in chosen:
        if topic not in available:
            raise ValueError(f'unknown expert topic {topic!r}; available: {available}')
        values = {}
        for layer in range(n_layers):
            key = f'{source}_{topic}'
            if key not in pl[str(layer)]:
                raise ValueError(f'missing {key} at L{layer}; no histogram-family fallback')
            c = np.asarray(pl[str(layer)][key], dtype=np.float64)
            if (c.shape != (n_experts,) or not np.isfinite(c).all() or
                    (c < 0).any() or not np.isfinite(c.sum())):
                raise ValueError(f'invalid {key} at L{layer}')
            values[layer] = c
        per[topic] = values
    return per, hashlib.sha256(raw).hexdigest()


def mask_digest(masks):
    if masks is None:
        return 'all-experts'
    h = hashlib.sha256()
    for layer in sorted(masks):
        h.update(int(layer).to_bytes(4, 'little'))
        m = masks[layer]
        if hasattr(m, 'detach'):
            m = m.detach().cpu().numpy()
        h.update(np.asarray(m, dtype=np.bool_).tobytes())
    return h.hexdigest()


def maxmin_counts(per: dict, frac: float, n_layers: int = 40, n_experts: int = 384) -> dict:
    """Per-layer scores whose top-N is a water-filling allocation across topics.

    Each layer's slots are handed out one at a time to whichever selected topic currently has the
    least of its routing mass covered, which is the greedy solution to "make the worst-served topic
    as well served as possible". An expert admitted for one topic counts for every topic that also
    routes to it, so overlap is not paid for twice and the topics converge on a common coverage
    rather than a spread.

    The result is returned as a score vector, not a set, so the caller's existing top-N selection
    and warm-start ordering work unchanged: an admitted expert scores above every rejected one and
    they are ordered by admission, while rejected experts keep their summed score squashed below 1
    so the warm start still fills the tail in a sensible order.
    """
    import math as _m
    n_keep = max(6, _m.ceil(frac * n_experts))
    topics = list(per)
    out = {}
    for L in range(n_layers):
        p = {}
        for t in topics:
            c = np.asarray(per[t][L], dtype=np.float64)
            tot = c.sum()
            p[t] = c / tot if tot > 0 else c
        order = {t: np.argsort(p[t])[::-1] for t in topics}
        ptr = {t: 0 for t in topics}
        # A topic with no mass in this layer would otherwise be the least covered forever and hand
        # every slot to its argsort of zeros; it has nothing to ask for, so it does not vote here.
        got = {t: (0.0 if p[t].sum() > 0 else float("inf")) for t in topics}
        if all(np.isinf(got[t]) for t in topics):
            out[L] = np.zeros(n_experts)
            continue
        admitted, seen = [], set()
        while len(admitted) < n_keep:
            t = min(topics, key=lambda t: got[t])
            while ptr[t] < n_experts and int(order[t][ptr[t]]) in seen:
                ptr[t] += 1
            if ptr[t] >= n_experts:
                # this topic has nothing left to ask for; take it out of the running
                got[t] = float("inf")
                if all(np.isinf(got[u]) for u in topics):
                    break
                continue
            e = int(order[t][ptr[t]]); ptr[t] += 1
            seen.add(e); admitted.append(e)
            for u in topics:
                got[u] += p[u][e]
        s = sum(p[t] for t in topics)
        score = s / (s.max() + 1e-12) * 0.999          # every rejected expert scores below 1.0
        for i, e in enumerate(admitted):
            score[e] = 1.0 + (len(admitted) - i)       # admitted, hottest first, all above 1.0
        out[L] = score
    return out

class ExpertProfile:
    def __init__(self, env, n_layers=40, n_experts=384):
        self.source = (env.get('DSV41_PRUNE_SOURCE') or 'counts').strip()
        self.rank = (env.get('DSV41_PRUNE_RANK') or 'sum').strip()
        self.requested = tuple(dict.fromkeys(t.strip() for t in
            (env.get('DSV41_EXPERT_TOPICS') or '').split(',') if t.strip()))
        if self.source not in SOURCES or self.rank not in RANKS:
            raise ValueError('expert selection requires source counts|saliency and rank sum|max|maxmin')
        self.static = self.source != 'counts' or self.rank == 'maxmin' or bool(self.requested)
        self.n_layers, self.n_experts = n_layers, n_experts
        self.topics = self.requested or (() if self.static else ('coding', 'general'))
        self.digest = None
        self.histograms = None
        self.coverage = None

    def validate(self, fraction, selection, layer_counts, adapt):
        if not self.static:
            return
        if not fraction or not 0 < fraction < 1:
            raise ValueError('topic expert profiles require pruned residency')
        if self.rank == 'maxmin' and (selection != 'uniform' or layer_counts is not None):
            raise ValueError('maxmin requires uniform selection and equal layer budgets')
        if adapt.use_db or adapt.swap or adapt.swap_prefill or adapt.decode_tokens or adapt.urgent:
            raise ValueError('topic expert profiles require PRUNE_ADAPT=0 and every swap trigger off; '
                             'router histories cannot replace saliency or admission priorities')

    def scores(self, path, fraction):
        if not self.static:
            raise ValueError('legacy selection uses the existing trace/demand path')
        per, self.digest = load_histograms(path, self.source, self.requested,
                                           self.n_layers, self.n_experts)
        self.histograms, self.topics = per, tuple(per)
        if self.rank == 'maxmin':
            return maxmin_counts(per, fraction, self.n_layers, self.n_experts)
        out = {}
        for layer in range(self.n_layers):
            values = []
            for c in per.values():
                row = c[layer]
                values.append(row / row.sum() if row.sum() > 0 else row)
            out[layer] = sum(values) if self.rank == 'sum' else np.maximum.reduce(values)
        return out

    def measure_coverage(self, keep):
        if self.histograms is None:
            return
        self.coverage = {}
        for topic, hist in self.histograms.items():
            values = [float(c[keep[layer]].sum() / c.sum()) for layer, c in hist.items() if c.sum() > 0]
            self.coverage[topic] = {'mean_layer_coverage': float(np.mean(values)) if values else None,
                                    'min_layer_coverage': min(values) if values else None}

    def boot_fields(self):
        return {'expert_profile_version': VERSION, 'prune_source': self.source,
                'prune_rank': self.rank, 'expert_topics': self.topics,
                'expert_profile_sha256': self.digest, 'expert_profile_static': self.static}

    def report(self):
        return {**self.boot_fields(), 'expert_topics': list(self.topics),
                'expert_topic_coverage': self.coverage}
