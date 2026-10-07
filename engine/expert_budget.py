"""Explicit layer budgets for controlled retention experiments, off by default."""
import math


def parse_layer_counts(raw, layers, experts, topk, fraction, selection):
    if not raw.strip():
        return None
    if not fraction or not 0 < fraction < 1 or selection != 'uniform':
        raise ValueError('DSV41_PRUNE_LAYER_COUNTS requires pruned uniform selection')
    try:
        counts = tuple(int(v.strip()) for v in raw.split(','))
    except ValueError as exc:
        raise ValueError('DSV41_PRUNE_LAYER_COUNTS must contain integer counts') from exc
    if len(counts) != layers or any(v < topk or v > experts for v in counts):
        raise ValueError(f'layer counts need {layers} entries, each between {topk} and {experts}')
    expected = max(topk, math.ceil(fraction * experts)) * layers
    if sum(counts) != expected:
        raise ValueError(f'layer counts must preserve the resident budget: {sum(counts)} != {expected}')
    return counts


def resident_budget(raw, *, dynamic, layers, experts, topk, fraction):
    """Optional exact global budget; static profiles retain their existing semantics."""
    if not raw.strip():
        return None
    if not dynamic or not fraction or not 0 < fraction < 1:
        raise ValueError('DSV41_RESIDENT_EXPERTS requires dynamic pruned allocation')
    try:
        budget = int(raw)
    except ValueError as exc:
        raise ValueError('DSV41_RESIDENT_EXPERTS must be an integer') from exc
    if not layers * topk <= budget <= layers * experts:
        raise ValueError('DSV41_RESIDENT_EXPERTS outside routing floor/model capacity')
    return budget
