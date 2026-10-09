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


# DSV41_RESIDENT_EXPERTS=auto leaves this many LRU slots unassigned. Dynamic swaps reuse the
# evicted expert's slot, so nothing in that mode needs them; the hand-set budgets kept 12-14 spare
# (FP4 9,574 of 9,586, EXL3 13,500 of 13,514) and a margin costs ~0.1% of the arena.
AUTO_MARGIN = 16


def resident_budget(raw, *, dynamic, layers, experts, topk, fraction):
    """The global resident budget under dynamic allocation: 'auto' unless an exact count is given.

    Unset (or 'auto') returns the string 'auto': the count is the arena's, which does not exist yet
    at validation time, so V41Engine resolves it with auto_budget() once the store has its LRU
    slots. An integer pins an exact count (experiments). Static profiles keep their semantics."""
    if not raw.strip():
        return 'auto' if dynamic else None
    if not dynamic or not fraction or not 0 < fraction < 1:
        raise ValueError('DSV41_RESIDENT_EXPERTS requires dynamic pruned allocation')
    if raw.strip().lower() == 'auto':
        return 'auto'
    try:
        budget = int(raw)
    except ValueError as exc:
        raise ValueError('DSV41_RESIDENT_EXPERTS must be an integer') from exc
    if not layers * topk <= budget <= layers * experts:
        raise ValueError('DSV41_RESIDENT_EXPERTS outside routing floor/model capacity')
    return budget


def auto_budget(lru_slots, *, layers, experts, topk, margin=AUTO_MARGIN):
    """DSV41_RESIDENT_EXPERTS=auto: every LRU slot the arena has, less a small margin, capped at
    the whole model. Follows the expert format and ARENA_GB without hand arithmetic (an EXL3 slot
    is 0.71x an FP4 one, and a budget copied across formats left a quarter of the arena empty)."""
    budget = min(int(lru_slots) - margin, layers * experts)
    if budget < layers * topk:
        raise ValueError(f'DSV41_RESIDENT_EXPERTS=auto: {lru_slots} LRU slots are below the routing floor '
                         f'({layers * topk} + {margin} spare); raise ARENA_GB')
    return budget
