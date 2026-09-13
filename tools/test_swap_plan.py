"""Stage 2 planner invariants, against a stub engine -- no model, no server, no CUDA needed."""
import os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch
import engine.v41_engine as V

nL, nE, WORLD, KEEP = 8, 384, 2, 0.6
rng = np.random.default_rng(17)

trace = {L: rng.random(nE) ** 3 for L in range(nL)}
# the keep set the trace produces, per ownership class, with a fixed budget each
mask = {}
for L in range(nL):
    m = torch.zeros(nE, dtype=torch.bool)
    for r in range(WORLD):
        own = np.flatnonzero(np.arange(nE) % WORLD == r)
        n_keep = int(round(KEEP * len(own)))
        m[own[np.argsort(-trace[L][own])[:n_keep]]] = True
    mask[L] = m

# demand that disagrees with the trace: this server has its own specialists
demand = np.zeros((nL, nE))
for L in range(nL):
    fav = rng.choice(nE, 100, replace=False)
    demand[L, fav] = rng.random(100) * 1e6

stub = types.SimpleNamespace(
    _prune_trace=trace,
    model_prune_mask=mask,
    ep=types.SimpleNamespace(world=WORLD, rank=0),
    # A real rank-0 arena contains only even experts.  Planning still has to cover odd experts:
    # rank 0 sends one global plan to rank 1, which applies its own half.
    store=types.SimpleNamespace(lru={(L, int(e)): 1 for L, m in mask.items()
                                     for e in torch.where(m)[0].tolist() if e % WORLD == 0}),
    model=types.SimpleNamespace(
        prune_miss_report=lambda: ({}, torch.tensor(demand), torch.tensor(demand))),
)
os.environ["DSV41_PRUNE_PRIOR"] = "1e3"     # make the observed half dominate for the test
swaps = V.V41Engine.plan_swaps(stub, max_swaps=200)
print(f"planned {len(swaps)} swaps")
assert swaps, "planner produced nothing on a DB that clearly disagrees with the trace"

# 1. every swap stays inside one ownership class
bad = [(L, o, i) for L, o, i, _ in swaps if (o % WORLD) != (i % WORLD)]
print(f"1. swaps crossing ownership classes: {len(bad)}  (must be 0)")
assert not bad

# 1b. rank 0 is the sole planner, so the plan must cover both ranks despite its local-only LRU.
owners = {o % WORLD for _L, o, _i, _g in swaps}
print(f"1b. ownership classes represented: {sorted(owners)}  (must be [0, 1])")
assert owners == set(range(WORLD))

# 2. every swap is a strict improvement, and they are ordered best-first
gains = [g for *_x, g in swaps]
print(f"2. all gains positive: {all(g > 0 for g in gains)}   ordered best-first: {gains == sorted(gains, reverse=True)}")
assert all(g > 0 for g in gains) and gains == sorted(gains, reverse=True)

# 3. no expert is both promoted and demoted, and none appears twice
seen = {}
dup = 0
for L, o, i, _ in swaps:
    for e in (o, i):
        if (L, e) in seen:
            dup += 1
        seen[(L, e)] = 1
print(f"3. experts touched more than once: {dup}  (must be 0 -- a slot moved twice would corrupt)")
assert dup == 0

# 4. THE invariant: applying every swap leaves each (layer, rank) budget unchanged
before = {(L, r): int((mask[L].numpy() & (np.arange(nE) % WORLD == r)).sum())
          for L in range(nL) for r in range(WORLD)}
for L, o, i, _ in swaps:
    mask[L][o] = False
    mask[L][i] = True
after = {(L, r): int((mask[L].numpy() & (np.arange(nE) % WORLD == r)).sum())
         for L in range(nL) for r in range(WORLD)}
drift = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
print(f"4. (layer, rank) budgets that drifted: {len(drift)}  (must be 0)")
assert not drift, drift

# 5. the swaps actually chase demand: promoted experts outrank demoted ones on observed demand
promoted = np.mean([demand[L, i] for L, _o, i, _ in swaps])
demoted = np.mean([demand[L, o] for L, o, _i, _ in swaps])
print(f"5. mean observed demand: promoted {promoted:,.0f}  vs demoted {demoted:,.0f}")
assert promoted > demoted
print("\nALL STAGE-2 PLANNER INVARIANTS HOLD")
