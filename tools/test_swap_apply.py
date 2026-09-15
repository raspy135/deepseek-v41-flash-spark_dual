"""apply_swaps must be all-or-nothing: a partial apply desyncs the pair silently."""
import os, sys, types; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch, numpy as np
import engine.v41_engine as V

nE, WORLD = 384, 2

def make(rank, resident, fail_on=None):
    lru = {(0, e): i for i, e in enumerate(resident)}
    st = types.SimpleNamespace(lru=dict(lru), slot_key={}, null_slot=9999)
    def load(key, slot):
        if fail_on is not None and key == fail_on:
            raise RuntimeError("simulated NVMe failure")
        st.slot_key[slot] = key
    st._load_into_slot = load
    mask = {0: torch.zeros(nE, dtype=torch.bool)}
    for e in resident: mask[0][e] = True
    eng = types.SimpleNamespace(
        store=st, ep=types.SimpleNamespace(world=WORLD, rank=rank),
        model_prune_mask=mask, fast=None, _swap_baseline=0.0, expert_generation=0,
        model=types.SimpleNamespace(prune_miss_report=lambda: ({}, torch.zeros(1), None)))
    return eng, st, mask

resident = [e for e in range(0, 40, 2)]        # rank-0-owned, resident
absent   = [e for e in range(40, 80, 2)]       # rank-0-owned, pruned
good = [(0, resident[i], absent[i], 1.0) for i in range(5)]

# 1. valid plan applies fully
eng, st, mask = make(0, resident)
n = V.V41Engine.apply_swaps(eng, good)
ok = all(not mask[0][o] and mask[0][i] for _L, o, i, _g in good)
moved = all((0, i) in st.lru and (0, o) not in st.lru for _L, o, i, _g in good)
print(f"1. valid plan: applied {n}, mask correct {ok}, slots moved {moved}")
assert n == 5 and ok and moved

# 2. the peer's pairs: mask updates, arena untouched
eng1, st1, mask1 = make(1, resident)
before = dict(st1.lru)
V.V41Engine.apply_swaps(eng1, good)
print(f"2. non-owning rank: arena untouched {st1.lru == before}, "
      f"mask still updated {all(not mask1[0][o] and mask1[0][i] for _L,o,i,_g in good)}")
assert st1.lru == before and all(mask1[0][i] for _L, _o, i, _g in good)

# 3. cross-ownership pair -> reject, touch nothing
eng, st, mask = make(0, resident)
before, mbefore = dict(st.lru), mask[0].clone()
try:
    V.V41Engine.apply_swaps(eng, [(0, resident[0], 41, 1.0)])   # 41 is odd = rank 1
    print("3. FAILED: cross-ownership plan was accepted"); sys.exit(1)
except RuntimeError as e:
    print(f"3. cross-ownership rejected: {str(e)[:52]}...  arena intact {st.lru == before}, "
          f"mask intact {torch.equal(mask[0], mbefore)}")
    assert st.lru == before and torch.equal(mask[0], mbefore)

# 4. evicting a non-resident expert -> reject before any load
eng, st, mask = make(0, resident)
before, mbefore = dict(st.lru), mask[0].clone()
try:
    V.V41Engine.apply_swaps(eng, good + [(0, 900, 902, 1.0)])
    print("4. FAILED: non-resident eviction accepted"); sys.exit(1)
except RuntimeError as e:
    print(f"4. non-resident eviction rejected: arena intact {st.lru == before}, "
          f"mask intact {torch.equal(mask[0], mbefore)}  <-- the GOOD swaps were not applied either")
    assert st.lru == before and torch.equal(mask[0], mbefore)

# 5. a load that fails mid-plan -> raise, and put the evicted expert back
eng, st, mask = make(0, resident, fail_on=(0, absent[2]))
try:
    V.V41Engine.apply_swaps(eng, good)
    print("5. FAILED: load error was swallowed"); sys.exit(1)
except RuntimeError as e:
    restored = (0, resident[2]) in st.lru
    print(f"5. load failure raised ({str(e)[:38]}...), evicted expert restored {restored}")
    assert restored
print("\nALL APPLY INVARIANTS HOLD -- no partial apply reaches the router")
