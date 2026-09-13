"""A restored prefix must survive into the decoder replay.

The prefix cache stores the tail of the replay buffer (`_rep`) and `_restore_prefix` puts it
back as one-element lists. `_decode_loop` then used to call `begin_prompt()`, which empties
`_rep`, on every request -- so the restored tail was dropped and only the recomputed suffix
refilled it. On a FULL cache hit there is no suffix at all and `_rep_tail` did `torch.cat([])`.

This pins the shape contract between the two: whatever `_restore_prefix` builds must be a valid
input to `_rep_tail` with no forward pass in between.
"""
import os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from engine.model import Model

W, DIM = 128, 64


def stub():
    m = types.SimpleNamespace()
    m.args = types.SimpleNamespace(window_size=W)
    m._rep, m._rep_end = None, 0
    for name in ("begin_prompt", "_rep_keep", "_rep_tail"):
        setattr(m, name, getattr(Model, name).__get__(m, types.SimpleNamespace))
    return m


def restored(n_rows):
    """Exactly what _restore_prefix writes: each saved tensor wrapped in a single-element list."""
    rep = {"h": torch.randn(n_rows, DIM), "pre_mix": torch.randn(n_rows, DIM),
           "topk": torch.randn(n_rows, 8), "cand": None}
    return {k: ([v.clone()] if v is not None else [None]) for k, v in rep.items()}


# 1. a full hit: restore, then straight to _rep_tail with nothing prefilled in between
m = stub()
m.begin_prompt()                       # previous request's state is dropped here, before restore
m._rep, m._rep_end = restored(W), 500
h, pre, topk, cand, start = m._rep_tail()
assert h.shape == (W, DIM), h.shape
assert cand is None
assert start == 500 - W, start
print(f"full hit:    _rep_tail -> h {tuple(h.shape)} start {start}")

# 2. a partial hit: restore, then one suffix chunk. The window must be the last W rows of the
#    JOINED sequence, not just the suffix -- that is the silent half of the bug.
m = stub()
m._rep, m._rep_end = restored(W), 500
suffix = 10
sh = types.SimpleNamespace(topk=torch.randn(suffix, 8), candidates=None)
m._rep_keep(torch.randn(suffix, DIM), torch.randn(suffix, DIM), sh, 500, suffix)
h, _, _, _, start = m._rep_tail()
assert h.shape == (W, DIM), f"replay window truncated to the suffix: {h.shape}"
assert start == 500 + suffix - W, start
print(f"partial hit: _rep_tail -> h {tuple(h.shape)} start {start} (suffix {suffix})")

# 3. and the reset really does empty it, so this test would catch the reset coming back
m = stub()
m._rep, m._rep_end = restored(W), 500
m.begin_prompt()
try:
    m._rep_tail()
except (ValueError, RuntimeError, IndexError):
    print("reset:       _rep_tail raises on an emptied buffer, as it did in production")
else:
    raise AssertionError("begin_prompt no longer empties _rep -- this test is testing nothing")

print("\nRESTORED PREFIX SURVIVES INTO THE REPLAY")
