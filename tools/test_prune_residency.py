"""Pruned mode must fail before warm start when either EP rank cannot hold its keep set."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.v41_engine import pruned_owned_counts, require_pruned_residency


ranked = [(layer, expert) for layer in range(2) for expert in range(8)]
assert pruned_owned_counts(ranked, 2) == [8, 8]
assert require_pruned_residency(ranked, 2, 8, 0.5, 16) == [8, 8]

# Both ranks execute this same check from the globally broadcast ranking.  Reporting every
# overflowing owner avoids the old asymmetric startup where one rank proceeded into a collective.
skewed = ranked + [(2, 1), (2, 3), (2, 5)]
try:
    require_pruned_residency(skewed, 2, 8, 0.6, 16)
except RuntimeError as exc:
    msg = str(exc)
    assert "rank 1 needs 11" in msg
    assert "8 LRU slots" in msg
    assert "transient ring" in msg
else:
    raise AssertionError("overcommitted pruned configuration was accepted")

print("PRUNED RESIDENCY INVARIANT HOLDS")
