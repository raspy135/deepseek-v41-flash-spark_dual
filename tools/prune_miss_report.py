"""Read results/prune_miss.npz (DSV41_PRUNE_MISS=1) and say what pruning is costing THIS traffic.

The prune set is ranked from a traced corpus (coding + general). On traffic unlike that corpus the
router keeps asking for experts that were dropped, and nothing downstream can see it: masking
happens before the topk, so the second-choice expert looks like an ordinary pick. This reads the
recorded preferences back and answers three things:

  * how often the router's first choice was unreachable, per layer
  * which specific experts it wanted, weighted by the score they would have carried
  * which swaps would recover the most, holding the per-layer budget fixed

Usage: python3 tools/prune_miss_report.py [npz] [--trace coverage.json] [--top N]
"""
import argparse
import json
import os
import sys

import numpy as np


def load_trace_counts(path):
    """Per-layer normalized frequencies the prune set was ranked from (v41_engine: 'sum')."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "experts", os.path.join(os.path.dirname(__file__), "..", "engine", "experts.py"))
        EX = importlib.util.module_from_spec(spec)
        sys.modules["experts"] = EX
        spec.loader.exec_module(EX)
        cc, cg = EX.category_counts(path, "coding"), EX.category_counts(path, "general")
        if len(cc) != 40 or len(cg) != 40:
            return None
        return {L: cc[L] / cc[L].sum() + cg[L] / cg[L].sum() for L in range(40)}
    except Exception as e:  # noqa: BLE001
        print(f"  (trace not loaded: {e})")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="?", default="results/prune_miss.npz")
    ap.add_argument("--trace", default="results/trace-union/stats/coverage.json")
    ap.add_argument("--keep", type=float, default=None, help="prune_keep in force (for the swap list)")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    d = np.load(a.npz)
    counts, mass = d["counts"], d["mass"]          # [n_layers, n_experts]
    nL, nE = counts.shape
    total_miss = counts.sum()
    print(f"{a.npz}: {nL} layers x {nE} experts")
    if total_miss == 0:
        print("no misses recorded -- every expert the router wanted was resident")
        return
    print(f"total missed (token, k) slots: {total_miss:,.0f}")
    print(f"distinct experts wanted but pruned: {(counts > 0).sum():,} of {nL * nE:,}")

    per_layer = counts.sum(axis=1)
    order = np.argsort(-per_layer)
    print(f"\nworst layers by missed slots:")
    for L in order[:8]:
        share = per_layer[L] / total_miss * 100
        print(f"  layer {L:>2}: {per_layer[L]:>10,.0f}  ({share:4.1f}% of all misses)  "
              f"{(counts[L] > 0).sum():>3} distinct experts")

    print(f"\ntop {a.top} experts the router wanted but could not reach "
          f"(ranked by the score mass they would have carried):")
    flat = [(mass[L, e], counts[L, e], L, e) for L in range(nL) for e in range(nE) if counts[L, e] > 0]
    flat.sort(reverse=True)
    for m, c, L, e in flat[:a.top]:
        print(f"  layer {L:>2} expert {e:>3}:  wanted {c:>9,.0f} times   score mass {m:>12,.1f}")

    trace = load_trace_counts(a.trace)
    if trace is None:
        return
    # Holding each layer's budget fixed, which kept experts would this traffic evict, and which
    # pruned ones would it promote? Rank by observed mass vs the trace score that put them in.
    print(f"\nswaps this traffic would make, holding the per-layer budget fixed:")
    shown = 0
    for L in order[:6]:
        tr = trace[L]
        wanted = [(mass[L, e], e) for e in range(nE) if counts[L, e] > 0]
        if not wanted:
            continue
        wanted.sort(reverse=True)
        # the kept set is the trace top-N; N is whatever the engine used
        n_keep = int(round((a.keep or 0.6) * nE))
        kept = set(np.argsort(-tr)[:n_keep].tolist())
        promote = [(m, e) for m, e in wanted if e not in kept][:3]
        if not promote:
            continue
        # the weakest kept experts by trace score are what a re-rank would drop
        demote = [e for e in np.argsort(tr).tolist() if e in kept][:3]
        print(f"  layer {L:>2}: promote {[int(e) for _, e in promote]} "
              f"(mass {[round(float(m)) for m, _ in promote]})  <-  drop {demote}")
        shown += 1
    if not shown:
        print("  none: every expert this traffic wanted is already kept")


if __name__ == "__main__":
    main()
