"""What is the expert pruning costing THIS server's traffic?

Reads the persisted routing-demand database (DSV41_PRUNE_MISS=1 records it; it accumulates
across restarts) and compares it against the keep set currently in force.

The prune set ships ranked from a traced corpus (coding + general, combined with "sum", which by
its own comment drops each workload's specialists -- the two top sets overlap by a Jaccard of only
0.18-0.31). This says whether that ranking fits what the server is actually asked to do.

Usage: python3 tools/prune_miss_report.py [db.npz] [--keep 0.6] [--top 12]
"""
import argparse
import os
import sys

import numpy as np


def trace_counts(path):
    """The shipped per-layer ranking (v41_engine's DSV41_PRUNE_RANK='sum')."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "experts", os.path.join(os.path.dirname(__file__), "..", "engine", "experts.py"))
    EX = importlib.util.module_from_spec(spec)
    sys.modules["experts"] = EX
    spec.loader.exec_module(EX)
    cc, cg = EX.category_counts(path, "coding"), EX.category_counts(path, "general")
    if len(cc) != 40 or len(cg) != 40:
        raise ValueError("per-layer trace npz files not found next to coverage.json")
    return {L: cc[L] / cc[L].sum() + cg[L] / cg[L].sum() for L in range(40)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", nargs="?", default=os.environ.get("DSV41_PRUNE_DB", "results/prune_demand.npz"))
    ap.add_argument("--trace", default="results/trace-union/stats/coverage.json")
    ap.add_argument("--keep", type=float, default=0.6)
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    if not os.path.exists(a.db):
        print(f"no demand database at {a.db}\n"
              f"run the server with DSV41_PRUNE_MISS=1 and put traffic through it first.")
        return
    d = np.load(a.db)
    demand, mass = d["counts"], d["mass"]
    nL, nE = demand.shape
    total = demand.sum()
    print(f"{a.db}: {nL} layers x {nE} experts, {total:,.0f} recorded routing slots")
    if total == 0:
        print("database is empty")
        return

    try:
        tr = trace_counts(a.trace)
    except Exception as e:  # noqa: BLE001
        print(f"cannot load the shipped trace ({e}) -- reporting demand only")
        tr = None

    n_keep = int(round(a.keep * nE))
    if tr is None:
        return
    # what the SHIPPED ranking keeps, and how much of this server's demand it serves
    served, lost, per_layer_loss = 0.0, 0.0, []
    swaps = []
    for L in range(nL):
        kept = np.argsort(-tr[L])[:n_keep]
        keep_mask = np.zeros(nE, bool)
        keep_mask[kept] = True
        s = demand[L][keep_mask].sum()
        m = demand[L][~keep_mask].sum()
        served += s
        lost += m
        per_layer_loss.append(m / max(demand[L].sum(), 1e-9))
        # which pruned experts does this traffic want most, and which kept ones does it not use?
        want = [(demand[L][e], e) for e in range(nE) if not keep_mask[e] and demand[L][e] > 0]
        want.sort(reverse=True)
        unused = [(demand[L][e], e) for e in range(nE) if keep_mask[e]]
        unused.sort()
        if want:
            swaps.append((per_layer_loss[L], L, want[:3], unused[:3]))

    print(f"\nagainst the shipped ranking at keep={a.keep:.2f} ({n_keep}/{nE} per layer):")
    print(f"  routing demand served by resident experts : {served/total*100:5.1f}%")
    print(f"  routing demand landing on pruned experts  : {lost/total*100:5.1f}%   <-- the cost")

    order = np.argsort(-np.array(per_layer_loss))
    print(f"\nworst layers (share of that layer's demand that cannot be routed):")
    for L in order[:8]:
        print(f"  layer {L:>2}: {per_layer_loss[L]*100:5.1f}%   "
              f"{int((demand[L] > 0).sum()):>3} experts wanted, "
              f"{int(((demand[L] > 0) & ~np.isin(np.arange(nE), np.argsort(-tr[L])[:n_keep])).sum()):>3} of them pruned")

    print(f"\nswaps this traffic would make, holding each layer's budget fixed:")
    swaps.sort(reverse=True)
    for loss, L, want, unused in swaps[:a.top]:
        pr = [int(e) for _, e in want]
        dr = [int(e) for _, e in unused]
        print(f"  layer {L:>2} ({loss*100:4.1f}% lost): promote {pr}  <-  drop {dr}")
    if not swaps:
        print("  none -- every expert this traffic wants is already resident")
    print(f"\nThe running server applies this automatically (DSV41_PRUNE_ADAPT=1, the default):\n"
          f"the blend weight rises as the database grows, so this report is what it is acting on.")


if __name__ == "__main__":
    main()
