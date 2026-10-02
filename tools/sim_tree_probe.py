"""Replay the DSV41_TREE_PROBE log: would a free-sibling tree have been faster?

The probe recorded, at every position of a depth-5 chain, the verifier's own argmax (`cand[i]`, the
token the target would emit -- that is what "correct" means) and the drafter's top-2 (`top2[i]`). A
tree's siblings are exactly those runner-ups, and they cost no extra drafter work: `top2[i][1]` is
already conditioned on the chain's prefix up to i.

So every shape can be replayed from one log:

  chainD   t0 -> a1 -> b1 -> ... -> z1        D drafts, rows = D + 1
  treeD    chainD + a runner-up sibling at every level; a sibling is a LEAF (no continuation),
           rows = D + 1 + D

Rows are what cost: the step is ~59.0 + 6.65 x rows ms, fit to the BLOCK=3/5/7 width sweep
(rows 4/6/8 -> 85.6/98.5/112.2 ms on the pair). So the honest readout is tok/s, not tokens/step --
a shape that adds tokens but adds rows can still lose.

    python tools/sim_tree_probe.py results/<dir>/spec-conf-rank0.json
"""
from __future__ import annotations

import argparse
import json


def chain_tokens(cand, top2, depth):
    acc = 0
    for i in range(depth):
        if i >= len(cand) or cand[i] != top2[i][0]:
            break
        acc += 1
    return acc + 1


def tree_tokens(cand, top2, depth):
    path = 0
    for i in range(depth):
        if i >= len(cand):
            break
        if cand[i] == top2[i][0]:
            path += 1
        elif cand[i] == top2[i][1]:
            return path + 2          # sibling accepted (path + 1) and its bonus token
        else:
            break
    return path + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report")
    args = ap.parse_args()
    rep = json.load(open(args.report))
    assert rep.get("tree_probe"), "collect with DSV41_TREE_PROBE=1"
    ms = lambda rows: 59.0 + 6.65 * rows
    shapes = [("chain1", "c", 1, 3), ("chain3", "c", 3, 4), ("chain5", "c", 5, 6),
              ("tree1", "t", 1, 3), ("tree2", "t", 2, 5), ("tree3", "t", 3, 6)]
    print(f"{'workload':10} {'steps':>6} | " + " ".join(f"{n:>7}" for n, _, _, _ in shapes) + "   (tok/s)")
    rate = {n: [0.0, 0] for n, _, _, _ in shapes}
    for run in rep["runs"]:
        log = run.get("tree_log") or []
        if not log:
            continue
        cells = []
        for name, kind, depth, rows in shapes:
            fn = chain_tokens if kind == "c" else tree_tokens
            tok = sum(fn(c, t, depth) for _d, _a, c, t in log) / len(log)
            rate[name][0] += tok * len(log)
            rate[name][1] += len(log)
            cells.append(f"{tok / (ms(rows) / 1000):7.2f}")
        print(f"{run['workload']:10} {len(log):6d} | " + " ".join(cells))
    print(f"{'ALL':10} {'':>6} | " +
          " ".join(f"{rate[n][0] / rate[n][1] / (ms(r) / 1000):7.2f}" for n, _, _, r in shapes))
    print("\nchainN = chain of N drafts; treeN = chainN + a runner-up sibling at every level")


if __name__ == "__main__":
    main()
