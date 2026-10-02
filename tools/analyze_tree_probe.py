"""Would a second candidate rescue a miss? Reads the DSV41_TREE_PROBE log.

For every verified position it has the verifier's own argmax (`cand[i]` -- the token the target would
have produced, which is what "correct" means here) and the drafter's top-2 (`top2[i]`). Position i is
reached only when positions 0..i-1 were accepted, and the chain stops at its first miss, so:

  reached(i) = a >= i
  hit(i)     = a > i          (position i was accepted)
  rescue(i)  = a == i and cand[i] == top2[i][1]   -- the #2 candidate held the target's token

The rescue rate is the number a shallow tree has to beat: a second branch costs a row (worth ~0.35-0.45
committed tokens at our step cost), and it only pays where misses are common and rescuable.

    python tools/analyze_tree_probe.py results/<dir>/spec-conf-rank0.json
"""
from __future__ import annotations

import argparse
import collections
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report")
    args = ap.parse_args()
    rep = json.load(open(args.report))
    assert rep.get("tree_probe"), "collect with DSV41_TREE_PROBE=1"
    print(f"{'workload':10} {'steps':>6} {'accepts/step':>12} | per position: reached, hit%, rescue-of-miss%, in-top2%")
    agg = collections.defaultdict(lambda: [0, 0, 0, 0])   # reached, hit, miss, rescued
    for run in rep["runs"]:
        log = run.get("tree_log") or []
        if not log:
            continue
        depth = len(log[0][2])
        per = [[0, 0, 0, 0] for _ in range(depth)]
        for d, a, cand, top2 in log:
            for i in range(min(d, len(cand))):
                if a < i:
                    continue                       # the chain stopped before this position
                per[i][0] += 1
                agg[i][0] += 1
                if a > i:                          # position i was accepted
                    per[i][1] += 1
                    agg[i][1] += 1
                else:                              # a == i: the chain stopped here
                    per[i][2] += 1
                    agg[i][2] += 1
                    if cand[i] == top2[i][1]:      # ... and the #2 candidate had the target's token
                        per[i][3] += 1
                        agg[i][3] += 1
        acc = sum(a for _, a, _, _ in log) / len(log)
        cells = []
        for i, (r, h, m, s) in enumerate(per):
            if not r:
                cells.append(f"p{i}: -")
                continue
            cells.append(f"p{i}: {r} {100*h/r:.0f}% res{100*s/m if m else 0:.0f}% (miss {m})")
        print(f"{run['workload']:10} {len(log):6d} {acc:12.2f} | " + "  ".join(cells))
    print("\nall runs, pooled per position:")
    for i, vals in sorted(agg.items()):
        r, h, m, s = vals
        if not r:
            continue
        print(f"  position {i}: reached {r:5d}  hit {100*h/r:5.1f}%  first-miss {m:5d}  "
              f"of which #2 held the target: {100*s/m if m else 0:5.1f}%")


if __name__ == "__main__":
    main()
