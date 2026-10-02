"""Replay the DSV41_TREE_PROBE log: acceptance rate and throughput, chain vs free-sibling tree.

The probe recorded, at every position of a depth-5 chain, the verifier's own argmax (`cand[i]`, the
token the target would emit) and the drafter's top-2 (`top2[i]`). The siblings in a tree are exactly
those runner-ups, and they cost no extra drafter work: `top2[i][1]` is already conditioned on the
chain's prefix up to i.

Readouts from one log:

  tokens/step   accepted drafts + the bonus token
  tok/s         tokens/step over the measured row cost (~59.0 + 6.65 x rows ms, fit to the BLOCK
                3/5/7 sweep: rows 4/6/8 -> 85.6/98.5/112.2 ms). A row is ~2.7 distinct experts.
  accept rate   accepted drafts / proposed drafts. A tree proposes more, so it must accept
                proportionally more to tie on tokens.
  rescue        steps where the top-1 missed and the runner-up held -- the only steps a tree can
                extend that a chain cannot.

    python tools/sim_tree_probe.py results/<dir>/spec-conf-rank0.json
"""
from __future__ import annotations

import argparse
import json


def step_metrics(cand, top2, depth, sib):
    """(chain_accepted, tree_accepted, sibling_used, tree_proposed) for one step."""
    chain = 0
    for i in range(depth):
        if i >= len(cand) or cand[i] != top2[i][0]:
            break
        chain += 1
    path, used = 0, 0
    for i in range(depth):
        if i >= len(cand):
            break
        if cand[i] == top2[i][0]:
            path += 1
        elif i in sib and cand[i] == top2[i][1]:
            path, used = path + 1, 1
            break
        else:
            break
    return chain, path, used, depth + len(sib)


ARMS = [("chain3", 3, (), 4), ("chain5", 5, (), 6), ("tree3_rb", 3, (0, 1), 6)]
MS = lambda rows: 59.0 + 6.65 * rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report")
    args = ap.parse_args()
    rep = json.load(open(args.report))
    assert rep.get("tree_probe"), "collect with DSV41_TREE_PROBE=1"
    hdr = (f"{'workload':10} {'steps':>6} | {'tokens/step':^27} | {'tok/s':^27} | "
           f"{'accept rate (acc/prop)':^27} | rescue")
    print(hdr)
    tot = {nm: {"t": 0.0, "a": 0.0, "n": 0} for nm, *_ in ARMS}
    tot_resc = [0, 0]
    for run in rep["runs"]:
        log = run.get("tree_log") or []
        if not log:
            continue
        n = len(log)
        tok = {nm: 0.0 for nm, *_ in ARMS}
        acc = {nm: 0.0 for nm, *_ in ARMS}
        resc = 0
        for _d, _a, cand, top2 in log:
            for nm, depth, sib, rows in ARMS:
                c, p, u, prop = step_metrics(cand, top2, depth, sib)
                accepted = p if sib else c
                tok[nm] += accepted + 1
                acc[nm] += accepted / (prop if sib else depth)
                if nm == "tree3_rb":
                    resc += u
        print(f"{run['workload']:10} {n:6d} | " +
              " ".join(f"{tok[nm]/n:8.2f} " for nm, *_ in ARMS) + "| " +
              " ".join(f"{tok[nm]/(MS(r)/1000)/n:8.2f} " for nm, _d, _s, r in ARMS) + "| " +
              " ".join(f"{acc[nm]/n*100:8.1f}% " for nm, *_ in ARMS) + f"| {resc/n*100:5.1f}%")
        for nm, _d, _s, rows in ARMS:
            tot[nm]["t"] += tok[nm]
            tot[nm]["a"] += acc[nm]
            tot[nm]["n"] += n
        tot_resc[0] += resc
        tot_resc[1] += n
    n = tot["chain3"]["n"]
    print(f"{'ALL':10} {n:6d} | " +
          " ".join(f"{tot[nm]['t']/n:8.2f} " for nm, *_ in ARMS) + "| " +
          " ".join(f"{tot[nm]['t']/n/(MS(r)/1000):8.2f} " for nm, _d, _s, r in ARMS) + "| " +
          " ".join(f"{tot[nm]['a']/n*100:8.1f}% " for nm, *_ in ARMS) +
          f"| {tot_resc[0]/tot_resc[1]*100:5.1f}%")
    print("\nchainN = chain of N drafts (N+1 rows); tree3_rb = chain3 + a2 + b2 (6 rows)")

    # Per-node acceptance. For a tree the branches are parallel, so count each node's own hit; the
    # accepted length is the sum, and accepted/proposed would serialise them.
    print("\ntree node acceptance (prose/code)")
    print(f"{'class':8} {'steps':>6} | {'P(a1)':>6} {'P(a2)':>6} {'P(b1|a1)':>8} {'P(b2|a1)':>8} "
          f"{'P(c1|a1b1)':>10} | {'accepted':>8} {'chain3':>7}")
    for cls, pred in (("prose", lambda w: w in ("explain", "story")),
                      ("code", lambda w: w not in ("explain", "story"))):
        n = c3 = a1s = a2s = b1s = b2s = c1s = 0.0
        for run in rep["runs"]:
            if not pred(run["workload"]):
                continue
            for _d, _a, cand, top2 in run.get("tree_log") or []:
                a1 = cand[0] == top2[0][0]
                b1 = a1 and len(cand) > 1 and cand[1] == top2[1][0]
                b2 = a1 and len(cand) > 1 and cand[1] == top2[1][1]
                c1 = b1 and len(cand) > 2 and cand[2] == top2[2][0]
                ch = 0
                for i in range(3):
                    if i < len(cand) and cand[i] == top2[i][0]:
                        ch += 1
                    else:
                        break
                c3 += ch
                a1s += a1
                a2s += cand[0] == top2[0][1]
                b1s += b1
                b2s += b2
                c1s += c1
                n += 1
        accepted = (a1s + a2s + b1s + b2s + c1s) / n
        print(f"{cls:8} {int(n):6d} | {a1s/n:6.3f} {a2s/n:6.3f} {b1s/a1s:8.3f} {b2s/a1s:8.3f} "
              f"{c1s/b1s:10.3f} | {accepted:8.3f} {c3/n:7.3f}")

    # Per-position coverage (conditioned on the chain prefix): how much the runner-up adds.
    cov = {}
    for run in rep["runs"]:
        for _d, _a, cand, top2 in run.get("tree_log") or []:
            for i in range(min(len(cand), len(top2))):
                s = cov.setdefault(i, [0, 0, 0, 0])
                s[0] += 1
                s[1] += cand[i] == top2[i][0]
                s[2] += cand[i] == top2[i][1]
                s[3] += cand[i] != top2[i][0] and cand[i] == top2[i][1]
    print("\nper-position (chain prefix held fixed)")
    print(f"{'pos':>3} {'n':>6}  {'top1':>7}  {'top2':>7}  {'top1|top2':>9}  {'rescue|miss':>11}  {'miss':>6}")
    for i in sorted(cov):
        n, h1, h2, r = cov[i]
        print(f"{i:3d} {n:6d}  {h1/n*100:6.1f}%  {h2/n*100:6.1f}%  {(h1+r)/n*100:8.1f}%  "
              f"{r/max(n-h1,1)*100:10.1f}%  {(1-h1/n)*100:5.1f}%")


if __name__ == "__main__":
    main()
