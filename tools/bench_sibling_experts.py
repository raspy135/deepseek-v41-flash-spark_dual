"""Does a sibling row actually cost as much as a fresh chain row? Measure the expert overlap.

The tree's whole case rests on this: a verify block reads each distinct (layer, expert) once, so the
bytes are the UNION of the block's routes. A sibling (`a2`, the runner-up at a position) forks from
the node it shares a parent with (`a1`), so its route may already be inside the union -- unlike a
chain's next token, which is a fresh context. If that is true the tree's extra rows are cheap; if
not, they cost exactly like chain rows and the tree cannot win.

No tree attention is needed to measure a row's routing: a candidate token at position p+1 attends to
the same prefix whether it is a sibling or a chain step, so a plain causal forward reproduces the
row exactly. Batching the rows in one kernel would not change any row's expert set -- only its
timing -- so the overlap measured here is the same overlap the one-load implementation would see.

For each sampled position p, with a1/a2 = the two candidates at p+1 and b1/b2 = the ones at p+2
given a1 (the tree's other sibling), and x = the text's own continuation:

  marginal(a2) = |route(a2) \\ route(a1)|        the root sibling
  marginal(b2) = |route(b2) \\ route(b1)|        the deep sibling
  marginal(x)  = |route(x_{p+2}) \\ route(x_{p+1})|   a chain neighbour

    python tools/bench_sibling_experts.py --max-len 192 --stride 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from engine.v41_engine import V41Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", default=None, help="file to sample positions from (default: a builtin mix)")
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--out", default="results/sibling-experts.json")
    args = ap.parse_args()

    md = os.environ.get("MODEL_DIR", os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
    eng = V41Engine(md, max_seq=4096, trace_stats=os.environ.get("TRACE_STATS", "results/trace-union/stats/coverage.json"),
                    spec=True, prune_keep=float(os.environ.get("PK", os.environ.get("PRUNE_KEEP", "0.61"))),
                    arena_gb=float(os.environ.get("AG", "90.5")),
                    transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "8")),
                    keep_free_gb=float(os.environ.get("KEEP_FREE_GB", "10")),
                    expert_format=os.environ.get("EXPERT_FORMAT", "fp4"))
    model, dev = eng.model, eng.device
    k = eng.args.n_activated_experts

    text = open(args.text).read() if args.text else (
        "def quicksort(items):\n    if len(items) <= 1:\n        return items\n"
        "    pivot = items[len(items) // 2]\n    left = [x for x in items if x < pivot]\n"
        "    return quicksort(left) + [pivot] + quicksort([x for x in items if x > pivot])\n\n"
        "The scheduler works best when the working set fits in cache; otherwise every miss costs a "
        "round trip to memory. In practice the ordering of the two loops matters more than the "
        "arithmetic, because the loads dominate and the ALU has slack. A wider verify block adds "
        "rows, and each row is another routing decision, so the expert bytes grow with the block.\n")
    ids = eng.tokenizer.encode(text, add_special_tokens=False)[:args.max_len]
    if len(ids) < 32:
        raise SystemExit("text too short")

    cap: dict[int, torch.Tensor] = {}

    def tap(name, L, t):
        if name == "route_idx":
            cap[L] = t.detach()

    model.tap = tap

    def trace(seq):
        cap.clear()
        eng._reset()
        model.begin_prompt()
        t = torch.tensor(seq, dtype=torch.long, device=dev)
        logits, _ = model.forward(t, 0, prefill=True, need_logits=True)
        return {L: cap[L].clone() for L in cap}, logits

    def last(t):                            # the last row's expert ids, per layer
        return {L: t[L][-1] for L in t}

    rows = []
    positions = list(range(8, len(ids) - 6, args.stride))
    for p in positions:
        pre = ids[:p + 1]
        _, lg = trace(pre)
        a1, a2 = int(lg[p].argmax()), int(lg[p].topk(2).indices[1])
        capA, lgA = trace(pre + [a1])                 # route(a1); logits at p+1 -> b1,b2
        b1, b2 = int(lgA[-1].argmax()), int(lgA[-1].topk(2).indices[1])
        capB, _ = trace(pre + [a1, a2])               # route(a2)  (root sibling)
        capC, _ = trace(pre + [a1, b1])               # route(b1)
        capD, _ = trace(pre + [a1, b2])               # route(b2)  (deep sibling)
        capE, _ = trace(ids[:p + 3])                  # rows p+1, p+2 = a chain neighbour pair
        rows.append(dict(a1=last(capA), a2=last(capB), b1=last(capC), b2=last(capD),
                         x1={L: capE[L][p + 1] for L in capE}, x2={L: capE[L][p + 2] for L in capE}))
        if len(rows) % 10 == 0:
            print(f"  {len(rows)}/{len(positions)} positions", flush=True)

    layers = sorted(rows[0]["a1"])

    def stats(base, alt):
        un = marg = jac = 0.0
        for r in rows:
            for L in layers:
                A = set(r[base][L].tolist()); B = set(r[alt][L].tolist())
                un += len(A | B); marg += len(B - A); jac += len(A & B) / max(len(A | B), 1)
        n = len(rows) * len(layers)
        return dict(union=round(un / n, 2), marginal=round(marg / n, 2), jaccard=round(jac / n, 3))

    chain = stats("x1", "x2")
    rep = dict(n_samples=len(rows), n_layers=len(layers), topk=k,
               root_sibling=stats("a1", "a2"), deep_sibling=stats("b1", "b2"), chain=chain)
    rep["root_ratio"] = round(rep["root_sibling"]["marginal"] / max(chain["marginal"], 1e-9), 3)
    rep["deep_ratio"] = round(rep["deep_sibling"]["marginal"] / max(chain["marginal"], 1e-9), 3)
    print(json.dumps(rep, indent=2))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
