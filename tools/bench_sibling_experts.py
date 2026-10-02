"""Measure the routed-expert bytes of the proposed six-row DSpark tree.

The proposal replaces the last two rows of a depth-5 chain

    root, a1, b1, c1, d1, e1

with runner-up leaves that DSpark already computes while choosing the main chain:

    root, a1, b1, c1, a2, b2

where a2 has prefix ``root`` and b2 has prefix ``root, a1``. Expert weights are
read once per distinct (layer, expert) in a verify block, so the decision metric
is the complete per-layer union for those six rows, not pairwise overlap.

This benchmark deliberately obtains a1..e1 and a2/b2 from the production
FastDecoder DSpark draft. It then runs the target backbone on the main chain and
on the two correct sibling prefixes. The old benchmark instead chose candidates
from target-model logits and accidentally evaluated a2 after ``root,a1,a2``;
both errors made its result unsuitable for deciding whether to build the tree.

Run on the idle pair with the production prune configuration and a five-token
draft (the script refuses shorter drafts):

    DSV41_BLOCK=5 DSV41_BLOCK_DYNAMIC=off \
      python tools/bench_sibling_experts.py --max-len 192 --stride 4
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

# FastDecoder records DSpark's second choice only when this observation-only flag
# is set at import time. It does not alter candidate selection or target routing.
os.environ["DSV41_TREE_PROBE"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from engine.fastdecode import T_DRAFT, TREE_PROBE  # noqa: E402
from engine.v41_engine import MAX_CHUNK, V41Engine  # noqa: E402


BUILTIN_TEXT = (
    "def quicksort(items):\n    if len(items) <= 1:\n        return items\n"
    "    pivot = items[len(items) // 2]\n    left = [x for x in items if x < pivot]\n"
    "    return quicksort(left) + [pivot] + quicksort([x for x in items if x > pivot])\n\n"
    "The scheduler works best when the working set fits in cache; otherwise every miss costs a "
    "round trip to memory. In practice the ordering of the two loops matters more than the "
    "arithmetic, because the loads dominate and the ALU has slack. A wider verify block adds "
    "rows, and each row is another routing decision, so the expert bytes grow with the block.\n"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", default=None, help="file to sample positions from (default: a builtin mix)")
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--out", default="results/sibling-experts.json")
    args = ap.parse_args()

    if not TREE_PROBE:
        raise SystemExit("DSV41_TREE_PROBE must be enabled before engine.fastdecode is imported")
    if T_DRAFT < 5:
        raise SystemExit(f"benchmark needs a five-token DSpark draft; configured depth is {T_DRAFT}")

    md = os.environ.get("MODEL_DIR", os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
    eng = V41Engine(md, max_seq=4096,
                    trace_stats=os.environ.get("TRACE_STATS", "results/trace-union/stats/coverage.json"),
                    spec=True, prune_keep=float(os.environ.get("PK", os.environ.get("PRUNE_KEEP", "0.61"))),
                    arena_gb=float(os.environ.get("AG", "90.5")),
                    transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "8")),
                    keep_free_gb=float(os.environ.get("KEEP_FREE_GB", "10")),
                    expert_format=os.environ.get("EXPERT_FORMAT", "fp4"))
    model, dev = eng.model, eng.device
    if eng.fast is None:
        raise SystemExit("benchmark requires the production FastDecoder DSpark path")
    k = eng.args.n_activated_experts

    if args.text:
        with open(args.text) as f:
            text = f.read()
    else:
        text = BUILTIN_TEXT
    ids = eng.tokenizer.encode(text, add_special_tokens=False)[:args.max_len]
    if len(ids) < 32:
        raise SystemExit("text too short")

    cap: dict[int, torch.Tensor] = {}
    capture_routes = False
    top2_reorders = 0

    def tap(name, layer, tensor):
        if capture_routes and name == "route_idx":
            cap[layer] = tensor.detach().clone()

    model.tap = tap

    def prefill(prefix: list[int]):
        """Reproduce the production prefill/replay state and return final logits and DSpark seed."""
        nonlocal capture_routes
        capture_routes = False
        cap.clear()
        eng._reset()
        model.begin_prompt()
        tokens = torch.tensor(prefix, dtype=torch.long, device=dev)
        if eng.swa_replay:
            for start in range(0, len(prefix), MAX_CHUNK):
                model.forward(tokens[start:start + MAX_CHUNK], start, prefill=True,
                              need_logits=False, encoder_only=True)
            logits, main_hidden, seed_pos = model.decoder_replay(need_logits=True)
        else:
            logits, main_hidden = model.forward(tokens, 0, prefill=True, need_logits=True)
            seed_pos = 0
        return logits, main_hidden, seed_pos

    def dspark_candidates(prefix: list[int]):
        """Return target root, DSpark top-1 chain, and DSpark runner-ups for this real step."""
        nonlocal top2_reorders
        logits, main_hidden, seed_pos = prefill(prefix)
        model.dspark_seed(main_hidden, seed_pos)
        root = int(logits[-1].argmax())
        drafts, _ = eng.fast.draft(root, len(prefix) - 1, 0.0)
        # draft() owns static buffers; copy before the next prefill or draft touches them.
        chain = [int(v) for v in drafts[:5].tolist()]
        top2 = []
        for raw_pair, tok in zip(eng.fast.d_top2[:5].tolist(), chain):
            pair = [int(v) for v in raw_pair]
            if tok not in pair:
                raise RuntimeError("DSpark's selected token is absent from its top-2 probe")
            if pair[0] != tok:
                # Older FastDecoder source can be present in a benchmark image.  Normalize the
                # exact-tie ordering here too, so the measurement remains self-checking.
                pair = [tok, pair[0]]
                top2_reorders += 1
            top2.append(pair)
        return root, chain, top2

    def trace_suffix(prefix: list[int], suffix: list[int]):
        """Route suffix rows after a production-equivalent prefix state."""
        nonlocal capture_routes
        prefill(prefix)
        cap.clear()
        capture_routes = True
        tokens = torch.tensor(suffix, dtype=torch.long, device=dev)
        model.forward(tokens, len(prefix), prefill=False, need_logits=False)
        capture_routes = False
        if len(cap) != eng.args.n_layers:
            raise RuntimeError(f"captured {len(cap)} routed layers, expected {eng.args.n_layers}")
        return {layer: routes for layer, routes in cap.items()}

    rows = []
    positions = list(range(8, len(ids) - 6, args.stride))
    for p in positions:
        prefix = ids[:p]
        root, chain, top2 = dspark_candidates(prefix)
        a1, b1, c1, d1, e1 = chain
        a2, b2 = top2[0][1], top2[1][1]

        main = trace_suffix(prefix, [root, a1, b1, c1, d1, e1])
        root_branch = trace_suffix(prefix, [root, a2])
        deep_branch = trace_suffix(prefix, [root, a1, b2])
        rows.append({
            "position": p,
            "tokens": {"root": root, "a1": a1, "b1": b1, "c1": c1,
                       "d1": d1, "e1": e1, "a2": a2, "b2": b2},
            "routes": {
                "root": {layer: main[layer][0] for layer in main},
                "a1": {layer: main[layer][1] for layer in main},
                "b1": {layer: main[layer][2] for layer in main},
                "c1": {layer: main[layer][3] for layer in main},
                "d1": {layer: main[layer][4] for layer in main},
                "e1": {layer: main[layer][5] for layer in main},
                "a2": {layer: root_branch[layer][1] for layer in root_branch},
                "b2": {layer: deep_branch[layer][2] for layer in deep_branch},
            },
        })
        if len(rows) % 5 == 0:
            print(f"  rank {eng.ep.rank}: {len(rows)}/{len(positions)} positions", flush=True)

    layers = sorted(rows[0]["routes"]["root"])

    def union_count(row, names, layer):
        union = set()
        for name in names:
            union.update(row["routes"][name][layer].tolist())
        return len(union)

    def distribution(values):
        ordered = sorted(values)
        return {
            "mean": round(statistics.fmean(values), 3),
            "p50": round(statistics.median(values), 3),
            "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
            "min": min(values),
            "max": max(values),
        }

    chain3_names = ("root", "a1", "b1", "c1")
    chain5_names = chain3_names + ("d1", "e1")
    tree_a2_names = chain3_names + ("a2",)
    tree_b2_names = chain3_names + ("b2",)
    tree_names = chain3_names + ("a2", "b2")
    base = [union_count(row, chain3_names, layer) for row in rows for layer in layers]
    chain5 = [union_count(row, chain5_names, layer) for row in rows for layer in layers]
    tree_a2 = [union_count(row, tree_a2_names, layer) for row in rows for layer in layers]
    tree_b2 = [union_count(row, tree_b2_names, layer) for row in rows for layer in layers]
    tree = [union_count(row, tree_names, layer) for row in rows for layer in layers]
    chain_tail = [wide - shallow for wide, shallow in zip(chain5, base)]
    a2_marginal = [wide - shallow for wide, shallow in zip(tree_a2, base)]
    b2_marginal = [wide - shallow for wide, shallow in zip(tree_b2, base)]
    tree_siblings = [wide - shallow for wide, shallow in zip(tree, base)]

    def pair_stats(first, second):
        unions, marginal, jaccard = [], [], []
        for row in rows:
            for layer in layers:
                left = set(row["routes"][first][layer].tolist())
                right = set(row["routes"][second][layer].tolist())
                unions.append(len(left | right))
                marginal.append(len(right - left))
                jaccard.append(len(left & right) / max(len(left | right), 1))
        return {"union": round(statistics.fmean(unions), 3),
                "marginal": round(statistics.fmean(marginal), 3),
                "jaccard": round(statistics.fmean(jaccard), 4)}

    chain_extra = statistics.fmean(chain_tail)
    base_total = statistics.fmean(base)
    a2_total = statistics.fmean(tree_a2)
    b2_total = statistics.fmean(tree_b2)
    tree_extra = statistics.fmean(tree_siblings)
    chain_total = statistics.fmean(chain5)
    tree_total = statistics.fmean(tree)
    rep = {
        "method": "production DSpark candidates; production prefill/replay; exact six-row route union",
        "n_samples": len(rows),
        "n_layers": len(layers),
        "topk": k,
        "draft_depth": T_DRAFT,
        "input_tokens": len(ids),
        "sample_stride": args.stride,
        "top2_tie_reorders": top2_reorders,
        "rows": {
            "chain3_rows4": distribution(base),
            "chain5_rows6": distribution(chain5),
            "tree_a2_rows5": distribution(tree_a2),
            "tree_b2_rows5": distribution(tree_b2),
            "tree3_rows6": distribution(tree),
            "chain_tail_marginal": distribution(chain_tail),
            "a2_marginal": distribution(a2_marginal),
            "b2_marginal": distribution(b2_marginal),
            "tree_siblings_marginal": distribution(tree_siblings),
        },
        "comparison": {
            "tree_a2_total_vs_chain3_pct": round(100 * (a2_total / base_total - 1), 3),
            "tree_b2_total_vs_chain3_pct": round(100 * (b2_total / base_total - 1), 3),
            "tree_total_vs_chain5_pct": round(100 * (tree_total / chain_total - 1), 3),
            "tree_extra_vs_chain_tail_pct": round(100 * (tree_extra / chain_extra - 1), 3),
            "sibling_extra_discount_pct": round(100 * (1 - tree_extra / chain_extra), 3),
        },
        "pair_diagnostics": {
            "root_siblings_a1_a2": pair_stats("a1", "a2"),
            "deep_siblings_b1_b2": pair_stats("b1", "b2"),
            "chain_tail_c1_d1": pair_stats("c1", "d1"),
            "chain_tail_d1_e1": pair_stats("d1", "e1"),
        },
        "samples": [{"position": row["position"], "tokens": row["tokens"]} for row in rows],
    }
    if eng.ep.rank == 0:
        print(json.dumps(rep, indent=2))
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
            f.write("\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
