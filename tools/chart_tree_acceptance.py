"""Depth-5 verify: which branch/node contributes the accepted tokens, as a chart.

Each step's verify block is the chain a1..e1. The drafter's top-1 at position i is the chain node;
its top-2 is the runner-up a sibling would occupy. A node's contribution to the accepted length is
its unconditional hit rate, which is the chain's leading-match probability times the conditional:
the tree's accepted length is the sum over nodes of these, because the branches are parallel.

Emits a standalone SVG (no dependencies -- the venv has no matplotlib) plus the table.

    python tools/chart_tree_acceptance.py results/<dir>/spec-conf-rank0.json
"""
from __future__ import annotations

import argparse
import json

POS = list("abcde")


def contributions(log, depth=5):
    """per-position (chain_uncond, sibling_uncond) contributions to accepted length."""
    chain = [0.0] * depth          # P(chain matched through position i)
    sib = [0.0] * depth            # P(reached i, top-1 missed, runner-up held)
    n = len(log)
    for _d, _a, cand, top2 in log:
        reached = True
        for i in range(depth):
            if not reached or i >= len(cand):
                break
            if cand[i] == top2[i][0]:
                chain[i] += 1
            else:
                if cand[i] == top2[i][1]:
                    sib[i] += 1
                reached = False
    return [c / n for c in chain], [s / n for s in sib], n


def bars(group, values, colors, x0, y0, w, h, vmax, title):
    """grouped bars: values is a list of (label, series) columns."""
    parts = [f'<text x="{x0}" y="{y0 - 8}" class="ttl">{title}</text>']
    bw, gap = 16, 10
    for gi, (pos, cols) in enumerate(group):
        gx = x0 + gi * (len(cols) * (bw + 4) + gap + 10)
        for si, (name, val) in enumerate(cols):
            bh = h * (val / vmax)
            parts.append(f'<rect x="{gx + si * (bw + 4)}" y="{y0 + h - bh:.1f}" width="{bw}" '
                         f'height="{bh:.1f}" fill="{colors[si]}"/>')
            parts.append(f'<text x="{gx + si * (bw + 4) + bw / 2:.0f}" y="{y0 + h - bh - 3:.1f}" '
                         f'class="val">{val:.2f}</text>')
        parts.append(f'<text x="{gx + (len(cols) * (bw + 4)) / 2:.0f}" y="{y0 + h + 14}" '
                     f'class="lab">{pos}</text>')
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report")
    ap.add_argument("--out", default="results/tree-probe/acceptance-depth5.svg")
    args = ap.parse_args()
    rep = json.load(open(args.report))
    assert rep.get("tree_probe"), "collect with DSV41_TREE_PROBE=1"

    data = {}
    for cls, pred in (("prose", lambda w: w in ("explain", "story")),
                      ("code", lambda w: w not in ("explain", "story"))):
        log = [e for r in rep["runs"] if pred(r["workload"]) for e in r["tree_log"]]
        data[cls] = contributions(log)

    print(f"{'':6} | {' '.join(f'pos{i}'.rjust(17) for i in range(5))}")
    print(f"{'':6} | {' '.join(('chain  sib'.rjust(17) for _ in range(5)))}")
    for cls in ("prose", "code"):
        c, s, n = data[cls]
        print(f"{cls:6} | " + " ".join(f"{c[i]:7.3f}{s[i]:8.3f}" for i in range(5)) +
              f"   (n={n}, chain {sum(c):.3f} sib {sum(s):.3f} total {sum(c) + sum(s):.3f})")

    # SVG
    vmax = max(max(c + s) for c, s, _ in data.values()) * 1.25
    body = []
    y = 60
    for cls in ("prose", "code"):
        c, s, n = data[cls]
        group = [(f"{POS[i]}", [("chain", c[i]), ("runner-up", s[i])]) for i in range(5)]
        body.append(bars(group, None, ["#3b82f6", "#f59e0b"], 120, y, 5 * 78, 150, vmax,
                         f"{cls}  (chain {sum(c):.2f} + runner-up {sum(s):.2f} = {sum(c) + sum(s):.2f} accepted/step)"))
        y += 230
    sw, sh = 620, y + 30
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{sw}" height="{sh}" font-family="system-ui,sans-serif">
<style>
.ttl {{ font-size: 14px; font-weight: 600; fill: #111; }}
.lab {{ font-size: 12px; fill: #555; text-anchor: middle; }}
.val {{ font-size: 10px; fill: #333; text-anchor: middle; }}
.leg {{ font-size: 12px; fill: #333; }}
</style>
<text x="20" y="26" class="ttl">Depth-5 verify: accepted-token contribution per node</text>
<text x="20" y="44" class="lab" style="text-anchor:start">blue = chain top-1   orange = runner-up (what a2/b2 would add)</text>
<rect x="20" y="60" width="12" height="12" fill="#3b82f6"/><text x="38" y="70" class="leg">chain (top-1)</text>
<rect x="150" y="60" width="12" height="12" fill="#f59e0b"/><text x="168" y="70" class="leg">runner-up (sibling)</text>
{chr(10).join(body)}
</svg>
'''
    with open(args.out, "w") as f:
        f.write(svg)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
