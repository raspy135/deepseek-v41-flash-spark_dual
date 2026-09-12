#!/usr/bin/env python3
"""generation_gate.py -- the quality gate RESULTS.md section 4.1 defines, as a runnable tool.

Why this exists rather than a loss number: teacher-forced loss scores the next token of text the
model is SHOWN, so it never lets an error compound and cannot see a configuration that has lost the
ability to stay on its own trajectory. The v0.3.0-wip default measured BETTER on loss
(1.5384 / 3.2087 against 1.5705 / 3.3790) and wrote `<!DOCTYPE><!DOCTYPE><!DOCTYPE>` for as long as
it was allowed. It was withdrawn on free generation, and every configuration since is gated here.

Two thresholds that RESULTS.md says were learned the hard way, and that this reimplements:
  * length. "A 300-token gate passed configurations that collapse at 900." Minimum 900 tokens.
  * structure. Repetition ratios alone pass output whose CSS has decayed into
    `inset - 00 1 pix - 00 1 pix`, so balanced tags, closed fences and unit spelling are checked too.

Usage:  python3 tools/generation_gate.py [--base http://127.0.0.1:8000] [--model deepseek]
        [--max-tokens 1200] [--only html,python] [--json out.json]
Exit 0 if every prompt passes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from collections import Counter

# Five prompts: a story and an essay at temperature 0.7, a Python module, a single-file HTML game
# and a JavaScript module at temperature 0 (RESULTS.md 4.1). The temperature split matters -- prose
# degeneration shows at 0.7, structural decay at 0.
# (question, exact answer). Kept to arithmetic a careful person does on paper: the gate is testing
# whether the model can carry a multi-step computation without drifting, not whether it is a
# calculator. Answers are checked as whole tokens so "56" does not match inside "560".
ARITH_ITEMS = [
    ("What is 17 * 23 + 456?", 847),
    ("What is 1024 - 377 + 89?", 736),
    ("What is 144 / 12 * 7?", 84),
    ("What is 2^10 + 2^8?", 1280),
    ("What is 45 * 61?", 2745),
    ("What is (320 + 180) / 4?", 125),
    ("What is 999 - 111 - 222?", 666),
    ("What is 13 * 13 * 3?", 507),
    ("If a train travels 87 km/h for 4 hours, how many km does it cover?", 348),
    ("What is the sum of the integers from 1 to 40?", 820),
]

PROMPTS = [
    dict(name="story", temperature=0.7, kind="prose", min_tokens=900,
         prompt="Write a short story of at least 1200 words about a lighthouse keeper who "
                "discovers that the light has been answering someone. Give it a beginning, a "
                "middle and an ending."),
    dict(name="essay", temperature=0.7, kind="prose", min_tokens=900,
         prompt="Write a detailed essay of at least 1200 words on why distributed systems are "
                "hard to reason about, with concrete examples and a conclusion."),
    dict(name="python", temperature=0.0, kind="python", min_tokens=900,
         prompt="Write a complete, runnable Python module implementing an LRU cache with a TTL "
                "per entry, thread safety, and a small test suite at the bottom. Include "
                "docstrings and type hints."),
    dict(name="html", temperature=0.0, kind="html", min_tokens=900,
         prompt="Write a complete single-file HTML game: tic-tac-toe on a 3x3 grid with inline "
                "CSS and JavaScript, a win check, and a reset button. Output only the file."),
    dict(name="javascript", temperature=0.0, kind="js", min_tokens=900,
         prompt="Write a complete JavaScript module implementing a priority queue with a binary "
                "heap, an iterator, JSDoc comments, and a set of assertions exercising it."),
    # Arithmetic is the one workload with a ground truth, which makes it the sharpest degeneration
    # probe here: the other four can only be checked for SHAPE (does it repeat, are the tags
    # balanced), and a model can stay perfectly well-formed while its answers rot. Long multi-step
    # work also compounds errors the way free generation does, which is the whole point of gating
    # on generation rather than teacher-forced loss.
    dict(name="arithmetic", temperature=0.0, kind="arithmetic", min_tokens=400,
         prompt="Solve each of these, showing your working step by step, and end each one with a "
                "line of the form 'ANSWER: <number>'.\n"
                + "\n".join(f"{i + 1}. {q}" for i, q in enumerate(q for q, _ in ARITH_ITEMS))),
]


def post(base, path, payload, timeout=1800):
    req = urllib.request.Request(base.rstrip("/") + path, method="POST",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# --------------------------------------------------------------------------- checks
def distinct_ratio(text: str) -> float:
    """Unique / total whitespace tokens. RESULTS.md 4.1 requires > 0.25."""
    toks = text.split()
    return len(set(toks)) / max(1, len(toks))


def max_line_repeat(text: str) -> float:
    """Share of non-blank lines taken by the single most common line. Must stay <= 0.30."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 1.0
    return Counter(lines).most_common(1)[0][1] / len(lines)


def structural(text: str, kind: str) -> list[str]:
    """Cheap structural intactness. Returns a list of failures (empty == intact).

    Deliberately not a parser: the failure mode being caught is decay -- an unbalanced <script>,
    a fence that never closes, `00 1 pix` where `1px` belongs -- not subtle invalidity.
    """
    bad = []
    if text.count("```") % 2:
        bad.append("unclosed code fence")
    # CSS/px decay: a number split from its unit, or a mangled unit spelling.
    if re.search(r"\b\d+\s+\d+\s*pix\b", text) or re.search(r"\bpix\b(?!el)", text):
        bad.append("decayed CSS unit (e.g. '00 1 pix')")
    if kind == "html":
        for tag in ("script", "style"):
            o, c = len(re.findall(rf"<{tag}\b", text, re.I)), len(re.findall(rf"</{tag}>", text, re.I))
            if o != c:
                bad.append(f"unbalanced <{tag}>: {o} open / {c} close")
        if "<!doctype" not in text.lower():
            bad.append("no doctype")
        for tag in ("html", "body"):
            o, c = len(re.findall(rf"<{tag}\b", text, re.I)), len(re.findall(rf"</{tag}>", text, re.I))
            if o and o != c:
                bad.append(f"unbalanced <{tag}>")
    if kind in ("js", "html", "python"):
        for a, b, label in (("{", "}", "braces"), ("(", ")", "parens"), ("[", "]", "brackets")):
            if abs(text.count(a) - text.count(b)) > 2:   # slack for braces inside strings/prose
                bad.append(f"unbalanced {label}: {text.count(a)} vs {text.count(b)}")
    return bad


def run_one(base, model, spec, max_tokens, seed):
    body = {"model": model, "temperature": spec["temperature"], "max_tokens": max_tokens,
            "seed": seed, "messages": [{"role": "user", "content": spec["prompt"]}]}
    t0 = time.perf_counter()
    d = post(base, "/v1/chat/completions", body)
    wall = time.perf_counter() - t0
    if "error" in d:
        return dict(name=spec["name"], ok=False, failures=[f"API error: {d['error']}"], text="")
    ch = d["choices"][0]
    text = ch["message"]["content"] or ""
    st = d.get("x_engine_stats") or {}
    n = st.get("completion_tokens") or d.get("usage", {}).get("completion_tokens") or 0

    failures = []
    if n < spec["min_tokens"]:
        # Short output is only a gate failure when the model stopped early; hitting max_tokens
        # is the harness's limit, not the model's, and 4.1 only applies the structural check
        # "where the generation finished on its own".
        if ch.get("finish_reason") != "length":
            failures.append(f"stopped at {n} tokens, below the {spec['min_tokens']}-token floor")
    if spec["kind"] == "arithmetic":
        # Whole-token match so 56 does not match inside 560, and so a model that merely restates
        # the question does not score. Correctness, not formatting, is the signal.
        got = sum(1 for _, ans in ARITH_ITEMS
                  if re.search(rf"(?<![\d.]){ans}(?![\d.])", text.replace(",", "")))
        failures.append(f"arithmetic {got}/{len(ARITH_ITEMS)} correct") if got < 8 else None
        spec_note = f"{got}/{len(ARITH_ITEMS)}"
    else:
        spec_note = ""
    dr = distinct_ratio(text)
    lr = max_line_repeat(text)
    if dr <= 0.25:
        failures.append(f"distinct-token ratio {dr:.3f} <= 0.25")
    if lr > 0.30:
        failures.append(f"most-repeated line is {lr:.0%} of lines (> 30%)")
    if ch.get("finish_reason") != "length":
        failures += structural(text, spec["kind"])

    return dict(name=spec["name"], ok=not failures, failures=failures, text=text, note=spec_note,
                tokens=n, finish=ch.get("finish_reason"), wall_s=round(wall, 1),
                distinct=round(dr, 3), line_repeat=round(lr, 3),
                tok_s=st.get("decode_tok_s"), accept=st.get("accept_len_mean"),
                hit_rate=st.get("expert_hit_rate"), nvme_gb=st.get("nvme_gb"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="deepseek")
    ap.add_argument("--max-tokens", type=int, default=1200)  # inside 4.1's 900-2,000 band
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--only", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    want = {s.strip() for s in a.only.split(",") if s.strip()}
    specs = [s for s in PROMPTS if not want or s["name"] in want]
    rows = []
    for s in specs:
        print(f"--- {s['name']} (temp {s['temperature']}, max {a.max_tokens}) ...", flush=True)
        r = run_one(a.base, a.model, s, a.max_tokens, a.seed)
        rows.append(r)
        mark = "PASS" if r["ok"] else "FAIL"
        print(f"    {mark}  {r.get('tokens')} tok  {r.get('tok_s')} tok/s  "
              f"distinct {r.get('distinct')}  line-repeat {r.get('line_repeat')}  "
              f"finish={r.get('finish')} {r.get('note') or ''}")
        for f in r["failures"]:
            print(f"      ! {f}")

    print("\n=== generation gate ===")
    print(f"  {'workload':<12} {'gate':<6} {'tok/s':>7} {'tokens':>7} {'accept':>7} {'distinct':>9} {'linerep':>8}")
    for r in rows:
        print(f"  {r['name']:<12} {'PASS' if r['ok'] else 'FAIL':<6} {str(r.get('tok_s')):>7} "
              f"{str(r.get('tokens')):>7} {str(r.get('accept')):>7} {str(r.get('distinct')):>9} "
              f"{str(r.get('line_repeat')):>8}")
    n_fail = sum(1 for r in rows if not r["ok"])
    print(f"  => {len(rows) - n_fail}/{len(rows)} passed")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"  wrote {a.json}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
