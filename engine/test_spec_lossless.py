"""Speculative decoding must not change what the model writes.

DSpark drafts are verified against the target model, so greedy decoding with speculation on has to
produce exactly the tokens greedy decoding without it produces. A mismatch means the verification is
reading logits that do not belong to the position it is checking, and the output that reaches a user
is partly the drafter's. Teacher-forced loss cannot see this: it never runs the decode loop.

Run: python engine/test_spec_lossless.py [--max-tokens 120]
The engine is built twice (spec on, spec off), so this costs two warm starts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from engine.v41_engine import V41Engine, log  # noqa: E402

PROMPTS = [
    "Write a complete single-file HTML tic-tac-toe game. Output only the HTML.",
    "Write a Python function that returns the n-th Fibonacci number, with a docstring.",
]


def run(eng, prompt: str, max_tokens: int):
    sys.path.insert(0, os.path.join(eng.model_dir, "encoding"))
    from encoding import encode_messages  # noqa: E402
    pr = encode_messages([{"role": "user", "content": prompt}], thinking_mode="chat")
    ids = eng.tokenizer.encode(pr if isinstance(pr, str) else pr[0], add_special_tokens=False)
    out = []
    for t in eng.generate(ids, max_tokens=max_tokens, temperature=0.0):
        out += t
    return out


def _await_memory(kw, timeout_s: int = 300):
    """Block until the previous arm's arena is actually back.

    Both arms run in ONE process, and the first engine holds ~70 GB of HOST memory for its expert
    arena. `del eng` returns immediately; the pages come back to MemAvailable some time later, and
    until they do the second engine's own preflight refuses to start:

        not enough host memory to start: MemAvailable 4.8 GB leaves -15.2 GB for the expert arena

    which is the guard doing its job on a test that did not wait. Polling is enough -- the release
    is prompt once the allocator actually returns the pages, it is simply not synchronous.
    """
    import time
    need = float(kw.get("keep_free_gb", 20.0)) + 40.0
    for _ in range(timeout_s):
        with open("/proc/meminfo") as f:
            avail = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable:")) / 1048576
        if avail >= need:
            return
        time.sleep(1)
    print(f"WARNING: only {avail:.1f} GiB available after {timeout_s}s; starting anyway",
          file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--engine-kwargs", default="{}")
    a = ap.parse_args()
    kw = json.loads(a.engine_kwargs)
    kw.setdefault("trace_stats", "results/trace-full-20260910/stats/coverage.json")

    outs = {}
    for spec in (False, True):
        _await_memory(kw)
        eng = V41Engine(a.model_dir, max_seq=8192, spec=spec, **kw)
        outs[spec] = [run(eng, p, a.max_tokens) for p in PROMPTS]
        tk = eng.tokenizer
        eng.close() if hasattr(eng, "close") else None
        del eng
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

    failed = 0
    for i, p in enumerate(PROMPTS):
        ref, got = outs[False][i], outs[True][i]
        n = min(len(ref), len(got))
        first = next((j for j in range(n) if ref[j] != got[j]), None)
        if first is None and len(ref) == len(got):
            log(f"PASS prompt {i}: {len(ref)} tokens identical with and without speculation")
            continue
        failed += 1
        log(f"FAIL prompt {i}: first divergence at token {first} of {n}")
        lo = max(0, (first or n) - 12)
        log("  without spec: " + repr(tk.decode(ref[lo:(first or n) + 24])))
        log("  with spec:    " + repr(tk.decode(got[lo:(first or n) + 24])))
    print(f"{len(PROMPTS) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
