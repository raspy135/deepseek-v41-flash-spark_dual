"""Two-rank losslessness and timing smoke test for request-local lookup drafting.

The first raw prompt is deliberately periodic so an exact 16-token suffix has a
known continuation. A second prompt plants a deliberately wrong continuation,
exercising rejection and the transition back to DSpark. The same loaded engine
runs each prompt with lookup disabled and enabled; greedy output must be
identical in every arm.

    GATE_ENV="DSV41_LOOKUP_DRAFT_NGRAM=16" GATE_IMAGE=<id> \
      GATE_LOG_DIR=results/<dir> GATE_SOURCE_ROOT=<snapshot> \
      bash tools/run_two_node_gate.sh bench_lookup_draft_tp.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

# Keep the disposable run from restoring prefixes or writing demand history.
for key in ("DSV41_PREFIX_CACHE", "DSV41_PREFIX_DISK", "DSV41_PRUNE_SWAP",
            "DSV41_PRUNE_SWAP_PREFILL"):
    os.environ[key] = "0"

sys.path[:0] = ["/app", "/app/tools"]
import engine.v41_engine as V  # noqa: E402


def main():
    assert V.LOOKUP_DRAFT_NGRAM >= 2
    V.save_prune_db = lambda *args, **kwargs: None
    engine = V.V41Engine(
        os.environ["MODEL_DIR"], max_seq=4096,
        trace_stats="/app/results/trace-union/stats/coverage.json",
        spec=True, prune_keep=float(os.environ.get("PRUNE_KEEP", ".61")),
        arena_gb=float(os.environ.get("ARENA_GB", "90.1")),
        transient_slots=int(os.environ.get("TRANSIENT_SLOTS", "16")),
        keep_free_gb=float(os.environ.get("KEEP_FREE_GB", "6")),
        expert_format=os.environ.get("EXPERT_FORMAT", "fp4"),
    )
    # Natural words keep the tokenizer from collapsing a repeated character run
    # into a tokenization unlike normal traffic.
    unit = ("The copper clock marks seventeen while the quiet river carries "
            "three silver leaves beyond the old stone bridge.\n")
    periodic_ids = engine.tokenizer.encode(unit * 24, add_special_tokens=False)
    suffix = periodic_ids[:16]
    wrong_tail = engine.tokenizer.encode(
        " purple helicopters calculate sideways under nine cardboard moons", add_special_tokens=False)
    filler = engine.tokenizer.encode(
        "This paragraph separates the records. Its wording is intentionally unrelated to the "
        "clock sentence, and it contains enough tokens to prevent an overlapping suffix match. ",
        add_special_tokens=False)

    def make_stale(wrong_first):
        return (engine.tokenizer.encode("Earlier record: ", add_special_tokens=False)
                + suffix + [wrong_first] + wrong_tail
                + filler
                + engine.tokenizer.encode("Current record: ", add_special_tokens=False)
                + suffix)

    def run(prompt_ids, enabled: bool, max_tokens: int):
        V.LOOKUP_DRAFT_NGRAM = 16 if enabled else 0
        output = []
        started = time.perf_counter()
        for burst in engine.generate(prompt_ids, max_tokens=max_tokens, temperature=0.0,
                                     seed=42, ignore_eos=True):
            output.extend(burst)
        elapsed = time.perf_counter() - started
        stats = dict(engine.last_stats)
        return output, {
            "enabled": enabled,
            "wall_s": round(elapsed, 4),
            "decode_tok_s": stats.get("decode_tok_s"),
            "steps": stats.get("steps"),
            "accept_len_mean": stats.get("accept_len_mean"),
            "lookup": stats.get("lookup_draft"),
            "sha256": hashlib.sha256(json.dumps(output).encode()).hexdigest(),
        }

    # Select a planted first continuation token that this exact prompt's target does not choose.
    # Special tokens make this settle on the first probe in practice; the loop makes rejection an
    # assertion rather than an assumption about model behavior.
    stale_ids = None
    wrong_first = None
    for candidate in (2, 0, 1, 100, 1000, 10000):
        probe_ids = make_stale(candidate)
        probe, _ = run(probe_ids, False, 1)
        if probe[0] != candidate:
            stale_ids, wrong_first = probe_ids, candidate
            break
    assert stale_ids is not None, "could not construct a rejected lookup continuation"

    # Capture/warm both paths before retaining the comparison.
    run(periodic_ids, False, 96)
    run(periodic_ids, True, 96)
    periodic_base, periodic_off = run(periodic_ids, False, 96)
    periodic_candidate, periodic_on = run(periodic_ids, True, 96)
    stale_base, stale_off = run(stale_ids, False, 48)
    stale_candidate, stale_on = run(stale_ids, True, 48)

    periodic_exact = periodic_base == periodic_candidate
    stale_exact = stale_base == stale_candidate
    identity = (periodic_off["sha256"], periodic_on["sha256"], periodic_exact,
                stale_off["sha256"], stale_on["sha256"], stale_exact)
    peers = engine.ep.gather_objects(identity)
    periodic_lookup = periodic_on.get("lookup") or {}
    stale_lookup = stale_on.get("lookup") or {}
    passed = periodic_exact and stale_exact and all(item == peers[0] for item in peers)
    passed = passed and periodic_lookup.get("hits", 0) > 0
    passed = passed and periodic_lookup.get("dspark_skipped", 0) > 0
    passed = passed and stale_lookup.get("hits", 0) > 0 and stale_lookup.get("misses", 0) > 0
    passed = passed and stale_lookup.get("accepted_tokens", 0) < stale_lookup.get("draft_tokens", 0)
    if engine.ep.rank == 0:
        print("LOOKUP_DRAFT " + json.dumps({
            "periodic": {"off": periodic_off, "on": periodic_on, "exact": periodic_exact},
            "stale": {"wrong_first": wrong_first, "off": stale_off, "on": stale_on,
                      "exact": stale_exact},
            "passed": passed,
        }), flush=True)
    assert all(engine.ep.gather_objects(passed))
    os._exit(0 if passed else 1)


if __name__ == "__main__":
    main()
