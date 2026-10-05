"""Two-rank losslessness and timing smoke test for request-local lookup drafting.

The first raw prompt is deliberately periodic so an exact 16-token suffix has a
known continuation. A second prompt exercises mixed hits and DSpark misses;
its planted continuation is not guaranteed to be rejected by the target. A
separate correctness case corrupts the first copied proposal deliberately,
requiring rejection and lookup-off output equality. The same loaded engine
runs each prompt with lookup disabled and enabled; greedy output must be
identical in every arm. Sampled cases check rank parity and repeated seeded
execution, not equality across proposal/RNG schedules. The exact distribution
check lives in engine/test_lookup_sampling.py.

    GATE_ENV="DSV41_LOOKUP_DRAFT_ENABLED=1 DSV41_LOOKUP_DRAFT_NGRAM=16" GATE_IMAGE=<id> \
      GATE_LOG_DIR=results/<dir> GATE_SOURCE_ROOT=<snapshot> \
      bash tools/run_two_node_gate.sh bench_lookup_draft_tp.py
"""
from __future__ import annotations

import hashlib
import argparse
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


def main(engine=None, argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tokens', type=int, default=32)
    ap.add_argument('--sample-tokens', type=int, default=8)
    ap.add_argument('--stale-tokens', type=int, default=32)
    ap.add_argument('--greedy-only', action='store_true')
    args = ap.parse_args(argv)
    external = engine is not None
    assert V.LOOKUP_DRAFT_NGRAM >= 2
    if engine is None:
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
    saved_flags = V.LOOKUP_DRAFT_ENABLED, V.LOOKUP_DRAFT_NGRAM
    saved_bypass = engine.draft_bypass_enabled
    policies = [p for p in (engine.depth_policy, engine.confidence_depth_policy) if p is not None]
    pins = [(p, p.pinned) for p in policies]
    try:
        # Timings can move an adaptive depth policy across a cutoff between repeats.
        # Fix width while checking seeded repeatability and isolating copy overhead.
        for policy, _ in pins:
            policy.pinned = 3
        engine.draft_bypass_enabled = False
        report = qualify(engine, args)
        if engine.ep.rank == 0:
            print("LOOKUP_DRAFT " + json.dumps(report), flush=True)
        assert all(engine.ep.gather_objects(report['passed']))
    finally:
        V.LOOKUP_DRAFT_ENABLED, V.LOOKUP_DRAFT_NGRAM = saved_flags
        engine.draft_bypass_enabled = saved_bypass
        for policy, pinned in pins:
            policy.pinned = pinned
    if external:
        return report
    os._exit(0 if report['passed'] else 1)


def qualify(engine, args):
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

    def run(prompt_ids, enabled: bool, max_tokens: int, temperature=0., top_p=1., seed=42):
        V.LOOKUP_DRAFT_ENABLED = enabled
        V.LOOKUP_DRAFT_NGRAM = 16
        output = []
        started = time.perf_counter()
        for burst in engine.generate(prompt_ids, max_tokens=max_tokens, temperature=temperature,
                                     top_p=top_p, seed=seed, ignore_eos=True):
            output.extend(burst)
        elapsed = time.perf_counter() - started
        stats = dict(engine.last_stats)
        return output, {
            "enabled": enabled,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
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
    run(periodic_ids, False, args.tokens)
    run(periodic_ids, True, args.tokens)
    periodic_base, periodic_off = run(periodic_ids, False, args.tokens)
    periodic_candidate, periodic_on = run(periodic_ids, True, args.tokens)
    stale_base, stale_off = run(stale_ids, False, args.stale_tokens)
    stale_candidate, stale_on = run(stale_ids, True, args.stale_tokens)

    # A record written into the prompt can become the target's preferred answer.
    # Therefore a naturally stale copy is not reliable rejection coverage. Force
    # one false proposal instead: a fixed ID absent from the entire greedy reply
    # cannot match any target row along that reply. q remains the proposal's delta.
    # This is a correctness-only fault injection, never a throughput comparison.
    from engine.lookup_draft import ExactDraftCache
    reference_tokens = set(periodic_base)
    forced_wrong = next(token for token in range(engine.args.vocab_size)
                        if token not in reference_tokens)
    original_propose = ExactDraftCache.propose
    injected = [False]
    def corrupt_first_copy(cache, depth):
        proposal = original_propose(cache, depth)
        if proposal is not None and not injected[0]:
            proposal = list(proposal)
            proposal[0] = forced_wrong
            injected[0] = True
            cache.stats['forced_corruption_count'] = 1
        return proposal
    try:
        ExactDraftCache.propose = corrupt_first_copy
        forced_candidate, forced_on = run(periodic_ids, True, args.tokens)
    finally:
        ExactDraftCache.propose = original_propose
    forced_exact = periodic_base == forced_candidate

    periodic_exact = periodic_base == periodic_candidate
    stale_exact = stale_base == stale_candidate
    identity = (periodic_off["sha256"], periodic_on["sha256"], periodic_exact,
                stale_off["sha256"], stale_on["sha256"], stale_exact,
                forced_on['sha256'], forced_exact)
    peers = engine.ep.gather_objects(identity)
    periodic_lookup = periodic_on.get("lookup") or {}
    stale_lookup = stale_on.get("lookup") or {}
    passed = periodic_exact and stale_exact and forced_exact and all(item == peers[0] for item in peers)
    # Only rank 0 owns the request-local index and its statistics; proposal IDs
    # are broadcast after the shared mode control. Rank 1 still checks hashes.
    if engine.ep.rank == 0:
        passed = passed and periodic_lookup.get("hits", 0) > 0
        passed = passed and periodic_lookup.get("dspark_skipped", 0) > 0
        passed = passed and stale_lookup.get("hits", 0) > 0 and stale_lookup.get("misses", 0) > 0
        forced_lookup = forced_on.get('lookup') or {}
        passed = passed and injected[0]
        passed = passed and forced_lookup.get('accepted_tokens', 0) < forced_lookup.get('draft_tokens', 0)
    sampled = []
    if not args.greedy_only:
        for temperature in (.1, .6, 1., 2.):
            for top_p in (.5, .95, 1.):
                # Each positive-temperature request may choose a different continuation;
                # output equality with lookup off is therefore not a correctness condition.
                _, off = run(periodic_ids, False, args.sample_tokens, temperature, top_p)
                first, on = run(periodic_ids, True, args.sample_tokens, temperature, top_p)
                second, repeat = run(periodic_ids, True, args.sample_tokens, temperature, top_p)
                item = {'off': off, 'on': on, 'repeat': repeat, 'repeat_exact': first == second}
                identities = engine.ep.gather_objects((off['sha256'], on['sha256'], repeat['sha256']))
                item['rank_exact'] = all(identity == identities[0] for identity in identities)
                passed = passed and item['repeat_exact'] and item['rank_exact']
                sampled.append(item)
        if engine.ep.rank == 0:
            passed = passed and sum((item['on'].get('lookup') or {}).get('hits', 0) for item in sampled) > 0
    passed = all(engine.ep.gather_objects(passed))
    return {
        "periodic": {"off": periodic_off, "on": periodic_on, "exact": periodic_exact},
        "stale": {"wrong_first": wrong_first, "off": stale_off, "on": stale_on,
                  "exact": stale_exact},
        "forced_rejection": {'purpose': 'correctness only; first copied ID deliberately corrupted',
                             'wrong_token': forced_wrong, 'on': forced_on, 'exact': forced_exact},
        "sampled": sampled,
        "fixed_depth": 3,
        "passed": passed,
    }


if __name__ == "__main__":
    main()
