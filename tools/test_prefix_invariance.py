"""A prompt served from the prefix cache must produce what a cold prefill of it produces.

This is the invariant the whole prefix cache rests on, and nothing tested it end to end: the
existing engine/test_prefix_cache.py checks the bookkeeping on CPU fixtures, which cannot see
whether the attention path is chunk-invariant. It is the reason the prefill GEMMs run on
fixed-size token tiles (engine/model.py: MM_TILE/ATTN_TILE), and the reason a faster attention
kernel cannot simply be switched on -- if resuming from a cached prefix diverges from prefilling
the whole prompt, the cache silently changes answers.

Drives a running server, so it measures the path that actually ships. The engine keeps exactly
one prefix, so sending an unrelated prompt is what makes the next run cold.

    python3 tools/test_prefix_invariance.py [--base http://127.0.0.1:8000]
"""
import argparse, json, sys, urllib.request

CODE = [f"def function_{i}(value): return value * {i+1} + {i%17}\n" for i in range(300)]
HEAD = "Explain the following code and identify patterns.\n"


def complete(base, prompt, n=48):
    """-> (text, prompt_tokens, tokens served from the prefix cache)."""
    body = json.dumps(dict(model="deepseek", prompt=prompt, max_tokens=n,
                           temperature=0, seed=42)).encode()
    req = urllib.request.Request(base + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    cached = int((d.get("x_engine_stats") or {}).get("prefix_cached_tokens") or 0)
    return d["choices"][0]["text"], d["usage"]["prompt_tokens"], cached


def generation(base):
    """The engine's expert-adaptation counter. Adaptation changes which experts are routable, so
    it legitimately changes the answer -- if it fires between the two runs being compared, a
    difference says nothing about chunk invariance."""
    req = urllib.request.Request(base + "/health")
    with urllib.request.urlopen(req, timeout=60) as r:
        return (json.loads(r.read())["engine_config"] or {}).get("expert_generation")


def attempt(base, full, half, other):
    complete(base, other, 8)                       # evict: the engine holds one prefix
    g0 = generation(base)
    cold, n_full, cold_cached = complete(base, full)
    complete(base, other, 8)                       # evict again
    _, n_half, _ = complete(base, half, 8)         # this run leaves `half` in the cache
    warm, n_full2, warm_cached = complete(base, full)   # ...so this one resumes from it
    return cold, warm, n_full, n_full2, n_half, cold_cached, warm_cached, g0, generation(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--tries", type=int, default=3)
    args = ap.parse_args()
    full = HEAD + "".join(CODE)
    half = HEAD + "".join(CODE[:150])
    other = "Write one sentence about tide pools.\n"

    for k in range(args.tries):
        (cold, warm, n_full, n_full2, n_half,
         cold_cached, warm_cached, g0, g1) = attempt(args.base, full, half, other)
        if g0 is None or g0 == g1:
            break
        # The resident expert set moved underneath the comparison. Not a failure -- but not
        # evidence either. Demand adaptation settles once the same prompts stop producing new
        # demand, so simply going round again usually finds a quiet window.
        print(f"attempt {k + 1}: expert generation moved {g0} -> {g1} mid-test; retrying")
    else:
        sys.exit("INCONCLUSIVE: expert adaptation fired on every attempt; cannot compare runs")

    assert n_full == n_full2, f"prompt lengths differ: {n_full} vs {n_full2}"
    # Equality alone is not enough: if the cache quietly stopped being used, BOTH runs would be
    # cold prefills and would of course agree. Assert the second run really resumed.
    assert cold_cached == 0, f"the 'cold' run reused {cold_cached} tokens -- eviction failed"
    assert warm_cached == n_half, \
        f"prefix cache NOT USED: resumed run reused {warm_cached} tokens, expected {n_half}"
    print(f"prompt {n_full} tokens, resumed after a {n_half}-token prefix "
          f"(cold reused {cold_cached}, warm reused {warm_cached}, expert generation {g0} stable)")
    if cold != warm:
        print("COLD:", repr(cold[:300]))
        print("WARM:", repr(warm[:300]))
        # where they diverge says how deep the difference is
        i = next((i for i, (a, b) in enumerate(zip(cold, warm)) if a != b), min(len(cold), len(warm)))
        print(f"first difference at character {i} of {min(len(cold), len(warm))}")
        sys.exit("PREFIX CACHE IS NOT INVARIANT: resuming changed the answer")
    print(f"cold == resumed, {len(cold)} chars identical")
    print("\nPREFIX CACHE IS USED AND INVARIANT")


if __name__ == "__main__":
    main()
