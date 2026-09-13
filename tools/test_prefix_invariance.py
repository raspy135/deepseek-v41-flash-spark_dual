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
    body = json.dumps(dict(model="deepseek", prompt=prompt, max_tokens=n,
                           temperature=0, seed=42)).encode()
    req = urllib.request.Request(base + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    return d["choices"][0]["text"], d["usage"]["prompt_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    full = HEAD + "".join(CODE)
    half = HEAD + "".join(CODE[:150])
    other = "Write one sentence about tide pools.\n"

    complete(args.base, other, 8)                    # evict: the engine holds one prefix
    cold, n_full = complete(args.base, full)
    complete(args.base, other, 8)                    # evict again
    _, n_half = complete(args.base, half, 8)         # this run leaves `half` in the cache
    warm, n_full2 = complete(args.base, full)        # ...so this one resumes from it

    assert n_full == n_full2, f"prompt lengths differ: {n_full} vs {n_full2}"
    print(f"prompt {n_full} tokens, resumed after a {n_half}-token prefix")
    if cold != warm:
        print("COLD:", repr(cold[:300]))
        print("WARM:", repr(warm[:300]))
        # where they diverge says how deep the difference is
        i = next((i for i, (a, b) in enumerate(zip(cold, warm)) if a != b), min(len(cold), len(warm)))
        print(f"first difference at character {i} of {min(len(cold), len(warm))}")
        sys.exit("PREFIX CACHE IS NOT INVARIANT: resuming changed the answer")
    print(f"cold == resumed, {len(cold)} chars identical")
    print("\nPREFIX CACHE IS INVARIANT")


if __name__ == "__main__":
    main()
