"""Cold-prefill breakdown and row-count A/B in one loaded TP2 process.

Prompts are bench/bench.py's random workload (unique numbered words), the same kind of
text TensorFold's published cold-prefill cells use. Each measured prompt is fresh, so
neither the prefix cache nor the Engram row cache has seen it; one prompt is repeated to
price the Engram reads separately. Rows per chunk are switched at runtime through
V.MAX_CHUNK (boot DSV41_PREFILL_CHUNK must be the largest row count tested). Expert swaps
are off, so the resident map is identical for every arm. Optionally captures a CUPTI
kernel trace of one cold prefill per row count.

Run with DSV41_PREFILL_TIMING=1 (per-chunk host/GPU envelopes) and DSV41_ATTN_TIMING=1
(GPU phase split) in GATE_ENV. Use the disposable two-node gate, not serving.
"""
import argparse
import hashlib
import json
import os
import random
import string
import sys
import time
sys.path[:0] = ['/app', '/app/tools']
# DSV41_BENCH_SERVING_LIKE=1 keeps the serving .env's adaptation, swaps and prefix cache (the map
# then moves, so the frozen-map assertion is skipped; demand saves stay suppressed).
SERVING_LIKE = os.environ.get('DSV41_BENCH_SERVING_LIKE', '0') == '1'
for key in (('DSV41_PREFIX_DISK', 'DSV41_PREFIX_RESPONSE', 'DSV41_GPU_TIMING', 'DSV41_STEP_TIMING')
            + (() if SERVING_LIKE else ('DSV41_PRUNE_SWAP', 'DSV41_PRUNE_SWAP_PREFILL', 'DSV41_PREFIX_CACHE'))):
    os.environ[key] = '0'
import torch
import engine.v41_engine as V
from engine.expert_profiles import mask_digest
from torch.profiler import profile, ProfilerActivity
from server.app import Tok, load_encoding_module, build_chat_prompt


def words(seed, n):
    """bench/bench.py::make_prompt, without the server round trip."""
    rnd = random.Random(seed)
    return [f"{i}:{''.join(rnd.choice(string.ascii_lowercase) for _ in range(rnd.randint(3, 9)))}"
            for i in range(n)]


def fit(e, enc, tok, seed, target):
    pool = words(seed, target * 3 + 64)
    count = max(16, target // 2)
    best = None
    for _ in range(8):
        _, ids, _, _ = build_chat_prompt({'messages': [{'role': 'user', 'content': ' '.join(pool[:count])}]},
                                         enc, tok, False, 75, e)
        if best is None or abs(len(ids) - target) < abs(len(best) - target):
            best = ids
        if abs(len(ids) - target) <= 16:
            break
        count = max(16, min(len(pool), int(count * target / len(ids))))
    return best


# Runtime-switchable prefill paths (module globals read at call time; both ranks switch together).
ARMS = {
    'base': dict(indexed=False, hc=False, index=False),
    'indexed': dict(indexed=True, hc=False, index=False),
    'hc': dict(indexed=False, hc=True, index=False),
    'index': dict(indexed=False, hc=False, index=True),
    'all': dict(indexed=True, hc=True, index=True),
    # + prefill-only dot_scaled experts (DSV41_FP4_PREFILL_DOT_SCALED), and its 32-row tiles
    'all_sc': dict(indexed=True, hc=True, index=True, scaled=True),
    'all_sct': dict(indexed=True, hc=True, index=True, scaled=True, tiles=True),
}


def arm_rows(name, default):
    """'base@1024' -> 1024 rows per chunk; plain names use --ab-rows."""
    return int(name.split('@')[1]) if '@' in name else default


def set_arm(e, name):
    import engine.model as M
    cfg = ARMS[name.split('@')[0]]
    M.PREFILL_ATTN_INDEXED, M.HC_PREFILL_FUSED, M.INDEX_FUSED = cfg['indexed'], cfg['hc'], cfg['index']
    import fp4_moe as K  # the module the engine's moe_fn reads at call time
    K.PREFILL_DOT_SCALED, K.PREFILL_SCALED_TILES = cfg.get('scaled', False), cfg.get('tiles', False)
    assert len(set(e.ep.gather_objects(json.dumps([name, cfg], sort_keys=True)))) == 1


def ab(e, args, enc, tok, run, seed_base, chown):
    """Each fresh prompt through every arm (rotated order); prefill time and the prefill logits."""
    names = args.arms.split(',')
    assert all(n.split('@')[0] in ARMS for n in names)
    captured = []
    original = e.model.decoder_replay
    def observe(*a, **kw):
        out = original(*a, **kw)
        captured.append(out[0][-1].detach().float().cpu())
        return out
    e.model.decoder_replay = observe
    for i, name in enumerate(names):  # compile each arm's kernels at every shape first
        set_arm(e, name)
        run(f'warm_{name}', fit(e, enc, tok, seed_base + 950 + i, 2 * args.ab_rows + 300),
            arm_rows(name, args.ab_rows))
    rows, logits = [], {}
    prompts = [(f'random{k}', fit(e, enc, tok, seed_base + 500 + k, args.tokens), 1) for k in range(args.prompts)]
    for path in [p for p in args.natural.split(',') if p]:
        text = open(path).read()
        # ~3.5 characters a token here; trim to the target and ask for a continuation.
        body = text[:int(args.tokens * 3.3)]
        _, ids, _, _ = build_chat_prompt({'messages': [{'role': 'user', 'content':
                                          'Summarize the following notes in five bullet points.\n\n' + body}]},
                                         enc, tok, False, 75, e)
        prompts.append((os.path.basename(path), ids, args.gen_tokens))
    for k, (pname, ids, gen) in enumerate(prompts):
        order = names[k % len(names):] + names[:k % len(names)]
        for name in order:
            set_arm(e, name)
            captured.clear()
            row = run(f'{pname}_{name}', ids, arm_rows(name, args.ab_rows), gen=gen)
            row.update(arm=name, prompt=k, prompt_name=pname)
            rows.append(row)
            logits[(k, name)] = captured[0]
    cmp = {}
    for name in names:
        if name == names[0]:
            continue
        d = []
        for k, (pname, _, _) in enumerate(prompts):
            a, b = logits[(k, names[0])], logits[(k, name)]
            ta = next(r['tokens'] for r in rows if r['prompt'] == k and r['arm'] == names[0])
            tb = next(r['tokens'] for r in rows if r['prompt'] == k and r['arm'] == name)
            common = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            d.append(dict(prompt=pname, tokens_equal=ta == tb, common_prefix=common, n_tokens=len(ta),
                          identical=bool(torch.equal(a, b)), max_abs=float((a - b).abs().max()),
                          top1_equal=bool(a.argmax() == b.argmax()),
                          top5_overlap=len(set(a.topk(5).indices.tolist()) & set(b.topk(5).indices.tolist())),
                          kl=float(torch.nn.functional.kl_div(b.log_softmax(-1), a.log_softmax(-1),
                                                              log_target=True, reduction='sum'))))
        cmp[name] = d
    summary = {}
    for name in names:
        ts = [r['prefill_s'] for r in rows if r['arm'] == name]
        summary[name] = dict(prefill_s=ts, mean_s=sum(ts) / len(ts))
    print('PREFILL_AB ' + json.dumps(dict(summary=summary, logits_vs_first=cmp)), flush=True)
    path = f'{args.out}/rank{e.ep.rank}-ab.json'
    with open(path, 'w') as f:
        json.dump(dict(config=e.config(), rows=rows, summary=summary, logits_vs_first=cmp), f, indent=1, default=str)
    chown(path)
    assert all(e.ep.gather_objects(True))
    os._exit(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rows', default='512,2048,1024,2048,512',
                    help='measured cold prompts, one per entry, in this order')
    ap.add_argument('--tokens', type=int, default=8192)
    ap.add_argument('--long', default='', help='extra cold prompts as rows:tokens, e.g. 2048:32768')
    ap.add_argument('--trace', default='512,2048', help='row counts to CUPTI-trace (one cold prompt each)')
    ap.add_argument('--repeat-rows', type=int, default=512, help='0 skips the hot-Engram repeat')
    ap.add_argument('--arms', default='', help='A/B mode: comma list of ARMS, interleaved per prompt')
    ap.add_argument('--prompts', type=int, default=4, help='A/B mode: fresh prompts')
    ap.add_argument('--ab-rows', type=int, default=2048, help='A/B mode: rows per chunk')
    ap.add_argument('--natural', default='', help='A/B mode: text files for natural prompts (also decoded)')
    ap.add_argument('--gen-tokens', type=int, default=64, help='A/B mode: greedy tokens decoded on natural prompts')
    ap.add_argument('--max-seq', type=int, default=0, help='context allocation (0: smallest power of two that fits)')
    args = ap.parse_args()
    rows = [int(r) for r in args.rows.split(',') if r]
    traced = [int(r) for r in args.trace.split(',') if r]
    longs = [tuple(int(v) for v in item.split(':')) for item in args.long.split(',') if item]
    biggest = max(rows + traced + [r for r, _ in longs] + [args.repeat_rows])
    assert V.MAX_CHUNK >= biggest, f'boot with DSV41_PREFILL_CHUNK>={biggest} (have {V.MAX_CHUNK})'
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    max_tokens = max([args.tokens] + [t for _, t in longs])
    e = V.V41Engine(root, max_seq=args.max_seq or max(65536, 1 << (max_tokens + 1024).bit_length()), arena_gb=90.2,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=8, keep_free_gb=6)
    os.makedirs(args.out, exist_ok=True)
    owner = os.stat('/app/results')
    tok, enc = Tok(root), load_encoding_module(root)
    maphash = mask_digest(e.model_prune_mask)
    seed_base = 7_000_000

    def chown(path):
        os.chown(path, owner.st_uid, owner.st_gid)

    def run(label, ids, chunk, trace=False, gen=1):
        V.MAX_CHUNK = chunk
        signature = hashlib.sha256(json.dumps([label, ids, chunk, trace]).encode()).hexdigest()
        assert len(set(e.ep.gather_objects(signature))) == 1, 'workload differs between ranks'
        torch.cuda.synchronize()
        prof = None
        if trace:
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                           record_shapes=False, with_stack=False, profile_memory=False)
            prof.start()
        t0 = time.perf_counter()
        out = []
        for burst in e.generate(ids, max_tokens=gen, temperature=0, seed=42, ignore_eos=True):
            out.extend(burst)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        if prof is not None:
            prof.stop()
            path = f'{args.out}/rank{e.ep.rank}-trace-{label}.json'
            prof.export_chrome_trace(path)
            chown(path)
        st = e.last_stats
        assert len(set(e.ep.gather_objects(json.dumps(out)))) == 1, 'rank disagreement'
        assert SERVING_LIKE or mask_digest(e.model_prune_mask) == maphash, 'expert map changed'
        keep = ('prompt_tokens', 'prefill_s', 'prefill_tok_s', 'prefill_budget', 'nvme_gb', 'nvme_read_s',
                'engram_rows', 'engram_s', 'engram_read_s', 'engram_cache', 'attn_s', 'moe_s', 'ep_s',
                'route_s', 'load_wait_s', 'sync_s', 'book_s', 'kernel_s', 'prefill_expert_misses',
                'prefill_chunks', 'attn_phases', 'decode_accounting', 'prefill_moe_timing')
        row = dict(label=label, rows=chunk, traced=trace, wall_s=round(wall, 3), first_token=out[:1], tokens=out,
                   **{k: st.get(k) for k in keep})
        print('PREFILL_ROW ' + json.dumps({k: row[k] for k in ('label', 'rows', 'prompt_tokens', 'prefill_s',
                                                               'prefill_tok_s', 'nvme_gb', 'engram_s',
                                                               'attn_s', 'moe_s', 'traced')}), flush=True)
        return row

    if args.arms:
        return ab(e, args, enc, tok, run, seed_base, chown)
    results = []
    # Compile every row shape (and the short last chunk) before anything is measured.
    for i, chunk in enumerate(sorted(set(rows + traced + [r for r, _ in longs] + [args.repeat_rows]) - {0})):
        results.append(run(f'warm{chunk}', fit(e, enc, tok, seed_base + 900 + i, 2048 + chunk + 100), chunk))
    seed = seed_base
    last = None
    for chunk in rows:
        seed += 1
        last = fit(e, enc, tok, seed, args.tokens)
        results.append(run(f'cold{chunk}_{seed}', last, chunk))
    if args.repeat_rows and last is not None:
        # Same tokens again: prefix reuse is off, so this recomputes everything, but the Engram
        # rows it needs were just read. The difference from a cold prompt is the read cost.
        results.append(run(f'hot{args.repeat_rows}_{seed}', last, args.repeat_rows))
    for chunk, ntok in longs:
        seed += 1
        results.append(run(f'long{chunk}_{ntok}_{seed}', fit(e, enc, tok, seed, ntok), chunk))
    for chunk in traced:
        seed += 1
        results.append(run(f'trace{chunk}_{seed}', fit(e, enc, tok, seed, args.tokens), chunk, trace=True))
    path = f'{args.out}/rank{e.ep.rank}-summary.json'
    with open(path, 'w') as f:
        json.dump(dict(config=e.config(), results=results), f, indent=1, default=str)
    chown(path)
    assert all(e.ep.gather_objects(True))
    os._exit(0)


if __name__ == '__main__':
    main()
