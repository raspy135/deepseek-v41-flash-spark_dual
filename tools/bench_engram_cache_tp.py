"""2-node row-cache A/B: cold application cache, warm repeat, exact token parity.

Cache-off arms release its arrays. Cache-on arrays are fully touched before timing
so the memory-pressure comparison includes the entire configured capacity. OS
page cache is never dropped. Frozen expert ranking; no prefix reuse or writes.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import gc
sys.path[:0] = ['/app', '/app/tools']
from bench_decode_timeline_tp import V, torch, Tok, load_encoding_module, build_chat_prompt
from bench_decode_block_tp import WORKLOADS
from engine.engram_cache import PackedRowCache, cache_budget_mb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--qualify-only', action='store_true', help='Short asymmetric-capacity deployment gate')
    args = ap.parse_args()
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=524288, arena_gb=90,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=16, keep_free_gb=6)
    tok, enc = Tok(root), load_encoding_module(root)
    eos = tok.token_to_id(enc.eos_token)
    budget = cache_budget_mb(e.ep.rank) * 1024**2 // len(e.tables)
    prompts = {}
    for name, prompt, _ in WORKLOADS:
        _, prompts[name], _, _ = build_chat_prompt(
            {'messages': [{'role': 'user', 'content': prompt}]}, enc, tok, False, 75, e)
    report = dict(config=e.config(), runs=[], mismatches=[],
                  note='Cold packed-cache arms start empty, OS page cache retained; full cache capacity touched before timing. Warm-repeat is a favorable upper bound.')
    references = {}
    timing = dict(active=False, wait_s=0., calls=0)
    submit = e.eg_pool.submit
    step = e.fast.step
    def timed_submit(*a, **kw):
        f = submit(*a, **kw)
        result = f.result
        def timed_result(*a, **kw):
            t = time.perf_counter()
            try:
                return result(*a, **kw)
            finally:
                if timing['active']:
                    timing['wait_s'] += time.perf_counter() - t
                    timing['calls'] += 1
        f.result = timed_result
        return f
    def timed_step(*a, **kw):
        timing['active'] = True
        try:
            return step(*a, **kw)
        finally:
            timing['active'] = False
    e.eg_pool.submit = timed_submit
    e.fast.step = timed_step

    def save():
        os.makedirs(args.out, exist_ok=True)
        path = f'{args.out}/cache-rank{e.ep.rank}.json'
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        owner = os.stat('/app/results')
        os.chown(path, owner.st_uid, owner.st_gid)

    def cache_totals():
        reports = [t.row_cache.report() for t in e.tables.values() if t.row_cache is not None]
        return {k: sum(r[k] for r in reports) for k in ('hits', 'misses', 'calls')}

    def run(name, mode, count, measured=True, temperature=0):
        for table in e.tables.values():
            if mode == 'off':
                table.row_cache = None
            elif mode == 'cold':
                if table.row_cache is None:
                    table.row_cache = PackedRowCache(budget, table.n_rows)
                table.row_cache.clear()
                table.row_cache.data.fill(0)  # include full memory pressure, outside timed request
            elif mode == 'warm':
                assert table.row_cache is not None
        gc.collect()
        e.ep.gather_objects(True)
        before = cache_totals()
        timing.update(active=False, wait_s=0., calls=0)
        out = []
        for burst in e.generate(prompts[name], max_tokens=count, temperature=temperature,
                                seed=42, stop_token_ids={eos}):
            out.extend(burst)
        digest = hashlib.sha256(json.dumps(out).encode()).hexdigest()
        parity = len(set(e.ep.gather_objects(digest))) == 1
        key = (name, count, temperature)
        equal = references.setdefault(key, out) == out
        equal = all(e.ep.gather_objects(equal))
        st = e.last_stats
        totals = cache_totals()
        cache = {k: totals[k] - before[k] for k in totals}
        cache['hit_rate'] = cache['hits'] / max(1, cache['hits'] + cache['misses'])
        with open('/proc/meminfo') as f:
            mem = {l.split(':')[0]: int(l.split()[1]) for l in f if l.startswith(('MemAvailable:', 'SwapFree:'))}
        item = dict(workload=name, mode=mode, measured=measured, temperature=temperature,
                    tokens=len(out), decode_tok_s=st['decode_tok_s'], decode_s=st['decode_s'],
                    prefill_s=st['prefill_s'], steps=st['steps'], accept_len_mean=st['accept_len_mean'],
                    future_wait_s=timing['wait_s'], future_calls=timing['calls'],
                    engram_read_s=st['engram_read_s'], cache=cache, memory_kib=mem,
                    rank_parity=parity, exact_tokens=equal, sha256=digest, token_ids=out)
        report['runs'].append(item)
        if not (parity and equal): report['mismatches'].append(f'{name}/{mode}/{count}')
        save()
        print('ENGRAM_CACHE_RUN ' + json.dumps({k: v for k, v in item.items() if k != 'token_ids'}), flush=True)
        assert parity and equal, 'cache changed output tokens'

    for name in (() if args.qualify_only else ('html', 'python', 'explain')):
        run(name, 'off', 128, False)
        run(name, 'cold', 128, False)
        for mode in ('off', 'cold', 'cold', 'off'):
            run(name, mode, args.max_tokens)
        run(name, 'cold', args.max_tokens, False)
        run(name, 'warm', args.max_tokens)
    if args.qualify_only:
        for mode in ('off', 'cold', 'warm'):
            run('html', mode, 128, False)
    # Sampled path checks fixed RNG consumption using a fixed verification depth.
    e.depth_policy.pinned = 3
    run('story_t07', 'off', 128, False, .7)
    run('story_t07', 'cold', 128, False, .7)
    if not args.qualify_only:
        from bench_engram_remote import measure_remote
        report['remote_probe'] = measure_remote(e)
    save()
    assert all(e.ep.gather_objects(True))
    print('ENGRAM_CACHE_PASS', flush=True)
    os._exit(0)


if __name__ == '__main__': main()
