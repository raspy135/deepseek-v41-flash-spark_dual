"""Disposable TP2 depth/cache comparison with one frozen dynamic expert map.

Use run_two_node_gate.sh after stopping serving. No production history writes.
All modes use the same five-token drafter; only verification depth changes.
Greedy output must match across depths/cache modes and both ranks.
"""
import argparse
import gc
import hashlib
import json
import os
import sys

sys.path[:0] = ['/app', '/app/tools']
from bench_decode_timeline_tp import V, Tok, load_encoding_module, build_chat_prompt
from engine.engram_cache import PackedRowCache, cache_budget_mb
from bench.bench import WORKLOADS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-tokens', type=int, default=384)
    args = ap.parse_args()
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=524288, arena_gb=90.2,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=8, keep_free_gb=6)
    assert e.dynamic_experts and e.depth_policy is not None
    assert not V.ADAPT.swap and not V.ADAPT.decode_tokens
    tok, enc = Tok(root), load_encoding_module(root)
    pol = e.depth_policy
    assert e.confidence_depth_policy is None
    from engine.expert_profiles import mask_digest
    map_hash = mask_digest(e.model_prune_mask)
    assert len(set(e.ep.gather_objects(map_hash))) == 1
    report = dict(config=e.config(), initial_map=map_hash, runs=[])
    references = {}
    budget = cache_budget_mb(e.ep.rank) * 1024**2 // len(e.tables)
    assert budget > 0
    prompts = {}
    for name in ('code', 'prose'):
        _, prompts[name], _, _ = build_chat_prompt(
            {'messages': [{'role': 'user', 'content': '[req 41420] ' + WORKLOADS[name]}]},
            enc, tok, False, 75, e)

    def save():
        path = f'{args.out}/depth-rank{e.ep.rank}.json'
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        owner = os.stat('/app/results')
        os.chown(path, owner.st_uid, owner.st_gid)

    def run(name, mode, cache, count, measured=True, temperature=0):
        pol.pinned = {'3': 3, '5': 5}.get(mode)
        for table in e.tables.values():
            if cache == 'off':
                table.row_cache = None
            elif cache == 'cold':
                table.row_cache = PackedRowCache(budget, table.n_rows)
                table.row_cache.data.fill(0)
            else:
                assert cache == 'warm' and table.row_cache is not None
        gc.collect()
        identity = (name, mode, cache, count, temperature, measured)
        assert len(set(e.ep.gather_objects(identity))) == 1
        out = []
        for burst in e.generate(prompts[name], max_tokens=count, temperature=temperature,
                                seed=42, ignore_eos=True):
            out.extend(burst)
        digest = hashlib.sha256(json.dumps(out).encode()).hexdigest()
        assert len(set(e.ep.gather_objects(digest))) == 1, 'rank outputs differ'
        assert mask_digest(e.model_prune_mask) == map_hash, 'resident map changed'
        key = (name, count, temperature, mode if temperature else 'greedy')
        exact = references.setdefault(key, digest) == digest
        st = dict(e.last_stats)
        item = dict(workload=name, mode=mode, cache=cache, measured=measured,
                    temperature=temperature, tokens=len(out), token_ids=out, sha256=digest,
                    exact=exact, decode_tok_s=st['decode_tok_s'], decode_s=st['decode_s'],
                    steps=st['steps'], accept_len_mean=st['accept_len_mean'],
                    spec_depth=st['spec_depth'], engram_read_s=st.get('engram_read_s'),
                    prune_miss_request=st.get('prune_miss_request'))
        report['runs'].append(item)
        save()
        print('RESIDENCY_DEPTH_RUN ' + json.dumps({k:v for k,v in item.items()
              if k not in ('token_ids', 'prune_miss_request')}), flush=True)
        assert exact, 'greedy output or repeated fixed-depth sampled output changed'

    # Capture both widths/parities and calibrate the process's measured step costs.
    for mode in ('3', '5'):
        run('code', mode, 'off', 128, False)
    for name in ('code', 'prose'):
        for mode in ('3', '5', 'adaptive', 'adaptive', '5', '3'):
            run(name, mode, 'off', args.max_tokens)
    # ABBA cache comparison at fixed depth; same experts, tokens and graph shapes.
    for cache in ('off', 'cold', 'cold', 'off'):
        run('code', '3', cache, args.max_tokens)
    run('code', '3', 'cold', args.max_tokens, False)
    run('code', '3', 'warm', args.max_tokens)
    # Sampling parity between cache implementations at a fixed schedule.
    for cache in ('off', 'cold'):
        run('code', '3', cache, 128, False, .6)
    save()
    assert all(e.ep.gather_objects(True))
    print('RESIDENCY_DEPTH_PASS', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
