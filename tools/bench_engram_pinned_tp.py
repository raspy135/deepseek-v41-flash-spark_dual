"""Engram H2D staging A/B in one loaded TP2 process: pageable vs DSV41_ENGRAM_PINNED.

The pageable copy synchronizes the stream between decode graph segments, so the next
segment's cudaGraphLaunch (0.55-1.05 ms of host time for 1,025/2,068 nodes) runs with the
GPU already drained. Pinned staging copies the same bytes non-blocking, so outputs must
be bit-identical; only wall time per step may move. Arms alternate ABBA on a frozen map at
fixed depth 3. Use the disposable two-node gate, not serving.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
sys.path[:0] = ['/app', '/app/tools']
for key in ('DSV41_PRUNE_SWAP', 'DSV41_PRUNE_SWAP_PREFILL', 'DSV41_PREFIX_CACHE',
            'DSV41_PREFIX_DISK', 'DSV41_PREFIX_RESPONSE', 'DSV41_GPU_TIMING', 'DSV41_STEP_TIMING'):
    os.environ[key] = '0'
os.environ['DSV41_PRUNE_ADAPT'] = '1'
import torch
import engine.v41_engine as V
from engine.expert_profiles import mask_digest
from server.app import Tok, load_encoding_module, build_chat_prompt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tokens', type=int, default=192)
    ap.add_argument('--blocks', type=int, default=2, help='ABBA blocks per workload')
    args = ap.parse_args()
    assert 32 <= args.tokens <= 1024 and 1 <= args.blocks <= 4
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=32768, arena_gb=90.2,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=8, keep_free_gb=6)
    e.confidence_depth_policy = None
    e.depth_policy.pinned = 3
    tables = list(e.tables.values())
    assert tables, 'engram tables missing'
    os.makedirs(args.out, exist_ok=True)
    tok, enc = Tok(root), load_encoding_module(root)
    from bench.bench import WORKLOADS
    ids = {name: build_chat_prompt({'messages': [{'role': 'user', 'content': '[req 70101] ' + WORKLOADS[name]}]},
                                   enc, tok, False, 75, e)[1] for name in ('prose', 'code')}
    signature = hashlib.sha256(json.dumps([ids, args.tokens, args.blocks]).encode()).hexdigest()
    assert len(set(e.ep.gather_objects(signature))) == 1, 'workload differs between ranks'
    maphash = mask_digest(e.model_prune_mask)
    report = {'config': e.config(), 'map': maphash, 'tokens': args.tokens, 'runs': []}
    refs = {}
    path = f'{args.out}/rank{e.ep.rank}.json'

    def save():
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)

    def generate(name, arm, n, measured):
        pinned = arm == 'pinned'
        for t in tables:
            t.pinned = pinned
        assert len(set(e.ep.gather_objects((arm, pinned)))) == 1, 'arm differs between ranks'
        engram_before = sum(t.stats['seconds'] for t in tables)
        output = []
        for burst in e.generate(ids[name], max_tokens=n, temperature=0, seed=42, ignore_eos=True):
            output.extend(burst)
        digest = hashlib.sha256(json.dumps(output).encode()).hexdigest()
        assert len(set(e.ep.gather_objects(digest))) == 1, 'rank disagreement'
        assert mask_digest(e.model_prune_mask) == maphash, 'expert map changed'
        stats = dict(e.last_stats)
        row = {'name': name, 'arm': arm, 'measured': measured, 'tokens': len(output), 'hash': digest,
               'exact': refs.setdefault((name, n), digest) == digest,
               'steps': stats['steps'], 'decode_s': stats['decode_s'],
               'decode_tok_s': stats['decode_tok_s'], 'accept_len_mean': stats['accept_len_mean'],
               'wall_ms_per_step': 1000 * stats['decode_s'] / max(1, stats['steps']),
               'engram_to_device_s': round(sum(t.stats['seconds'] for t in tables) - engram_before, 4)}
        report['runs'].append(row)
        save()
        print('ENGRAM_PINNED_RUN ' + json.dumps(row), flush=True)
        assert row['exact'], f'{name} output changed under {arm}: staging must not change bits'

    # Warm both workloads' graphs and the row cache before anything is measured.
    for name in ('prose', 'code'):
        generate(name, 'pageable', 32, False)
        generate(name, 'pinned', 32, False)
    order = ['pageable', 'pinned', 'pinned', 'pageable'] * args.blocks
    for arm in order:
        for name in ('prose', 'code'):
            generate(name, arm, args.tokens, True)
    summary = {}
    for name in ('prose', 'code'):
        for arm in ('pageable', 'pinned'):
            rows = [r for r in report['runs'] if r['measured'] and r['name'] == name and r['arm'] == arm]
            ms = [r['wall_ms_per_step'] for r in rows]
            summary[f'{name}/{arm}'] = {'n': len(rows), 'ms_per_step_mean': statistics.mean(ms),
                                        'ms_per_step_min': min(ms), 'ms_per_step_max': max(ms),
                                        'tok_s_mean': statistics.mean(r['decode_tok_s'] for r in rows),
                                        'engram_to_device_s_mean': statistics.mean(r['engram_to_device_s'] for r in rows)}
    report['summary'] = summary
    report['completed'] = True
    save()
    owner = os.stat('/app/results')
    os.chown(path, owner.st_uid, owner.st_gid)
    print('ENGRAM_PINNED_SUMMARY ' + json.dumps(summary), flush=True)
    assert all(e.ep.gather_objects(True))
    os._exit(0)


if __name__ == '__main__':
    main()
