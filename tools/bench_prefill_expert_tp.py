"""Paired prefill-only dot_scaled quality/timing gate on a frozen TP2 map.

All other serving numerics remain unchanged. Both ranks switch the same explicit
prefill flag at request boundaries; decode graphs can be reused because they do
not consult that flag. No prefix reuse, expert adaptation, or demand persistence.
Run only in disposable gate containers, with serving stopped.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
sys.path[:0] = ['/app', '/app/tools']
for key in ('DSV41_PRUNE_SWAP', 'DSV41_PRUNE_SWAP_PREFILL', 'DSV41_PREFIX_CACHE',
            'DSV41_PREFIX_DISK', 'DSV41_PREFIX_RESPONSE', 'DSV41_GPU_TIMING',
            'DSV41_STEP_TIMING', 'DSV41_PREFILL_GRAPHS'):
    os.environ[key] = '0'
os.environ['DSV41_PRUNE_ADAPT'] = '1'
# The serving shadow predictor requires swaps to be enabled. This gate freezes
# the resident map, so disable prediction as well as its persistent updates.
os.environ['DSV41_PREDICTIVE_PREFILL'] = 'off'
import torch
import fp4_moe as K
import engine.v41_engine as V
from engine.expert_profiles import mask_digest
from server.app import Tok, load_encoding_module, build_chat_prompt
from prefill_expert_probes import grade, short_probes, retrieval_probe
from bench_prefill_trace_tp import fit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--suite', choices=('short', 'long', 'timing', 'all'), default='all')
    ap.add_argument('--max-seq', type=int, default=65536)
    ap.add_argument('--repeat', type=int, default=1)
    args = ap.parse_args()
    assert not K.DOT_SCALED and K.CUDA_DECODE
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=args.max_seq, arena_gb=90.2,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=8, keep_free_gb=6)
    tok, enc = Tok(root), load_encoding_module(root)
    eos = tok.token_to_id(enc.eos_token)
    assert eos is not None
    maphash = mask_digest(e.model_prune_mask)
    os.makedirs(args.out, exist_ok=True)
    owner = os.stat('/app/results')
    path = f'{args.out}/rank{e.ep.rank}.json'
    result = dict(config=e.config(), map_sha256=maphash, suite=args.suite, rows=[])
    def save():
        with open(path, 'w') as f:
            json.dump(result, f, indent=1, ensure_ascii=False, default=str)
        os.chown(path, owner.st_uid, owner.st_gid)
    def render(body):
        return build_chat_prompt(body, enc, tok, False, 75, e)[1]
    def arm(name):
        assert len(set(e.ep.gather_objects(name))) == 1
        K.PREFILL_DOT_SCALED = name == 'scaled'
        e.fp4_prefill_dot_scaled = K.PREFILL_DOT_SCALED
        assert not K.DOT_SCALED and K.CUDA_DECODE
    def run(name, p, ids, rep=0, warm=False):
        arm(name)
        signature = hashlib.sha256(json.dumps([p, ids, rep], sort_keys=True).encode()).hexdigest()
        assert len(set(e.ep.gather_objects(signature))) == 1
        out = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        for burst in e.generate(ids, max_tokens=p['limit'], temperature=0, seed=42,
                                stop_token_ids={eos}, ignore_eos=p['family']=='timing'):
            out.extend(burst)
        torch.cuda.synchronize()
        wall = time.perf_counter()-start
        assert len(set(e.ep.gather_objects(hashlib.sha256(json.dumps(out).encode()).hexdigest()))) == 1, 'rank output disagreement'
        assert mask_digest(e.model_prune_mask) == maphash, 'expert map moved'
        text = tok.decode([t for t in out if t != eos])
        st = e.last_stats
        row = dict(arm=name, name=p['name'], family=p['family'], repeat=rep, warm=warm,
                   prompt_tokens=len(ids), prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                   text=text, tokens=out, eos=eos in out, limit=p['limit'], wall_s=wall,
                   grade=None if p['family']=='timing' else grade(p, text, enc))
        for key in ('prefill_s', 'prefill_tok_s', 'decode_tok_s', 'accept_len_mean', 'decode_s',
                    'prefill_budget', 'prefill_moe_timing', 'prefix_cached_tokens', 'prefill_expert_misses'):
            row[key] = st.get(key)
        assert not st.get('prefix_cached_tokens'), 'prefix reused'
        result['rows'].append(row)
        save()
        if e.ep.rank == 0:
            print('EXPERT_AB '+json.dumps({k:row[k] for k in ('arm','name','warm','prompt_tokens','prefill_s','grade','eos')},
                                        ensure_ascii=False), flush=True)
    # Compile both arithmetic arms at encoder/replay widths before measurements.
    for name in ('base', 'scaled'):
        run(name, dict(name='warm-long', family='timing', limit=2),
            fit(e, enc, tok, 810091, 4400), warm=True)
        for p in short_probes()[:1]:
            run(name, p, render(p['body']), warm=True)
    probes = []
    if args.suite in ('short', 'all'):
        probes += [(p, render(p['body'])) for p in short_probes()]
    if args.suite in ('long', 'all'):
        for i, (size, pos) in enumerate(((8192,.1),(8192,.5),(8192,.9),(16384,.1),(16384,.9),(32768,.1),(32768,.5),(32768,.9))):
            probes.append(retrieval_probe(size, pos, 814100+i, render))
    if args.suite in ('timing', 'all'):
        for i, size in enumerate((8192,8192,8192,8192,32768,32768)):
            probes.append((dict(name=f'random-{size}-{i}', family='timing', limit=1),
                           fit(e, enc, tok, 815000+i, size)))
    for rep in range(args.repeat):
        for i, (p, ids) in enumerate(probes):
            for name in (('base','scaled') if (i+rep)%2==0 else ('scaled','base')):
                run(name, p, ids, rep)
    summary = {}
    for family in sorted({p['family'] for p,_ in probes}):
        summary[family] = {}
        for name in ('base', 'scaled'):
            rows = [r for r in result['rows'] if not r['warm'] and r['arm']==name and r['family']==family]
            summary[family][name] = dict(n=len(rows), passed=sum(bool(r['grade'] and r['grade']['pass_']) for r in rows),
                                         mean_prefill_s=statistics.mean(r['prefill_s'] for r in rows))
    result['summary'] = summary
    save()
    if e.ep.rank == 0:
        print('EXPERT_SUMMARY '+json.dumps(summary), flush=True)
    assert all(e.ep.gather_objects(True))
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    # Engine construction allocates arenas that loader threads populate. Creating
    # those tensors inside thread-local inference_mode makes their writes illegal.
    # generate()/model methods manage inference mode themselves.
    main()
