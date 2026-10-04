"""Disposable two-node qualification and ABBA timing of confidence vs adaptive depth.

Set DSV41_BLOCK_CONFIDENCE=1, DSV41_BLOCK=5, DSV41_BLOCK_DYNAMIC=3,5.
Both arms share the compiled confidence head; only confidence pays its readback and
second control broadcast. This isolates policy savings, not head-compute overhead.
No serving configuration or prune rankings are written.
"""
import argparse
import hashlib
import json
import os
import sys
sys.path[:0] = ['/app', '/app/tools']
from bench_decode_timeline_tp import V, torch, Tok, load_encoding_module, build_chat_prompt
from bench_decode_block_tp import WORKLOADS
import engine.fastdecode as FD


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-tokens', type=int, default=512)
    args = ap.parse_args()
    assert FD.CONFIDENCE_DEPTHS == (1, 3, 5)
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=524288, arena_gb=90,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=16, keep_free_gb=6)
    tok, enc = Tok(root), load_encoding_module(root)
    eos = tok.token_to_id(enc.eos_token)
    confidence = e.confidence_depth_policy
    choose = confidence.choose
    report = dict(config=e.config(), runs=[], qualification=[], mismatches=[],
                  note='Both arms compute confidence head; only confidence reads it before verify.')
    prompts = {}
    for name, prompt, _ in WORKLOADS:
        _, prompts[name], _, _ = build_chat_prompt(
            {'messages': [{'role': 'user', 'content': prompt}]}, enc, tok, False, 75, e)

    def save():
        os.makedirs(args.out, exist_ok=True)
        path = f'{args.out}/confidence-rank{e.ep.rank}.json'
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        owner = os.stat('/app/results')
        os.chown(path, owner.st_uid, owner.st_gid)

    def run(name, mode, count, temperature=0, group='qualification'):
        e.confidence_depth_policy = None if mode == 'adaptive' else confidence
        confidence.pinned = {'pin1': 1, 'pin3': 3, 'pin5': 5}.get(mode)
        confidence.choose = choose
        if mode == 'alternate':
            state = [0]
            def alternate(logits):
                confidence.pinned = (1, 3, 5)[state[0] % 3]
                state[0] += 1
                return choose(logits)
            confidence.choose = alternate
        out = []
        for burst in e.generate(prompts[name], max_tokens=count, temperature=temperature,
                                seed=42, stop_token_ids={eos}):
            out.extend(burst)
        confidence.pinned, confidence.choose = None, choose
        digest = hashlib.sha256(json.dumps(out).encode()).hexdigest()
        parity = len(set(e.ep.gather_objects(digest))) == 1
        st = e.last_stats
        item = dict(workload=name, mode=mode, temperature=temperature, tokens=len(out),
                    decode_tok_s=st['decode_tok_s'], decode_s=st['decode_s'],
                    spec_depth=st['spec_depth'], sha256=digest, rank_parity=parity,
                    token_ids=out)
        report[group].append(item)
        if not parity:
            report['mismatches'].append(f'{name}/{mode}/rank-parity')
        print('CONFIDENCE_RUN ' + json.dumps({k: v for k, v in item.items() if k != 'token_ids'}), flush=True)
        save()
        assert parity, 'rank token mismatch'
        return out

    # Actual width switching, including on first graph capture. No ahead-of-position capture:
    # it can overwrite the KV ring and corrupt subsequent attention.
    reference = run('html', 'alternate', 256)
    for mode in ('pin1', 'pin3', 'pin5', 'adaptive', 'confidence'):
        out = run('html', mode, 256)
        equal = all(e.ep.gather_objects(out == reference))
        if not equal:
            report['mismatches'].append(f'html/{mode}/width-parity')
        save()
        assert equal, f'greedy output differs at {mode}'

    # Sampled requests must bypass confidence lookahead altogether. Pin the legacy
    # schedule so the existing sampler's random stream is the same in both arms.
    e.depth_policy.pinned = 3
    a = run('story_t07', 'adaptive', 128, .7)
    b = run('story_t07', 'confidence', 128, .7)
    equal = all(e.ep.gather_objects(a == b))
    if not equal:
        report['mismatches'].append('sampled-fallback')
    save()
    assert equal, 'sampled fallback changed tokens'
    e.depth_policy.pinned = None

    for name in ('html', 'python', 'explain'):
        for mode in ('adaptive', 'confidence'):
            run(name, mode, 128)
        reference = None
        for mode in ('adaptive', 'confidence', 'confidence', 'adaptive'):
            out = run(name, mode, args.max_tokens, group='runs')
            reference = out if reference is None else reference
            equal = all(e.ep.gather_objects(out == reference))
            if not equal:
                report['mismatches'].append(f'{name}/{mode}/measured-parity')
            save()
            assert equal, f'{name}: greedy output differs'
    report['graph_keys'] = sorted(str(k) for k in e.fast.graphs)
    report['peak_allocated_gb'] = torch.cuda.max_memory_allocated() / 1e9
    save()
    assert all(e.ep.gather_objects(True))
    print('CONFIDENCE_PASS', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
