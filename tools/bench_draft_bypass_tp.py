"""Bounded two-rank qualification, then optional ABBA timing of no-draft decode.

No serving engine is reused. The first qualification checks 64 greedy tokens
with bypass off, forced on, alternating and adaptive. Positive-temperature
checks cover four temperatures and three nucleus settings. Pinned proposal
schedules must repeat with one seed; adaptive requests check rank agreement,
because measured timing can legitimately change their proposal/RNG schedule.

The bypass uses the existing two-row graph and discards its dummy row. Direct
root/dummy logits are not compared by restoring live cache/hash internals: normal
fresh-prefill rollouts exercise compressor rollback, MTP state and mode switches.
Timing remains opt-in after qualification, with matching full-length warmups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
# Match the existing block benchmark prompts without importing its timeline
# module, which changes diagnostic environment selectors at import time.
WORKLOADS = (
    ('html', 'Write a simple, self-contained HTML page for a neighborhood coffee shop. '
             'Include a heading, a short introduction, three menu items with prices, '
             'opening hours, and a footer. Use a small embedded CSS stylesheet with '
             'a cream background, dark text, and comfortable spacing. No JavaScript, '
             'external assets, or complicated layout. Format it readably, roughly '
             '50 lines. Output only the complete HTML, without explanations.', 0.),
    ('python', 'Write a Python module implementing an LRU cache class with get, put and a '
               'max-size eviction policy, plus five unittest test cases. Output only the code.', 0.),
    ('explain', 'Explain in plain prose, for a curious non-specialist, why the sky is blue and '
                'why sunsets are red. Use about five paragraphs and no lists or headings.', 0.),
    ('story', 'Write an original short story, about 400 words, about a lighthouse keeper who '
              'finds an unusual object washed ashore. Literary tone, no title.', 0.),
    ('story_t07', 'Write an original short story, about 400 words, about a lighthouse keeper who '
                  'finds an unusual object washed ashore. Literary tone, no title.', .7),
)


def main(engine=None, argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--qualification-tokens', type=int, default=64)
    ap.add_argument('--sample-tokens', type=int, default=16)
    ap.add_argument('--qualification-only', action='store_true')
    ap.add_argument('--greedy-only', action='store_true')
    ap.add_argument('--max-tokens', type=int, default=96)
    ap.add_argument('--max-seq', type=int, default=4096)
    ap.add_argument('--workloads', default='explain,python')
    ap.add_argument('--measured-modes', default='on,adaptive')
    args = ap.parse_args(argv)
    if min(args.qualification_tokens, args.sample_tokens, args.max_tokens) < 2:
        ap.error('token budgets must be at least 2')
    names = args.workloads.split(',')
    modes = args.measured_modes.split(',')
    if not set(modes) <= {'on', 'adaptive'}:
        ap.error('measured modes must be on and/or adaptive')
    external_engine = engine is not None
    if not external_engine:
        # Standalone setup must precede module-level graph width/feature resolution.
        for key in ('DSV41_PREFIX_CACHE', 'DSV41_PREFIX_DISK', 'DSV41_PREFIX_RESPONSE',
                    'DSV41_PRUNE_SWAP', 'DSV41_PRUNE_SWAP_PREFILL', 'DSV41_ADAPT_URGENT'):
            os.environ[key] = '0'
        os.environ.update(DSV41_ADAPT_SENSITIVITY='off', DSV41_DRAFT_BYPASS='1',
            DSV41_BLOCK_CONFIDENCE='1', DSV41_BLOCK_DYNAMIC='3,5', DSV41_BLOCK='5',
            DSV41_LOOKUP_DRAFT_ENABLED='0', DSV41_SPEC_CONF='0', DSV41_TREE_PROBE='0')
    import torch
    import engine.v41_engine as V
    from server.app import Tok, load_encoding_module, build_chat_prompt
    from engine.draft_bypass import DraftBypassPolicy
    if not set(names) <= {name for name, _, _ in WORKLOADS}:
        ap.error('unknown workload')
    if not external_engine:
        V.save_prune_db = lambda *a, **kw: None
        root = os.environ['MODEL_DIR']
        engine = V.V41Engine(root, max_seq=args.max_seq,
            arena_gb=float(os.environ.get('ARENA_GB', '90.1')),
            trace_stats='/app/results/trace-union/stats/coverage.json',
            spec=True, prune_keep=float(os.environ.get('PRUNE_KEEP', '.61')),
            transient_slots=int(os.environ.get('TRANSIENT_SLOTS', '16')),
            keep_free_gb=float(os.environ.get('KEEP_FREE_GB', '6')))
    root = engine.model_dir
    tok, enc = Tok(root), load_encoding_module(root)
    eos = tok.token_to_id(enc.eos_token)
    assert engine.confidence_depth_policy is not None
    prompts = {}
    for name, prompt, _ in WORKLOADS:
        _, prompts[name], _, _ = build_chat_prompt(
            {'messages': [{'role': 'user', 'content': prompt}]}, enc, tok, False, 75, engine)
    report = {'config': engine.config(), 'qualification': [], 'runs': [],
              'mismatches': [], 'summary': {},
              'note': 'Bypass reuses width2; adaptive sampled schedules need not reproduce exact seeds.'}

    def save():
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f'bypass-rank{engine.ep.rank}.json'
        path.write_text(json.dumps(report, indent=2) + '\n')
        os.chmod(path, 0o600)
        owner = os.stat('/app/results')
        os.chown(path, owner.st_uid, owner.st_gid)

    def run(name, mode, count, temperature=0., top_p=.95, group='qualification', pin_width=None):
        original_enabled = engine.draft_bypass_enabled
        had_pinned = hasattr(engine, '_bypass_pinned')
        original_pinned = getattr(engine, '_bypass_pinned', None)
        original_width = engine.confidence_depth_policy.pinned
        original_lookup = V.LOOKUP_DRAFT_ENABLED
        V.LOOKUP_DRAFT_ENABLED = False
        engine.draft_bypass_enabled = mode != 'off'
        engine._bypass_pinned = True if mode == 'on' else None
        engine.confidence_depth_policy.pinned = pin_width
        original_decide = DraftBypassPolicy.decide
        if mode == 'alternate':
            # Choose before proposals. Frequent return to DSpark checks its seeded
            # KV after discarded dummy rows and one-token compressor rollbacks.
            DraftBypassPolicy.decide = lambda policy: sum(policy.steps.values()) % 3 == 0
        output = []
        try:
            for burst in engine.generate(prompts[name], max_tokens=count, temperature=temperature,
                    top_p=top_p, seed=42, stop_token_ids={eos}):
                output.extend(burst)
        finally:
            DraftBypassPolicy.decide = original_decide
            engine.draft_bypass_enabled = original_enabled
            if had_pinned:
                engine._bypass_pinned = original_pinned
            else:
                del engine._bypass_pinned
            engine.confidence_depth_policy.pinned = original_width
            V.LOOKUP_DRAFT_ENABLED = original_lookup
        st = engine.last_stats
        digest = hashlib.sha256(json.dumps(output).encode()).hexdigest()
        parity = len(set(engine.ep.gather_objects(digest))) == 1
        item = {'workload': name, 'mode': mode, 'tokens': len(output),
                'temperature': temperature, 'top_p': top_p, 'pin_width': pin_width,
                'sha256': digest, 'rank_parity': parity,
                'decode_tok_s': st['decode_tok_s'], 'decode_s': st['decode_s'],
                'steps': st['steps'], 'accept_len_mean': st.get('accept_len_mean'),
                'draft_bypass': st.get('draft_bypass'), 'spec_depth': st.get('spec_depth')}
        report[group].append(item)
        if not parity:
            report['mismatches'].append(f'{name}/{mode}/rank-parity')
        # Rank 0 owns decisions and therefore the policy observation counters.
        if engine.ep.rank == 0 and mode == 'on':
            steps = (item['draft_bypass'] or {}).get('steps', {})
            assert steps.get('draft', 0) == 0 and steps.get('bypass', 0) > 0, 'bypass was not exercised'
        print('DRAFT_BYPASS_RUN ' + json.dumps(item), flush=True)
        save()
        assert parity, 'rank output mismatch'
        return output

    # Fresh requests prevent comparisons from sharing committed KV state. Force-on
    # advances by one token, exercising both compressor start parities naturally.
    for name in ('explain', 'python'):
        reference = run(name, 'off', args.qualification_tokens)
        for mode in ('on', 'alternate', 'adaptive'):
            output = run(name, mode, args.qualification_tokens)
            exact = all(engine.ep.gather_objects(output == reference))
            if not exact:
                report['mismatches'].append(f'{name}/{mode}/greedy-equality')
            save()
            assert exact, f'{name}/{mode}: greedy output changed'
    if not args.greedy_only:
        for temperature in (.1, .6, 1., 2.):
            for top_p in (.5, .95, 1.):
                for mode in ('off', 'on'):
                    # Fixed DSpark width protects this repeat check from wall-time
                    # changes to the confidence controller's graph schedule.
                    first = run('story_t07', mode, args.sample_tokens, temperature, top_p, pin_width=3)
                    repeat = run('story_t07', mode, args.sample_tokens, temperature, top_p, pin_width=3)
                    exact = all(engine.ep.gather_objects(first == repeat))
                    if not exact:
                        report['mismatches'].append(f'{mode}/{temperature}/{top_p}/seed-repeat')
                    save()
                    assert exact, 'pinned sampled mode did not repeat'
                run('story_t07', 'adaptive', args.sample_tokens, temperature, top_p)

    if not args.qualification_only:
        for name in names:
            for candidate in modes:
                for mode in ('off', candidate):
                    run(name, mode, args.max_tokens, group='qualification')
                reference = None
                measured = []
                for mode in ('off', candidate, candidate, 'off'):
                    output = run(name, mode, args.max_tokens, group='runs')
                    reference = output if reference is None else reference
                    exact = all(engine.ep.gather_objects(output == reference))
                    assert exact, f'{name}/{mode}: measured greedy output changed'
                    measured.append(report['runs'][-1])
                by_mode = {mode: [r for r in measured if r['mode'] == mode] for mode in ('off', candidate)}
                report['summary'][f'{name}/{candidate}'] = {
                    mode: {'median_tok_s': statistics.median(r['decode_tok_s'] for r in rows),
                           'median_ms_step': statistics.median(1000 * r['decode_s'] / r['steps'] for r in rows),
                           'median_tokens': statistics.median(r['tokens'] for r in rows)}
                    for mode, rows in by_mode.items()}
                save()
    report['graph_keys'] = sorted(str(key) for key in engine.fast.graphs)
    report['peak_allocated_gb'] = torch.cuda.max_memory_allocated() / 1e9
    save()
    assert all(engine.ep.gather_objects(not report['mismatches']))
    print('DRAFT_BYPASS_PASS', flush=True)
    if external_engine:
        return report
    os._exit(0)


if __name__ == '__main__':
    main()
