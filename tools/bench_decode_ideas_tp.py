"""Screen decode experiments in one disposable two-rank model load.

Head-only/lookup/staging microbenchmarks run first. This driver qualifies copies
and root-only verification, then measures BF16/FP8 draft heads in ABBA order.
The target head, attention, routing mask and expert residency are held fixed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path[:0] = ['/app', '/app/tools']
from bench_decode_timeline_tp import V, torch, Tok, load_encoding_module, build_chat_prompt
from bench_decode_block_tp import WORKLOADS
from bench_lookup_draft_tp import main as qualify_lookup
from bench_draft_bypass_tp import main as qualify_bypass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tokens', type=int, default=256)
    ap.add_argument('--qualification-only', action='store_true')
    args = ap.parse_args()
    V.save_prune_db = lambda *a, **kw: None
    e = V.V41Engine(os.environ['MODEL_DIR'], max_seq=524288, arena_gb=90.1,
                    trace_stats='/app/results/trace-union/stats/coverage.json',
                    spec=True, prune_keep=.61, transient_slots=16, keep_free_gb=6)
    draft = e.fast.draft_head
    assert draft is not e.fast.head, 'set TP_DRAFT_HEAD=1 DRAFT_HEAD_FMT=fp8'
    # Warm/qualify the original head first. Both copies remain allocated in every arm.
    e.fast.draft_head = e.fast.head
    report = dict(config=e.config(), qualification={}, runs=[], source='host snapshot',
                  note='Fixed target BF16 and attention FP8; extra draft head allocated in both arms.')
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save():
        (out_dir / f'ideas-rank{e.ep.rank}.json').write_text(json.dumps(report, indent=2))

    report['qualification']['lookup'] = qualify_lookup(engine=e, argv=['--tokens', '32', '--sample-tokens', '8'])
    save()
    report['qualification']['bypass'] = qualify_bypass(engine=e, argv=['--out', str(out_dir / 'bypass'), '--qualification-only',
                                                    '--qualification-tokens', '32', '--sample-tokens', '8'])
    save()
    V.LOOKUP_DRAFT_ENABLED = False
    e.draft_bypass_enabled = False
    e._bypass_pinned = None
    e.confidence_depth_policy.pinned = None
    head_graphs = {'bf16': e.fast.draft_graphs}
    current = ['bf16']

    def head_mode(mode):
        head_graphs[current[0]] = e.fast.draft_graphs
        e.fast.draft_head = e.fast.head if mode == 'bf16' else draft
        e.fast.draft_graphs = head_graphs.get(mode)
        if e.fast.draft_graphs is None:
            # Draft capture reads MTP ring history, never writes it. Capture both
            # modes here; subsequent fresh prefill establishes actual request state.
            e.fast._capture_draft_graphs()
        current[0] = mode

    tok, enc = Tok(os.environ['MODEL_DIR']), load_encoding_module(os.environ['MODEL_DIR'])
    prompts = {}
    for name, prompt, _ in WORKLOADS:
        _, prompts[name], _, _ = build_chat_prompt({'messages':[{'role':'user','content':prompt}]},
                                                   enc, tok, False, 75, e)
    unit = ('The copper clock marks seventeen while the quiet river carries '
            'three silver leaves beyond the old stone bridge.\n')
    prompts['periodic'] = e.tokenizer.encode(unit * 24, add_special_tokens=False)

    def run(name, mode, count, temperature, measured=False, *, experiment='head',
            copy=False, bypass=None, pin=None):
        head_mode(mode)
        V.LOOKUP_DRAFT_ENABLED = copy
        e.draft_bypass_enabled = bypass is not None
        e._bypass_pinned = {'on': True, 'off': False, 'adaptive': None}.get(bypass)
        e.confidence_depth_policy.pinned = pin
        tokens = []
        started = time.perf_counter()
        for burst in e.generate(prompts[name], max_tokens=count, temperature=temperature,
                                top_p=.95, seed=42):
            tokens.extend(burst)
        digest = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        parity = len(set(e.ep.gather_objects(digest))) == 1
        assert parity, (name, mode, temperature, 'rank mismatch')
        row = dict(workload=name, head=mode, experiment=experiment,
                   copy=copy, bypass=bypass, temperature=temperature, tokens=len(tokens),
                   wall_s=time.perf_counter()-started, decode_tok_s=e.last_stats['decode_tok_s'],
                   decode_s=e.last_stats['decode_s'], acceptance=e.last_stats['accept_len_mean'],
                   depth=e.last_stats['spec_depth'], lookup=e.last_stats.get('lookup_draft'),
                   bypass_stats=e.last_stats.get('draft_bypass'), sha256=digest, rank_parity=parity)
        if measured:
            report['runs'].append(row)
            save()
        print('IDEA_RUN ' + json.dumps(row), flush=True)
        return tokens

    a = run('python', 'bf16', 64, 0)
    b = run('python', 'fp8', 64, 0)
    assert all(e.ep.gather_objects(a == b)), 'draft-head changed greedy target tokens'
    report['qualification']['head_greedy_equal'] = True
    for temperature in (.1, .6, 1., 2.):
        run('story_t07', 'fp8', 16, temperature)
    save()
    if not args.qualification_only:
        # Short warmed ABBA screens price copying and skipping DSpark before any
        # longer measurements. Their sampled RNG schedules differ legitimately.
        for copy in (False, True):
            run('periodic','bf16',32,1.,copy=copy,pin=3)
        for copy in (False, True, True, False):
            run('periodic','bf16',96,1.,True,experiment='copy',copy=copy,pin=3)
        for bypass in ('off','on'):
            run('explain','bf16',32,1.,bypass=bypass)
        for bypass in ('off','on','on','off'):
            run('explain','bf16',96,1.,True,experiment='bypass',bypass=bypass)
        for name, temperature in (('html',0), ('python',0), ('explain',1.)):
            for mode in ('bf16','fp8'):
                run(name,mode,64,temperature)
            reference = None
            for mode in ('bf16','fp8','fp8','bf16'):
                output = run(name,mode,args.tokens,temperature,True)
                if temperature == 0:
                    reference = output if reference is None else reference
                    assert all(e.ep.gather_objects(output == reference)), 'greedy ABBA changed output'
    report['peak_allocated_gb'] = torch.cuda.max_memory_allocated()/1e9
    save()
    print('DECODE_IDEAS_PASS', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
