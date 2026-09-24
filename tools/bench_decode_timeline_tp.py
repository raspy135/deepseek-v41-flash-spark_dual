"""Bounded end-to-end decode timeline. Use the disposable two-node gate, not serving."""
import argparse
import hashlib
import json
import os
import sys
sys.path[:0] = ['/app', '/app/tools']
for key in ('DSV41_PRUNE_SWAP', 'DSV41_PRUNE_SWAP_PREFILL', 'DSV41_PREFIX_CACHE',
            'DSV41_PREFIX_DISK', 'DSV41_PREFIX_RESPONSE', 'DSV41_GPU_TIMING', 'DSV41_STEP_TIMING'):
    os.environ[key] = '0'
os.environ['DSV41_PRUNE_ADAPT'] = '1'
import torch
import engine.v41_engine as V
from torch.profiler import profile, ProfilerActivity
from server.app import Tok, load_encoding_module, build_chat_prompt
from bench_decode_timeline_hooks import DecodeTimeline
from bench_decode_timeline_analysis import analyze

HTML_PROMPT = ('Write a simple, self-contained HTML page for a neighborhood coffee shop. '
               'Include a heading, a short introduction, three menu items with prices, '
               'opening hours, and a footer. Use a small embedded CSS stylesheet with '
               'a cream background, dark text, and comfortable spacing. No JavaScript, '
               'external assets, or complicated layout. Format it readably, roughly '
               '50 lines. Output only the complete HTML, without explanations.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture', help='Trusted LOCAL server capture only; pickle can execute code')
    ap.add_argument('--out', required=True)
    ap.add_argument('--bursts', type=int, default=8)
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--warmup-tokens', type=int, default=128)
    args = ap.parse_args()
    assert 1 <= args.bursts <= 16
    assert 1 <= args.warmup_tokens < args.max_tokens <= 2048
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=524288, arena_gb=90,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=16, keep_free_gb=6)
    kwargs = dict(max_tokens=args.max_tokens, temperature=0, seed=42)
    if args.capture:
        cap = torch.load(args.capture, map_location='cpu', weights_only=False)
        if cap.get('vl') is not None or cap.get('kwargs', {}).get('grammar') is not None:
            raise ValueError('vision/grammar captures require their full server path')
        ids = list(cap['prompt_ids'])
        kwargs = dict(cap['kwargs'])
        kwargs['max_tokens'] = args.max_tokens
        kwargs['seed'] = kwargs.get('seed') if kwargs.get('seed') is not None else 42
    else:
        tok, enc = Tok(root), load_encoding_module(root)
        prompt = HTML_PROMPT
        _, ids, _, _ = build_chat_prompt({'messages': [{'role': 'user', 'content': prompt}]}, enc, tok, False, 75, e)
    signature = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    run_signature = (signature, args.max_tokens, args.warmup_tokens, args.bursts)
    assert len(set(e.ep.gather_objects(run_signature))) == 1, 'workload differs between ranks'
    outputs, stats = [], []
    for _ in range(2):
        out = []
        for burst in e.generate(ids, **kwargs):
            out.extend(burst)
        outputs.append(out)
        stats.append(dict(e.last_stats))
        print('TIMELINE_BASELINE ' + json.dumps({'rank': e.ep.rank, 'pass': len(outputs),
              'prefill_s': e.last_stats['prefill_s'], 'decode_s': e.last_stats['decode_s'],
              'completion_tokens': len(out)}), flush=True)
    generator = e.generate(ids, **kwargs)
    output = []
    # Start later in the response, not during first-token/early decode behavior.
    # Warm graph variants are retained from preceding generations.
    while len(output) < args.warmup_tokens:
        try:
            output.extend(next(generator))
        except StopIteration as exc:
            raise ValueError('response ended before the profiling window; choose a longer workload') from exc
    trace_start_tokens = len(output)
    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                   record_shapes=False, with_stack=False, profile_memory=False)
    bursts = 0
    with DecodeTimeline(e) as timeline:
        prof.start()
        timeline.active = True
        with timeline.span('decode/window'):
            for _ in range(args.bursts):
                try:
                    with timeline.span('decode/burst'):
                        output.extend(next(generator))
                    bursts += 1
                except StopIteration:
                    break
            torch.cuda.synchronize()
        timeline.active = False
        prof.stop()
        trace_end_tokens = len(output)
        for burst in generator:
            output.extend(burst)
        metadata = dict(rank=e.ep.rank, prompt_tokens=len(ids), prompt_sha256=signature,
                        source='local_capture' if args.capture else 'synthetic', bursts=bursts,
                        max_tokens=args.max_tokens, completion_tokens=len(output),
                        trace_start_tokens=trace_start_tokens, trace_end_tokens=trace_end_tokens,
                        unprofiled_stats=stats, profiled_stats=e.last_stats,
                        exact_output_matches=[output == baseline for baseline in outputs],
                        config=e.config())
        path = f'{args.out}/rank{e.ep.rank}.json'
        timeline.export(prof, path, metadata)
    with open(path) as f:
        report = analyze(json.load(f))
    with os.fdopen(os.open(f'{args.out}/rank{e.ep.rank}-summary.json',
                          os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
        json.dump(report, f, indent=2)
    # Containers run as root; retain private permissions but hand artifacts to
    # the owner of the results bind mount so the local user can inspect them.
    owner = os.stat('/app/results')
    for artifact in (path, f'{args.out}/rank{e.ep.rank}-summary.json'):
        os.chown(artifact, owner.st_uid, owner.st_gid)
    print('DECODE_TIMELINE ' + json.dumps(report), flush=True)
    assert all(e.ep.gather_objects(True))
    os._exit(0)


if __name__ == '__main__':
    main()
