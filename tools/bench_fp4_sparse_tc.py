"""Bounded cold native-FP4 sparse-TC qualification, then component timing.

Candidate module API: optional prepare(arena, **kwargs), then
up(arena, case, out, **kwargs) and down(arena, case, out, h=None,
partial=False, **kwargs). The case dictionaries and native CUDA baseline are
those of bench_fp4_pairbatch. Preparation must occur outside capture; no
serving integration or model request is performed by this driver.

Require a successful raw-group proof before allocating CUDA tensors. Check
intermediate BF16 h, raw/rounded FP32 down parts and routed sums, including
mixed nulls, before timing T4/U6, T6/U6 and T4/U20. Any discrepancy aborts the
campaign and leaves a report. No full-engine benchmark follows automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
CELLS = ((4, 6), (6, 6), (4, 20))


def compare(actual, reference):
    import torch
    assert actual.shape == reference.shape and actual.dtype == reference.dtype
    bit_dtype = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    bits_equal = actual.view(bit_dtype) == reference.view(bit_dtype)
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    return {'bit_exact': bool(bits_equal.all()), 'finite': finite,
        'changed_bit_values': int((~bits_equal).sum()), 'elements': actual.numel(),
        'numeric_changed_values': int((actual != reference).sum()),
        'max_abs_error': float((actual.float() - reference.float()).abs().max()) if finite else None}


def cpu_checks():
    import torch
    ref = torch.tensor([1., 0., -2.], dtype=torch.float32)
    assert compare(ref.clone(), ref)['bit_exact']
    flipped = ref.clone(); flipped[1] = -0.
    result = compare(flipped, ref)
    assert not result['bit_exact'] and result['numeric_changed_values'] == 0
    changed = ref.clone(); changed[0] += .25
    assert compare(changed, ref)['max_abs_error'] == .25
    shared = ref.clone(); snapshot = shared.clone(); shared.zero_()
    assert compare(snapshot, ref)['bit_exact']
    assert compare(ref.bfloat16().clone(), ref.bfloat16())['bit_exact']
    return {'passed': True, 'checks': 5, 'cuda_used': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cpu-checks', action='store_true')
    ap.add_argument('--candidate', help='module implementing the documented adapter API')
    ap.add_argument('--candidate-kwargs', default='{}')
    ap.add_argument('--group-proof', type=Path,
                    help='JSON report with passed=true from the raw32-K group proof')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--calls', type=int, default=16)
    ap.add_argument('--quartets', type=int, choices=(6,), default=6)
    ap.add_argument('--qualify-only', action='store_true')
    ap.add_argument('--memory-limit-gb', type=float, default=3)
    args = ap.parse_args()
    if args.cpu_checks:
        print(json.dumps(cpu_checks())); return
    if args.candidate is None or args.out is None or args.group_proof is None:
        ap.error('--candidate, --group-proof and --out are required')
    if args.calls < 8 or args.memory_limit_gb <= 0:
        ap.error('calls must be >=8 and memory-limit-gb positive')
    kwargs = json.loads(args.candidate_kwargs)
    if not isinstance(kwargs, dict):
        ap.error('candidate-kwargs must be a JSON object')
    proof = json.loads(args.group_proof.read_text())
    if proof.get('passed') is not True:
        ap.error('raw group proof has not passed; no CUDA/model allocation performed')

    import torch
    import fp4_moe_cuda as CUDA
    import bench_fp4_pairbatch as B
    candidate = importlib.import_module(args.candidate)
    assert callable(candidate.up) and callable(candidate.down)
    assert CUDA.RELAXED_REDUCE and CUDA.V2_FMA_SCALE == 0
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    arena = B.load_actual_weights()
    if hasattr(candidate, 'prepare'):
        candidate.prepare(arena, **kwargs)
    assert torch.cuda.max_memory_reserved() <= args.memory_limit_gb * 1024**3
    functions = {
        'baseline': {'up': B.up, 'down': B.down},
        'candidate': {
            'up': lambda a, c, out: candidate.up(a, c, out, **kwargs),
            'down': lambda a, c, out, h=None, partial=False: candidate.down(
                a, c, out, h=h, partial=partial, **kwargs)},
    }
    paths = [Path(__file__), Path(candidate.__file__), Path(B.__file__), Path(CUDA.SOURCE)]
    report = {'workload': __doc__, 'device': torch.cuda.get_device_name(),
        'arena_bytes': arena.bytes_per_slot * arena.slots, 'native_bytes_per_slot': arena.bytes_per_slot,
        'copies': 4, 'actual_experts_per_copy': 32, 'tp_layout': 'output', 'tp_rank': 0,
        'candidate': args.candidate, 'candidate_kwargs': kwargs,
        'group_proof': {'path': str(args.group_proof),
            'sha256': hashlib.sha256(args.group_proof.read_bytes()).hexdigest(), 'report': proof},
        'calls': args.calls, 'quartets': args.quartets, 'checks': [], 'timing': [],
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        'completed': False, 'passed': False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    save()
    generator = torch.Generator().manual_seed(20261008)
    cases = {(rows, distinct): [B.make_case(arena, rows, distinct, call, generator)
                               for call in range(args.calls)] for rows, distinct in CELLS}

    def qualify(case, *, distinct, mixed_null, case_index):
        outputs = []
        for arm in ('baseline', 'candidate'):
            f = functions[arm]
            h = torch.zeros(case['rows'] * B.TOPK, arena.w1.shape[1], dtype=torch.bfloat16, device='cuda')
            f['up'](arena, case, h)
            # Identical synthetic peer half; test the candidate's actual local
            # upstream h rather than an unrelated random down input.
            full_h = torch.cat((h, case['hf'][:, arena.w1.shape[1]:]), dim=1).contiguous()
            result = {'h_bf16': h.clone()}
            for partial in (False, True):
                parts = torch.empty(case['rows'] * B.TOPK, arena.w2.shape[1], device='cuda')
                f['down'](arena, case, parts, h=full_h, partial=partial)
                suffix = 'raw' if partial else 'rounded'
                result['parts_' + suffix] = parts.clone()
                final = parts.view(B.TOPK, case['rows'], -1).sum(dim=0)
                result['sum_' + suffix] = final.clone()
                result['sum_bf16_' + suffix] = final.bfloat16().clone()
            outputs.append(result)
        for name, reference in outputs[0].items():
            row = {'rows': case['rows'], 'distinct_requested': distinct,
                'mixed_null': mixed_null, 'case': case_index, 'output': name,
                **compare(outputs[1][name], reference)}
            report['checks'].append(row)
        save()

    # All cold inputs are qualified, not merely the first case that happened to
    # warm the kernel. Mixed null routing must pass before any timed replay.
    for (rows, distinct), group in cases.items():
        for index, case in enumerate(group):
            qualify(case, distinct=distinct, mixed_null=False, case_index=index)
    for rows, distinct in ((4, 6), (6, 6)):
        qualify(B.make_case(arena, rows, distinct, 0, generator, null=True),
                distinct=distinct, mixed_null=True, case_index=0)
    if not all(row['bit_exact'] and row['finite'] for row in report['checks']):
        report['failure'] = 'intermediate/down/null numerical qualification failed; timing skipped'
        report['completed'] = True; save()
        print('SPARSE_TC_NUMERICAL_FAILURE ' + str(args.out), flush=True)
        raise SystemExit(1)
    report['numerical_qualified'] = True; save()
    if args.qualify_only:
        report['completed'] = report['passed'] = True; save(); return

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    up_bytes = sum(t[0].numel() for t in (arena.w1, arena.s1, arena.w3, arena.s3))
    down_bytes = sum(t[0].numel() for t in (arena.w2, arena.s2))
    for (rows, distinct), group in cases.items():
        graphs, expected = {}, {}
        # Use a separate output per graph. Snapshot all eager references before
        # another arm runs, including adapters with internal shared scratch.
        for arm, f in functions.items():
            for operation in ('up', 'down'):
                out = torch.zeros(rows * B.TOPK,
                    arena.w1.shape[1] if operation == 'up' else arena.w2.shape[1],
                    dtype=torch.bfloat16 if operation == 'up' else torch.float32, device='cuda')
                with torch.cuda.stream(stream):
                    for case in group:
                        f[operation](arena, case, out)
                    expected[arm, operation] = out.clone()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        for case in group:
                            f[operation](arena, case, out)
                graphs[arm, operation] = graph, out
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        samples = {key: [] for key in graphs}
        graph_checks = {key: [] for key in graphs}
        with torch.cuda.stream(stream):
            for quartet in range(args.quartets):
                arms = ('baseline', 'candidate', 'candidate', 'baseline')
                if quartet % 2:
                    arms = ('candidate', 'baseline', 'baseline', 'candidate')
                for operation in ('up', 'down'):
                    for arm in arms:
                        graph, out = graphs[arm, operation]
                        # The full preceding sweep makes the first timed call
                        # cold even when the previous arm ended at the same set.
                        graph.replay()
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record(); graph.replay(); end.record(); end.synchronize()
                        samples[arm, operation].append(begin.elapsed_time(end) * 1000 / len(group))
                        snapshot = out.clone()
                        graph_checks[arm, operation].append(compare(snapshot, expected[arm, operation]))
                        del snapshot
        assert all(row['bit_exact'] for checks in graph_checks.values() for row in checks), 'graph changed eager output'
        assert all(compare(expected['candidate', op], expected['baseline', op])['bit_exact']
                   for op in ('up', 'down')), 'timing input changed candidate numerics'
        timing = {'rows': rows, 'distinct': distinct, 'arms': {}}
        for arm in functions:
            up_us = statistics.median(samples[arm, 'up'])
            down_us = statistics.median(samples[arm, 'down'])
            timing['arms'][arm] = {'up_us': up_us, 'down_us': down_us,
                'total_us': up_us + down_us, 'samples_up_us': samples[arm, 'up'],
                'samples_down_us': samples[arm, 'down'],
                'minimum_payload_up_GBps': distinct * up_bytes / up_us / 1000,
                'minimum_payload_down_GBps': distinct * down_bytes / down_us / 1000,
                'minimum_payload_total_GBps': distinct * arena.bytes_per_slot / (up_us + down_us) / 1000,
                'graph_bit_exact': True}
        timing['baseline_over_candidate'] = (timing['arms']['baseline']['total_us']
                                              / timing['arms']['candidate']['total_us'])
        report['timing'].append(timing); save()
        print('SPARSE_TC_CELL ' + json.dumps(timing), flush=True)
    assert torch.cuda.max_memory_reserved() <= args.memory_limit_gb * 1024**3
    report['completed'] = report['passed'] = True; save()


if __name__ == '__main__':
    main()
