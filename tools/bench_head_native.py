"""Cold native BF16 vocabulary-head screen without loading the serving engine.

Only one TP vocabulary shard is loaded. Production semantics are BF16 inputs
and weights, fixed 16-row GEMM, BF16-rounded logits widened to FP32. TP's
disjoint vocabulary all-gather is excluded. Candidates implement
``project(x, weight, **kwargs)`` and return complete local FP32 logits.

Example candidate: --candidate 'tc=head_native:project:{"backend":"tc"}'
With no candidate, this is a bounded production-baseline measurement. Native
weights are retained; this script does not quantize them or change services.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]


def candidate_spec(spec):
    """[label=]module:function[:JSON object], without evaluating Python text."""
    label, body = spec.split('=', 1) if '=' in spec else (None, spec)
    parts = body.split(':', 2)
    if len(parts) < 2 or not all(parts[:2]):
        raise ValueError('candidate needs [label=]module:function[:JSON kwargs]')
    kwargs = json.loads(parts[2]) if len(parts) == 3 else {}
    if not isinstance(kwargs, dict):
        raise ValueError('candidate kwargs must be a JSON object')
    return label or parts[0] + '.' + parts[1] + ':' + json.dumps(kwargs, sort_keys=True), parts[0], parts[1], kwargs


def compare_logits(actual, expected, *, offset=0, topk=10):
    """Exactness and local-vocabulary ranking, retaining concrete token IDs."""
    import torch
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype == torch.float32
    assert torch.isfinite(expected).all(), 'production reference is not finite'
    finite = bool(torch.isfinite(actual).all())
    k = min(topk, expected.shape[-1])
    ei = expected.topk(k, -1).indices
    ai = actual.topk(k, -1).indices
    ea, aa = expected.argmax(-1), actual.argmax(-1)
    delta = actual - expected
    return {
        'exact': bool(torch.equal(actual, expected)), 'finite': finite,
        'changed_elements': int(torch.count_nonzero(actual != expected)),
        'elements': actual.numel(),
        'max_abs_error': float(delta.abs().max()) if finite else None,
        'rms_error': float(delta.square().mean().sqrt()) if finite else None,
        'relative_l2_error': float(delta.norm() / expected.norm().clamp_min(1e-30)) if finite else None,
        'bf16_rounded_exact': bool(torch.equal(actual.bfloat16(), expected.bfloat16())),
        'argmax_changed_rows': int(torch.count_nonzero(aa != ea)),
        'reference_argmax_token_ids': (ea + offset).cpu().tolist(),
        'candidate_argmax_token_ids': (aa + offset).cpu().tolist(),
        'topk': k,
        'topk_order_changed_rows': int((ai != ei).any(-1).sum()),
        'topk_set_changed_rows': int((ai.sort(-1).values != ei.sort(-1).values).any(-1).sum()),
        'topk_overlap_per_row': (ai[:, :, None] == ei[:, None, :]).any(-1).sum(-1).cpu().tolist(),
    }


def cpu_checks():
    """Configuration, production row padding and output ownership; no CUDA."""
    import torch
    import v41_ref as R
    assert candidate_spec('tc=head_native:project:{"backend":"tc","bk":256}') == (
        'tc', 'head_native', 'project', {'backend': 'tc', 'bk': 256})
    for bad in ('project', ':project', 'head:project:[]'):
        try:
            candidate_spec(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    torch.manual_seed(781)
    before = R.MM_TILE
    R.MM_TILE = 16
    try:
        x, weight = torch.randn(16, 32).bfloat16(), torch.randn(24, 32).bfloat16()
        full = R.head_logits(x, weight).clone()
        for rows in (1, 2, 4, 6):
            actual = R.head_logits(x[:rows], weight).clone()
            result = compare_logits(actual, full[:rows])
            assert result['exact'] and result['argmax_changed_rows'] == 0
        # An output reused by a kernel must be snapshotted before another call.
        shared = full.clone()
        snapshot = shared.clone()
        shared.zero_()
        assert torch.equal(snapshot, full)
        changed = full.clone(); changed[0, 0] += 1
        assert not compare_logits(changed, full)['exact']
        return {'passed': True, 'checks': 10, 'cuda_used': False}
    finally:
        R.MM_TILE = before


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cpu-checks', action='store_true')
    ap.add_argument('--model-dir', default=os.environ.get('MODEL_DIR'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--rank', type=int, choices=(0, 1), default=int(os.environ.get('RANK', '0')))
    ap.add_argument('--rows', default='1,2,4,6')
    ap.add_argument('--cases', type=int, default=6)
    ap.add_argument('--scales', default='.1,1,4')
    ap.add_argument('--weight-copies', type=int, choices=(1, 2), default=2)
    ap.add_argument('--cache-mb', type=int, default=64)
    ap.add_argument('--quartets', type=int, default=12)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--memory-limit-gb', type=float, default=3)
    ap.add_argument('--candidate', action='append', default=[])
    ap.add_argument('--activations', type=Path,
                    help='optional CPU tensor .pt, at least 16 rows; no prompt text is loaded')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    if args.cpu_checks:
        print(json.dumps(cpu_checks(), indent=2)); return
    if not args.model_dir or args.out is None:
        ap.error('--model-dir (or MODEL_DIR) and --out are required')
    try:
        widths = tuple(sorted({int(value) for value in args.rows.split(',')}))
        scales = tuple(float(value) for value in args.scales.split(','))
        specs = [candidate_spec(value) for value in args.candidate]
    except ValueError as exc:
        ap.error(str(exc))
    if not widths or min(widths) < 1 or max(widths) > 16:
        ap.error('rows must be in 1..16')
    if not scales or any(value <= 0 for value in scales):
        ap.error('scales must be positive')
    if args.cases < 1 or args.quartets < 1 or args.warmup < 1 or args.cache_mb < 0:
        ap.error('cases/quartets/warmup must be positive and cache-mb nonnegative')
    if len({name for name, *_ in specs}) != len(specs) or any(name == 'production' for name, *_ in specs):
        ap.error('candidate labels must be unique and cannot be production')

    import torch
    from safetensors import safe_open
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources
    import v41_ref as R
    # Exactly the BF16 reduction guard and row padding set by engine/model.py.
    R.MM_TILE, R.HC_MM_TILE = 16, 32
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device(args.device)
    root = Path(args.model_dir).expanduser()
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    shard_file = root / index['head.weight']
    with safe_open(str(shard_file), framework='pt', device='cpu') as source:
        sliced = source.get_slice('head.weight')
        n, k = sliced.get_shape()
        assert n % 2 == 0, 'TP2 vocabulary must divide evenly'
        cpu_weight = sliced[args.rank * (n // 2):(args.rank + 1) * (n // 2)]
        assert cpu_weight.dtype == torch.bfloat16, ('native checkpoint dtype', cpu_weight.dtype)
        weight = cpu_weight.to(device).contiguous()
    del cpu_weight
    weights = [weight] + [weight.clone() for _ in range(args.weight_copies - 1)]
    offset = args.rank * (n // 2)
    sample_rows = torch.tensor([0, 1, n // 8, n // 4, n // 2 - 2, n // 2 - 1], device=device)
    sentinels = [w[sample_rows].clone() for w in weights]
    generator = torch.Generator(device=device).manual_seed(20261008)
    if args.activations:
        source_x = torch.load(args.activations, map_location='cpu', weights_only=True)
        if not isinstance(source_x, torch.Tensor) or source_x.ndim != 2 or source_x.shape[1] != k or source_x.shape[0] < 16:
            raise ValueError('activations must be a [>=16, head K] tensor')
        banks = [source_x.roll(-16 * i, 0)[:16].to(device=device, dtype=torch.bfloat16).contiguous()
                 for i in range(args.cases)]
        activation_kind = 'captured-tensor'
    else:
        banks = [(torch.randn(16, k, device=device, generator=generator) * scales[i % len(scales)]).bfloat16()
                 for i in range(args.cases)]
        activation_kind = 'seeded-random-bf16-not-model-activation'
    functions = {'production': lambda x, w: R.head_logits(x, w)}
    source_paths = [HERE / 'v41_ref.py', HERE.parent / 'engine/tensor_parallel.py', HERE.parent / 'engine/model.py']
    for name, module_name, function_name, kwargs in specs:
        module = importlib.import_module(module_name)
        fn = getattr(module, function_name)
        functions[name] = lambda x, w, fn=fn, kwargs=kwargs: fn(x, w, **kwargs)
        if getattr(module, '__file__', None): source_paths.append(Path(module.__file__))
        if module_name == 'head_native' and kwargs.get('backend', '').startswith('packed'):
            source_paths.append(Path(importlib.import_module('head_packed').__file__))
            backend_module = {'packed-remat': 'head_packed_remat', 'packed-word': 'head_packed_word'}.get(kwargs['backend'])
            if backend_module:
                source_paths.append(Path(importlib.import_module(backend_module).__file__))

    report = {
        'workload': 'native-bf16-local-TP2-head', 'device': torch.cuda.get_device_name(device),
        'rank': args.rank, 'world': 2, 'full_shape': [n, k], 'local_shape': list(weight.shape),
        'checkpoint_dtype': str(weight.dtype), 'weight_copies': len(weights),
        'weight_bytes_per_copy': weight.numel() * weight.element_size(),
        'activation': activation_kind, 'cases': args.cases, 'scales': scales,
        'common_reference_rows': 16, 'MM_TILE': R.MM_TILE,
        'precision': {'bf16_reduced_precision_reduction': False, 'fp16_reduced_precision_reduction': False,
                      'allow_tf32': False, 'output': 'BF16 round then FP32 widen'},
        'excludes': ['vocabulary_all_gather', 'normalization', 'DSpark', 'acceptance', 'serving'],
        'candidate_specs': args.candidate, 'correctness': [], 'rejections': [], 'timing': [],
        'packed_weights': [], 'completed': False,
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.out.write_text(json.dumps(report, indent=2) + '\n')

    def output(fn, x, w):
        value = fn(x, w)
        assert isinstance(value, torch.Tensor) and value.shape == (x.shape[0], w.shape[0])
        assert value.dtype == torch.float32, ('candidate must return BF16-rounded FP32 logits', value.dtype)
        return value

    verified_packed = set()
    def packed_metadata():
        """Inspect the experiment cache and compare every original BF16 bit.

        Slicing all three group-aligned tables keeps escape slots global while
        decoding only 4096 rows at once. No full-head temporary is retained.
        """
        import copy
        metadata = []
        for module_name in sorted({'head_native', *(spec[1] for spec in specs)}):
            cache = getattr(sys.modules.get(module_name), '_packed', {})
            for wi, w in enumerate(weights):
                entry = cache.get(id(w))
                if entry is None:
                    continue
                packed = entry[1]
                packed_module = sys.modules.get(type(packed).__module__)
                packed_source = getattr(packed_module, '__file__', None)
                if packed_source:
                    path = Path(packed_source)
                    report['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                key = module_name, id(packed), id(w)
                if key not in verified_packed:
                    for first in range(0, w.shape[0], 4096):
                        last = min(first + 4096, w.shape[0])
                        if hasattr(packed, 'dequant_rows'):
                            decoded = packed.dequant_rows(first, last)
                        else:
                            part = copy.copy(packed)
                            part.shape = (last - first, w.shape[1])
                            part.low, part.delta, part.header = (getattr(packed, name)[first:last]
                                                                 for name in ('low', 'delta', 'header'))
                            decoded = part.dequant()
                            del part
                        assert torch.equal(decoded.view(torch.int16), w[first:last].view(torch.int16)), 'lossless packed head changed checkpoint bits'
                        del decoded
                    verified_packed.add(key)
                native_bytes = w.numel() * w.element_size()
                metadata.append({'module': module_name, 'weight_copy': wi, 'shape': list(packed.shape),
                    'native_bytes': native_bytes, 'stored_bytes': packed.stored_bytes,
                    'fraction_of_native_bytes': packed.stored_bytes / native_bytes,
                    'saved_bytes': native_bytes - packed.stored_bytes,
                    'escape_groups': packed.escape_groups,
                    'total_groups': packed.header.numel(),
                    'reconstruction_bit_exact': True,
                    'reconstruction_check': 'all rows, int16 bits, 4096-row temporary chunks'})
        report['packed_weights'] = metadata
        return {(row['module'], row['weight_copy']): row['stored_bytes'] for row in metadata}

    save()

    # Every result is cloned immediately: a candidate can reuse an output
    # workspace, including across widths or captured graphs.
    for case, x in enumerate(banks):
        reference = output(functions['production'], x, weight).clone()
        for arm, fn in list(functions.items()):
            try:
                common = output(fn, x, weight).clone()
                for rows in widths:
                    actual = output(fn, x[:rows], weight).clone()
                    row = {'arm': arm, 'case': case, 'rows': rows,
                           'scale': None if args.activations else scales[case % len(scales)],
                           **compare_logits(actual, reference[:rows], offset=offset),
                           'row_invariant': bool(torch.equal(actual, common[:rows])),
                           'row_invariance_max_abs_error': float((actual - common[:rows]).abs().max())}
                    report['correctness'].append(row)
            except (OutOfResources, CompilationError) as exc:
                # Rejected launch plans are evidence too. These errors occur
                # before a launch; CUDA runtime failures still abort the screen.
                rejection = {'arm': arm, 'case': case, 'error_type': type(exc).__name__, 'error': str(exc)}
                report['rejections'].append(rejection)
                functions.pop(arm)
                print('HEAD_REJECTED ' + json.dumps(rejection), flush=True)
        save()
    packed_metadata(); save()
    assert all(row['exact'] and row['row_invariant'] for row in report['correctness'] if row['arm'] == 'production'), 'production row padding changed'
    print('HEAD_CORRECTNESS ' + json.dumps({arm: {
        'exact': all(r['exact'] for r in report['correctness'] if r['arm'] == arm),
        'row_invariant': all(r['row_invariant'] for r in report['correctness'] if r['arm'] == arm),
        'max_abs_error': max((r['max_abs_error'] or 0) for r in report['correctness'] if r['arm'] == arm),
        'argmax_changed_rows': sum(r['argmax_changed_rows'] for r in report['correctness'] if r['arm'] == arm)}
        for arm in functions}), flush=True)

    stream = torch.cuda.Stream(device=device)  # one cuBLAS workspace for all captures
    trash = torch.zeros(args.cache_mb * 1024**2 // 4, device=device, dtype=torch.float32)
    flush = None
    if trash.numel():
        flush = torch.cuda.CUDAGraph()
        with torch.cuda.graph(flush, stream=stream): trash.add_(1)
    report['cold_policy'] = {'cache_flush_bytes': trash.numel() * trash.element_size(),
        'flush_excluded_from_events': True, 'alternating_weight_storage': len(weights),
        'distinct_input_each_quartet': True,
        'order': 'forward then reverse candidates (ABBA with two arms)'}

    for rows in widths:
        buffers = [torch.empty(rows, k, device=device, dtype=torch.bfloat16) for _ in weights]
        graphs = {}
        for wi, w in enumerate(weights):
            buffers[wi].copy_(banks[0][:rows])
            for arm, fn in functions.items():
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(args.warmup): output(fn, buffers[wi], w)
                torch.cuda.current_stream(device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    result = output(fn, buffers[wi], w)
                graphs[(arm, wi)] = (graph, result)
        torch.cuda.synchronize(device)
        packed_bytes = packed_metadata()
        assert torch.cuda.max_memory_reserved(device) <= args.memory_limit_gb * 1024**3, 'microbenchmark memory limit exceeded'
        samples = {arm: [] for arm in functions}
        graph_checks = {arm: [] for arm in functions}
        order = list(functions) + list(functions)[::-1]
        for quartet in range(args.quartets):
            case = quartet % len(banks)
            wi = quartet % len(weights)
            buffers[wi].copy_(banks[case][:rows])
            reference = output(functions['production'], banks[case][:rows], weights[wi]).clone()
            for arm in order:
                graph, result = graphs[(arm, wi)]
                if flush is not None: flush.replay()
                begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(); graph.replay(); finish.record(); finish.synchronize()
                samples[arm].append(begin.elapsed_time(finish))
                # Snapshot before another arm can overwrite a shared output.
                graph_result = result.clone()
                graph_checks[arm].append(compare_logits(graph_result, reference, offset=offset))
        base = statistics.median(samples['production'])
        for arm in functions:
            ms = statistics.median(samples[arm])
            packed_module = next((module for name, module, function, kwargs in specs
                if name == arm and (kwargs.get('backend', '').startswith('packed')
                                    or function == 'project_native' and (module, 0) in packed_bytes)), None)
            stored_bytes = packed_bytes.get((packed_module, 0), weight.numel() * weight.element_size())
            row = {'arm': arm, 'rows': rows, 'median_ms': ms, 'samples_ms': samples[arm],
                'padded_production_speedup': base / ms,
                'weight_GBps': weight.numel() * weight.element_size() / ms / 1e6,
                'stored_weight_bytes': stored_bytes, 'stored_weight_GBps': stored_bytes / ms / 1e6,
                'graph_exact': all(c['exact'] for c in graph_checks[arm]),
                'graph_finite': all(c['finite'] for c in graph_checks[arm]),
                'graph_max_abs_error': max((c['max_abs_error'] or 0) for c in graph_checks[arm]),
                'graph_argmax_changed_rows': sum(c['argmax_changed_rows'] for c in graph_checks[arm])}
            report['timing'].append(row)
            print('HEAD_COLD ' + json.dumps({k: v for k, v in row.items() if k != 'samples_ms'}), flush=True)
        save()
        del graphs, buffers
    report['weight_sentinels_unchanged'] = all(torch.equal(w[sample_rows], sentinel) for w, sentinel in zip(weights, sentinels))
    report['memory'] = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    report['completed'] = True
    save()
    assert report['weight_sentinels_unchanged'], 'sampled native weight rows changed'
    print('HEAD_NATIVE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
