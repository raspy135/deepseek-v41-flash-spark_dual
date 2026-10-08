"""Functional native packed-head prefill/shape qualification, no serving changes.

Loads only one actual TP2 BF16 vocabulary shard. This is not a performance or
full-model acceptance test. All matrix shapes are checked before reporting a
failure, so a differing cuBLAS plan leaves useful numerical evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def comparison(actual, reference):
    """Chunk diagnostics to avoid another full 512-row vocabulary temporary."""
    import torch
    from bench_head_native import compare_logits
    chunks = [compare_logits(actual[first:first + 16], reference[first:first + 16])
              for first in range(0, actual.shape[0], 16)]
    return {
        'exact': all(row['exact'] for row in chunks),
        'finite': all(row['finite'] for row in chunks),
        'changed_elements': sum(row['changed_elements'] for row in chunks),
        'elements': actual.numel(),
        'max_abs_error': max(row['max_abs_error'] or 0 for row in chunks),
        'argmax_changed_rows': sum(row['argmax_changed_rows'] for row in chunks),
        'top10_order_changed_rows': sum(row['topk_order_changed_rows'] for row in chunks),
        'top10_set_changed_rows': sum(row['topk_set_changed_rows'] for row in chunks),
        'reference_argmax_token_ids': [token for row in chunks for token in row['reference_argmax_token_ids']],
        'candidate_argmax_token_ids': [token for row in chunks for token in row['candidate_argmax_token_ids']],
        'output_hash': digest(actual), 'reference_hash': digest(reference),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', default=os.environ.get('MODEL_DIR'), required=not os.environ.get('MODEL_DIR'))
    ap.add_argument('--rank', type=int, choices=(0, 1), default=0)
    ap.add_argument('--rows', default='17,32,128,512')
    ap.add_argument('--scales', default='1,4')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    import v41_ref as R
    from engine.native_head import make_packed_head
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    R.MM_TILE, R.HC_MM_TILE = 16, 32
    torch.cuda.reset_peak_memory_stats()
    root = Path(args.model_dir)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(str(root / index['head.weight']), framework='pt', device='cpu') as source:
        sliced = source.get_slice('head.weight')
        n, k = sliced.get_shape()
        assert n % 2 == 0
        original = sliced[args.rank * (n // 2):(args.rank + 1) * (n // 2)].cuda().contiguous()
    assert original.dtype == torch.bfloat16
    packed = make_packed_head(original)
    report = {
        'workload': 'actual native BF16 local TP2 head functional fallback',
        'device': torch.cuda.get_device_name(), 'rank': args.rank,
        'shape': list(original.shape), 'native_bytes': original.numel() * original.element_size(),
        'packed_bytes': packed.stored_bytes, 'precision': 'BF16 sources / guarded FP32 accumulate / BF16 round / FP32 widen',
        'rows': [], 'shape_checks': [], 'completed': False,
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            HERE.parent / 'engine/native_head.py', HERE / 'head_packed_tiles.py',
            HERE / 'head_packed_gluon.py', HERE / 'v41_ref.py')},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    save()
    # Reconstruction is independent of GEMM numerical equivalence. Check every
    # original bit with small row chunks, including rare exponent escapes.
    for first in range(0, original.shape[0], 4096):
        last = min(first + 4096, original.shape[0])
        decoded = packed.dequant_rows(first, last)
        assert torch.equal(decoded.view(torch.int16), original[first:last].view(torch.int16))
        del decoded
    report['reconstruction_bit_exact'] = True
    generator = torch.Generator(device='cuda').manual_seed(20261008)
    for rows in tuple(int(value) for value in args.rows.split(',')):
        assert rows > 16
        for scale in tuple(float(value) for value in args.scales.split(',')):
            x = (torch.randn(rows, k, generator=generator, device='cuda') * scale).bfloat16()
            reference = F.linear(x, original).float()
            actual = R.head_logits(x, packed)
            assert actual.shape == reference.shape and actual.dtype == torch.float32
            row = {'rows': rows, 'scale': scale, **comparison(actual, reference)}
            report['rows'].append(row); save()
            print('PREFILL_HEAD_CHECK ' + json.dumps({key: value for key, value in row.items()
                if not key.endswith('token_ids')}), flush=True)
            del reference, actual, x
    # The wrapper flattens leading dimensions. The relevant decode contract is
    # equality to the fixed-row padded 2D native path. The original 3D path did
    # not apply R.mm padding; retain its separate diagnostic without assuming it.
    x = torch.randn(2, 2, k, generator=generator, device='cuda', dtype=torch.bfloat16)
    flat_reference = R.head_logits(x.reshape(4, k), original)
    actual = R.head_logits(x, packed)
    unpadded_reference = R.head_logits(x, original).reshape(4, -1)
    report['shape_checks'].append({'input_shape': list(x.shape), 'output_shape': list(actual.shape),
        'vs_flattened_padded_native': comparison(actual.reshape(4, -1), flat_reference),
        'vs_original_unpadded_3d_native': comparison(actual.reshape(4, -1), unpadded_reference)})
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    report['completed'] = True
    report['passed'] = all(row['exact'] for row in report['rows']) and all(
        row['vs_flattened_padded_native']['exact'] for row in report['shape_checks'])
    save()
    print('HEAD_FALLBACK_RESULT ' + json.dumps({key: report[key] for key in (
        'passed', 'peak_allocated_bytes', 'peak_reserved_bytes')}), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
