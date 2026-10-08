"""CPU-only zero-order entropy of unchanged checkpoint FP4 codes.

This samples expert zero at layers 0/20/39, all three expert matrices. The
entropy bound excludes coding headers, scales and GPU decompression costs; it
is not a universal bound on conditional or predictive compression.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    import torch
    from safetensors import safe_open
    index = json.loads((args.model_dir / 'model.safetensors.index.json').read_text())['weight_map']
    report = {'workload': __doc__, 'cuda_used': False, 'stored_bits_per_symbol': 4,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'matrices': []}
    for layer in (0, 20, 39):
        for matrix in ('w1', 'w2', 'w3'):
            key = f'layers.{layer}.ffn.experts.0.{matrix}.weight'
            with safe_open(str(args.model_dir / index[key]), framework='pt', device='cpu') as source:
                packed = source.get_tensor(key).view(torch.uint8).reshape(-1)
                counts = (torch.bincount((packed & 15).long(), minlength=16)
                          + torch.bincount((packed >> 4).long(), minlength=16))
            probabilities = counts.double() / counts.sum()
            nonzero = probabilities[probabilities > 0]
            entropy = float(-(nonzero * nonzero.log2()).sum())
            report['matrices'].append({'key': key, 'packed_bytes': packed.numel(),
                'symbol_counts': counts.tolist(), 'entropy_bits_per_symbol': entropy,
                'ideal_zero_order_payload_saving_fraction': 1 - entropy / 4})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'matrices': len(report['matrices']), 'entropy_range': (
        min(row['entropy_bits_per_symbol'] for row in report['matrices']),
        max(row['entropy_bits_per_symbol'] for row in report['matrices'])), 'cuda_used': False}))


if __name__ == '__main__':
    main()
