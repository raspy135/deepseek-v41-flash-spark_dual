"""Price the checkpoint's local vocabulary shard without loading a serving engine.

This measures local BF16 versus FP8 draft-head graph replay, including FP32 logits
conversion. It excludes vocabulary all-gather, DSpark, verification and acceptance;
it cannot establish a full-engine speedup. Each ABBA quartet shares one activation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import v41_ref as R
from engine.tensor_parallel import VocabParallelHead, draft_head_bytes, make_tp_draft_head


def capture(x, head):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            R.head_logits(x, head)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = R.head_logits(x, head)
    torch.cuda.current_stream().wait_stream(stream)
    return graph, output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', default=os.environ.get('MODEL_DIR'), required=not os.environ.get('MODEL_DIR'))
    ap.add_argument('--rank', type=int, choices=(0, 1), default=0)
    ap.add_argument('--rows', type=int, default=5)
    ap.add_argument('--quartets', type=int, default=25)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.rows < 1 or args.rows > 16 or args.quartets < 1 or args.warmup < 0:
        ap.error('rows must be 1..16, quartets positive and warmup nonnegative')
    os.environ.update(DSV41_HEAD_FMT='bf16', DSV41_HEAD_FP32='0',
                      DSV41_DRAFT_HEAD_FMT='fp8', DSV41_TP_DRAFT_HEAD='1')
    root = Path(args.model_dir)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(root / index['head.weight'], framework='pt', device='cpu') as f:
        sliced = f.get_slice('head.weight')
        n, k = sliced.get_shape()
        if n % 2:
            raise ValueError('head vocabulary must divide across two ranks')
        local = sliced[args.rank * (n // 2):(args.rank + 1) * (n // 2)].to('cuda').bfloat16()
    target = VocabParallelHead(local, 2)
    sentinel = local[:128].clone()
    torch.manual_seed(1107)
    x = torch.randn(args.rows, k, dtype=torch.bfloat16, device='cuda')
    original_logits = R.head_logits(x, target.local).clone()
    draft = make_tp_draft_head(target)
    assert draft is not None
    heads = {'bf16': target.local, 'fp8': draft.local}
    graphs = {fmt: capture(x, head) for fmt, head in heads.items()}
    for _ in range(args.warmup):
        for graph, _ in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    samples = {fmt: [] for fmt in graphs}
    for _ in range(args.quartets):
        for fmt in ('bf16', 'fp8', 'fp8', 'bf16'):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[fmt][0].replay()
            end.record()
            end.synchronize()
            samples[fmt].append(start.elapsed_time(end))
    assert torch.equal(local[:128], sentinel), 'draft construction/replay mutated verifier weights'
    assert torch.equal(R.head_logits(x, target.local), original_logits), 'verifier logits changed'
    assert torch.isfinite(graphs['fp8'][1]).all(), 'draft logits are not finite'
    output = {
        'workload': 'checkpoint-local-vocab-shard-graph-replay',
        'excludes': ['vocab_all_gather', 'DSpark', 'verification', 'acceptance'],
        'rank': args.rank, 'world': 2, 'shape': [n // 2, k], 'rows': args.rows,
        'activation': 'seeded-random-bf16-not-captured-model-activation',
        'device': torch.cuda.get_device_name(), 'warmup': args.warmup,
        'order': 'ABBA', 'quartets': args.quartets,
        'stored_bytes': {'bf16': draft_head_bytes(target), 'fp8': draft_head_bytes(draft)},
        'extra_resident_bytes': draft_head_bytes(draft),
        'samples_ms': samples,
        'median_ms': {fmt: statistics.median(values) for fmt, values in samples.items()},
        'verifier_logits_identical': True,
        'source_sha256': {str(path.relative_to(HERE.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (HERE / 'v41_ref.py', HERE / 'fp8_linear.py',
                                       HERE.parent / 'engine/tensor_parallel.py')},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({key: output[key] for key in ('shape', 'rows', 'stored_bytes', 'median_ms',
                                                 'verifier_logits_identical')}), flush=True)


if __name__ == '__main__':
    main()
