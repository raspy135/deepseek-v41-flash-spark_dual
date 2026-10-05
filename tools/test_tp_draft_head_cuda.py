"""Tiny two-rank TP draft-head collective and changing-input CUDA graph gate.

Use run_two_node_gate.sh while serving is stopped. No checkpoint is loaded.
The FP8 draft is checked against independently computed concatenated shard logits;
only the BF16 verifier is required to remain bit-identical to its original output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import v41_ref as R
from engine.dist import EPDistributed
from engine.tensor_parallel import (TP_DRAFT_HEAD_VERSION, VocabParallelHead,
                                   draft_head_bytes, make_tp_draft_head, shard)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    os.environ.update(DSV41_HEAD_FMT='bf16', DSV41_HEAD_FP32='0',
                      DSV41_DRAFT_HEAD_FMT='fp8', DSV41_TP_DRAFT_HEAD='1')
    ep = EPDistributed()
    if ep.world != 2:
        raise ValueError('this gate requires two ranks')
    ep.init('cuda')
    torch.manual_seed(114)
    full = torch.randn(1024, 512, dtype=torch.bfloat16, device='cuda') * .05
    target = VocabParallelHead(shard(full, 0, ep.rank, ep.world), ep.world)
    before = target.local.clone()
    draft = make_tp_draft_head(target)
    assert draft is not None
    # Independently construct both shards, without a collective, to check row ordering.
    independent = [R.make_draft_head(part.contiguous()) for part in full.chunk(2)]
    results = []
    for rows in (1, 3, 5):
        torch.manual_seed(29 + rows)
        x = torch.randn(rows, 512, dtype=torch.bfloat16, device='cuda')
        original = R.head_logits(x, target.local).clone()
        expected = torch.cat([R.head_logits(x, part) for part in independent], dim=-1)
        actual = R.head_logits(x, draft)
        assert torch.equal(actual, expected), (ep.rank, rows, 'draft gather')
        assert torch.equal(R.head_logits(x, target.local), original), 'verifier logits changed'
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                R.head_logits(x, draft)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = R.head_logits(x, draft)
        for scale in (.75, -1.):
            x.mul_(scale)
            graph.replay()
            torch.cuda.synchronize()
            expected = torch.cat([R.head_logits(x, part) for part in independent], dim=-1)
            assert torch.equal(captured, expected), (ep.rank, rows, 'graph replay')
            assert torch.isfinite(captured).all()
        results.append({'rows': rows, 'graph_hash': hashlib.sha256(
            captured.cpu().numpy().tobytes()).hexdigest()})
    assert torch.equal(before, target.local), 'verifier weight shard changed'
    report = {'tp_draft_head_version': TP_DRAFT_HEAD_VERSION, 'checks': results,
              'verifier_unchanged': True, 'draft_bytes': draft_head_bytes(draft),
              'target_bytes': draft_head_bytes(target)}
    gathered = ep.gather_objects(report)
    assert gathered[0] == gathered[1], 'TP draft-head ranks disagree'
    if ep.rank == 0:
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + '\n')
        print('TP_DRAFT_HEAD_GATE_PASSED ' + json.dumps(report), flush=True)
    torch.cuda.synchronize()
    sys.stdout.flush()
    # NCCL graph objects remain live; disposable gates exit after synchronization.
    os._exit(0)


if __name__ == '__main__':
    main()
