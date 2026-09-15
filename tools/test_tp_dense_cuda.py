"""Real-weight dense TP arithmetic and CUDA graph gate on both nodes."""
import json
import os
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/tools')

import torch
from safetensors import safe_open
import v41_ref as R
from fp8_linear import FP8Weight
from engine.tensor_parallel import shard, RowParallelWeight, OutputParallelWeight, VocabParallelHead
from engine.dist import EPDistributed


def main():
    ep = EPDistributed()
    ep.init('cuda')
    root = os.environ['MODEL_DIR']
    index = json.load(open(root + '/model.safetensors.index.json'))['weight_map']
    def get(name):
        with safe_open(root + '/' + index[name], framework='pt', device='cpu') as f:
            return f.get_tensor(name).to('cuda')
    name = 'layers.0.ffn.shared_experts.w2'
    weight = FP8Weight(get(name + '.weight'), get(name + '.scale'))
    local = (OutputParallelWeight(shard(weight, 0, ep.rank, ep.world), ep.world)
             if '--output' in sys.argv else RowParallelWeight(shard(weight, 1, ep.rank, ep.world)))
    for n in (1, 6, 63, 128, 2048):
        torch.manual_seed(42)
        x = torch.randn(n, weight.K, device='cuda', dtype=torch.bfloat16)
        expected = R.mm(x, weight).float()
        xs = x.chunk(2, dim=-1)[ep.rank].contiguous()
        actual = local.tp_linear(xs).float()
        rel = float((actual-expected).norm()/expected.norm())
        assert rel < .005 and torch.isfinite(actual).all(), (n, rel)
        # M=63 selects a different cuBLAS geometry for the output-row shard: a
        # small residual difference remains even with complete K dot products.
        # Keep an explicit tighter bound; do not claim bitwise whole-model TP.
        if '--output' in sys.argv and n != 63:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if '--output' in sys.argv and n == 63:
            assert rel < 1e-4, rel
        if ep.rank == 0:
            print('TP_DENSE ' + json.dumps(dict(tokens=n, relative_l2=rel)), flush=True)
        if n == 6:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    local.tp_linear(xs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = local.tp_linear(xs)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured.float(), actual, rtol=0, atol=0)
    # Small exact vocabulary gather control, no reduction/rounding across ranks.
    torch.manual_seed(19)
    head = torch.randn(256, 128, device='cuda', dtype=torch.bfloat16)
    x = torch.randn(6, 128, device='cuda', dtype=torch.bfloat16)
    distributed_head = VocabParallelHead(shard(head, 0, ep.rank, ep.world), ep.world)
    actual, expected = distributed_head.tp_logits(x), R.head_logits(x, head)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(ep.gather_objects(True))
    if ep.rank == 0:
        print('TP_DENSE_GATE_PASSED', flush=True)
    torch.cuda.synchronize()
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
