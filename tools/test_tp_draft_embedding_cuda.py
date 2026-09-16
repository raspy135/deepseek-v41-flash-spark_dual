"""Real draft weights: exact TP output, embedding lookup and dynamic graph replay.

Run with tools/run_two_node_gate.sh while serving is stopped.
"""
import json
import os
import sys
sys.path[:0] = ['/app', '/app/tools']
import torch
from safetensors import safe_open
import fp4_moe as K
from engine.dist import EPDistributed
from engine.tensor_parallel import FeatureParallelEmbedding, shard


def main():
    os.environ['DSV41_TP_EXPERT_LAYOUT'] = 'output'
    ep = EPDistributed()
    ep.init('cuda')
    root = os.environ['MODEL_DIR']
    with open(root + '/model.safetensors.index.json') as f:
        index = json.load(f)['weight_map']
    def get(name):
        with safe_open(root + '/' + index[name], framework='pt', device='cpu') as f:
            return f.get_tensor(name)
    weight = get('embed.weight')
    embedding = FeatureParallelEmbedding(shard(weight, 1, ep.rank, 2).cuda(), 2)
    for count in (1, 3, 4, 1024):
        ids = torch.arange(count, device='cuda') * 7
        expected = weight[ids.cpu()].cuda()
        actual = embedding[ids]
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    ids = torch.tensor([0, 3, 17], device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            embedding[ids]
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = embedding[ids]
    for delta in (1, 37):
        ids.add_(delta)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured.view(torch.int16), weight[ids.cpu()].cuda().view(torch.int16))
    if ep.rank == 0:
        print('EMBEDDING_EXACT_AND_GRAPH_PASS bytes_saved=' + str(weight.numel()), flush=True)
    del weight, embedding, graph, captured
    full = K.ExpertArena(3, 'cuda')
    tp = K.ExpertArena(3, 'cuda', ep.rank, ep.world)
    # Discover MTP naming from the checkpoint rather than assuming a layer offset.
    prefixes = sorted(n[:-len('w1.weight')] for n in index
                      if 'mtp' in n and '.experts.0.w1.weight' in n)
    assert len(prefixes) == 3, prefixes
    for prefix in prefixes:
        for expert in range(3):
            p = prefix.replace('.experts.0.', f'.experts.{expert}.')
            values = [get(p+n) for n in ('w1.weight', 'w1.scale', 'w2.weight', 'w2.scale', 'w3.weight', 'w3.scale')]
            full.load_slot(expert, *values)
            tp.load_slot(expert, *values)
        for count in (1, 3, 4, 16):
            torch.manual_seed(42)
            x = torch.randn(count, K.DIM, dtype=torch.bfloat16, device='cuda')
            slots = torch.arange(3, dtype=torch.int32, device='cuda').repeat(count, 1)
            weights = torch.rand(count, 3, device='cuda')
            weights /= weights.sum(-1, keepdim=True)
            def run(arena):
                return K.moe_forward(x, slots, weights, arena, out_dtype=torch.float32)
            expected, actual = run(full), run(tp)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run(tp)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = run(tp)
            x.mul_(0.75)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, run(full), rtol=0, atol=0)
            if ep.rank == 0:
                print(f'DRAFT_EXACT_AND_GRAPH_PASS {prefix} tokens={count}', flush=True)
    assert all(ep.gather_objects(True))
    if ep.rank == 0:
        print(f'DRAFT_EMBEDDING_GATE_PASSED draft_bytes_saved={384 * (full.bytes_per_slot - tp.bytes_per_slot)}', flush=True)
    torch.cuda.synchronize()
    sys.stdout.flush()
    # Live CUDA graph objects retain NCCL work; disposable gates exit after synchronization.
    os._exit(0)


if __name__ == '__main__':
    main()
