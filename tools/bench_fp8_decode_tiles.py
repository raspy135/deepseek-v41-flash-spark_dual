"""Decode FP8 scheduling sweep with real weights and unchanged K reduction tiles."""
import json
import os
import statistics
import sys
import time
sys.path[:0] = ['/app', '/app/tools']
import torch
import triton
from safetensors import safe_open
from engine.dist import EPDistributed
from fp8_linear import FP8Weight, fp8_linear, _fp8_linear_kernel


def main():
    ep = EPDistributed()
    ep.init('cuda')
    root = os.environ['MODEL_DIR']
    with open(os.path.join(root, 'model.safetensors.index.json')) as f:
        index = json.load(f)['weight_map']
    generator = torch.Generator(device='cuda').manual_seed(20260916)
    names = ('attn.wq_a', 'attn.wq_b', 'attn.wkv', 'attn.wo_b',
             'ffn.shared_experts.w1', 'ffn.shared_experts.w2')
    configs = [(128, 4, 3), (128, 4, 1), (64, 4, 2), (256, 4, 2),
               (128, 8, 2), (128, 4, 2), (128, 4, 3)]
    for name in names:
        prefix = 'layers.0.' + name
        tensors = []
        for suffix in ('.weight', '.scale'):
            key = prefix + suffix
            with safe_open(os.path.join(root, index[key]), framework='pt', device='cpu') as f:
                tensors.append(f.get_tensor(key).cuda())
        weight = FP8Weight(*tensors)
        x = torch.randn(4, weight.K, generator=generator, device='cuda', dtype=torch.bfloat16)
        reference = fp8_linear(x, weight)
        y = torch.empty_like(reference)
        for order, (bn, warps, stages) in enumerate(configs):
            def run():
                _fp8_linear_kernel[(triton.cdiv(weight.N, bn), 1)](
                    x, weight.w, weight.s, y, 4, weight.N, weight.K,
                    x.stride(0), weight.w.stride(0), weight.s.stride(0), y.stride(0),
                    BLOCK_M=16, BLOCK_N=bn, BLOCK_K=128 if weight.K % 128 == 0 else 64,
                    num_warps=warps, num_stages=stages)
            for _ in range(3):
                run()
            exact = torch.equal(y, reference)
            delta = float((y.float()-reference.float()).abs().max())
            agreed = all(ep.gather_objects(exact))
            timings = []
            if agreed:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                for _ in range(5):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(50):
                        graph.replay()
                    torch.cuda.synchronize()
                    timings.append((time.perf_counter()-start)*20)
            ranks = ep.gather_objects(statistics.median(timings) if timings else None)
            if ep.rank == 0:
                print('FP8_TILE ' + json.dumps({'name': name, 'shape': weight.shape,
                      'order': order, 'config': [bn, warps, stages], 'exact': agreed,
                      'delta': delta, 'rank_ms': ranks}), flush=True)
        del reference, y, x, weight, tensors
    assert all(ep.gather_objects(True))
    os._exit(0)


if __name__ == '__main__':
    main()
