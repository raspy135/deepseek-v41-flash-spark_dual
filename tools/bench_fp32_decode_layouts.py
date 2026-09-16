"""FP32 decode projection layout/padding probes, real weights, TF32 disabled.

Benchmark only. Numerically different candidates are reported, never promoted.
"""
import json
import os
import statistics
import sys
sys.path[:0] = ['/app', '/app/tools']
import torch
import torch.nn.functional as F
from safetensors import safe_open
from engine.dist import EPDistributed


def main():
    ep = EPDistributed()
    ep.init('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    root = os.environ['MODEL_DIR']
    with open(os.path.join(root, 'model.safetensors.index.json')) as f:
        index = json.load(f)['weight_map']
    names = ['layers.0.hc_attn_fn', 'layers.0.hc_ffn_fn', 'layers.0.ffn.gate.weight']
    names += sorted(k for k in index if k.endswith('attn.compressor.wgate.weight'))[:1]
    generator = torch.Generator(device='cuda').manual_seed(20260916)
    for name in names:
        with safe_open(os.path.join(root, index[name]), framework='pt', device='cpu') as f:
            source = f.get_tensor(name)
        w = source.cuda().float()
        n, k = w.shape
        # Serving h starts BF16; FP32 projections must not re-quantize weights.
        x = torch.randn(4, k, generator=generator, device='cuda', dtype=torch.bfloat16).float()
        base_x = F.pad(x, (0, 0, 0, 12))
        reference = F.linear(base_x, w)[:4]
        configs = [('baseline', 16, n, False)]
        configs += [('transpose', 16, n, True)]
        configs += [(f'rows-{m}', m, n, False) for m in (32, 64, 128, 256)]
        configs += [(f'cols-{nn}', 16, nn, False) for nn in (32, 64, 128, 512) if nn > n]
        configs += [('baseline-repeat', 16, n, False)]
        for label, m, nn, transposed in configs:
            xx = F.pad(x, (0, 0, 0, m-4))
            ww = F.pad(w, (0, 0, 0, nn-n))
            if transposed:
                ww = ww.t().contiguous().t()
            for _ in range(3):
                output = F.linear(xx, ww)[:4, :n]
            exact = torch.equal(reference, output)
            delta = float((reference-output).abs().max())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = F.linear(xx, ww)[:4, :n]
            times = []
            for _ in range(5):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(50):
                    graph.replay()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end)/50)
            ranks = ep.gather_objects({'ms': statistics.median(times), 'exact': exact, 'delta': delta})
            if ep.rank == 0:
                print('FP32_LAYOUT ' + json.dumps({'name': name, 'stored_dtype': str(source.dtype),
                      'shape': [n, k], 'label': label, 'ranks': ranks}), flush=True)
    assert all(ep.gather_objects(True))
    os._exit(0)


if __name__ == '__main__':
    main()
