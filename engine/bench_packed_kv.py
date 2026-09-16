"""Interleaved CUDA-graph timing of QDQ/write and indexed gather (milliseconds)."""
import json
import statistics
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import v41_ref as R
from engine.packed_kv import write, gather


def captured(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def main():
    torch.manual_seed(1)
    for t in (4, 1024):
        rows = 16384
        dense = torch.randn(rows, 512, device='cuda', dtype=torch.bfloat16)
        packed = torch.empty(rows, 36, device='cuda', dtype=torch.int64)
        write(packed, dense, 0)
        dense.copy_(R.fp4_qdq(dense, 16, 'e4m3'))
        ids = torch.randint(rows, (t, 512), device='cuda')
        x = torch.randn(t, 512, device='cuda', dtype=torch.bfloat16)
        dst = torch.arange(t, device='cuda')
        grid = R.FP4_GRID.to('cuda')
        def baseline_write():
            dense[dst] = R.fp4_qdq(x, 16, 'e4m3', grid)
        fns = {'bf16_gather': lambda: dense[ids],
               'packed_gather': lambda: gather(packed, ids),
               'bf16_write': baseline_write,
               'packed_write': lambda: write(packed, x, dst)}
        graphs = {name: captured(fn) for name, fn in fns.items()}
        samples = {name: [] for name in graphs}
        for trial in range(10):
            for name in list(graphs)[::1 if trial % 2 else -1]:
                a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                for _ in range(10):
                    graphs[name][0].replay()
                b.record(); b.synchronize()
                samples[name].append(a.elapsed_time(b) / 10)
        print(json.dumps({'tokens': t, 'median_ms': {k: statistics.median(v)
                                                  for k, v in samples.items()}}), flush=True)


if __name__ == '__main__':
    main()
