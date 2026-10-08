"""Cold-weight, same-arithmetic decode schedule sweep. Experimental, not a preset."""
import json, os, sys, statistics
sys.path[:0] = ['/app', '/app/tools']
import torch
import triton
from safetensors import safe_open
import fp8_linear as F
from fp8_decode_experiments import _bit_linear_kernel

root = os.environ['MODEL_DIR']
idx = json.load(open(root + '/model.safetensors.index.json'))['weight_map']
torch.manual_seed(71)
report = []

def load(name):
    ts = []
    for suffix in ('weight', 'scale'):
        key = 'layers.0.attn.' + name + '.' + suffix
        with safe_open(root + '/' + idx[key], framework='pt', device='cpu') as f:
            ts.append(f.get_tensor(key).cuda())
    w = F.FP8Weight(*ts)
    return w.shard(0, 0, 2) if name in ('wq_b', 'wo_b') else w

def run(x, w, bn, warps, stages, bit=False):
    y = torch.empty((x.shape[0], w.N), device=x.device, dtype=torch.float32)
    (_bit_linear_kernel if bit else F._fp8_linear_kernel)[(triton.cdiv(w.N, bn), 1)](
        x, w.w, w.s, y, x.shape[0], w.N, w.K,
        x.stride(0), w.w.stride(0), w.s.stride(0), y.stride(0),
        BLOCK_M=16, BLOCK_N=bn, BLOCK_K=128, ACT_QDQ=True,
        num_warps=warps, num_stages=stages)
    return y

def time(fn, ws):
    for w in ws: fn(w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for w in ws: fn(w)
    times = []
    for _ in range(5):
        graph.replay()
        a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        a.record(); graph.replay(); b.record(); b.synchronize()
        times.append(a.elapsed_time(b) / len(ws))
    return statistics.median(times)

for name in ('wq_b', 'wo_b', 'wq_a', 'wkv'):
    w = load(name)
    ws = [F.FP8Weight(w.w.clone(), w.s.clone()) for _ in range(16)]
    bn = F.decode_block_n(w.N, 1, w.w.device)
    for rows in (1,4,6):
        x = torch.randn(rows, w.K, device='cuda', dtype=torch.bfloat16) * .5
        ref = F.fp8_linear(x, w, act_qdq=True, out_dtype=torch.float32)
        configs = [(False,3), (False,5), (True,3), (True,5)]
        for rnd in range(2):
            for bit, stages in (configs if rnd == 0 else configs[::-1]):
                y = run(x, w, bn, 4, stages, bit)
                exact = torch.equal(y, ref)
                ms = time(lambda v: run(x, v, bn, 4, stages, bit), ws) if exact else None
                row = dict(name=name, rows=rows, bn=bn, bit=bit, stages=stages, exact=exact, maxerr=float((y-ref).abs().max()), ms=ms, rnd=rnd)
                report.append(row); print(json.dumps(row), flush=True)
                with open('/out/bits.json', 'w') as f: json.dump(report, f, indent=2)
    del ws, w
print('BITS_PASS', flush=True)
