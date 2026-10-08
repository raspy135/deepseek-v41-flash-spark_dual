"""Cold-weight, same-arithmetic decode schedule sweep. Experimental, not a preset."""
import json, os, sys, statistics
sys.path[:0] = ['/app', '/app/tools']
import torch
import triton
from safetensors import safe_open
import fp8_linear as F

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

def run(x, w, bn, warps, stages):
    y = torch.empty((x.shape[0], w.N), device=x.device, dtype=torch.float32)
    F._fp8_linear_kernel[(triton.cdiv(w.N, bn), 1)](
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

for name in ('wq_b', 'wo_b', 'wq_a'):
    w = load(name)
    ws = [F.FP8Weight(w.w.clone(), w.s.clone()) for _ in range(16)]
    x = torch.randn(4, w.K, device='cuda', dtype=torch.bfloat16) * .5
    ref = F.fp8_linear(x, w, act_qdq=True, out_dtype=torch.float32)
    configs = [(bn, warps, stages) for bn in (16,32,64,128) for warps in (2,4,8) for stages in (1,3,5)]
    for bn, warps, stages in configs:
        try:
            y = run(x, w, bn, warps, stages)
            exact = torch.equal(y, ref)
            ms = time(lambda v: run(x, v, bn, warps, stages), ws) if exact else None
            row = dict(name=name, bn=bn, warps=warps, stages=stages, exact=exact, ms=ms)
        except Exception as e:
            row = dict(name=name, bn=bn, warps=warps, stages=stages, error=str(e)[:400])
        report.append(row)
        print(json.dumps(row), flush=True)
        with open('/out/schedule.json', 'w') as f: json.dump(report, f, indent=2)
    del ws, w
print('SCHEDULE_PASS', flush=True)
