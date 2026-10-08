"""Small accuracy/latency screen, including FP64 error and masked attention rows.

No model or expert adaptation. A synthetic kernel screen cannot establish answer
quality; its purpose is to reject slow/inaccurate candidates before model loading.
"""
import json
import os
import statistics
import sys
sys.path[:0] = ['/app', '/app/tools']
import torch
from engine.decode_lean import LeanOps
import decode_attn_staged as S

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.manual_seed(418)
lean = LeanOps('cuda', 16, 16)
report = []

def reference(q, k, mask, sink):
    s = (q.double() @ k.double().transpose(1, 2)) * (q.shape[-1] ** -.5)
    s.masked_fill_(~mask[:, None], -float('inf'))
    mx = s.amax(-1, keepdim=True).clamp_min(-1e30)
    p = (s-mx).exp()
    return p / (p.sum(-1, keepdim=True)+(sink.double()[None,:,None]-mx).exp()) @ k.double()

for t, n in ((4,128),(4,640),(4,3200),(6,640),(1,133)):
    q = torch.randn(t,32,512,device='cuda',dtype=torch.bfloat16).float()
    k = torch.randn(t,n,512,device='cuda',dtype=torch.bfloat16).float()
    mask = torch.rand(t,n,device='cuda') > .15
    if t>1: mask[0,:] = False  # all-masked rows must remain finite zero
    sink = torch.randn(32,device='cuda')
    scale = 512 ** -.5
    def baseline():
        p = lean.attn_probs(q @ k.transpose(1,2),mask,sink,scale)
        return p @ k
    def candidate():
        return S.attention(q,k,mask,sink,scale,lean)
    def bf16x3():
        return S.attention(q,k,mask,sink,scale,lean,True)
    def split4():
        return S.attention(q,k,mask,sink,scale,lean,split=4)
    ref = reference(q,k,mask,sink)
    graphs, outputs = {}, {}
    for name,fn in [('baseline',baseline),('staged',candidate),('split4',split4)]:
        for _ in range(3): fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): outputs[name] = fn()
        graph.replay()
        graphs[name] = graph
    times = {name:[] for name in graphs}
    for _ in range(5):
        for name in ('baseline','staged','split4','split4','staged','baseline'):
            a,b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            a.record()
            for _ in range(20): graphs[name].replay()
            b.record(); b.synchronize()
            times[name].append(a.elapsed_time(b)/20)
    row = {'rows':t,'keys':n,'ms':{name:statistics.median(v) for name,v in times.items()}}
    for name,out in outputs.items():
        assert torch.isfinite(out).all()
        if t>1: assert torch.count_nonzero(out[0]) == 0
        row[name+'_relative_l2_vs_fp64'] = float((out.double()-ref).norm()/ref.norm().clamp_min(1e-30))
        row[name+'_max_abs_vs_fp64'] = float((out.double()-ref).abs().max())
    row['bf16_mismatch_fraction'] = float((outputs['baseline'].bfloat16()!=outputs['staged'].bfloat16()).float().mean())
    assert row['staged_relative_l2_vs_fp64'] < 2e-6, row
    assert row['split4_relative_l2_vs_fp64'] < 2e-6, row
    if t>1:
        # A query must not change when alone or in a speculative verification block.
        alone=S.attention(q[1:2],k[1:2],mask[1:2],sink,scale,lean)
        assert torch.equal(alone,outputs['staged'][1:2]), 'row invariance'
        alone=S.attention(q[1:2],k[1:2],mask[1:2],sink,scale,lean,split=4)
        assert torch.equal(alone,outputs['split4'][1:2]), 'split row invariance'
    report.append(row)
    print(json.dumps(row),flush=True)
os.makedirs('/out',exist_ok=True)
with open('/out/attention-staged.json','w') as f: json.dump(report,f,indent=2)
print('STAGED_ATTN_SCREEN_PASS',flush=True)
