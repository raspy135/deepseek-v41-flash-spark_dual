"""Real checkpoint cold-weight A/B; pipeline scheduling must preserve every FP32 result."""
import json, os, sys, statistics
sys.path[:0] = ['/app', '/app/tools']
import torch
from safetensors import safe_open
import fp8_linear as F

root = os.environ['MODEL_DIR']
idx = json.load(open(root + '/model.safetensors.index.json'))['weight_map']
torch.manual_seed(71)
report = []
def load(layer, name):
    ts = []
    for suffix in ('weight', 'scale'):
        key = f'layers.{layer}.' + name + '.' + suffix
        with safe_open(root + '/' + idx[key], framework='pt', device='cpu') as f:
            ts.append(f.get_tensor(key).cuda())
    return F.FP8Weight(*ts)

def time(fn, ws):
    for w in ws: fn(w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for w in ws: fn(w)
    times = []
    for _ in range(7):
        graph.replay()
        a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        a.record(); graph.replay(); b.record(); b.synchronize()
        times.append(a.elapsed_time(b) / len(ws))
    return statistics.median(times)

for layer in (0,10,20,30,39):
    for name in ('qkv', 'wq_b', 'wo_b', 'wo_a', 'shared13', 'shared2'):
        if name=='qkv':
            w,_ = F.concat_rows(load(layer,'attn.wq_a'),load(layer,'attn.wkv'))
        elif name=='shared13':
            w,_ = F.concat_rows(*[load(layer,'ffn.shared_experts.'+s).shard(0,0,2) for s in ('w1','w3')])
        elif name=='shared2': w=load(layer,'ffn.shared_experts.w2').shard(1,0,2)
        else: w=load(layer,'attn.'+name).shard(0,0,2)
        if name=='wo_a':
            w=F.FP8GroupedWeight(w.w,w.s,4,1024)
            fn=lambda x,v:F.fp8_grouped_linear(x,v)
        else: fn=lambda x,v:F.fp8_linear(x,v,act_qdq=True,out_dtype=torch.float32)
        # All selected layers are correctness checks; layer zero carries the timing sweep.
        ws=([F.FP8GroupedWeight(w.w.clone(),w.s.clone(),w.G,w.R) if name=='wo_a'
             else F.FP8Weight(w.w.clone(),w.s.clone()) for _ in range(16)] if layer==0 else [])
        for rows in (1,4,6,16):
            x=torch.randn((rows,4,w.K) if name=='wo_a' else (rows,w.K),device='cuda',dtype=torch.bfloat16)*.5
            F.DECODE_PIPELINE=False; ref=fn(x,w)
            F.DECODE_PIPELINE=True; got=fn(x,w)
            assert torch.equal(ref,got),(layer,name,rows,float((ref-got).abs().max()))
            # Position/width invariant when used by speculative verification and drafting.
            assert torch.equal(got[:1],fn(x[:1],w)),(layer,name,rows,'row invariance')
            if layer==0 and rows<16:
                times={False:[],True:[]}
                for arm in (False,True,True,False):
                    F.DECODE_PIPELINE=arm;times[arm].append(time(lambda v:fn(x,v),ws))
                row=dict(layer=layer,name=name,rows=rows,baseline_ms=statistics.median(times[False]),pipeline_ms=statistics.median(times[True]))
                report.append(row);print(json.dumps(row),flush=True)
                with open('/out/pipeline.json','w') as f:json.dump(report,f,indent=2)
        print('LAYER_SHAPE_EXACT',layer,name,flush=True)
        del ws,w
print('PIPELINE_MICRO_PASS',flush=True)
