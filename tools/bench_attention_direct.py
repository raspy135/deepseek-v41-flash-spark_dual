"""Screen direct packed-key reads against staged attention, including gather cost."""
import json, statistics, sys
sys.path[:0]=['/app','/app/tools']
import torch
from engine.decode_lean import LeanOps
from engine.packed_kv import write
from decode_attn_staged import attention,attention_direct
torch.manual_seed(871)
lean=LeanOps('cuda',16,16)
report=[]
for t,n2 in ((4,0),(4,1024),(4,3072),(6,1024)):
    q=torch.randn(t,32,512,device='cuda',dtype=torch.bfloat16).float()
    ring=torch.randn(256,512,device='cuda',dtype=torch.bfloat16)
    slots=torch.randint(0,256,(t,128),device='cuda')
    packed=torch.empty(4096,36,device='cuda',dtype=torch.int64)
    write(packed,torch.randn(4096,512,device='cuda',dtype=torch.bfloat16),0)
    ids=torch.randint(0,4096,(t,n2),device='cuda') if n2 else None
    mask=torch.rand(t,128+n2,device='cuda')>.2
    mask[0]=False
    sink=torch.randn(32,device='cuda')
    def staged():
        keys=lean.keys_f32(ring,slots,packed if n2 else None,ids)
        return attention(q,keys,mask,sink,512**-.5,lean)
    def direct():
        return attention_direct(q,ring,slots,packed if n2 else None,ids,mask,sink,512**-.5,lean)
    def bf16():
        keys=lean.keys_f32(ring,slots,packed if n2 else None,ids,dtype=torch.bfloat16)
        return attention(q,keys,mask,sink,512**-.5,lean)
    k32=lean.keys_f32(ring,slots,packed if n2 else None,ids)
    k16=lean.keys_f32(ring,slots,packed if n2 else None,ids,dtype=torch.bfloat16)
    assert torch.equal(k32.view(torch.int32),k16.float().view(torch.int32)), 'key staging changed bits'
    graphs={};outs={}
    for name,fn in [('staged',staged),('direct',direct),('bf16',bf16)]:
        for _ in range(3):fn()
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):outs[name]=fn()
        g.replay();graphs[name]=g
    delta=outs['staged']-outs['direct']
    rel=float(delta.norm()/outs['staged'].norm().clamp_min(1e-30))
    assert torch.isfinite(outs['direct']).all() and rel < 2e-6, (t,n2,rel)
    times={k:[] for k in graphs}
    for _ in range(5):
        for name in ('staged','direct','bf16','bf16','direct','staged'):
            a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
            a.record()
            for _ in range(20):graphs[name].replay()
            b.record();b.synchronize();times[name].append(a.elapsed_time(b)/20)
    row=dict(rows=t,keys=128+n2,ms={k:statistics.median(v) for k,v in times.items()},
        exact=torch.equal(outs['staged'],outs['direct']),relative_l2=rel,max_abs=float(delta.abs().max()),
        bf16_mismatch_fraction=float((outs['staged'].bfloat16()!=outs['direct'].bfloat16()).float().mean()),
        eliminated_key_buffer_bytes=t*(128+n2)*512*4)
    row['bf16_staging_exact']=torch.equal(outs['staged'],outs['bf16'])
    row['bf16_staging_relative_l2']=float((outs['staged']-outs['bf16']).norm()/outs['staged'].norm())
    assert row['bf16_staging_relative_l2']<2e-6
    report.append(row);print(json.dumps(row),flush=True)
    with open('/out/attention-direct.json','w') as f:json.dump(report,f,indent=2)
print('DIRECT_SCREEN_PASS',flush=True)
