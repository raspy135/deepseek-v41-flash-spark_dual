"""Isolate native attention batch-one arithmetic before another model load."""
import json
import sys
sys.path[:0] = ['/app','/app/tools']
import torch
from engine.decode_lean import LeanOps

torch.manual_seed(718)
torch.backends.cuda.matmul.allow_tf32 = False
lean=LeanOps('cuda',16,32)
report=[]
for n in (128,1152,3200):
    q=torch.randn(6,32,512,device='cuda',dtype=torch.bfloat16).float()
    k=torch.randn(6,n,512,device='cuda',dtype=torch.bfloat16).float()
    sink=torch.randn(32,device='cuda')
    mask=torch.ones(6,n,device='cuda',dtype=torch.bool)
    scores=torch.einsum('thd,tnd->thn',q,k)
    probs=lean.attn_probs(scores,mask,sink,512**-.5)
    values=torch.einsum('thn,tnd->thd',probs,k)
    for t in (1,2,3,4,5):
        s=torch.einsum('thd,tnd->thn',q[:t],k[:t])
        p=lean.attn_probs(s,mask[:t],sink,512**-.5)
        v=torch.einsum('thn,tnd->thd',p,k[:t])
        row=dict(keys=n,rows=t,score_exact=torch.equal(s,scores[:t]),
                 max_score=float((s-scores[:t]).abs().max()),
                 value_exact=torch.equal(v,values[:t]),
                 max_value=float((v-values[:t]).abs().max()),
                 bf16_differences=int((v.bfloat16()!=values[:t].bfloat16()).sum()))
        if t==1:
            ss=torch.einsum('thd,tnd->thn',q[:1].expand(2,-1,-1),k[:1].expand(2,-1,-1))[:1]
            pp=lean.attn_probs(ss,mask[:1],sink,512**-.5)
            vv=torch.einsum('thn,tnd->thd',pp.expand(2,-1,-1),k[:1].expand(2,-1,-1))[:1]
            row.update(batch2_score_exact=torch.equal(ss,scores[:1]),
                       batch2_value_exact=torch.equal(vv,values[:1]),
                       batch2_bf16_differences=int((vv.bfloat16()!=values[:1].bfloat16()).sum()))
        report.append(row)
print(json.dumps(report,indent=2),flush=True)
