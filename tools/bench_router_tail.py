"""Exact router-tail qualification and CUDA-graph timing, no model load."""
import sys,json,statistics
sys.path[:0]=['/app','/app/tools']
import torch
from engine.decode_lean import LeanOps
from triton.testing import do_bench_cudagraph

torch.manual_seed(712)
ops=LeanOps('cuda',16,32)
w=torch.randn(384,5120,device='cuda')*.03
bias=torch.randn(384,device='cuda')*.01
keep=torch.rand(384,device='cuda')>.4
report=[]
for t in (1,2,4,6,8):
 y=torch.randn(t,5120,device='cuda',dtype=torch.bfloat16)
 ids=torch.empty((t,6),device='cuda',dtype=torch.int64)
 out=torch.empty((t,6),device='cuda')
 def run(fused):
  ops.fused_router_tail=fused
  return ops.router(y,w,bias,keep,6,1.5,ids,out)
 # Test changing activations/masks, including ties, zeros and extreme logits.
 for i in range(80):
  y.normal_();y.mul_(10**((i%9)-4));keep.copy_(torch.rand_like(keep,dtype=torch.float32)>.4)
  if i%13==0:y.zero_()
  scores=run(False);ri=ids.clone();rw=out.clone();rs=scores.clone()
  scores=run(True)
  assert torch.equal(ids,ri) and torch.equal(out,rw) and torch.equal(scores,rs),(t,i,(out-rw).abs().max().item())
 y.normal_()
 timing={}
 for f in (False,True,True,False):
  run(f)
  timing.setdefault(str(f),[]).append(do_bench_cudagraph(lambda:run(f),rep=100)*1000)
 report.append(dict(rows=t,baseline_us=statistics.median(timing['False']),fused_us=statistics.median(timing['True']),exact=True))
print(json.dumps(report,indent=2))
