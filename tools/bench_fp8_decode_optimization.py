"""Cold-weight graph microbenchmark for exact FP8 decode candidates."""
import sys,os,json,statistics
sys.path[:0]=['/app','/app/tools']
import torch
from safetensors import safe_open
import fp8_linear as F
from fp8_decode_experiments import qdq,parallel
from v41_ref import act_qdq_fp8
root=os.environ['MODEL_DIR'];idx=json.load(open(root+'/model.safetensors.index.json'))['weight_map']
report=[]
torch.manual_seed(20261007)
def weight(name,shard):
 ts=[]
 for suffix in ('weight','scale'):
  key='layers.0.attn.'+name+'.'+suffix
  with safe_open(root+'/'+idx[key],framework='pt',device='cpu') as f:ts.append(f.get_tensor(key).cuda())
 w=F.FP8Weight(*ts)
 return w.shard(0,0,2) if shard else w

def time_fn(fn,ws):
 # Distinct weight addresses exceed L2; never report a repeatedly cached single matrix.
 for w in ws:fn(w)
 torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):
  for w in ws:fn(w)
 samples=[]
 for _ in range(4):
  g.replay();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  a.record();g.replay();b.record();b.synchronize();samples.append(a.elapsed_time(b)/len(ws))
 return statistics.median(samples)
for name,shard in [('wq_a',False),('wkv',False),('wq_b',True),('wo_b',True)]:
 w=weight(name,shard);ws=[F.FP8Weight(w.w.clone(),w.s.clone()) for _ in range(16)]
 for rows in (1,4,6):
  x=torch.randn(rows,w.K,device='cuda',dtype=torch.bfloat16)*.5
  assert torch.equal(qdq(x),act_qdq_fp8(x))
  for act in (False,True):
   refs=F.fp8_linear(x,w,act_qdq=act,out_dtype=torch.float32)
   arms={'baseline':lambda v:F.fp8_linear(x,v,act_qdq=act)}
   if act:arms['qdq_once']=lambda v:F.fp8_linear(qdq(x),v,act_qdq=False)
   for bn in (32,64,128):
    z=parallel(x,w,act,bn,dtype=torch.float32)
    exact=torch.equal(z,refs)
    print('CHECK',name,rows,act,bn,exact,'max',float((z-refs).abs().max()),flush=True)
    if exact:arms['parallel'+str(bn)]=lambda v,bn=bn:parallel(x,v,act,bn)
   if act:assert torch.equal(F.fp8_linear(qdq(x),w,out_dtype=torch.float32),refs)
   times={key:[] for key in arms}
   for rnd in range(2):
    for key in (list(arms) if rnd==0 else list(arms)[::-1]):times[key].append(time_fn(arms[key],ws))
   row={'name':name,'shape':w.shape,'rows':rows,'act_qdq':act,'ms':{k:statistics.median(v) for k,v in times.items()}}
   report.append(row);print('RESULT '+json.dumps(row),flush=True)
   with open('/out/micro.json','w') as f:json.dump(report,f,indent=2)
 del ws,w
print('MICRO_PASS',flush=True)
