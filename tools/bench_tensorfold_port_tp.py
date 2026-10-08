"""Frozen-map TP2 exactness, throughput and memory gate for the TensorFold-inspired changes."""
import argparse,hashlib,json,os,sys,time
sys.path[:0]=['/app','/app/tools']
# A failed assertion must not hang in NCCL teardown with the peer waiting.
def fatal(kind, error, tb):
 import traceback
 traceback.print_exception(kind,error,tb);sys.stdout.flush();sys.stderr.flush();os._exit(1)
sys.excepthook=fatal
import torch
from bench_decode_timeline_tp import V,Tok,load_encoding_module,build_chat_prompt
from engine import l2pf
from engine.prefill_budget import PrefillBudget
from engine.expert_profiles import mask_digest
from bench.bench import WORKLOADS

a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--memory-only',action='store_true');a.add_argument('--attention-only',action='store_true');args=a.parse_args()
V.save_prune_db=lambda *a,**k:None
os.makedirs(args.out,exist_ok=True)
root=os.environ['MODEL_DIR']
e=V.V41Engine(root,max_seq=65536,arena_gb=90.2,trace_stats='/app/results/trace-union/stats/coverage.json',spec=True,prune_keep=.61,transient_slots=8,keep_free_gb=6)
assert e.dynamic_experts and not V.ADAPT.swap and not V.ADAPT.decode_tokens
maphash=mask_digest(e.model_prune_mask)
tok,enc=Tok(root),load_encoding_module(root)
prompts={}
for name in ('code','prose'):
 _,prompts[name],_,_=build_chat_prompt({'messages':[{'role':'user','content':'[req 41420] '+WORKLOADS[name]}]},enc,tok,False,75,e)
_,longids,_,_=build_chat_prompt({'messages':[{'role':'user','content':('The archive records the weather, train arrivals, and library opening hours for each town.\n'*650)+'Summarize the subjects recorded in the archive.'}]},enc,tok,False,75,e)
prompts['long']=longids
report={'config':e.config(),'map':maphash,'runs':[]};refs={};reply_refs={}
def save():
 with open(f'{args.out}/gate-rank{e.ep.rank}.json','w') as f:json.dump(report,f,indent=2)
def variant(fused,mode,pace,adapt=False,limit=8,mb=2):
 torch.cuda.synchronize()
 e.fast.release_graphs()
 e.fast.lean.fused_router_tail=fused
 l2pf.MODE=mode;l2pf.PACE_GBPS=pace;l2pf.MB=mb
 e.fast.graph_limit=limit
 # Force a small chunk only for the chunk-invariance gate; use real reports otherwise.
 e.prefill_budget=PrefillBudget(adapt,100 if adapt else 4,256,2)
 e.depth_policy.pinned=3
 assert len(set(e.ep.gather_objects((fused,mode,pace,adapt,limit,mb))))==1

def run(name,label,n=160,measured=True):
 torch.cuda.reset_peak_memory_stats()
 out=[]
 for b in e.generate(prompts[name],max_tokens=n,temperature=0,seed=42,ignore_eos=True):out.extend(b)
 digest=hashlib.sha256(json.dumps(out).encode()).hexdigest()
 assert len(set(e.ep.gather_objects(digest)))==1,'rank divergence'
 key=(name,n)
 exact=refs.setdefault(key,digest)==digest
 reply=tok.decode(out).split('<｜end▁of▁sentence｜>')[0]
 reply_exact=reply_refs.setdefault(key,reply)==reply
 assert mask_digest(e.model_prune_mask)==maphash
 st=e.last_stats
 row=dict(name=name,label=label,measured=measured,tokens=n,hash=digest,exact=exact,reply_exact=reply_exact,token_ids=out,text=tok.decode(out),decode_tok_s=st['decode_tok_s'],prefill_s=st['prefill_s'],memory=e.fast.memory_report(),budget=e.prefill_budget_last)
 report['runs'].append(row);save();print('PORT_RUN '+json.dumps({k:v for k,v in row.items() if k!='token_ids'}),flush=True)
 # The existing sparse indexer documents chunk-dependent tie breaking beyond 1K.
 # Require bit equality for router/prefetch/graph changes at fixed chunk size;
 # record the long chunk comparison separately for answer review.
 if name!='long':assert exact,'output changed'
 else:assert reply_exact,'normal long-prompt answer changed'
if args.attention_only:
 for mb in (0,4,8,12,12,8,4,0):
  variant(True,'touch',0)
  e.fast.attn_prefetch_mb=mb
  run('code','attention'+str(mb),64,False)
  for name in ('code','prose'):run(name,'attention'+str(mb),192)
 print('PORT_ATTN_GATE_PASS',flush=True)
elif not args.memory_only:
 for fused,mode,pace,label in [(False,'touch',0,'baseline'),(True,'touch',0,'router'),(True,'bulk',0,'bulk2'),(True,'bulk',0,'bulk4'),(True,'bulk',0,'bulk4'),(True,'bulk',0,'bulk2'),(True,'touch',0,'router'),(False,'touch',0,'baseline')]:
  variant(fused,mode,pace,mb=4 if label=='bulk4' else 2)
  run('code',label,64,False)
  for name in ('code','prose'):run(name,label)
variant(False,'touch',0)
run('long','chunk512',32,False)
variant(True,'touch',0,True)
run('long','chunk256',32,False)
# A two-entry cache holds one width/parity pair; alternate widths to force eviction.
variant(True,'touch',0,False,2)
for depth in (3,5,3):
 e.depth_policy.pinned=depth
 run('code','eviction-depth'+str(depth),96,False)
assert e.fast.graph_evictions>0
assert len(e.fast.graphs)<=2
report['passed']=True;save()
print('PORT_GATE_PASS',flush=True)
assert all(e.ep.gather_objects(True))
sys.stdout.flush();os._exit(0)
