"""Frozen expert map, exact-output TP2 decode A/B for native FP8 scheduling."""
import argparse, hashlib, json, os, sys
sys.path[:0] = ['/app', '/app/tools']
def fatal(kind, error, tb):
    import traceback
    traceback.print_exception(kind, error, tb)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
sys.excepthook = fatal
import torch
from bench_decode_timeline_tp import V, Tok, load_encoding_module, build_chat_prompt
from engine.expert_profiles import mask_digest
from bench.bench import WORKLOADS
import fp8_linear as F

ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
V.save_prune_db=lambda *a,**kw:None
os.makedirs(args.out,exist_ok=True)
root=os.environ['MODEL_DIR']
e=V.V41Engine(root,max_seq=32768,arena_gb=90.2,trace_stats='/app/results/trace-union/stats/coverage.json',spec=True,prune_keep=.61,transient_slots=8,keep_free_gb=6)
assert e.dynamic_experts and not V.ADAPT.swap and not V.ADAPT.decode_tokens
maphash=mask_digest(e.model_prune_mask)
tok,enc=Tok(root),load_encoding_module(root)
prompts={}
for i,name in enumerate(('prose','code','angry-birds')):
    prompt=WORKLOADS[name] if name=='angry-birds' else f'[req {70100+i}] '+WORKLOADS[name]
    _,prompts[name],_,_=build_chat_prompt({'messages':[{'role':'user','content':prompt}]},enc,tok,False,75,e)
report={'config':e.config(),'map':maphash,'runs':[]};refs={}
def run(name,arm,depth,n,measured):
    out=[]
    for b in e.generate(prompts[name],max_tokens=n,temperature=0,seed=42,ignore_eos=True):out.extend(b)
    digest=hashlib.sha256(json.dumps(out).encode()).hexdigest()
    assert len(set(e.ep.gather_objects(digest)))==1,'rank divergence'
    key=(name,depth,n)
    exact=refs.setdefault(key,digest)==digest
    assert mask_digest(e.model_prune_mask)==maphash
    row=dict(name=name,pipeline=arm,depth=depth,measured=measured,tokens=len(out),hash=digest,exact=exact,text=tok.decode(out),stats=e.last_stats,memory=e.fast.memory_report())
    report['runs'].append(row)
    with open(f'{args.out}/rank{e.ep.rank}.json','w') as f:json.dump(report,f,indent=2)
    print('PIPELINE_TP_RUN '+json.dumps({k:v for k,v in row.items() if k not in ('text','memory')}),flush=True)
    assert exact,'output changed'

for depth,arms in ((3,(False,True,True,False)),(5,(False,True))):
    for arm in arms:
        torch.cuda.synchronize();e.fast.release_graphs()
        F.DECODE_PIPELINE=arm;e.depth_policy.pinned=depth
        assert len(set(e.ep.gather_objects((arm,depth))))==1
        # Prime this graph width and all three prompts before collecting timings.
        for name in prompts:run(name,arm,depth,64,False)
        if depth==3:
            for name in prompts:run(name,arm,depth,192,True)
report['passed']=True
with open(f'{args.out}/rank{e.ep.rank}.json','w') as f:json.dump(report,f,indent=2)
print('PIPELINE_TP_PASS',flush=True)
assert all(e.ep.gather_objects(True))
sys.stdout.flush();os._exit(0)
