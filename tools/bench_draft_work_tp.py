"""One bounded TP2 A/B/A of reduced draft work, with frozen expert residency."""
import argparse, hashlib, json, os, sys
sys.path[:0] = ['/app', '/app/tools']
def fatal(kind, error, tb):
    import traceback
    message=''.join(traceback.format_exception(kind,error,tb))
    os.write(2,message.encode('utf-8','backslashreplace'))
    if 'args' in globals():
        with open(args.out+'/failure-rank'+os.environ.get('RANK','?')+'.txt','w') as f:f.write(message)
    sys.stdout.flush();sys.stderr.flush();os._exit(1)
sys.excepthook=fatal
import torch
from bench_decode_timeline_tp import V,Tok,load_encoding_module,build_chat_prompt
from engine.expert_profiles import mask_digest
from engine.tensor_parallel import shard_attention
from bench.bench import WORKLOADS

ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--resume');args=ap.parse_args()
# Fail preflight before spending time loading the model. Each rank owns its result file.
prior=json.load(open(args.resume.replace('rank0.json','rank'+os.environ.get('RANK','0')+'.json'))) if args.resume else None
V.save_prune_db=lambda *a,**kw:None
os.makedirs(args.out,exist_ok=True)
root=os.environ['MODEL_DIR']
e=V.V41Engine(root,max_seq=32768,arena_gb=90.2,trace_stats='/app/results/trace-union/stats/coverage.json',spec=True,prune_keep=.61,transient_slots=8,keep_free_gb=6)
e.depth_policy.pinned=3
assert e.dynamic_experts and not V.ADAPT.swap and not V.ADAPT.decode_tokens
maphash=mask_digest(e.model_prune_mask)
tok,enc=Tok(root),load_encoding_module(root)
prompts={}
for name in ('prose','code'):
    _,prompts[name],_,_=build_chat_prompt({'messages':[{'role':'user','content':'[req 70101] '+WORKLOADS[name]}]},enc,tok,False,75,e)
# Preserve the original objects only in this disposable diagnostic. Serving loads one layout.
attrs=('wq_b','attn_sink','wo_a','wo_b','tp_heads','tp_groups')
weights=e.fast.W.mtp
original=[{a:getattr(w,a) for a in attrs if hasattr(w,a)} for w in weights]
for w in weights:shard_attention(w,e.args,e.ep.rank,e.ep.world)
sharded=[{a:getattr(w,a) for a in attrs} for w in weights]
report={'config':e.config(),'map':maphash,'runs':[]};refs={}
if args.resume:
    assert prior['map']==maphash
    for r in prior['runs']:refs[r['name'],r['tokens']]=r['hash']
    report['prior']=args.resume
def save():
    with open(f'{args.out}/rank{e.ep.rank}.json','w') as f:json.dump(report,f,indent=2)
def run(name,arm,n,measured):
    out=[]
    for b in e.generate(prompts[name],max_tokens=n,temperature=0,seed=42,ignore_eos=True):out.extend(b)
    digest=hashlib.sha256(json.dumps(out).encode()).hexdigest()
    assert len(set(e.ep.gather_objects(digest)))==1,'rank divergence'
    exact=refs.setdefault((name,n),digest)==digest
    assert mask_digest(e.model_prune_mask)==maphash
    row=dict(name=name,arm=arm,measured=measured,tokens=len(out),hash=digest,exact=exact,text=tok.decode(out),stats=e.last_stats,memory=e.fast.memory_report())
    report['runs'].append(row);save()
    print('DRAFT_WORK_RUN '+json.dumps({k:v for k,v in row.items() if k not in ('text','memory')}),flush=True)
    assert exact,'target output changed'

for arm in (('combined','baseline') if args.resume else ('baseline','combined','baseline')):
    print('DRAFT_WORK_STAGE '+arm+' release',flush=True)
    torch.cuda.synchronize();e.fast.release_graphs()
    state=sharded if arm=='combined' else original
    for w,s in zip(weights,state):
        for a in attrs:
            if a in s:setattr(w,a,s[a])
            elif hasattr(w,a):delattr(w,a)
    e.fast.draft_markov_topk=128 if arm=='combined' else 0
    assert len(set(e.ep.gather_objects((arm,e.fast.draft_markov_topk))))==1
    print('DRAFT_WORK_STAGE '+arm+' warm',flush=True)
    run('code',arm,64,False)
    for name in prompts:run(name,arm,192,True)
report['passed']=True;save()
print('DRAFT_WORK_PASS',flush=True)
assert all(e.ep.gather_objects(True))
sys.stdout.flush();os._exit(0)
