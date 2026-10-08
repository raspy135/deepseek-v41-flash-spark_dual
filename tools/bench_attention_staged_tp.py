"""One frozen-map A/B/A throughput check plus short structural answer checks.

The staged path changes accumulation, so record output differences and objective
grades instead of interpreting a collapse-only gate as quality equivalence.
"""
import argparse, hashlib, json, os, sys
sys.path[:0] = ['/app','/app/tools']
def fatal(kind,error,tb):
    import traceback
    traceback.print_exception(kind,error,tb)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
sys.excepthook=fatal
import torch
from bench_decode_timeline_tp import V,Tok,load_encoding_module,build_chat_prompt
from engine.expert_profiles import mask_digest
from bench.bench import WORKLOADS

ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True)
ap.add_argument('--mode',type=int,choices=(1,2),default=1); args=ap.parse_args()
os.makedirs(args.out,exist_ok=True)
V.save_prune_db=lambda *a,**kw:None
root=os.environ['MODEL_DIR']
e=V.V41Engine(root,max_seq=32768,arena_gb=90.2,
    trace_stats='/app/results/trace-union/stats/coverage.json',spec=True,
    prune_keep=.61,transient_slots=8,keep_free_gb=6)
assert e.dynamic_experts and not V.ADAPT.swap and not V.ADAPT.decode_tokens
e.depth_policy.pinned=3
maphash=mask_digest(e.model_prune_mask)
tok,enc=Tok(root),load_encoding_module(root)
eos=tok.token_to_id(enc.eos_token)
prompts={name:'[req 70101] '+WORKLOADS[name] for name in ('prose','code')}
prompts.update({
    'nesting':'Output only a JSON value with exactly eight nested objects, each having the single key "n". The innermost value is 0. No markdown or explanation.',
    'copy':'Output exactly the following text and nothing else, preserving punctuation and capitalization: A9-zQ_27 / café / 日本語 / {"ok":true}',
    'arithmetic':'Output only a JSON object with two keys: "sum" is the sum of all integers from 1 through 99, and "count" is how many of those integers are even. No markdown.'})
ids={name:build_chat_prompt({'messages':[{'role':'user','content':p}]},enc,tok,False,75,e)[1] for name,p in prompts.items()}
expected=0
for _ in range(8):expected={'n':expected}
def grade(name,text):
    if name=='copy':return text.strip()=='A9-zQ_27 / café / 日本語 / {"ok":true}'
    try:obj=json.loads(text)
    except ValueError:return False
    return obj==(expected if name=='nesting' else {'sum':4950,'count':49})
report={'config':e.config(),'candidate_mode':args.mode,'map':maphash,'runs':[]}; refs={}
def save():
    with open(f'{args.out}/rank{e.ep.rank}.json','w') as f:json.dump(report,f,indent=2)
def run(name,arm,n,measured):
    out=[]
    kw={'ignore_eos':True} if name in ('prose','code') else {'stop_token_ids':{eos}}
    for b in e.generate(ids[name],max_tokens=n,temperature=0,seed=42,**kw):out.extend(b)
    digest=hashlib.sha256(json.dumps(out).encode()).hexdigest()
    assert len(set(e.ep.gather_objects(digest)))==1,'rank divergence'
    assert mask_digest(e.model_prune_mask)==maphash
    text=tok.decode([t for t in out if t!=eos])
    row=dict(name=name,arm=arm,measured=measured,tokens=len(out),hash=digest,
        exact=refs.setdefault((name,n),digest)==digest,text=text,stats=e.last_stats,
        grade=None if name in ('prose','code') else grade(name,text))
    report['runs'].append(row);save()
    print('STAGED_TP_RUN '+json.dumps({k:v for k,v in row.items() if k!='text'}),flush=True)
for arm in ('baseline','staged','baseline_repeat'):
    torch.cuda.synchronize();e.fast.release_graphs()
    e.fast.staged_attention=args.mode if arm=='staged' else 0
    assert len(set(e.ep.gather_objects((arm,e.fast.staged_attention))))==1
    run('code',arm,32,False)
    for name in ('prose','code'):run(name,arm,192,True)
    if arm!='baseline_repeat':
        for name in ('nesting','copy','arithmetic'):run(name,arm,128,False)
report['completed']=True;save()
print('STAGED_TP_COMPLETE',flush=True)
assert all(e.ep.gather_objects(True))
sys.stdout.flush();os._exit(0)
