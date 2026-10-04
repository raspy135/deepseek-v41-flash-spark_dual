"""TP2 rolling-trigger gate: isolated monitoring cost, real swap safety, natural misses.

No prefix reuse, prefill swapping or saved demand writes. Monitoring A/B pins draft
width and suppresses swaps to isolate its cost. Forced-trigger runs synthesize ONLY
miss-counter input, then use the real planner/load path; they are correctness tests,
not evidence of natural miss rates or improved output quality.
"""
import os
import sys
sys.path[:0]=['/app','/app/tools']
for key in ('DSV41_PREFIX_CACHE','DSV41_PREFIX_DISK','DSV41_PREFIX_RESPONSE','DSV41_PRUNE_SWAP_PREFILL'):
    os.environ[key]='0'
os.environ['DSV41_PRUNE_SWAP']='1'
import argparse
from dataclasses import replace
import hashlib
import json
import time
import torch
import engine.v41_engine as V
from server.app import Tok, load_encoding_module, build_chat_prompt

PROMPTS={
 'html':'Write a complete small HTML page for a coffee shop with embedded CSS, a heading, three menu items, opening hours and a footer. Output only HTML.',
 'python':'Write a Python module implementing an LRU cache class with get, put and a max-size eviction policy, plus five unittest test cases. Output only the code.',
 'prose':'Explain in plain prose, for a curious non-specialist, why the sky is blue and why sunsets are red. Use about five paragraphs and no lists or headings.',
 'story':'Write an original short story about a lighthouse keeper who finds an unusual object washed ashore. Literary tone, no title.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True)
    ap.add_argument('--max-tokens',type=int,default=512)
    ap.add_argument('--overhead-only',action='store_true',help='Fully warm prose, then monitor-only ABBA')
    args=ap.parse_args()
    V.save_prune_db=lambda *a,**kw:None
    base=V.ADAPT
    assert base.urgent and base.swap
    root=os.environ['MODEL_DIR']
    e=V.V41Engine(root,max_seq=524288,arena_gb=90,
        trace_stats='/app/results/trace-union/stats/coverage.json',spec=True,
        prune_keep=.61,transient_slots=16,keep_free_gb=6)
    tok,enc=Tok(root),load_encoding_module(root)
    eos=tok.token_to_id(enc.eos_token)
    ids={name:build_chat_prompt({'messages':[{'role':'user','content':prompt}]},enc,tok,False,75,e)[1]
         for name,prompt in PROMPTS.items()}
    e.depth_policy.pinned=3
    e.confidence_depth_policy.pinned=3
    report={'config':e.config(),'runs':[],'mismatches':[]}
    references={}
    real_snapshot=e.model.decode_miss_snapshot
    real_apply=e.apply_swaps
    applied=[]
    def tracked_apply(swaps):
        result=real_apply(swaps)
        applied.append(list(swaps))
        return result
    e.apply_swaps=tracked_apply
    def save():
        os.makedirs(args.out,exist_ok=True)
        path=f'{args.out}/urgent-rank{e.ep.rank}.json'
        with open(path,'w') as f:json.dump(report,f,indent=2)
        owner=os.stat('/app/results');os.chown(path,owner.st_uid,owner.st_gid)
    def run(name,mode,count,measured=False,temp=0):
        V.ADAPT=replace(base, urgent=mode!='off',urgent_miss=1.0 if mode=='monitor' else .10,
                        decode_tokens=100000 if mode in ('off','monitor') else base.decode_tokens)
        e.model.decode_miss_snapshot=real_snapshot
        if mode=='forced':
            def forced():
                snap=real_snapshot()
                return None if snap is None else (.20*snap[1],snap[1])
            e.model.decode_miss_snapshot=forced
        applied.clear()
        out=[]
        start=time.perf_counter()
        for burst in e.generate(ids[name],max_tokens=count,temperature=temp,seed=42,stop_token_ids={eos}):
            out.extend(burst)
        elapsed=time.perf_counter()-start
        digest=hashlib.sha256(json.dumps(out).encode()).hexdigest()
        parity=len(set(e.ep.gather_objects(digest)))==1
        exact=True
        if mode in ('off','monitor'):
            exact=references.setdefault((name,count,temp),out)==out
        exact=all(e.ep.gather_objects(exact))
        st=e.last_stats
        events=st['decode_adaptation']
        item=dict(workload=name,mode=mode,measured=measured,temperature=temp,tokens=len(out),
            decode_tok_s=st['decode_tok_s'],decode_s=st['decode_s'],steps=st['steps'],
            wall_s=elapsed,adaptation=events,rank_parity=parity,exact_monitor_output=exact,
            sha256=digest,token_ids=out,plans=[list(x) for x in applied])
        report['runs'].append(item)
        if not (parity and exact):report['mismatches'].append(f'{name}/{mode}')
        save()
        print('URGENT_RUN '+json.dumps({k:v for k,v in item.items() if k not in ('token_ids','plans','adaptation')}
              | {'passes':events['passes'],'last_check':events['checks'][-1:]},),flush=True)
        assert parity and exact
        if mode in ('off','monitor'):assert not applied
        if mode=='forced':
            assert events['passes'], 'forced trigger never reached planner'
            assert any(p['swaps']>0 for p in events['passes']), 'no real loads qualified'
            assert all(p['reason']=='urgent' and p['swaps']<=64 for p in events['passes'])
            assert all(b['output_tokens']-a['output_tokens']>=150 for a,b in zip(events['passes'],events['passes'][1:]))
        masks=hashlib.sha256(b''.join(m.cpu().numpy().tobytes() for m in e.model_prune_mask.values())).hexdigest()
        assert len(set(e.ep.gather_objects(masks)))==1,'rank masks differ'
        # Restore residency for the next comparison, without saving any demand DB.
        for plan in reversed(applied):
            real_apply([(L,new,old,gain) for L,old,new,gain in plan])
        return item
    if args.overhead_only:
        # Full-length warmups matter: 128-token warmups leave later Engram rows cold
        # in the first 512-token baseline. Do not attribute that cache warming to the monitor.
        for mode in ('off','monitor'):
            run('prose',mode,args.max_tokens)
        for mode in ('off','monitor','monitor','off'):
            run('prose',mode,args.max_tokens,True)
        print('URGENT_OVERHEAD_PASS',flush=True)
        assert all(e.ep.gather_objects(True))
        os._exit(0)
    for name in ('html','python','prose'):
        run(name,'off',128)
        run(name,'monitor',128)
        for mode in ('off','monitor','monitor','off'):
            run(name,mode,args.max_tokens,True)
    run('prose','forced',256)
    run('story','forced',128,temp=.7)
    for name in ('python','prose','story'):
        run(name,'natural',256)
    V.ADAPT=base
    e.model.decode_miss_snapshot=real_snapshot
    save()
    assert all(e.ep.gather_objects(True))
    print('URGENT_ADAPT_PASS',flush=True)
    os._exit(0)


if __name__=='__main__':main()
