"""Short router A/B/A attached to an already loaded, frozen TP2 qualification.

One component arithmetic change only. Token hashes and actual wall milliseconds
per step are kept beside throughput, since changed acceptance can inflate tok/s.
Three structural answers are graded; these are not a broad quality evaluation.
"""
import hashlib
import json
import os
from pathlib import Path

import torch


def run_router_followup(e, directory):
    """Restore caller state even when qualification fails partway through."""
    lean = e.fast.lean
    original_router, original_weights = lean.router_bf16, dict(lean.router_weights)
    original_confidence = e.confidence_depth_policy
    original_pinned = e.depth_policy.pinned
    try:
        return _run_router_followup(e, directory)
    finally:
        if lean.router_bf16 != original_router:
            torch.cuda.synchronize()
            e.fast.release_graphs()
        lean.router_bf16 = original_router
        lean.router_weights = original_weights
        e.confidence_depth_policy = original_confidence
        e.depth_policy.pinned = original_pinned


def _run_router_followup(e, directory):
    from server.app import Tok, load_encoding_module, build_chat_prompt
    from bench.bench import WORKLOADS
    from engine.expert_profiles import mask_digest
    from engine.v41_engine import ADAPT
    assert e.fast.lean is not None
    assert e.depth_policy is not None
    assert not ADAPT.swap and not ADAPT.decode_tokens
    outdir = Path(directory)
    outdir.mkdir(parents=True, exist_ok=True)
    lean = e.fast.lean
    lean.router_bf16 = True
    lean.prepare_router_weights([w.gate_w for w in e.W.layers])
    saved_confidence = e.confidence_depth_policy
    e.confidence_depth_policy = None
    e.depth_policy.pinned = 3
    tok, enc = Tok(os.environ['MODEL_DIR']), load_encoding_module(os.environ['MODEL_DIR'])
    eos = tok.token_to_id(enc.eos_token)
    prompts = {name:'[req 70101] '+WORKLOADS[name] for name in ('prose', 'code')}
    prompts.update({
        'nesting':'Output only a JSON value with exactly eight nested objects, each having the single key "n". The innermost value is 0. No markdown or explanation.',
        'copy':'Output exactly the following text and nothing else, preserving punctuation and capitalization: A9-zQ_27 / café / 日本語 / {"ok":true}',
        'arithmetic':'Output only a JSON object with two keys: "sum" is the sum of all integers from 1 through 99, and "count" is how many of those integers are even. No markdown.'})
    ids = {name:build_chat_prompt({'messages':[{'role':'user','content':p}]},enc,tok,False,75,e)[1]
           for name,p in prompts.items()}
    expected = 0
    for _ in range(8):
        expected = {'n':expected}
    def grade(name, text):
        if name=='copy':
            return text.strip()=='A9-zQ_27 / café / 日本語 / {"ok":true}'
        try:
            value=json.loads(text)
        except ValueError:
            return False
        return value==(expected if name=='nesting' else {'sum':4950,'count':49})
    maphash = mask_digest(e.model_prune_mask)
    report = {'config':e.config(), 'map':maphash, 'runs':[], 'component':'router'}
    refs = {}
    def save():
        (outdir / f'router-rank{e.ep.rank}.json').write_text(json.dumps(report,indent=2))
    def generate(name, arm, n, measured):
        output=[]
        opts={'ignore_eos':True} if name in ('prose','code') else {'stop_token_ids':{eos}}
        for burst in e.generate(ids[name],max_tokens=n,temperature=0,seed=42,**opts):
            output.extend(burst)
        digest=hashlib.sha256(json.dumps(output).encode()).hexdigest()
        assert len(set(e.ep.gather_objects(digest)))==1, 'rank disagreement'
        assert mask_digest(e.model_prune_mask)==maphash, 'expert map changed'
        text=tok.decode([t for t in output if t!=eos])
        stats=dict(e.last_stats)
        row={'name':name,'arm':arm,'measured':measured,'tokens':len(output),'hash':digest,
             'exact':refs.setdefault((name,n),digest)==digest,'stats':stats,'text':text,
             'wall_ms_per_step':1000*stats['decode_s']/max(1,stats['steps']),
             'grade':None if name in ('prose','code') else grade(name,text)}
        report['runs'].append(row)
        save()
        print('ROUTER_TP_RUN '+json.dumps({k:v for k,v in row.items() if k!='text'}),flush=True)
    try:
        for arm in ('baseline','bf16','baseline_repeat'):
            torch.cuda.synchronize()
            e.fast.release_graphs()
            lean.router_bf16=arm=='bf16'
            assert len(set(e.ep.gather_objects((arm,lean.router_bf16))))==1
            generate('code',arm,32,False)
            for name in ('prose','code'):
                generate(name,arm,192,True)
            if arm!='baseline_repeat':
                for name in ('nesting','copy','arithmetic'):
                    generate(name,arm,128,False)
        report['completed']=True
        save()
    finally:
        e.confidence_depth_policy=saved_confidence
        e.depth_policy.pinned=None
    print('ROUTER_TP_COMPLETE',flush=True)
