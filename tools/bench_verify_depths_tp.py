"""Four short native generations to price newly available intermediate depths.

Called with the same loaded frozen-map model as odd-width qualification. Both
arms use the same confidence-control envelope, pinned to a depth; no scheduler
gain is inferred from a projection microbenchmark.
"""
import hashlib
import json
from pathlib import Path


def run_verify_depths(e, directory):
    policy = e.confidence_depth_policy
    original_pinned = policy.pinned
    try:
        return _run_verify_depths(e, directory)
    finally:
        policy.pinned = original_pinned


def _run_verify_depths(e, directory):
    from server.app import Tok, load_encoding_module, build_chat_prompt
    from engine.expert_profiles import mask_digest
    from bench.bench import WORKLOADS
    assert e.confidence_depth_policy is not None
    assert tuple(e.confidence_depth_policy.depths)==(1,2,3,4,5)
    assert not e.fast.lean.router_bf16
    tok, enc = Tok(e.model_dir), load_encoding_module(e.model_dir)
    paths = Path(directory)
    paths.mkdir(parents=True, exist_ok=True)
    report={'config':e.config(),'runs':[],'exact':True}
    map_hash=mask_digest(e.model_prune_mask)
    references={}
    policy=e.confidence_depth_policy
    for name, depths in (('prose',(3,2)), ('code',(5,4))):
        ids=build_chat_prompt({'messages':[{'role':'user','content':'[req 70101] '+WORKLOADS[name]}]},
                             enc,tok,False,75,e)[1]
        for depth in depths:
            policy.pinned=depth
            for count in (32,192):
                output=[t for burst in e.generate(ids,max_tokens=count,temperature=0,seed=42,
                                                  ignore_eos=True) for t in burst]
                digest=hashlib.sha256(json.dumps(output).encode()).hexdigest()
                assert len(set(e.ep.gather_objects(digest)))==1, 'rank disagreement'
                assert mask_digest(e.model_prune_mask)==map_hash, 'expert map changed'
                if count==32:
                    continue
                exact=references.setdefault(name,output)==output
                report['exact'] &= exact
                stats=dict(e.last_stats)
                row={'name':name,'depth':depth,'hash':digest,'tokens':len(output),
                     'exact':exact,'stats':stats,'text':tok.decode(output),
                     'wall_ms_per_step':1000*stats['decode_s']/max(1,stats['steps'])}
                report['runs'].append(row)
                (paths/f'depths-rank{e.ep.rank}.json').write_text(json.dumps(report,indent=2))
                print('VERIFY_DEPTH_RUN '+json.dumps({k:v for k,v in row.items() if k!='text'}),flush=True)
    policy.pinned=None
    report['completed']=True
    (paths/f'depths-rank{e.ep.rank}.json').write_text(json.dumps(report,indent=2))
    print('VERIFY_DEPTH_COMPLETE '+json.dumps({'exact':report['exact']}),flush=True)
    return report
