"""Tiny standalone CUDA gate for runtime decode probes; no model loaded.

Root launches this only after coordinating machine ownership.  It captures a
synthetic pure matrix multiplication and timing-only stateful/collective sites,
then changes graph inputs in place.  Probe replay must use the latest captured
input snapshots without touching original inputs, weights, or state.  Tensor
and cold-flush allocations stay below 4 MiB. Total tracked GPU allocations also
include cuBLAS per-stream workspaces; CUDA context memory is separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def shared_pool_gate(torch, stream):
    """Three retained capture keys/phases sharing a pool, with temporary reuse.

    Small different matrices make a corrupted input/expected pair detectable
    even if both were overwritten together by another graph's snapshots.
    Frozen oracle copies are allocated outside every sharedpool capture.
    """
    from engine.decode_probe import DecodeProbe, ProbeConfig
    config=ProbeConfig(names=('shared.*',),max_cases=8,snapshot_budget_mb=1,
                       warmup=1,repeats=1,calls=2,flush_mb=0)
    probe=DecodeProbe(config,device='cuda',isolation_stream=stream)
    probe.arm();pool=torch.cuda.graph_pool_handle();graphs=[];frozen={}
    result={'passed':False,'snapshot_address_overlaps':[],'preservation':[]}
    try:
        for index,(key,phase,tokens) in enumerate((('draft','draft.greedy',5),
                                                 ('old','verify',4),('new','verify',4))):
            x=torch.randn((tokens,32),device='cuda')
            weight=torch.randn((32,16),device='cuda')*(index+1)
            torch.mm(x,weight);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph();probe.begin_capture(key)
            with torch.cuda.graph(graph,pool=pool,stream=stream):
                output=probe.operation('shared.'+key,lambda x=x,w=weight:torch.mm(x,w),
                    inputs=(x,),replay=lambda private_x,w=weight:torch.mm(private_x,w),
                    weight_refs=(weight,),pure=True,metadata={'phase':phase})
                temporary=torch.empty(65536*(index+1),device='cuda')
                temporary.fill_(index+7)
            del output,temporary
            probe.end_capture();graphs.append((key,phase,graph,x))
        # No two diagnostics snapshots should own overlapping address ranges.
        spans=[]
        for key,cases in probe._records.items():
            for case in cases:
                for field,tree in (('input',case.private_inputs),('expected',case.expected)):
                    for tensor in probe._tensor_leaves(tree):
                        spans.append((key,field,tensor.data_ptr(),probe._storage_bytes(tensor)))
        for i,a in enumerate(spans):
            for b in spans[i+1:]:
                if a[2]<b[2]+b[3] and b[2]<a[2]+a[3]:
                    result['snapshot_address_overlaps'].append({'first':list(a),'second':list(b)})
        for index,(key,phase,graph,x) in enumerate(graphs):
            x.fill_(index+.25);graph.replay();probe.mark_replay(key,phase);torch.cuda.synchronize()
            case=probe._records[key][0]
            frozen[key]=(case.private_inputs[0].clone(),case.expected.clone())
        # Repeated later graph use must not change older key/phase snapshots.
        key,phase,graph,x=graphs[-1]
        for value in (2.25,3.25,4.25):
            x.fill_(value);graph.replay();probe.mark_replay(key,phase);torch.cuda.synchronize()
        for key in ('draft','old'):
            case=probe._records[key][0]
            entry={'key':key,'input_preserved':torch.equal(case.private_inputs[0],frozen[key][0]),
                   'expected_preserved':torch.equal(case.expected,frozen[key][1])}
            result['preservation'].append(entry)
        isolated=probe.profile_idle();result['report']=isolated
        result['passed']=(not result['snapshot_address_overlaps'] and
            all(r['input_preserved'] and r['expected_preserved'] for r in result['preservation']) and
            all(c.get('isolation',{}).get('exact',False) for c in isolated['cases']))
        return result
    finally:
        for _,_,graph,_ in graphs:graph.reset()
        probe.disarm()


def run(args):
    import torch
    from engine.decode_probe import DecodeProbe, ProbeConfig
    torch.cuda.set_device(0)
    torch.manual_seed(891)
    report={'device':torch.cuda.get_device_name(),'torch':torch.__version__,
            'passed':False,'completed':False,'checks':[],
            'source_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__),Path(__file__).resolve().parents[1]/'engine/decode_probe.py')}}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    def save():args.out.write_text(json.dumps(report,indent=2)+'\n')
    def check(name, condition):
        report['checks'].append({'name':name,'passed':bool(condition)});save()
        assert condition,name
    save()
    config=ProbeConfig(names=('pure.mm','pure.mm.second','stateful.acc','collective.fake'),max_cases=8,
        snapshot_budget_mb=1,warmup=1,repeats=3,calls=4,flush_mb=1)
    probe=DecodeProbe(config,device='cuda')
    # Preserve a genuine gap-strided input rather than a dense transpose only.
    storage=torch.randn((4,64),device='cuda');x=storage[:,::2]
    w=torch.randn((32,16),device='cuda');counter=torch.zeros((),device='cuda')
    torch.mm(x,w);torch.cuda.synchronize()
    report['allocated_after_default_stream_warmup_bytes']=torch.cuda.memory_allocated()
    probe.arm();probe.begin_capture('synthetic.strided')
    graph=torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            output=probe.operation('pure.mm',lambda:torch.mm(x,w),inputs=(x,),
                replay=lambda private_x:torch.mm(private_x,w),weight_refs=(w,),pure=True,
                metadata={'synthetic':True})
            output_second=probe.operation('pure.mm.second',lambda:torch.mm(x,w)*.5,inputs=(x,),
                replay=lambda private_x:torch.mm(private_x,w)*.5,weight_refs=(w,),pure=True)
            probe.operation('stateful.acc',lambda:counter.add_(1),inputs=(counter,),
                replay=lambda private_counter:private_counter.add_(1),stateful=True,pure=True)
            probe.operation('collective.fake',lambda:counter.add_(1),inputs=(counter,),
                replay=lambda private_counter:private_counter.add_(1),collective=True,pure=True)
        probe.end_capture(success=True)
    except Exception:
        graph.reset();probe.end_capture(success=False);probe.disarm();raise
    try:
        live=[]
        for mutation in range(3):
            x.fill_(float(mutation+1));graph.replay();probe.mark_replay('synthetic.strided')
            torch.cuda.synchronize()
            check(f'graph_output_mutation_{mutation}',torch.equal(output,torch.mm(x,w)))
            live.append(probe.collect_live())
        before_w=w.clone();before_counter=counter.clone()
        # Original tensor is now different from the last graph execution's input.
        # A safe replay must retain the graph snapshot, not dereference this live x.
        x.fill_(91.);after_live_x=x.clone()
        idle=probe.profile_idle()
        check('idle_preserves_live_input',torch.equal(x,after_live_x))
        check('idle_preserves_live_weight',torch.equal(w,before_w))
        check('idle_skips_stateful_collective',torch.equal(counter,before_counter))
        report['live_reports']=live;report['idle_report']=idle
        check('report_json_serializable',isinstance(json.dumps(idle),str))
        cases={case['name']:case for case in idle['cases']}
        check('pure_replay_matches_actual_snapshot',cases['pure.mm']['isolation']['exact'])
        check('second_pure_case_exact',cases['pure.mm.second']['isolation']['exact'])
        check('stateful_not_replay_eligible',not cases['stateful.acc']['replay_eligible'])
        check('collective_not_replay_eligible',not cases['collective.fake']['replay_eligible'])
        check('last_live_replay_count',all(case['replay_count']==3 for case in cases.values()))
        check('original_gap_stride_retained',x.stride()==(64,2))
        # Another tiny run must reuse the same stream/cachedcuBLAS workspace.
        isolation_stream=probe._isolation_stream
        allocated_before_second=torch.cuda.memory_allocated()
        second_idle=probe.profile_idle()
        allocated_after_second=torch.cuda.memory_allocated()
        check('isolation_stream_reused',probe._isolation_stream is isolation_stream)
        check('repeat_isolation_allocation_growth_under_1MiB',
              allocated_after_second-allocated_before_second<1024*1024)
        report['second_idle_report']=second_idle
        # A fresh graph replay changes snapshots and invalidates oldisolation.
        graph.replay();probe.mark_replay('synthetic.strided');torch.cuda.synchronize()
        stale_report=probe.collect_live()
        check('new_graph_replay_marks_old_isolation_stale',all(
            case['isolation']['stale'] for case in stale_report['cases'] if case['replay_eligible']))
        report['post_replay_report']=stale_report
        workload_bytes=(sum(t.untyped_storage().nbytes() for t in
            (storage,w,counter,output,output_second,before_w,before_counter,after_live_x))+
            idle['coverage'].get('snapshot_reserved_bytes',idle['coverage']['snapshot_bytes'])+
            config.flush_mb*1024*1024+
            sum(case.get('isolation',{}).get('private_input_bytes',0) for case in idle['cases']))
        report['explicit_workload_snapshot_flush_bytes']=workload_bytes
        report['allocation_note']='Tracked peak includes cuBLAS per-stream workspaces; explicit workload budget reported separately.'
        check('explicit_workload_under_4MiB',workload_bytes<4*1024*1024)
        check('standalone_peak_under_128MiB',torch.cuda.max_memory_allocated()<128*1024*1024)
        if args.shared_pool:
            report['shared_pool']=shared_pool_gate(torch,probe._isolation_stream);save()
            check('shared_pool_snapshots_preserved_and_all_phases_exact',report['shared_pool']['passed'])
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        report['completed']=True;report['passed']=True;save()
    finally:
        # Explicit capture nodes still reference case events/snapshot addresses.
        # Destroy that graph before releasing the probe's private references.
        graph.reset()
        probe.disarm()
    print(json.dumps({'passed':report['passed'],'checks':len(report['checks']),
                      'peak_allocated_bytes':report['peak_allocated_bytes'],'out':str(args.out)}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--shared-pool',action='store_true',
                    help='additional three-key/multiphase sharedgraphpool snapshot regression')
    run(ap.parse_args())


if __name__=='__main__':main()
