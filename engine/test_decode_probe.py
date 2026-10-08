"""CPU-only runtime probe contracts; synthetic events, no CUDA initialization."""
from __future__ import annotations

import json
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from engine.decode_probe import DecodeProbe, ProbeConfig


class FakeEvents:
    def __init__(self):
        self.calls=[]
        self.tick=0

    def event(self, **kwargs):
        owner=self
        owner.calls.append(('create',kwargs))
        class Event:
            def record(self, *args, **kwargs):
                owner.tick+=1
                self.stamp=owner.tick
                owner.calls.append(('record',self.stamp))
            def synchronize(self):owner.calls.append(('event_sync',))
            def elapsed_time(self, other):return other.stamp-self.stamp
        return Event()

    def synchronize(self):self.calls.append(('sync',))


class ConfigTests(unittest.TestCase):
    def test_unknown_and_invalid_values_rejected_before_work(self):
        invalid=({'surprise':1},{'names':'dense'},{'max_cases':-1},
                 {'snapshot_budget_mb':-1},{'calls':0},{'repeats':0},
                 {'warmup':-1},{'warmup':17},{'repeats':17},
                 {'snapshot_budget_mb':0},{'snapshot_budget_mb':257},
                 {'max_cases':0},{'max_cases':1025},{'calls':65},
                 {'flush_mb':-1},{'flush_mb':257},{'action':'arbitrary'})
        for payload in invalid:
            with self.subTest(payload=payload),self.assertRaises((TypeError,ValueError)):
                ProbeConfig.from_dict(payload)

    def test_serialized_config_round_trip(self):
        payload={'action':'arm','names':['pure.*'],'max_cases':3,
                 'snapshot_budget_mb':1,'warmup':1,'repeats':2,'calls':4,'flush_mb':1}
        config=ProbeConfig.from_dict(payload)
        self.assertEqual(config.names,('pure.*',))
        self.assertEqual(config.max_cases,3)


class ProbeCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch

    def probe(self, **changes):
        fields={'names':('*',),'max_cases':4,'snapshot_budget_mb':1,
                'warmup':1,'repeats':2,'calls':2,'flush_mb':1}
        fields.update(changes)
        events=FakeEvents()
        probe=DecodeProbe(ProbeConfig(**fields),device='cpu',event_factory=events.event,
                          synchronize=events.synchronize,torch_module=self.torch)
        return probe,events

    def capture(self, probe, name, call, key='k', **kwargs):
        probe.begin_capture(key)
        try:
            result=probe.operation(name,call,**kwargs)
        except BaseException:
            probe.end_capture(success=False)
            raise
        probe.end_capture(success=True)
        return result

    def test_disarmed_is_identity_without_events_or_extra_calls(self):
        probe,events=self.probe()
        result=object();calls=[]
        self.assertIs(probe.operation('pure.identity',lambda:calls.append('call') or result),result)
        self.assertEqual(calls,['call'])
        self.assertEqual(events.calls,[])

    def test_events_deferred_and_return_value_unchanged(self):
        probe,events=self.probe();probe.arm()
        x=self.torch.arange(12,dtype=self.torch.float32).reshape(3,4)
        output=x+1
        actual=self.capture(probe,'pure.add',lambda:output,inputs=(x,),
                            replay=lambda private_x:private_x+1,pure=True)
        self.assertIs(actual,output)
        self.assertFalse(any(call[0].endswith('sync') for call in events.calls))
        probe.mark_replay('k')
        report=probe.collect_live()
        self.assertEqual(len(report['cases']),1)
        case=report['cases'][0]
        self.assertTrue(case['replay_eligible'])
        self.assertEqual(case['replay_count'],1)
        self.assertGreaterEqual(case['live_ms'],0.)
        self.assertIsInstance(json.dumps(report),str)

    def test_stateful_collective_never_isolation_replayed(self):
        probe,events=self.probe();probe.arm()
        x=self.torch.ones(2);calls=[]
        probe.begin_capture('k')
        for flag in ('stateful','collective'):
            probe.operation(flag,lambda:x.add_(1),inputs=(x,),pure=True,
                replay=lambda private_x:calls.append('unsafe replay') or private_x.add_(1),
                **{flag:True})
        probe.end_capture(success=True);probe.mark_replay('k')
        report=probe.collect_live()
        self.assertEqual(len(report['cases']),2)
        self.assertTrue(all(not case['replay_eligible'] for case in report['cases']))
        self.assertTrue(all(case['skip_reason'] for case in report['cases']))
        self.assertEqual(calls,[])
        self.assertEqual(x.tolist(),[3.,3.])

    def test_capture_failure_cleanup_and_next_capture_works(self):
        probe,events=self.probe();probe.arm()
        def bad():raise RuntimeError('synthetic operation failure')
        with self.assertRaisesRegex(RuntimeError,'synthetic'):
            self.capture(probe,'bad',bad,key='aborted')
        out=self.torch.ones(2)
        self.assertIs(self.capture(probe,'next',lambda:out,key='next'),out)
        probe.mark_replay('next')
        report=probe.collect_live()
        self.assertEqual([case['name'] for case in report['cases']],['next'])
        probe.disarm(clear=True)
        self.assertEqual(probe.collect_live()['cases'],[])

    def test_snapshot_case_count_bounded_but_all_operation_timings_retained(self):
        probe,events=self.probe(max_cases=1);probe.arm()
        probe.begin_capture('k');calls=[]
        x=self.torch.ones(2)
        for name in ('one','two','three'):
            probe.operation(name,lambda:calls.append('call') or x+1,inputs=(x,),
                            replay=lambda private_x:private_x+1,pure=True)
        probe.end_capture(success=True);probe.mark_replay('k')
        self.assertEqual(calls,['call']*3)
        cases=probe.collect_live()['cases']
        self.assertEqual(len(cases),3)
        self.assertEqual(sum(case['replay_eligible'] for case in cases),1)
        self.assertTrue(all(case['live_ms'] is not None for case in cases))
        self.assertTrue(all(case['skip_reason']=='case_limit' for case in cases[1:]))

    def test_capture_without_live_replay_has_no_usable_measurement(self):
        probe,events=self.probe();probe.arm()
        x=self.torch.ones(2)
        self.capture(probe,'unreplayed',lambda:x+1,inputs=(x,),
                     replay=lambda private_x:private_x+1,pure=True)
        case=probe.collect_live()['cases'][0]
        self.assertIsNone(case['live_ms'])
        self.assertEqual(case['replay_count'],0)

    def test_report_serialization_preserves_explicit_root_only(self):
        probe,_=self.probe();probe.arm()
        x=self.torch.ones(2)
        self.capture(probe,'recorded.root',lambda:x+1,
                     metadata={'parent':None,'kind':'dense.mm','weight':None,
                               'shape':None,'prompt':'must not appear'})
        probe.mark_replay('k')
        metadata=json.loads(json.dumps(probe.collect_live()))['cases'][0]['metadata']
        self.assertEqual(metadata,{'parent':None,'kind':'dense.mm'})
        # Missing parent still identifies a legacy report; do not manufacture a
        # root for operations whose caller did not supply nesting metadata.
        self.capture(probe,'legacy.root',lambda:x+1,key='legacy',metadata={'kind':'dense.mm'})
        probe.mark_replay('legacy')
        legacy=json.loads(json.dumps(probe.collect_live()))['cases'][1]['metadata']
        self.assertNotIn('parent',legacy)

    def test_snapshot_budget_rejects_combined_inputs_and_outputs(self):
        probe,events=self.probe();probe.arm()
        # Input alone fits; input+expected output exceeds1MiB. It must not leave
        # a partial eligible case or silently count only numel of one tensor.
        x=self.torch.ones(400,400)
        probe.begin_capture('k')
        probe.operation('large',lambda:x+1,inputs=(x,),replay=lambda sx:sx+1,pure=True)
        small=self.torch.ones(200,200)
        probe.operation('small',lambda:small+1,inputs=(small,),replay=lambda sx:sx+1,pure=True)
        probe.operation('another_large',lambda:x+1,inputs=(x,),replay=lambda sx:sx+1,pure=True)
        probe.end_capture(success=True)
        probe.mark_replay('k')
        report=probe.collect_live();case=report['cases'][0]
        self.assertFalse(case['replay_eligible'])
        self.assertIn('budget',case['skip_reason'].lower())
        # Even rejected inputs remain referenced by capturecopy nodes: count
        # their bytes until the parent destroys the instrumented CUDAgraph.
        self.assertGreater(case['snapshot_bytes'],0)
        self.assertTrue(report['cases'][1]['replay_eligible'])
        self.assertFalse(report['cases'][2]['replay_eligible'])
        self.assertLessEqual(report['coverage']['snapshot_bytes'],1024*1024)
        probe.clear_captures()
        self.assertEqual(probe.status()['coverage']['snapshot_bytes'],0)
        self.assertTrue(probe.armed)

    def test_gap_stride_metadata_and_overlap_not_assumed_safe(self):
        for name,x in (('gap',self.torch.arange(128.).reshape(4,32)[:,::2]),
                       ('overlap',self.torch.ones(1,16).expand(4,16))):
            probe,events=self.probe();probe.arm()
            self.capture(probe,name,lambda:x+1,inputs=(x,),replay=lambda sx:sx+1,pure=True)
            probe.mark_replay('k');case=probe.collect_live()['cases'][0]
            self.assertEqual(tuple(case['inputs'][0]['stride']),x.stride())
            if name=='overlap':
                self.assertFalse(case['replay_eligible'])
                self.assertTrue(case['skip_reason'])

    def test_default_event_factory_uses_graph_visible_events(self):
        # No CUDA call escapes: Event constructor/records are fakes.
        events=FakeEvents()
        with patch.object(self.torch.cuda,'Event',side_effect=events.event) as factory:
            probe=DecodeProbe(ProbeConfig(names=('*',)),device='cpu',
                              synchronize=events.synchronize,torch_module=self.torch)
            probe.arm();self.capture(probe,'stateful',lambda:None,stateful=True)
        self.assertGreaterEqual(factory.call_count,2)
        for call in factory.call_args_list:
            self.assertTrue(call.kwargs.get('enable_timing'))
            self.assertTrue(call.kwargs.get('external'))

    def test_isolation_stream_lazy_and_provided_stream_survives_capture_cleanup(self):
        events=FakeEvents();provided=object()
        with patch.object(self.torch.cuda,'Stream',side_effect=AssertionError('eager streamallocation')) as stream:
            lazy=DecodeProbe(ProbeConfig(),device='cpu',event_factory=events.event,
                             synchronize=events.synchronize,torch_module=self.torch)
            lazy.arm();lazy.status();lazy.clear_captures()
            self.assertIsNone(lazy._isolation_stream)
            borrowed=DecodeProbe(ProbeConfig(),device='cpu',event_factory=events.event,
                synchronize=events.synchronize,torch_module=self.torch,isolation_stream=provided)
            borrowed.arm();borrowed.clear_captures()
            self.assertIs(borrowed._isolation_stream,provided)
            stream.assert_not_called()

    def test_isolation_freshness_and_narrower_run_drop_old_measurements(self):
        events=FakeEvents()
        config=ProbeConfig(names=('*',),flush_mb=0,calls=1,repeats=1)
        # Merely constructing a cuda device object does not initialize CUDA.
        # Isolation itself is mocked so this still executes entirely on CPU.
        probe=DecodeProbe(config,device='cpu',event_factory=events.event,
                          synchronize=events.synchronize,torch_module=self.torch)
        probe.arm();probe.begin_capture('k');x=self.torch.ones(2)
        for name in ('first','second'):
            probe.operation(name,lambda:x+1,inputs=(x,),replay=lambda sx:sx+1,pure=True)
        probe.end_capture();probe.mark_replay('k')
        probe.device=self.torch.device('cuda')  # metadata only; mockedisolation below
        with patch.object(probe,'_isolate_case',return_value={'exact':True,'warm_ms':1.,'cold_ms':2.}):
            report=probe.profile_idle()
            self.assertTrue(all(not c['isolation']['stale'] for c in report['cases']))
            narrow=ProbeConfig(names=('first',),flush_mb=0,calls=1,repeats=1)
            report=probe.profile_idle(narrow)
        self.assertIn('isolation',report['cases'][0])
        self.assertNotIn('isolation',report['cases'][1])
        probe.mark_replay('k')
        report=probe.collect_live()
        self.assertTrue(report['cases'][0]['isolation']['stale'])

    def test_snapshots_do_not_retain_original_autograd_storage_owners(self):
        import gc
        import weakref
        probe,_=self.probe();probe.arm()
        x=self.torch.ones(4,requires_grad=True);original=weakref.ref(x)
        self.capture(probe,'grad_input',lambda:x+1,inputs=(x,),replay=lambda sx:sx+1,pure=True)
        del x;gc.collect()
        self.assertIsNone(original())
        private=probe._records['k'][0].private_inputs[0]
        self.assertIsNone(private.grad_fn)

    def test_mixed_dtype_snapshot_ranges_disjoint_and_duplicate_inputs_keep_alias(self):
        probe,_=self.probe();probe.arm();probe.begin_capture('slab')
        x=self.torch.arange(8,dtype=self.torch.float32)
        y=self.torch.arange(8,dtype=self.torch.float16)
        probe.operation('mixed',lambda:(x+1,y+1),inputs=(x,y),
                        replay=lambda sx,sy:(sx+1,sy+1),pure=True)
        probe.operation('duplicate',lambda:x+x,inputs=(x,x),
                        replay=lambda sx,sy:sx+sy,pure=True)
        probe.end_capture();cases=probe._records['slab']
        self.assertTrue(all(case.replay_eligible for case in cases))
        self.assertIs(cases[1].private_inputs[0],cases[1].private_inputs[1])
        spans=[]
        for case in cases:
            unique={id(t):t for t in probe._tensor_leaves((case.private_inputs,case.expected))}
            spans.extend((t.data_ptr(),probe._storage_bytes(t)) for t in unique.values())
        spans.sort()
        self.assertTrue(all(start+length<=next_start for (start,length),(next_start,_) in
                            zip(spans,spans[1:])))
        self.assertEqual(cases[0].private_inputs[0].dtype,self.torch.float32)
        self.assertEqual(cases[0].private_inputs[1].dtype,self.torch.float16)

    def test_persistent_slab_allocated_before_capture_and_reused_only_after_clear(self):
        probe,_=self.probe();probe.arm();x=self.torch.ones(2);output=x+1
        probe.begin_capture('first')
        arena=probe._snapshot_arena;arena_ptr=arena.data_ptr()
        with patch.object(self.torch,'empty_strided',side_effect=AssertionError('capture snapshotmalloc')):
            probe.operation('first',lambda:output,inputs=(x,),replay=lambda sx:sx+1,pure=True)
        probe.end_capture();first=probe._records['first'][0].private_inputs[0]
        frozen=first.clone()
        probe.begin_capture('second')
        self.assertEqual(probe._snapshot_arena.data_ptr(),arena_ptr)
        x.fill_(9.);probe.operation('second',lambda:x+1,inputs=(x,),replay=lambda sx:sx+1,pure=True)
        probe.end_capture();second=probe._records['second'][0].private_inputs[0]
        self.assertTrue(self.torch.equal(first,frozen))
        self.assertNotEqual(first.data_ptr(),second.data_ptr())
        status=probe.status()['coverage']
        self.assertEqual(status['snapshot_reserved_bytes'],1024*1024)
        self.assertGreater(status['snapshot_consumed_bytes'],status['snapshot_bytes'])
        probe.clear_captures()
        self.assertIs(probe._snapshot_arena,arena)
        self.assertEqual(probe.status()['coverage']['snapshot_consumed_bytes'],0)
        probe.begin_capture('third')
        probe.operation('third',lambda:x+1,inputs=(x,),replay=lambda sx:sx+1,pure=True)
        probe.end_capture()
        self.assertEqual(probe._records['third'][0].private_inputs[0].data_ptr(),first.data_ptr())
        probe.disarm()
        self.assertIsNone(probe._snapshot_arena)


class HooksCPU(unittest.TestCase):
    """Exercise real patch adapters with stub kernels, not checkpoint imports."""
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch

    probe=ProbeCPU.probe

    def fixture(self):
        import engine
        from engine.decode_probe_hooks import capture_hooks
        torch=self.torch;calls=[]
        weight=torch.eye(4)
        ref=ModuleType('v41_ref');dense=ModuleType('fp8_linear')
        decoder=ModuleType('engine.fastdecode');comm=ModuleType('engine.comm')
        decoder.l2pf=SimpleNamespace(touch=lambda *a,**kw:None)
        def mm(x,w,**kwargs):
            calls.append('mm')
            result=torch.mm(x,getattr(w,'local',w))
            if kwargs.get('out') is not None:
                kwargs['out'].copy_(result)
                return kwargs['out']
            return result
        ref.mm=ref.head_logits=ref.fp8_linear=ref.rmsnorm=mm
        dense.fp8_linear=mm
        comm.all_gather_fast=lambda x:x
        decoder._lin=mm
        class FakeDecoder:
            def _layer_a(self,layer,x):return ref.mm(x,weight)
            def _final(self,x,w=None):return ref.head_logits(x,weight if w is None else w)
            def _draft(self,argmax,x):return ref.head_logits(x,weight)
        def default_method(self,*args,**kwargs):return torch.ones(2)
        for attr in ('_layer_b','_attention','_compressed','_indexer','_hc_mixes',
                     '_hc_pre_rn','_moe_merged','_shared_ffn','_routed_experts'):
            setattr(FakeDecoder,attr,default_method)
        fd=FakeDecoder()
        fd.W=SimpleNamespace(layers=[SimpleNamespace(weight=weight)],mtp=[])
        fd.head=fd.draft_head=fd.markov_head_bf16=weight
        fd.lean=None;fd.m=SimpleNamespace(moe_fn=lambda x:x)
        stack=ExitStack()
        stack.enter_context(patch.dict('sys.modules',{'v41_ref':ref,'fp8_linear':dense,
            'engine.fastdecode':decoder,'engine.comm':comm}))
        stack.enter_context(patch.object(engine,'fastdecode',decoder,create=True))
        stack.enter_context(patch.object(engine,'comm',comm,create=True))
        return stack,capture_hooks,fd,ref,weight,calls

    def test_hooks_restore_module_functions_and_inherited_methods_on_failure(self):
        probe,_=self.probe();probe.arm()
        stack,hooks,fd,ref,weight,calls=self.fixture()
        original=ref.mm;inherited=fd._layer_a.__func__
        with stack:
            with self.assertRaisesRegex(RuntimeError,'synthetic hook failure'):
                with hooks(fd,probe,'failed'):
                    self.assertIsNot(ref.mm,original)
                    fd._layer_a(0,self.torch.ones(2,4))
                    raise RuntimeError('synthetic hook failure')
            self.assertIs(ref.mm,original)
            self.assertNotIn('_layer_a',vars(fd))
            self.assertIs(fd._layer_a.__func__,inherited)
            self.assertTrue(probe.armed)
            self.assertEqual(probe.collect_live()['cases'],[])

    def test_dense_pure_tp_collective_and_out_argument_classification(self):
        probe,_=self.probe();probe.arm()
        stack,hooks,fd,ref,weight,calls=self.fixture()
        x=self.torch.arange(8.).reshape(2,4);out=self.torch.empty_like(x)
        tp=SimpleNamespace(local=weight,tp_logits=lambda:None)
        with stack, hooks(fd,probe,'k'):
            self.assertTrue(self.torch.equal(fd._layer_a(0,x),x))
            ref.mm(x,weight,out=out)
            fd._final(x,tp)
        probe.mark_replay('k');cases=probe.collect_live()['cases']
        dense_cases=[case for case in cases if '/dense.mm/' in case['name']]
        head_cases=[case for case in cases if '/head/' in case['name'] and '/span#' not in case['name']]
        self.assertEqual(calls,['mm','mm','mm'])
        self.assertEqual(len(dense_cases),2)
        self.assertTrue(dense_cases[0]['replay_eligible'])
        self.assertFalse(dense_cases[1]['replay_eligible'])
        self.assertEqual(dense_cases[1]['skip_reason'],'stateful_timing_only')
        self.assertEqual(len(head_cases),1)
        self.assertFalse(head_cases[0]['replay_eligible'])
        self.assertEqual(head_cases[0]['skip_reason'],'collective_timing_only')

    def test_draft_markers_separate_greedy_and_sampled_measurements(self):
        probe,_=self.probe();probe.arm()
        stack,hooks,fd,ref,weight,calls=self.fixture()
        with stack,hooks(fd,probe,'k'):
            fd._draft(True,self.torch.ones(2,4))
            fd._draft(False,self.torch.ones(2,4))
        probe.mark_replay('k',phase='draft.greedy')
        cases=probe.collect_live()['cases']
        greedy=[c for c in cases if c['phase']=='draft.greedy']
        sampled=[c for c in cases if c['phase']=='draft.sampled']
        self.assertTrue(greedy and sampled)
        self.assertTrue(all(c['live_ms'] is not None and c['replay_count']==1 for c in greedy))
        self.assertTrue(all(c['live_ms'] is None and c['replay_count']==0 for c in sampled))

    def test_nested_dispatch_parents_survive_scope_path_resets(self):
        probe,_=self.probe();probe.arm()
        stack,hooks,fd,ref,weight,calls=self.fixture()
        original=ref.mm
        ref.fp8_linear=original
        ref.mm=lambda x,w:ref.fp8_linear(x,w)
        ref.qlinear=lambda x,w:ref.mm(x,w)
        fd._attention=lambda x:ref.qlinear(x,weight)
        # _final resets its name path to verify/head, while still nested under
        # the layer span. The parent must follow execution, not name prefixes.
        fd._layer_a=lambda layer,x:fd._final(fd._attention(x))
        x=self.torch.arange(8.).reshape(2,4)
        with stack,hooks(fd,probe,'nested'):
            actual=fd._layer_a(0,x)
            sibling=ref.mm(x,weight)
        self.assertTrue(self.torch.equal(actual,x))
        self.assertTrue(self.torch.equal(sibling,x))
        self.assertEqual(calls,['mm','mm','mm'])
        probe.mark_replay('nested')
        cases=probe.collect_live()['cases']
        layer,attention,projection,dense,leaf,head_span,head,sibling_dense,sibling_leaf=cases
        self.assertIn('parent',layer['metadata'])
        self.assertIsNone(layer['metadata']['parent'])
        for child,parent in ((attention,layer),(projection,attention),(dense,projection),
                             (leaf,dense),(head_span,layer),(head,head_span),
                             (sibling_leaf,sibling_dense)):
            self.assertEqual(child['metadata']['parent'],parent['name'])
        self.assertEqual(head_span['metadata']['scope'],'verify/head')
        self.assertIn('parent',sibling_dense['metadata'])
        self.assertIsNone(sibling_dense['metadata']['parent'])

    def test_parent_stack_unwinds_after_caught_nested_failure(self):
        stack,hooks,fd,ref,weight,calls=self.fixture()

        class RecordingProbe:
            def __init__(self):self.records=[]
            def begin_capture(self,key):pass
            def operation(self,name,call,**kwargs):
                self.records.append((name,dict(kwargs['metadata'])))
                return call()
            def end_capture(self,success=True):self.success=success

        probe=RecordingProbe()
        def fail(x):raise RuntimeError('nested dispatch failed')
        fd._attention=fail
        fd._layer_a=lambda layer,x:fd._attention(x)
        x=self.torch.ones(2,4)
        with stack,hooks(fd,probe,'failed-child'):
            with self.assertRaisesRegex(RuntimeError,'nested dispatch failed'):
                fd._layer_a(0,x)
            ref.mm(x,weight)
        layer,attention,sibling=probe.records
        self.assertEqual(attention[1]['parent'],layer[0])
        self.assertIsNone(sibling[1]['parent'])
        self.assertEqual(sibling[1]['scope'],'verify')
        self.assertEqual(calls,['mm'])
        self.assertTrue(probe.success)


if __name__=='__main__':unittest.main()
