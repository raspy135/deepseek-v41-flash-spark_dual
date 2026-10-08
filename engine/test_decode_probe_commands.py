"""CPU mocked TP command lifecycle; never construct weights or a CUDA context."""
from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from engine.decode_probe import ProbeConfig
from engine.v41_engine import V41Engine
from engine.decode_probe_phases import DecodeLoopPhases


class FakeProbe:
    def __init__(self, config, events, *, run_error=False):
        self.config=config;self.events=events;self.active=True;self.run_error=run_error
        self.last_config=None

    def arm(self):
        self.events.append('arm');self.active=True
        return {'status':'armed'}

    def disarm(self):
        self.events.append('disarm');self.active=False

    def status(self):
        self.events.append('status')
        return {'status':'armed' if self.active else 'stopped'}

    def profile_idle(self, config):
        self.events.append('run');self.last_config=config
        if self.run_error:raise RuntimeError('private activation values must not be serialized')
        return {'status':'complete'}


class CommandCPU(unittest.TestCase):
    def engine(self, *, rank=0, probe=None, peer_ok=True):
        events=[]
        eng=object.__new__(V41Engine)
        eng.device='cpu'
        eng._decode_probe_phases=None
        eng._decode_probe_reply={'version':1,'action':'status','ok':True,'ranks':[]}
        eng.fast=SimpleNamespace(use_graphs=True,decode_probe=probe,capture_warmup_stream=object(),
            release_graphs=lambda:events.append('release_graphs'))
        def gather(local):
            events.append('gather')
            peer={'rank':1-rank,'ok':peer_ok}
            return [local,peer] if rank==0 else [peer,local]
        eng.ep=SimpleNamespace(rank=rank,gather_objects=Mock(side_effect=gather))
        return eng,events

    def test_run_local_error_still_reaches_rank_gather_and_redacts_exception(self):
        eng,events=self.engine()
        probe=FakeProbe(ProbeConfig(),events,run_error=True);eng.fast.decode_probe=probe
        reply=eng.decode_probe_command({'action':'run'})
        eng.ep.gather_objects.assert_called_once()
        self.assertEqual(events,['run','gather'])
        self.assertFalse(reply['ok'])
        self.assertEqual(reply['ranks'][0]['error'],'RuntimeError')
        self.assertNotIn('activation',json.dumps(reply))
        self.assertIs(eng.fast.decode_probe,probe)
        self.assertTrue(probe.active)

    def test_action_only_run_inherits_armed_configuration_and_explicit_overrides(self):
        eng,events=self.engine()
        armed=ProbeConfig(names=('custom.dense*',),max_cases=7,snapshot_budget_mb=2,
                          warmup=1,repeats=5,calls=3,flush_mb=0)
        probe=FakeProbe(armed,events);eng.fast.decode_probe=probe
        self.assertTrue(eng.decode_probe_command({'action':'run'})['ok'])
        self.assertEqual(probe.last_config.names,armed.names)
        for field in ('max_cases','snapshot_budget_mb','warmup','repeats','calls','flush_mb'):
            self.assertEqual(getattr(probe.last_config,field),getattr(armed,field))
        self.assertTrue(eng.decode_probe_command({'action':'run','repeats':2})['ok'])
        self.assertEqual(probe.last_config.repeats,2)
        self.assertEqual(probe.last_config.names,armed.names)
        self.assertEqual(eng.ep.gather_objects.call_count,2)

    def test_either_rank_arm_failure_cleans_both_local_controllers_symmetrically(self):
        for rank in (0,1):
            with self.subTest(rank=rank):
                eng,events=self.engine(rank=rank,peer_ok=(rank==1))
                old=FakeProbe(ProbeConfig(),events);eng.fast.decode_probe=old
                created=[]
                def factory(config,device,isolation_stream=None):
                    self.assertIs(isolation_stream,eng.fast.capture_warmup_stream)
                    if rank==1:raise MemoryError('local allocation rejected')
                    new=FakeProbe(config,events);created.append(new);return new
                with patch('engine.decode_probe.DecodeProbe',side_effect=factory):
                    reply=eng.decode_probe_command({'action':'arm'})
                self.assertFalse(reply['ok'])
                eng.ep.gather_objects.assert_called_once()
                self.assertIsNone(eng.fast.decode_probe)
                self.assertFalse(old.active)
                self.assertEqual(events[0:2],['release_graphs','disarm'])
                self.assertEqual(events[-2:],['release_graphs','disarm'])
                if created:self.assertFalse(created[0].active)

    def test_stop_destroys_captured_graphs_before_releasing_owner_references(self):
        eng,events=self.engine()
        probe=FakeProbe(ProbeConfig(),events);eng.fast.decode_probe=probe
        eng._decode_probe_phases=object()
        reply=eng.decode_probe_command({'action':'stop'})
        self.assertTrue(reply['ok'])
        self.assertEqual(events,['release_graphs','disarm','gather'])
        self.assertIsNone(eng.fast.decode_probe)
        self.assertIsNone(eng._decode_probe_phases)

    def test_status_poll_reads_only_cached_cpu_metadata(self):
        eng,events=self.engine()
        probe=FakeProbe(ProbeConfig(),events);eng.fast.decode_probe=probe
        eng.fast.release_graphs=Mock(side_effect=AssertionError('GET cleared graph'))
        probe.profile_idle=Mock(side_effect=AssertionError('GET ran GPU isolation'))
        with patch('torch.cuda.synchronize',side_effect=AssertionError('GET synchronizedGPU')):
            reply=eng.decode_probe_status()
        self.assertEqual(reply['local_status'],{'status':'armed'})
        self.assertEqual(events,['status'])
        eng.ep.gather_objects.assert_not_called()
        eng.fast.release_graphs.assert_not_called()
        probe.profile_idle.assert_not_called()

    def test_unsupported_local_engine_and_invalid_options_gather_error(self):
        for unsupported in (False,True):
            with self.subTest(unsupported=unsupported):
                eng,events=self.engine()
                if unsupported:eng.fast=None
                reply=eng.decode_probe_command({'action':'run','unknown_option':True})
                self.assertFalse(reply['ok'])
                eng.ep.gather_objects.assert_called_once()
                self.assertEqual(events,['gather'])


class LoopPhasesCPU(unittest.TestCase):
    def test_allocation_request_and_pool_growth_counts_are_distinct_and_capture_flagged(self):
        samples=iter(({'allocation.all.allocated':10,'allocated_bytes.all.allocated':100,
                       'num_device_alloc':2,'num_device_free':1},
                      {'allocation.all.allocated':15,'allocated_bytes.all.allocated':500,
                       'num_device_alloc':2,'num_device_free':1},
                      {'allocation.all.allocated':18,'allocated_bytes.all.allocated':900,
                       'num_device_alloc':3,'num_device_free':1}))
        events=[]
        class Event:
            def __init__(self,**kwargs):self.kwargs=kwargs
            def record(self):self.stamp=len(events);events.append('record')
            def elapsed_time(self,other):return other.stamp-self.stamp
        gpu=SimpleNamespace(Event=Event,memory_stats=lambda device:next(samples))
        base=SimpleNamespace(steps=4,acc={'draft':.1,'verify':.2},
                             start=Mock(),mark=Mock(),table=Mock(return_value='base table'))
        captures=[9]
        phases=DecodeLoopPhases(base,'cpu',lambda:captures[0],torch_module=SimpleNamespace(cuda=gpu))
        phases.start();phases.mark('draft');captures[0]+=1;phases.mark('verify')
        report=phases.report();draft,verify=report['phases']
        self.assertEqual(draft['allocation_requests'],5)
        self.assertEqual(draft['requested_bytes'],400)
        self.assertEqual(draft['device_allocations'],0)
        self.assertEqual(draft['capture_intervals'],0)
        self.assertEqual(verify['device_allocations'],1)
        self.assertEqual(verify['capture_intervals'],1)
        self.assertEqual(draft['host_total_ms'],100.)
        self.assertEqual(verify['host_total_ms'],200.)
        self.assertEqual(events,['record']*3)
        phases.steps=7;self.assertEqual(base.steps,7)
        self.assertEqual(phases.table(('note',)),'base table')

    def test_event_history_is_bounded_while_allocation_totals_accumulate(self):
        tick=[0]
        class Event:
            def __init__(self,**kw):pass
            def record(self):tick[0]+=1;self.stamp=tick[0]
            def elapsed_time(self,other):return other.stamp-self.stamp
        gpu=SimpleNamespace(Event=Event,memory_stats=lambda device:{
            'allocation.all.allocated':tick[0],
            'allocated_bytes.all.allocated':tick[0]*32})
        base=SimpleNamespace(steps=1,acc={'loop':.1},start=lambda:None,mark=lambda name:None)
        phases=DecodeLoopPhases(base,'cpu',lambda:0,torch_module=SimpleNamespace(cuda=gpu))
        phases.start()
        for _ in range(70):phases.mark('loop')
        row=phases.report()['phases'][0]
        self.assertEqual(row['intervals'],70)
        self.assertEqual(row['stream_samples'],64)
        self.assertEqual(row['allocation_requests'],71)
        self.assertEqual(row['capture_intervals'],0)


if __name__=='__main__':unittest.main()
