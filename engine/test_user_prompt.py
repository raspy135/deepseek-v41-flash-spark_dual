import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

from engine import model as M
from engine import v41_engine as V
from engine.adapt_config import resolve
from engine.user_prompt import UserPrompt, capped_retention_plan, retention_plan, validate_ranges


class UserPromptTest(unittest.TestCase):
    def test_global_cap_fair_layers_and_deferred_absentees_are_not_protected(self):
        keeps={0:np.array([True,True,False,False]),1:np.array([True,True,False,False])}
        mass=np.array([[1.,0.,50.,49.],[.05,0.,.8,.15]])
        p=capped_retention_plan(keeps,mass,{0:np.arange(4),1:np.arange(4)},1)
        self.assertEqual([(L,b) for L,a,b,g in p['swaps']],[(1,2)])
        self.assertEqual(p['deferred'],3)
        for L,ids in p['protected'].items():
            post=keeps[L].copy()
            for l,a,b,g in p['swaps']:
                if l==L:post[a]=False;post[b]=True
            self.assertTrue(post[ids].all())
            self.assertEqual(int(post.sum()),2)
        zero=capped_retention_plan(keeps,mass,{0:np.arange(4),1:np.arange(4)},0)
        self.assertEqual(zero['swaps'],[])
        self.assertEqual(len(capped_retention_plan(keeps,mass,{0:np.arange(4),1:np.arange(4)})['swaps']),4)

    def test_stream_budget_resets_only_for_next_request(self):
        focus=UserPrompt(True,1,4,'cpu',100)
        focus.consume(80);focus.consume(20)
        self.assertEqual(focus.remaining_loads,0)
        with self.assertRaises(RuntimeError):focus.consume(1)
        focus.reset([],1)
        self.assertEqual(focus.remaining_loads,100)
        self.assertIsNone(UserPrompt(True,1,4,'cpu').remaining_loads)
        for bad in (-1,True,1.5):
            with self.assertRaises(ValueError):UserPrompt(True,1,4,'cpu',bad)

    def test_disabled_streaming_has_no_active_budget_or_context_selection(self):
        focus=UserPrompt(False,1,4,'cpu',100,True)
        focus.reset([[9990,10000]],10000)
        self.assertIsNone(focus.row_mask(0,10000))
        self.assertIsNone(focus.mass)
        self.assertEqual(focus.report()['selected_tokens'],0)
        self.assertIsNone(focus.remaining_loads)
        self.assertIsNone(focus.report()['resident_load_cap'])
        focus.reset([],10000)
        self.assertIsNone(focus.remaining_loads)

    def test_cold_stream_limit_uses_common_cache_and_rank_zero_scores(self):
        class Pair:
            packet=None
            calls=0
            rank=0
            def gather_objects(self, value):
                # Expert 2 is cached on both ranks; expert 3 is cached on only one.
                self.calls += 1
                return [[2, 3], [2]]
            def broadcast_obj(self, value):
                self.calls += 1
                if self.rank == 0:self.packet=value
                return self.packet
        pair=Pair();results=[]
        for rank in (0, 1):
            pair.rank=rank
            focus=UserPrompt(True, 1, 5, 'cpu', 1)
            focus.reset([[0, 2]], 2)
            keep=torch.tensor([True, True, False, False, False])
            logits=torch.tensor([[1.,2.,8.,10.,3.],[1.,2.,8.,3.,9.]])
            store=NS(ep=pair,transient_map={(0,2):7},transient_slots=3)
            allowed=focus.streaming_keep(0,logits if rank==0 else logits.flip(1),
                logits,torch.tensor([True,True]),keep,2,store)
            self.assertEqual(allowed.tolist(),[True,True,True,True,False])
            self.assertEqual(focus.loads_used,1)
            self.assertEqual(focus.remaining_loads,0)
            self.assertIsNone(focus.row_mask(0,2))  # Later calls use resident routing.
            results.append(allowed.tolist())
        self.assertEqual(results[0],results[1])
        self.assertEqual(pair.calls,4)

    def test_cached_stream_expert_does_not_spend_cold_load_budget(self):
        focus=UserPrompt(True,1,4,'cpu',1)
        store=NS(ep=NS(rank=0,gather_objects=lambda x:[x,x],broadcast_obj=lambda x:x),
                 transient_map={(0,3):7},transient_slots=2)
        logits=torch.tensor([[1.,2.,0.,9.]])
        allowed=focus.streaming_keep(0,logits,logits,torch.tensor([True]),
            torch.tensor([True,True,False,False]),1,store)
        self.assertEqual(allowed.tolist(),[True,True,False,True])
        self.assertEqual(focus.remaining_loads,1)

    def test_model_and_real_resolver_stop_cold_loads_at_cap_on_both_ranks(self):
        from concurrent.futures import ThreadPoolExecutor
        from engine.test_ep2 import bare_store
        packets=[]
        for rank in (0,1):
            read=[0]
            def broadcast(value):
                if rank==0:packets.append(value);return value
                value=packets[read[0]];read[0]+=1;return value
            store=bare_store(rank,2,slots=12,transient=3)
            store.ep=NS(rank=rank,tensor_parallel=True,owns=lambda L,e:True,
                        gather_objects=lambda value:[value,value],broadcast_obj=broadcast)
            store.lru.update({(0,0):0,(0,1):1})
            store.slot_key.update({0:(0,0),1:(0,1)})
            loads=[];store._load_into_slot=lambda key,slot:loads.append(key)
            focus=UserPrompt(True,1,4,'cpu',1,True);focus.reset([[0,2]],2)
            y=torch.tensor([[1.,0.]],dtype=torch.bfloat16)
            w=NS(gate_w=torch.tensor([[0.,0.],[1.,0.],[2.,0.],[10.,0.]]),
                 gate_bias=torch.zeros(4),sh_w1=None,sh_w2=None,sh_w3=None)
            routes=[]
            def compute(x,slots,weights,arena,limit,**kw):
                routes.append([store.slot_key[int(s)][1] for s in slots.flatten()])
                return torch.zeros_like(x).float()
            model=NS(args=NS(n_activated_experts=2,route_scale=1.,swiglu_limit=10.),
                     stream_layers=(),user_prompt=focus,_moe_start=0,
                     prune_mask={0:torch.tensor([True,True,False,False])},
                     slot_lut=torch.tensor([[0,1,11,11]],dtype=torch.int32),
                     prefill_routes={},_tap=lambda *a:None,moe_fn=compute,
                     stats={'ep_s':0.,'ep_calls':0,'moe_s':0.})
            with ThreadPoolExecutor(1) as pool:
                store.pool=pool
                with patch.object(M,'PRUNE_MISS',False),patch.object(M.R,'expert_ffn',return_value=torch.zeros_like(y)):
                    M.Model.moe(model,y,w,0,True,store,None,4)
                    model._moe_start=1
                    M.Model.moe(model,y,w,0,True,store,None,4)
            self.assertEqual(loads,[(0,3)])
            self.assertEqual(routes,[[3,1],[1,0]])
            self.assertEqual(store.stats['prefill_misses'],1)
            self.assertEqual(store.stats['resolves'],1)
            self.assertEqual(focus.loads_used,1)

    def test_disabled_prompt_mode_uses_only_resident_lut_for_every_prefill_row(self):
        y=torch.tensor([[1.,0.]]*8,dtype=torch.bfloat16)
        focus=UserPrompt(False,1,4,'cpu',100,True)
        focus.reset([[4,8]],8)
        w=NS(gate_w=torch.tensor([[0.,0.],[1.,0.],[2.,0.],[10.,0.]]),
             gate_bias=torch.zeros(4),sh_w1=None,sh_w2=None,sh_w3=None)
        calls=[]
        def compute(x,slots,weights,arena,limit,**kwargs):
            calls.append(slots.tolist());return torch.zeros_like(x).float()
        def resolve(*args):raise AssertionError('resident prefill must not stream weight loads')
        model=NS(args=NS(n_activated_experts=2,route_scale=1.,swiglu_limit=10.),
                 stream_layers=(),user_prompt=focus,_moe_start=0,
                 prune_mask={0:torch.tensor([True,True,False,False])},
                 slot_lut=torch.tensor([[0,1,9,9]],dtype=torch.int32),
                 prefill_routes={},_tap=lambda *a:None,moe_fn=compute,
                 stats={'ep_s':0.,'ep_calls':0,'moe_s':0.})
        store=NS(null_slot=9,resolve=resolve,ep=NS(tensor_parallel=True))
        with patch.object(M,'PRUNE_MISS',False),patch.object(M.R,'expert_ffn',return_value=torch.zeros_like(y)):
            M.Model.moe(model,y,w,0,True,store,None,4)
        self.assertEqual(calls,[[[1,0]]*8])
        self.assertEqual(focus.rows,[0])
        self.assertEqual(focus.stream_batches,[0])

    def test_priority_capacity_ties_and_protection(self):
        swaps, protected, overflow = retention_plan([1, 1, 0, 0], [0, .1, .8, .8], [0, 9, 8, 7])
        self.assertEqual(protected, [2, 3])
        self.assertEqual([(a,b) for a,b,g in swaps], [(0,2), (1,3)])
        self.assertEqual(overflow, 1)
        self.assertEqual(retention_plan([1,1,0], [0,0,0], [0,1,2]), ([], [], 0))

    def test_scope_and_next_request_release(self):
        focus = UserPrompt(True, 2, 4, 'cpu')
        focus.reset([[2,4], [7,8]], 10)
        self.assertIsNone(focus.row_mask(0, 2))
        self.assertEqual(focus.row_mask(1,4).tolist(), [False,True,True,False])
        focus.protected = {0:[1]}
        focus.reset([], 10)
        self.assertEqual(focus.protected, {})
        self.assertIsNone(focus.mass)
        for ranges in ([[0,11]], [[3,2]], [[0,3],[2,4]], [[True,2]], [[1]]):
            with self.assertRaises(ValueError):
                validate_ranges(ranges, 10)

    def test_streams_only_user_rows_and_prefill_before_current_compute(self):
        events = []
        y = torch.tensor([[1.,0.]] * 3, dtype=torch.bfloat16)
        w = NS(gate_w=torch.tensor([[0.,0.],[1.,0.],[2.,0.],[10.,0.]]),
               gate_bias=torch.zeros(4), sh_w1=None, sh_w2=None, sh_w3=None)
        focus = UserPrompt(True, 1, 4, 'cpu')
        focus.reset([[1,2]], 3)
        def load(L, ids, prefill):
            events.append(('load', ids.tolist(), prefill))
            return ids.to(torch.int32)
        def compute(x, slots, weights, arena, limit, **kwargs):
            events.append(('compute', slots.tolist()))
            self.assertNotIn('routing_ids', kwargs)
            return torch.zeros_like(x).float()
        model = NS(args=NS(n_activated_experts=2,route_scale=1.,swiglu_limit=10.),
            stream_layers=(), user_prompt=focus, _moe_start=0,
            prune_mask={0:torch.tensor([True,True,False,False])},
            slot_lut=torch.tensor([[0,1,9,9]],dtype=torch.int32),
            prefill_routes={0:(torch.zeros(4,dtype=torch.int32),torch.tensor([9]))},
            _tap=lambda *a:None, moe_fn=compute, stats={'ep_s':0.,'ep_calls':0,'moe_s':0.})
        store = NS(null_slot=9,resolve=load,ep=NS(tensor_parallel=True))
        with patch.object(M,'PRUNE_MISS',False), patch.object(M.R,'expert_ffn',return_value=torch.zeros_like(y)):
            M.Model.moe(model,y,w,0,True,store,None,4)
            self.assertEqual(events[0],('load',[[1,0],[3,2],[1,0]],True))
            self.assertEqual(events[1][0],'compute')
            self.assertEqual(focus.rows,[1])
            self.assertEqual(torch.nonzero(focus.mass[0]).flatten().tolist(),[2,3])
            events.clear()
            M.Model.moe(model,y,w,0,False,store,None,4)
            self.assertEqual(events[0],('compute',[[1,0],[1,0],[1,0]]))
            self.assertEqual(focus.rows,[1])  # decode contributes nothing

    def test_adaptive_planner_cannot_evict_protected_experts(self):
        model = NS(prune_miss_report=lambda: ({},torch.ones(1,4),torch.tensor([[0.,1.,8.,9.]])))
        eng = NS(_prune_trace={0:np.ones(4)},model=model,
            model_prune_mask={0:torch.tensor([True,True,False,False])},
            ep=NS(tensor_parallel=True,world=2), user_prompt=NS(protected={0:[0]}))
        cfg=resolve({'DSV41_PRUNE_METRIC':'score','DSV41_PRUNE_UNIT':'request','DSV41_ADAPT_PRIOR':'0'})
        with patch.object(V,'ADAPT',cfg):
            swaps=V.V41Engine.plan_swaps(eng,min_gain=0)
        self.assertEqual([(a,b) for L,a,b,g in swaps],[(1,3)])

    def test_empty_request_still_collects_on_both_ranks(self):
        class Pair:
            packet=None
            calls=0
            def broadcast_obj(self,p):
                self.calls+=1
                if self.rank==0: self.packet=p
                return self.packet
        ep=Pair()
        for rank in (0,1):
            ep.rank=rank
            eng=NS(user_prompt_stream=True,user_prompt=UserPrompt(True,1,4,'cpu'),
                   ep=ep,apply_swaps=lambda swaps:len(swaps))
            self.assertEqual(V.V41Engine.promote_user_prompt(eng),0)
        self.assertEqual(ep.calls,2)

    def test_rank_zero_protection_and_swaps_are_authoritative(self):
        class Pair:
            packet=None
            def broadcast_obj(self,p):
                if self.rank==0: self.packet=p
                return self.packet
        ep=Pair()
        results=[]
        for rank in (0,1):
            ep.rank=rank
            focus=UserPrompt(True,1,4,'cpu'); focus.reset([[0,1]],1)
            # Deliberately different peer evidence: it must obey rank 0's plan.
            focus.mass[0]=torch.tensor([0.,0.,.2,.8] if rank==0 else [9.,0.,0.,0.])
            keep=torch.tensor([True,True,False,False])
            changes=[]
            def apply(swaps):
                changes.extend(swaps)
                for L,a,b,g in swaps: keep[a]=False; keep[b]=True
                return len(swaps)
            eng=NS(user_prompt_stream=True,user_prompt=focus,ep=ep,apply_swaps=apply,
                _prune_trace={0:np.ones(4)},model_prune_mask={0:keep},
                model=NS(prune_miss_report=lambda:({},torch.ones(1,4),torch.ones(1,4))))
            V.V41Engine.promote_user_prompt(eng)
            results.append((keep.tolist(),focus.protected,changes))
        self.assertEqual(results[0],results[1])
        self.assertEqual(results[0][0],[False,False,True,True])

    def test_first_logits_refresh_uses_new_residents_without_recording_twice(self):
        events=[]
        focus=UserPrompt(True,1,4,'cpu'); focus.reset([[0,1]],1)
        focus.promoted=1
        model=NS(_prefix_replay_only=False)
        model.begin_prompt=lambda:events.append('reset-prompt-state')
        def forward(ids,s,**kwargs):
            self.assertFalse(focus.streaming)
            self.assertTrue(model._prefix_replay_only)
            events.append(('resident-prefill',ids.tolist(),s))
        model.forward=forward
        def replay(need_logits):
            self.assertTrue(model._prefix_replay_only)
            self.assertFalse(focus.recording)
            focus.record(0,torch.tensor([[2]]),torch.ones(1,1),torch.tensor([True]))
            events.append('new-resident-logits')
            return torch.tensor([[0.,1.]]),'new-hidden',7
        model.decoder_replay=replay
        model.dspark_seed=lambda h,p:events.append((h,p))
        eng=NS(model=model,user_prompt=focus,spec=True,caches=NS(len=2,_chunk_inputs={1:2},pending={0:3}))
        result=V.V41Engine.refresh_user_prompt_logits(eng,torch.tensor([[1.,0.]]),torch.tensor([7,8]),[(0,2)])
        self.assertEqual(int(result.argmax()),1)
        self.assertEqual(events,['reset-prompt-state',('resident-prefill',[7,8],0),'new-resident-logits',('new-hidden',7)])
        self.assertEqual(float(focus.mass.sum()),0.)
        self.assertTrue(focus.recording)
        self.assertTrue(focus.streaming)
        self.assertFalse(model._prefix_replay_only)


if __name__ == '__main__':
    unittest.main()
