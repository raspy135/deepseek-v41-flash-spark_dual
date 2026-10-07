import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

from engine import v41_engine as V
from engine.global_residency import global_plan, initial_selection, layer_weights, prompt_plan, stream_spans, streaming_moe
from engine.user_prompt import UserPrompt


class GlobalResidencyTest(unittest.TestCase):
    def test_global_boot_allocation_has_no_per_layer_quota(self):
        selected=initial_selection({0:np.array([.4,.3,.2,.1]),1:np.array([.6,.15,.15,.1])},5,1)
        self.assertEqual(selected,{0:[0,1,2],1:[0,1]})
        self.assertEqual(sum(map(len,selected.values())),5)
        with self.assertRaises(ValueError):initial_selection({0:np.ones(4),1:np.ones(4)},1,1)

    def test_stream_sector_batches_preserve_every_routing_contribution(self):
        ids=torch.tensor([[0,2],[2,3],[0,4],[4,5],[0,2]])
        keep=torch.tensor([True,False,False,False,False,False])
        spans=stream_spans(ids.numpy(),keep.numpy(),2)
        self.assertEqual(spans,[(0,2),(2,4),(4,5)])
        self.assertEqual(stream_spans(ids.numpy(),keep.numpy(),5),[(0,5)])
        class Pair:
            packet=None
            calls=0
            def broadcast_obj(self,packet):
                self.calls+=1
                if self.rank==0:self.packet=packet
                return self.packet
        ep=Pair();out=[]
        for rank in (0,1):
            ep.rank=rank;events=[]
            def resolve(L,route,prefill):
                self.assertTrue(prefill)
                self.assertLessEqual(len(set(route[~keep[route]].tolist())),2)
                events.append('load');return route.to(torch.int32)
            def compute(x,slots,weights,arena,limit,**kwargs):
                events.append('compute')
                return (slots*weights).sum(1,keepdim=True).expand_as(x)
            model=NS(prune_mask={0:keep},user_prompt=UserPrompt(True,1,6,'cpu',100,True),
                     args=NS(swiglu_limit=10),moe_fn=compute)
            store=NS(ep=ep,transient_slots=2,null_slot=9,resolve=resolve)
            y=torch.ones(5,2);w=torch.full((5,2),.5)
            actual=streaming_moe(model,y,ids,w,0,store,None)
            torch.testing.assert_close(actual,(ids*w).sum(1,keepdim=True).expand_as(y))
            self.assertEqual(events,['load','compute']*3)
            self.assertEqual(model.user_prompt.stream_batches,[3])
            out.append(actual)
        self.assertEqual(ep.calls,2)
        torch.testing.assert_close(*out)
        # A one-batch call still participates in the plan collective.
        ep.rank=0;streaming_moe(model,y[:1],ids[:1],w[:1],0,store,None)
        self.assertEqual(ep.calls,3)

    def test_quiet_layer_donates_and_total_and_floor_stay_fixed(self):
        keeps = {0:np.array([1,1,0,0],dtype=bool),1:np.array([1,1,0,0],dtype=bool)}
        scores = {0:np.array([.1,.1,.9,.8]),1:np.array([0.,0.,0.,0.])}
        swaps, counts = global_plan(keeps,scores,1,100)
        self.assertEqual(swaps[0][:4],(1,0,0,2))
        self.assertEqual(counts,{0:3,1:1})
        self.assertEqual(sum(counts.values()),4)
        self.assertEqual(len(swaps),2)  # At the donor floor, replace within the busy layer.
        self.assertTrue(all(v>=1 for v in counts.values()))

    def test_protected_sector_and_global_load_limit(self):
        keeps={L:np.array([1,1,0,0],dtype=bool) for L in range(2)}
        scores={0:np.array([.1,.1,.9,.8]),1:np.array([0.,0.,0.,0.])}
        swaps,counts=global_plan(keeps,scores,1,1,protected={1:[0]})
        self.assertEqual(swaps[0][:4],(1,1,0,2))
        self.assertEqual(len(swaps),1)
        self.assertEqual(global_plan(keeps,scores,1,0)[0],[])

    def test_user_mass_units_and_layer_discount(self):
        keeps={L:np.array([1,1,0,0],dtype=bool) for L in range(2)}
        mass=np.array([[0.,0.,100.,0.],[0.,0.,1.,0.]])
        fallback={L:np.ones(4) for L in range(2)}
        p=prompt_plan(keeps,mass,fallback,1,(.95,1.),1)
        self.assertEqual(p['swaps'][0][2:4],(1,2))
        self.assertEqual(p['cross_layer'],1)
        self.assertEqual(p['protected'],{0:[],1:[2]})
        self.assertEqual(p['deferred'],1)
        self.assertEqual(layer_weights('.95,1',2),(.95,1.))
        for bad in ('0,1','nan,1','1','inf,1'):
            with self.assertRaises(ValueError):layer_weights(bad,2)

    @staticmethod
    def engine(experts=4, residents=2):
        null = 9 if experts == 4 else residents*2
        masks={L:torch.tensor([True]*residents+[False]*(experts-residents)) for L in range(2)}
        lru={(L,e):L*residents+e for L in range(2) for e in range(residents)}
        lut=torch.tensor([[L*residents+e if e<residents else null for e in range(experts)]
                          for L in range(2)],dtype=torch.int32)
        loads=[]
        store=NS(lru=lru,slot_key={s:k for k,s in lru.items()},null_slot=null,
                 _load_into_slot=lambda key,slot:loads.append((key,slot)))
        routes={L:(torch.tensor([min(e,residents) for e in range(experts)],dtype=torch.int32),
                   torch.tensor([L*residents+e for e in range(residents)]+[null],dtype=torch.int32)) for L in range(2)}
        focus=UserPrompt(True,2,experts,'cpu',100,True)
        eng=NS(ep=NS(tensor_parallel=True),store=store,user_prompt=focus,
               dynamic_layer_floor=1,model_prune_mask=masks,fast=NS(lut=lut),expert_generation=0,
               model=NS(prefill_routes=routes,prune_miss_report=lambda:({},torch.ones(2,experts),None)))
        return eng,loads

    def test_both_tp_ranks_update_sector_directory_and_graph_addresses(self):
        snapshots=[]
        for rank in (0,1):
            eng,loads=self.engine();eng.ep.rank=rank
            addr=eng.fast.lut.data_ptr(); masks=[m.data_ptr() for m in eng.model_prune_mask.values()]
            with patch('torch.cuda.is_available',return_value=False):
                self.assertEqual(V.V41Engine.apply_swaps(eng,[(1,0,0,2,.8)]),1)
            self.assertEqual(loads,[((0,2),2)])
            self.assertEqual(eng.store.slot_key[2],(0,2))
            self.assertEqual(eng.resident_layer_counts,[3,1])
            self.assertEqual(eng.user_prompt.remaining_loads,100)
            self.assertEqual(eng.user_prompt.resident_loads,1)
            self.assertEqual(addr,eng.fast.lut.data_ptr())
            self.assertEqual(masks,[m.data_ptr() for m in eng.model_prune_mask.values()])
            for L,mask in eng.model_prune_mask.items():
                ids,slots=eng.model.prefill_routes[L]
                self.assertEqual(len(slots),int(mask.sum())+1)
                for e in range(4):
                    self.assertEqual(int(slots[ids[e]]),int(eng.fast.lut[L,e]))
            snapshots.append((dict(eng.store.lru),eng.fast.lut.tolist(),eng.resident_layer_counts))
        self.assertEqual(snapshots[0],snapshots[1])

    def test_invalid_whole_plan_touches_no_sector(self):
        for swaps in ([(1,0,0,2,.8),(1,1,0,3,.7)],
                      [(1,0,0,2,.8),(1,0,0,3,.7)],
                      [(1,0,0,2,.8),(0,1,0,1,.7)]):
            eng,loads=self.engine();before=dict(eng.store.lru)
            with self.assertRaises(RuntimeError):V.V41Engine.apply_swaps(eng,swaps)
            self.assertEqual(eng.store.lru,before)
            self.assertEqual(loads,[])
        eng,loads=self.engine();eng.user_prompt.protected={1:[0]}
        with self.assertRaises(RuntimeError):V.V41Engine.apply_swaps(eng,[(1,0,0,2,.8)])
        self.assertEqual(loads,[])

    def test_historical_prefix_survives_cross_layer_sector_transfer_on_both_ranks(self):
        from engine.test_prefix_cache import _fixture
        for rank in (0, 1):
            e = _fixture()
            g, loads = self.engine()
            for name in ('ep', 'store', 'fast', 'model_prune_mask', 'dynamic_layer_floor',
                         'expert_generation'):
                setattr(e, name, getattr(g, name))
            e.ep.rank = rank
            e.user_prompt = UserPrompt(False, 2, 4, 'cpu', 100, True)
            e.model.prefill_routes = g.model.prefill_routes
            e.model.prune_miss_report = g.model.prune_miss_report
            with patch.dict('os.environ', {'DSV41_PREFIX_CACHE': '1'}):
                e._save_prefix([10, 11, 12, 13, 14, 15], 6)
                saved = e._prefix_cache
                with patch('torch.cuda.is_available', return_value=False):
                    self.assertEqual(e.apply_global_swaps([(1, 0, 0, 2, .8)]), 1)
                self.assertEqual(e.resident_layer_counts, [3, 1])
                self.assertIs(e._prefix_cache, saved)
                self.assertEqual(e._restore_prefix([10, 11, 12, 13, 14, 15, 16]), 6)
                self.assertEqual(e.caches.len, 6)
                self.assertTrue(torch.equal(e.model._rep['h'][0], saved['rep']['h']))
                self.assertIsNone(e.user_prompt.remaining_loads)
                self.assertEqual(e.user_prompt.resident_loads, 1)
                # Restoring encoder history must not restore old expert-sector maps.
                self.assertEqual(loads, [((0, 2), 2)])
                self.assertEqual(e.store.slot_key[2], (0, 2))
                self.assertEqual(int(e.fast.lut[0, 2]), 2)

    def test_adaptation_can_load_over_100_after_stream_budget_exhaustion(self):
        eng, loads = self.engine(experts=256, residents=128)
        eng.user_prompt.consume(100)
        swaps=[(1,e,0,128+e,1.) for e in range(101)]
        with patch('torch.cuda.is_available',return_value=False):
            self.assertEqual(V.V41Engine.apply_swaps(eng,swaps),101)
        self.assertEqual(len(loads),101)
        self.assertEqual(eng.resident_layer_counts,[229,27])
        self.assertEqual(eng.user_prompt.resident_loads,101)
        self.assertEqual(eng.user_prompt.loads_used,100)

    def test_load_failure_is_fatal(self):
        eng,_=self.engine()
        def fail(key,slot):raise OSError('disk unavailable')
        eng.store._load_into_slot=fail
        with self.assertRaisesRegex(RuntimeError,'TP pair must stop'):
            V.V41Engine.apply_swaps(eng,[(1,0,0,2,.8)])


if __name__ == '__main__':
    unittest.main()
