import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from engine import v41_engine as V
from engine.adapt_config import resolve
from engine.predictive_prefill import PredictivePrefill, prompt_features, normalize
from engine.test_decode_adapt import fake_engine


class PredictivePrefillTest(unittest.TestCase):
    def bank(self, mode='shadow', **kw):
        env={'DSV41_PREDICTIVE_PREFILL':mode,'DSV41_PREDICTIVE_MIN_SAMPLES':'1',
             'DSV41_PREDICTIVE_CAPACITY':'3', **kw}
        return PredictivePrefill(2,16,env)

    def test_features_distinguish_new_suffix_and_order_without_storing_text(self):
        prompt=list(range(2000))*100
        a,key=prompt_features(prompt,198000)
        b,_=prompt_features(prompt[:-2000]+list(range(5000,7000)),198000)
        c,_=prompt_features(prompt,0)
        self.assertLess(float(a@b),.85)
        self.assertNotEqual(key,prompt_features(prompt,0)[1])
        self.assertTrue(np.isfinite(a).all())
        self.assertAlmostEqual(float(np.linalg.norm(a)),1.,places=5)
        self.assertFalse(np.array_equal(prompt_features(list(range(80)),0)[0],
                                       prompt_features(list(reversed(range(80))),0)[0]))

    def test_long_unrelated_contexts_are_not_false_exact_neighbors(self):
        rng=np.random.default_rng(42)
        a=prompt_features(rng.integers(0,60000,200000).tolist(),198000)[0]
        b=prompt_features(rng.integers(60000,120000,200000).tolist(),198000)[0]
        self.assertLess(float(a@b),.85)

    def test_bank_bounded_duplicates_replace_and_targets_are_normalized(self):
        b=self.bank();rows=np.arange(32,dtype=float).reshape(2,16)+1
        for i in range(6):
            f,k=prompt_features(list(range(i,i+60)),0);b.observe(f,k,rows,rows*3)
        self.assertEqual(len(b.keys),3)
        b.observe(f,k,rows*4,rows)
        self.assertEqual(len(b.keys),3)
        demand,report=b.predict(f)
        self.assertEqual(report['status'],'predicted')
        np.testing.assert_allclose(demand[0],normalize(rows),atol=1e-7)
        np.testing.assert_allclose(demand[1].sum(axis=1),1,atol=1e-7)
        cold=self.bank(DSV41_PREDICTIVE_MIN_SAMPLES='2')
        cold.observe(f,k,rows,rows);self.assertIsNone(cold.predict(f)[0])
        self.assertIsNone(b.predict(-f)[0])

    def test_persistence_rejects_incompatible_data_without_overwriting(self):
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d)/'bank.npz');cfg=Path(d)/'config';cfg.write_text('{}')
            b=self.bank(DSV41_PREDICTIVE_DB=path);b.configure({'x':1},cfg,cfg)
            f,k=prompt_features(list(range(60)),0);a=np.ones((2,16));b.observe(f,k,a,a);b.save()
            old=Path(path).read_bytes()
            restored=self.bank(DSV41_PREDICTIVE_DB=path);restored.configure({'x':1},cfg,cfg)
            np.testing.assert_array_equal(restored.features,b.features)
            bad=self.bank(DSV41_PREDICTIVE_DB=path);bad.configure({'x':2},cfg,cfg)
            self.assertFalse(bad.writable);self.assertEqual(bad.keys,[]);bad.save()
            self.assertEqual(Path(path).read_bytes(),old)
            with np.load(path,allow_pickle=False) as z:
                self.assertEqual(set(z.files),{'version','identity','keys','features','counts','mass'})

    @staticmethod
    def engine():
        db=np.zeros((2,16));db[:,:8]=.05
        req=np.zeros((2,16));req[:,12]=500;req[:,0]=1
        e=fake_engine(db,req,[True]*8+[False]*8)
        e.dynamic_experts=True;e.dynamic_layer_floor=1;e.prune_layer_weights=(1.,1.)
        e.user_prompt=NS(protected={})
        e.plan_swaps=lambda **kw:V.V41Engine.plan_swaps(e,**kw)
        e._images=None;e._miss_at_request_start=(0.,0.)
        e.model.miss_snapshot=lambda:(100.,1000.)
        return e,db,req

    def test_provisional_plan_equals_actual_fold_and_does_not_double_count(self):
        e,db,req=self.engine()
        cfg=resolve({'DSV41_ADAPT_SENSITIVITY':'high','DSV41_PRUNE_METRIC':'score'})
        with patch.object(V,'ADAPT',cfg):
            provisional=e.plan_swaps(max_swaps=512,request_demand=(req,req*3))
            self.assertTrue(provisional)
            np.testing.assert_array_equal(e.model.prune_miss_report()[1],db)
            np.testing.assert_array_equal(e.model._req_counts,req)
            model=NS(_want_counts=torch.tensor(db),_want_mass=torch.tensor(db),
                     _req_counts=torch.tensor(req),_req_mass=torch.tensor(req*3))
            from engine import model as M
            with patch.object(M,'PRUNE_UNIT_REQUEST',True):
                M.Model.flush_request_demand(model,cfg.halflife)
            e.model.prune_miss_report=lambda:({},model._want_counts,model._want_mass)
            actual=e.plan_swaps(max_swaps=512)
            self.assertEqual([s[:4] for s in provisional],[s[:4] for s in actual])
            np.testing.assert_allclose([s[4] for s in provisional],[s[4] for s in actual])
            self.assertEqual(float(model._req_counts.sum()),0)

    def test_shadow_evaluates_before_learning_and_never_loads(self):
        e,db,req=self.engine();e.ep.broadcast_obj=lambda x:x
        e.apply_swaps=lambda _:self.fail('shadow loaded weights')
        b=self.bank();b.save=lambda:None
        tokens=list(range(60));f,k=prompt_features(tokens,0);b.observe(f,k,req,req)
        with patch.object(V,'ADAPT',resolve({'DSV41_ADAPT_SENSITIVITY':'high','DSV41_PRUNE_METRIC':'score'})):
            b.begin(e,tokens,0)
            self.assertGreater(b.last['planned_swaps'],0);self.assertEqual(b.last['swaps'],0)
            b.finish(e)
        self.assertEqual(b.last['promotion_precision'],1.)
        self.assertEqual(b.last['promotion_recall'],1.)
        self.assertEqual(b.last['samples'],1);self.assertEqual(b.last['samples_after'],1)
        np.testing.assert_array_equal(e.model.prune_miss_report()[1],db)
        np.testing.assert_array_equal(e.model._req_counts,req)

    def test_both_ranks_get_same_plan_and_skips_still_broadcast(self):
        packet=[];applied=[[],[]]
        cfg=resolve({'DSV41_ADAPT_SENSITIVITY':'high','DSV41_PRUNE_METRIC':'score'})
        for rank in (0,1):
            e,db,req=self.engine();e.ep.rank=rank
            e.ep.broadcast_obj=(lambda x:packet.append(x) or x) if rank==0 else lambda _:packet[-1]
            e.apply_swaps=lambda p:applied[rank].extend(p) or len(p)
            b=self.bank('apply')
            f,k=prompt_features(list(range(60)),0)
            if rank==0:b.observe(f,k,req,req)
            with patch.object(V,'ADAPT',cfg):b.begin(e,list(range(60)),0)
        self.assertTrue(applied[0]);self.assertEqual(applied[0],applied[1])
        for vision,prefix,status in [(None,60,'skip_short_or_cached'),([object()],0,'skip_vision')]:
            e,_,_=self.engine();e._images=vision;calls=[]
            e.ep.broadcast_obj=lambda x:calls.append(x) or x
            b=self.bank('apply')
            with patch.object(V,'ADAPT',cfg):b.begin(e,list(range(60)),prefix)
            self.assertEqual(len(calls),1);self.assertEqual(b.last['status'],status)
            self.assertFalse(calls[0]['swaps'])

    def test_prediction_error_falls_back_before_any_mutation(self):
        e,_,req=self.engine();e.ep.broadcast_obj=lambda x:x
        e.apply_swaps=lambda _:self.fail('must not apply on predictor error')
        b=self.bank('apply');b.predict=lambda _:(_ for _ in ()).throw(ValueError('bad'))
        with patch.object(V,'ADAPT',resolve({'DSV41_ADAPT_SENSITIVITY':'high'})):
            b.begin(e,list(range(60)),0)
        self.assertEqual(b.last['status'],'prediction_error');self.assertIsNone(b.pending)


if __name__=='__main__':unittest.main()
