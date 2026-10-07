"""Selection objectives, invalid profiles, budget preservation and rank safety; no weights."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from engine.adapt_config import resolve
from engine.expert_profiles import ExpertProfile, load_histograms, mask_digest, maxmin_counts
from engine.v41_engine import build_keep_masks


class ExpertProfilesTest(unittest.TestCase):
    def fixture(self, root, layers=2, experts=12):
        a = np.arange(1, experts + 1, dtype=float)
        d = {'per_layer': {str(l): {'counts_quiet': a.tolist(), 'counts_loud': a[::-1].tolist(),
             'saliency_quiet': a[::-1].tolist(), 'saliency_loud': a.tolist()}
             for l in range(layers)}}
        p = Path(root) / 'coverage.json';p.write_text(json.dumps(d));return p,d

    def test_reader_selects_family_without_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            p,d=self.fixture(root)
            c,digest=load_histograms(p,'counts',('quiet',),2,12)
            s,_=load_histograms(p,'saliency',('quiet',),2,12)
            self.assertEqual(list(c),['quiet']);self.assertEqual(len(digest),64)
            self.assertEqual(c['quiet'][0].argmax(),11);self.assertEqual(s['quiet'][0].argmax(),0)
            del d['per_layer']['1']['saliency_quiet'];p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'missing saliency_quiet'):
                load_histograms(p,'saliency',('quiet',),2,12)
            with self.assertRaisesRegex(ValueError,'unknown expert topic'):
                load_histograms(p,'counts',('absent',),2,12)

    def test_invalid_shapes_values_and_layers_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for bad in ([1,2], [float('nan')]*12, [-1]*12):
                p,d=self.fixture(root);d['per_layer']['0']['saliency_quiet']=bad;p.write_text(json.dumps(d))
                with self.assertRaisesRegex(ValueError,'invalid'):
                    load_histograms(p,'saliency',('quiet',),2,12)
            p,d=self.fixture(root);del d['per_layer']['1'];p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'exactly layers'):
                load_histograms(p,'counts',(),2,12)

    def test_maxmin_protects_worst_topic_at_same_budget(self):
        a=np.array([.4,.1,.1,.1,.1,.1,.1,0,0,0,0,0])
        b=np.array([0,0,0,0,0,0,0,.45,.11,.11,.11,.11])
        summed=np.argsort(a+b)[::-1][:6]
        balanced=np.argsort(maxmin_counts({'a':{0:a},'b':{0:b}},.5,1,12)[0])[::-1][:6]
        self.assertEqual(len(set(balanced)),6)
        self.assertGreater(min(a[balanced].sum(),b[balanced].sum()),min(a[summed].sum(),b[summed].sum()))

    def test_empty_topic_does_not_take_slots_from_active_topic(self):
        a=np.arange(1,13,dtype=float)
        s=maxmin_counts({'empty':{0:np.zeros(12)},'active':{0:a}},.5,1,12)[0]
        self.assertEqual(set(np.argsort(s)[::-1][:6]),set(range(6,12)))

    def test_normalizing_topics_removes_trace_length_advantage(self):
        a=np.arange(1,13,dtype=float);b=a[::-1].copy()
        first=maxmin_counts({'a':{0:a},'b':{0:b}},.5,1,12)[0]
        scaled=maxmin_counts({'a':{0:a*1000},'b':{0:b}},.5,1,12)[0]
        np.testing.assert_allclose(first,scaled)

    def test_full_native_budget_and_all_admitted_ids_preserved(self):
        rng=np.random.default_rng(6)
        per={t:{l:rng.random(384) for l in range(40)} for t in ('a','b','c')}
        scores=maxmin_counts(per,.61)
        masks,keep=build_keep_masks(scores,.61,'uniform','cpu')
        self.assertEqual(sum(len(v) for v in keep.values()),9400)
        for l in range(40):
            self.assertEqual(len(keep[l]),235);self.assertEqual(int(masks[l].sum()),235)
            self.assertEqual(set(keep[l]),set(np.flatnonzero(scores[l]>=1)))

    def test_same_cardinality_different_ids_have_different_digest(self):
        self.assertNotEqual(mask_digest({0:np.array([True,False])}),mask_digest({0:np.array([False,True])}))
        self.assertEqual(mask_digest({0:np.array([True,False])}),mask_digest({0:np.array([1,0],dtype=np.int64)}))

    def test_static_profile_cannot_blend_or_swap_admission_priorities(self):
        p=ExpertProfile({'DSV41_PRUNE_SOURCE':'saliency','DSV41_PRUNE_RANK':'maxmin'})
        frozen=resolve({'DSV41_ADAPT_SENSITIVITY':'off','DSV41_PRUNE_ADAPT':'0'})
        p.validate(.61,'uniform',None,frozen)
        with self.assertRaisesRegex(ValueError,'every swap trigger'):
            p.validate(.61,'uniform',None,resolve({'DSV41_ADAPT_SENSITIVITY':'off'}))
        with self.assertRaisesRegex(ValueError,'every swap trigger'):
            p.validate(.61,'uniform',None,resolve({'DSV41_ADAPT_SENSITIVITY':'high','DSV41_PRUNE_ADAPT':'0'}))
        for selection,counts in [('global',None),('uniform',(235,)*40)]:
            with self.assertRaisesRegex(ValueError,'equal layer budgets'):
                p.validate(.61,selection,counts,frozen)

    def test_empty_options_preserve_legacy_mode_and_default_topics(self):
        p=ExpertProfile({'DSV41_PRUNE_SOURCE':'','DSV41_PRUNE_RANK':'','DSV41_EXPERT_TOPICS':''})
        self.assertFalse(p.static);self.assertEqual(p.topics,('coding','general'))
        p.validate(.61,'uniform',None,resolve({'DSV41_ADAPT_SENSITIVITY':'high'}))
        self.assertEqual(p.boot_fields()['prune_source'],'counts')
        self.assertNotEqual(p.boot_fields(),ExpertProfile({'DSV41_PRUNE_SOURCE':'saliency'}).boot_fields())


if __name__=='__main__':unittest.main()
