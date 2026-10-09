"""Verify budget preservation and placement, without loading model weights."""
import unittest

import numpy as np

from engine.expert_budget import parse_layer_counts
from engine.v41_engine import build_keep_masks


class ExpertBudgetTest(unittest.TestCase):
    def test_disabled_and_valid_edge_allocation(self):
        self.assertIsNone(parse_layer_counts('', 40, 384, 6, .61, 'uniform'))
        quotas = (330,) * 5 + (204,) * 10 + (203,) * 20 + (330,) * 5
        got = parse_layer_counts(','.join(map(str, quotas)), 40, 384, 6, .61, 'uniform')
        self.assertEqual(sum(got), 9400)
        ranking = {i: np.arange(384, dtype=float) for i in range(40)}
        masks, kept = build_keep_masks(ranking, .61, 'uniform', 'cpu', layer_counts=got)
        self.assertEqual(sum(int(m.sum()) for m in masks.values()), 9400)
        for layer, quota in enumerate(quotas):
            self.assertEqual(len(kept[layer]), quota)
            self.assertTrue(masks[layer][384-quota:].all())
            self.assertFalse(masks[layer][:384-quota].any())

    def test_invalid_budgets_fail_before_serving(self):
        for raw, fraction, selection in (
            ('235,' * 39 + '234', .61, 'uniform'),
            ('235,' * 39 + '235', 1., 'uniform'),
            ('235,' * 39 + '235', .61, 'global'),
            ('235,235', .61, 'uniform'),
            ('235,' * 39 + 'x', .61, 'uniform'),
            ('235,' * 38 + '465,5', .61, 'uniform'),
        ):
            with self.subTest(raw=raw, fraction=fraction, selection=selection):
                with self.assertRaises(ValueError):
                    parse_layer_counts(raw, 40, 384, 6, fraction, selection)


class ExactResidentBudgetTest(unittest.TestCase):
    def test_exact_budget_is_not_rounded_per_layer(self):
        from engine.expert_budget import resident_budget
        from engine.global_residency import initial_selection
        import numpy as np
        n=resident_budget('9500',dynamic=True,layers=40,experts=384,topk=6,fraction=.61)
        self.assertEqual(n,9500)
        keep=initial_selection({L:np.arange(384,dtype=float)+L for L in range(40)},n,6)
        self.assertEqual(sum(map(len,keep.values())),9500)
        self.assertGreaterEqual(min(map(len,keep.values())),6)

    def test_invalid_budget_and_fixed_mode_rejected(self):
        from engine.expert_budget import resident_budget
        args=dict(dynamic=True,layers=40,experts=384,topk=6,fraction=.61)
        self.assertEqual(resident_budget('',**args),'auto')   # unset = auto under dynamic allocation
        for raw in ('239','15361','9500.5'):
            with self.assertRaises(ValueError):resident_budget(raw,**args)
        args['dynamic']=False
        with self.assertRaises(ValueError):resident_budget('9500',**args)
        self.assertIsNone(resident_budget('',**args))         # static profiles: PRUNE_KEEP decides


class AutoResidentBudgetTest(unittest.TestCase):
    def test_auto_follows_the_arena(self):
        from engine.expert_budget import resident_budget, auto_budget, AUTO_MARGIN
        args = dict(dynamic=True, layers=40, experts=384, topk=6, fraction=.61)
        self.assertEqual(resident_budget('auto', **args), 'auto')
        self.assertEqual(resident_budget(' AUTO ', **args), 'auto')
        kw = dict(layers=40, experts=384, topk=6)
        self.assertEqual(auto_budget(13514, **kw), 13514 - AUTO_MARGIN)   # EXL3 at ARENA_GB=90.2
        self.assertEqual(auto_budget(9586, **kw), 9586 - AUTO_MARGIN)     # FP4 at ARENA_GB=90.2
        self.assertEqual(auto_budget(20000, **kw), 15360)                 # everything fits: all of it
        with self.assertRaises(ValueError):
            auto_budget(40 * 6 + AUTO_MARGIN - 1, **kw)

    def test_auto_still_needs_dynamic_allocation(self):
        from engine.expert_budget import resident_budget
        with self.assertRaises(ValueError):
            resident_budget('auto', dynamic=False, layers=40, experts=384, topk=6, fraction=.61)


if __name__ == '__main__':
    unittest.main()
