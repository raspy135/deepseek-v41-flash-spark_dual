"""CPU admission checks plus small real-CUDA capture/state/aliasing regression tests."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from engine.prefill_graphs import PrefillFFNGraphs, enabled


class AdmissionTest(unittest.TestCase):
    def test_disabled_demand_recording_stays_disabled(self):
        p = PrefillFFNGraphs.__new__(PrefillFFNGraphs)
        p.model = SimpleNamespace(_want_counts=None)
        self.assertEqual(p._counter_copies(), [])

    def test_flag(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(enabled())
            os.environ['DSV41_PREFILL_GRAPHS'] = '1'
            self.assertTrue(enabled())
            os.environ['DSV41_PREFILL_GRAPHS'] = 'yes'
            with self.assertRaises(ValueError):
                enabled()

    def test_only_stable_prefill_shapes(self):
        p = PrefillFFNGraphs.__new__(PrefillFFNGraphs)
        p.model = SimpleNamespace(args=SimpleNamespace(n_layers=40, n_routed_experts=384), image_mask=None)
        p.rows, p.enabled, p.fallbacks = (128, 2048), True, 0
        h = SimpleNamespace(shape=(2048, 4, 16))
        self.assertTrue(p.eligible(h, 0, True, 384))
        self.assertFalse(p.eligible(h, 0, False, 384))
        self.assertFalse(p.eligible(h, 40, True, 128))
        self.assertFalse(p.eligible(SimpleNamespace(shape=(127, 4, 16)), 0, True, 384))
        p.model.image_mask = object()
        self.assertFalse(p.eligible(h, 0, True, 384))
        p.model.image_mask = None
        p.model._prefix_replay_only = True
        self.assertFalse(p.eligible(h, 0, True, 384))
        p.model._prefix_replay_only = False
        p.enabled = False
        self.assertFalse(p.eligible(h, 0, True, 384))


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class CaptureTest(unittest.TestCase):
    def test_changed_inputs_counters_shared_pool_and_owned_outputs(self):
        class Model:
            dev = torch.device('cuda')
            args = SimpleNamespace(window_size=8, n_routed_experts=16, n_activated_experts=2)
            tap = None
            def __init__(self):
                self.stats = {'moe_s': 0., 'ep_calls': 0}
                self._want_counts = torch.zeros((2, 16), device=self.dev)
                self._rec_counts = self._want_counts  # recorder alias must be restored once
                self.weights = torch.tensor([2., 3.], device=self.dev)
            def alloc_prune_miss(self, *args):
                pass
            def _ffn(self, h, pre, w, L, *args):
                self._want_counts[L].add_(1)
                self.stats['ep_calls'] += 1
                return h * self.weights[L] + pre[..., None], pre + 1
        m = Model()
        p = PrefillFFNGraphs(m, 32)
        store = SimpleNamespace(null_slot=0)
        saved_outputs = []
        for T, L, value in ((32, 0, 1.), (32, 1, 2.), (8, 0, 3.), (32, 0, 4.), (8, 0, 5.)):
            h = torch.full((T, 4, 16), value, device='cuda')
            pre = torch.full((T, 4), .5, device='cuda')
            out, mix = p.run(h, pre, None, L, store, None, 16)
            self.assertTrue(torch.equal(out, h * m.weights[L] + pre[..., None]))
            self.assertTrue(torch.equal(mix, pre + 1))
            saved_outputs.append((out, out.clone()))
            self.assertTrue(all(torch.equal(a, b) for a, b in saved_outputs))
        self.assertEqual(p.captures, 3)
        self.assertEqual(p.replays, 5)
        self.assertTrue(torch.equal(m._want_counts[:, 0], torch.tensor([4., 1.], device='cuda')))
        self.assertEqual(m.stats['ep_calls'], 5)
        # Adaptation must change values at stable addresses and remain visible on replay.
        m.weights[0] = 7.
        out, _ = p.run(h, pre, None, 0, store, None, 16)
        self.assertTrue(torch.equal(out, h * 7. + pre[..., None]))
        borrowed, borrowed_pre = p.run(h, pre, None, 0, store, None, 16, retain=False)
        expected = borrowed.clone() * 3. + borrowed_pre.clone()[..., None]
        # A new graph's warmup must not consume/overwrite an aliased caller input twice.
        out, _ = p.run(borrowed, borrowed_pre, None, 1, store, None, 16)
        self.assertTrue(torch.equal(out, expected))


if __name__ == '__main__':
    unittest.main()
