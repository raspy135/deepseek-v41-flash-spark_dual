"""Fused DSV41_PRUNE_MISS accounting against the torch spelling in Model._record_prune_miss.

    python3 -m unittest engine.test_prune_miss_fused

Synthetic router blocks only; no checkpoint. Integer-seeded accumulators must match exactly,
score mass to float64 rounding (the kernel sums in a different order, see prune_miss_fused).
"""
from __future__ import annotations

import os
import sys
import types
import unittest

import torch

sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")]
from engine import model as M  # noqa: E402
from engine import prune_miss_fused as PMF  # noqa: E402

N_LAYERS, E, K = 3, 384, 6


def _fake(seed_counts: bool):
    m = types.SimpleNamespace(args=types.SimpleNamespace(n_layers=N_LAYERS))
    m._want_counts = None
    m.alloc_prune_miss = lambda n, dev: M.Model.alloc_prune_miss(m, n, dev)
    m.alloc_prune_miss(E, "cuda")
    if seed_counts:
        gen = torch.Generator(device="cuda").manual_seed(5)
        for t in (m._rec_counts, m._want_phase[0], m._want_phase[1]):
            t.copy_(torch.randint(0, 50, t.shape, generator=gen, device="cuda").double())
        m._want_mass.copy_(torch.rand(m._want_mass.shape, generator=gen, device="cuda",
                                      dtype=torch.float64) * 10)
    return m


def _block(t, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.nn.functional.softplus(torch.randn(t, E, generator=gen, device="cuda")).sqrt()
    bias = torch.randn(E, generator=gen, device="cuda") * 0.1
    keep = torch.rand(E, generator=gen, device="cuda") < 0.61
    return scores + bias, scores, keep


def _state(m):
    return {"counts": m._rec_counts, "mass": m._want_mass, "phase0": m._want_phase[0],
            "phase1": m._want_phase[1], "miss_tot": m._miss_tot, "miss_phase": m._miss_phase}


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class PruneMissFusedTest(unittest.TestCase):
    def setUp(self):
        self._flag = M.PRUNE_MISS_FUSED

    def tearDown(self):
        M.PRUNE_MISS_FUSED = self._flag

    def _compare(self, ref, got):
        for key, a in _state(ref).items():
            b = _state(got)[key]
            if key == "mass":
                torch.testing.assert_close(b, a, rtol=1e-12, atol=1e-12, msg=key)
            else:
                self.assertTrue(torch.equal(a, b), key)

    def test_matches_torch(self):
        for seeded in (False, True):
            ref, got = _fake(seeded), _fake(seeded)
            step = 0
            for t in (1, 4, 6, 10, 16):
                for L in range(N_LAYERS):
                    for decode in (False, True):
                        step += 1
                        logits, scores, keep = _block(t, step)
                        M.PRUNE_MISS_FUSED = False
                        M.Model._record_prune_miss(ref, logits, scores, keep, L, K, decode=decode)
                        M.PRUNE_MISS_FUSED = True
                        self.assertTrue(PMF.supported(logits, scores, keep))
                        M.Model._record_prune_miss(got, logits, scores, keep, L, K, decode=decode)
            torch.cuda.synchronize()
            with self.subTest(seeded=seeded):
                self._compare(ref, got)

    def test_large_blocks_fall_back(self):
        logits, scores, keep = _block(PMF.MAX_ROWS + 1, 99)
        self.assertFalse(PMF.supported(logits, scores, keep))
        ref, got = _fake(False), _fake(False)
        M.PRUNE_MISS_FUSED = False
        M.Model._record_prune_miss(ref, logits, scores, keep, 0, K)
        M.PRUNE_MISS_FUSED = True
        M.Model._record_prune_miss(got, logits, scores, keep, 0, K)
        self._compare(ref, got)

    def test_graph_replay_accumulates(self):
        ref, got = _fake(False), _fake(False)
        logits, scores, keep = _block(4, 7)
        M.PRUNE_MISS_FUSED = True
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):   # compile outside capture, then undo its effect
            M.Model._record_prune_miss(got, logits, scores, keep, 1, K, decode=True)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        for t in _state(got).values():
            t.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            M.Model._record_prune_miss(got, logits, scores, keep, 1, K, decode=True)
        for t in _state(got).values():   # capture does not execute; start from zero
            t.zero_()
        for seed in range(5):
            new_logits, new_scores, new_keep = _block(4, 100 + seed)
            logits.copy_(new_logits); scores.copy_(new_scores); keep.copy_(new_keep)
            graph.replay()
            M.PRUNE_MISS_FUSED = False
            M.Model._record_prune_miss(ref, new_logits, new_scores, new_keep, 1, K, decode=True)
            M.PRUNE_MISS_FUSED = True
        torch.cuda.synchronize()
        self._compare(ref, got)
        self.assertEqual(float(got._miss_tot[1, 1]), 5 * 4 * K)


if __name__ == "__main__":
    unittest.main()
