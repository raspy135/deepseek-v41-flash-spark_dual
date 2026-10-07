"""Score placement, request previews, and incompatible-history rejection; no model weights."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from engine import model as M, v41_engine as V
from engine.adapt_config import resolve


class PruneScoreTest(unittest.TestCase):
    def config(self, metric="score", **env):
        return resolve({"DSV41_ADAPT_SENSITIVITY": "high", "DSV41_ADAPT_PRIOR": "0",
                        "DSV41_PRUNE_METRIC": metric, **env})

    def test_rare_strong_expert_beats_frequent_weak_expert(self):
        counts, mass = np.array([[.9, .1]]), np.array([[.1, .9]])
        trace = {0: np.ones(2)}
        for metric, winner in (("frequency", 0), ("score", 1)):
            with patch.object(V, "ADAPT", self.config(metric)):
                ranked, w = V.blend_demand(trace, (counts, mass), 0)
                self.assertEqual(int(ranked[0].argmax()), winner)
                self.assertEqual(w, 1.)

    def test_score_confidence_uses_request_votes_and_empty_scores_fall_back(self):
        with patch.object(V, "ADAPT", self.config()):
            ranked, w = V.blend_demand({0: np.array([3., 1.])},
                                       (np.array([[100., 0.]]), np.array([[0., 1.]])), 4.)
            self.assertAlmostEqual(w, .2)
            np.testing.assert_allclose(ranked[0], [.6, .4])
            ranked, w = V.blend_demand({0: np.array([3., 1.])},
                                       (np.ones((1, 2)), np.zeros((1, 2))), 0.)
            np.testing.assert_allclose(ranked[0], [.75, .25])
            self.assertEqual(w, 0.)

    def test_preview_uses_pending_scores_without_mutating_history(self):
        counts = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64)
        mass = counts.clone()
        req = torch.tensor([[99., 1., 0., 0.]], dtype=torch.float64)
        req_mass = torch.tensor([[1., 99., 0., 0.]], dtype=torch.float64)
        model = SimpleNamespace(prune_miss_report=lambda: ({}, counts, mass),
                                _req_counts=req, _req_mass=req_mass)
        eng = SimpleNamespace(_prune_trace={0: np.ones(4)}, model=model,
                              model_prune_mask={0: torch.tensor([True, False, False, False])},
                              ep=SimpleNamespace(tensor_parallel=True, world=2))
        saved = [x.clone() for x in (counts, mass, req, req_mass)]
        with patch.object(V, "ADAPT", self.config()):
            self.assertEqual(V.V41Engine.plan_swaps(eng), [])
            swaps = V.V41Engine.plan_swaps(eng, pending_request=True)
            self.assertEqual([(L, out, into) for L, out, into, _ in swaps], [(0, 0, 1)])
        for old, now in zip(saved, (counts, mass, req, req_mass)):
            torch.testing.assert_close(now, old)

    def test_legacy_request_scores_rejected_but_frequency_counts_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'history.npz')
            counts, mass = np.array([[.9, .1]]), np.array([[1000., 20.]])
            np.savez(path, counts=counts, mass=mass, unit=[1])
            with patch.object(V, "ADAPT", self.config()):
                self.assertIsNone(V.load_prune_db(path))
            with patch.object(V, "ADAPT", self.config("frequency")):
                got_counts, got_mass = V.load_prune_db(path)
                np.testing.assert_array_equal(got_counts, counts)
                self.assertEqual(float(got_mass.sum()), 0.)
            # Loading never edits the incompatible file.
            with np.load(path) as d:
                np.testing.assert_array_equal(d['mass'], mass)

    def test_new_history_round_trips_and_wrong_units_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'history.npz')
            counts, mass = np.array([[.9, .1]]), np.array([[.1, .9]])
            with patch.object(V, "ADAPT", self.config()):
                V.save_prune_db(counts, mass, path)
                c, m = V.load_prune_db(path)
                np.testing.assert_array_equal(c, counts)
                np.testing.assert_array_equal(m, mass)
            with patch.object(V, "ADAPT", resolve({"DSV41_PRUNE_METRIC": "score"})):
                self.assertIsNone(V.load_prune_db(path))

    def test_recorder_measures_unmasked_demand_and_score_weighted_misses(self):
        model = SimpleNamespace(args=SimpleNamespace(n_layers=1), _want_counts=None)
        model.alloc_prune_miss = lambda n, dev: M.Model.alloc_prune_miss(model, n, dev)
        with patch.object(M, 'PRUNE_UNIT_REQUEST', True), patch.object(M, 'PRUNE_MISS_FUSED', False):
            M.Model._record_prune_miss(model, torch.tensor([[5., 4., 3., 2.]]),
                                      torch.tensor([[1., 9., 2., 2.]]),
                                      torch.tensor([True, False, True, True]), 0, 2)
        torch.testing.assert_close(model._req_counts, torch.tensor([[1., 1., 0., 0.]], dtype=torch.float64))
        torch.testing.assert_close(model._req_mass, torch.tensor([[1., 9., 0., 0.]], dtype=torch.float64))
        self.assertEqual(float(model._want_mass.sum()), 0.)
        self.assertEqual(tuple(model._miss_tot[0].tolist()), (1., 2.))
        self.assertEqual(tuple(model._miss_mass_tot[0].tolist()), (9., 10.))


if __name__ == '__main__':
    unittest.main()
