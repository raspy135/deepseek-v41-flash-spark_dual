"""Critical rescue must prioritize rare impact and preserve the transient/broadcast contract."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from engine.critical_stream import CriticalStream, select


class CriticalStreamTest(unittest.TestCase):
    def test_rare_high_contribution_beats_frequent_marginal_miss(self):
        # Expert 1 appears on 20 rows but has tiny output. Expert 2 appears once.
        ids = np.array([[0, 1]] * 20 + [[0, 2]])
        weights = np.array([[.5, .5]] * 21)
        self.assertEqual(select(ids, weights, [True, False, False], [1, .1, 10],
                                [10, 10, 10], threshold=.25, cap=1), [2])

    def test_unknown_expert_is_not_assumed_critical_or_useless(self):
        self.assertEqual(select([[0, 1]], [[.5, .5]], [True, False], [1, 100],
                                [10, 0], threshold=.1, cap=1), [])

    def test_unique_cap_and_stable_tie_break(self):
        ids = [[0, 2], [0, 1], [0, 2]]
        weights = [[.2, .8]] * 3
        self.assertEqual(select(ids, weights, [True, False, False], [1, 1, 1],
                                [10, 10, 10], threshold=.25, cap=1), [1])
        self.assertEqual(select(ids, weights, [True, False, False], [1, 1, 1],
                                [10, 10, 10], threshold=.25, cap=2), [1, 2])
        self.assertEqual(select(ids, weights, [True, False, False], [1, 1, 1],
                                [10, 10, 10], threshold=.25, cap=0), [])

    def profile(self, tmp, **env):
        path = Path(tmp) / 'norms.npz'
        np.savez(path, norms=[[1., 10., 1.]], samples=[[10., 10., 10.]])
        config = {'DSV41_CRITICAL_PREFILL': '1', 'DSV41_CRITICAL_PROFILE': str(path),
                  'DSV41_CRITICAL_BUDGET': '1', **env}
        with patch.dict(os.environ, config, clear=True):
            return CriticalStream(1, 3, 16)

    def test_budget_and_empty_plans_still_broadcast_on_both_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, peer = self.profile(tmp), self.profile(tmp)
            packets = []
            def send(packet):
                packets.append(packet)
                return packet
            rank0 = SimpleNamespace(rank=0, broadcast_obj=send)
            rank1 = SimpleNamespace(rank=1, broadcast_obj=lambda _: packets[-1])
            args = (0, torch.tensor([[3., 2., 1.]]), torch.tensor([[1., 1., 1.]]),
                    torch.tensor([True, False, True]), 2)
            self.assertEqual(root.plan(*args, rank0), [1])
            self.assertEqual(peer.plan(*args, rank1), [1])
            self.assertEqual(root.plan(*args, rank0), [])
            self.assertEqual(peer.plan(*args, rank1), [])
            self.assertEqual(len(packets), 2)
            self.assertEqual(root.report(), peer.report())
            root.reset()
            self.assertEqual(root.remaining, 1)
            self.assertEqual(root.rescued, 0)

    def test_profile_digest_and_transient_cap_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = self.profile(tmp)
            self.assertEqual(len(policy.boot_fields()['critical_profile_sha256']), 64)
            with self.assertRaises(ValueError):
                self.profile(tmp, DSV41_CRITICAL_PER_LAYER='17')
            with self.assertRaises(ValueError):
                self.profile(tmp, DSV41_CRITICAL_SHARE='nan')

    def test_bad_profile_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'bad.npz'
            np.savez(p, norms=[[float('nan')]], samples=[[3.]])
            with patch.dict(os.environ, {'DSV41_CRITICAL_PREFILL': '1',
                                        'DSV41_CRITICAL_PROFILE': str(p)}, clear=True):
                with self.assertRaises(ValueError):
                    CriticalStream(1, 1, 16)


if __name__ == '__main__':
    unittest.main()
