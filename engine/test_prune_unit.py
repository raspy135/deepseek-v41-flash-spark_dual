"""CPU check: request-unit demand accounting counts every request once.

DSV41_PRUNE_UNIT=request normalizes each request's per-layer routed counts to a distribution and
ages the database once per request, so a 3-token prompt and a 10k-token prompt contribute the same
mass. This test drives the two methods that do it (Model.reset_request_demand /
Model.flush_request_demand) on a stub -- no weights, no GPU -- and asserts the properties the
reformulation is supposed to have:

  * a request's per-layer distribution sums to 1 regardless of how many tokens it moved;
  * a long request and a short request with the same distribution are worth the same;
  * the EWMA ages by exactly 0.5**(1/H) per request.

The slot path is unchanged and must stay so: with the flag off, flush is a no-op and the counts are
still raw routing slots.
"""

import os
import sys
import types
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine import model as M


def _stub(n_layers=2, n_experts=4):
    z = lambda *sh: torch.zeros(*sh, dtype=torch.float64)  # noqa: E731
    s = types.SimpleNamespace()
    s._want_counts = z(n_layers, n_experts)
    s._want_mass = z(n_layers, n_experts)
    s._req_counts = z(n_layers, n_experts)
    return s


class PruneUnitRequestTest(unittest.TestCase):
    def setUp(self):
        self._saved = M.PRUNE_UNIT_REQUEST
        M.PRUNE_UNIT_REQUEST = True

    def tearDown(self):
        M.PRUNE_UNIT_REQUEST = self._saved

    def test_one_request_is_one_unit(self):
        s = _stub()
        s._req_counts[0] = torch.tensor([3.0, 1.0, 0.0, 0.0])
        M.Model.flush_request_demand(s, 0)          # H=0 -> no decay
        self.assertAlmostEqual(float(s._want_counts[0].sum()), 1.0, places=12)
        self.assertAlmostEqual(float(s._want_counts[0, 0]), 0.75, places=12)
        self.assertEqual(float(s._want_counts[1].sum()), 0.0)

    def test_long_and_short_requests_weigh_the_same(self):
        s = _stub()
        # same distribution, one 4-slot request and one 4000-slot request
        s._req_counts[0] = torch.tensor([1.0, 1.0, 1.0, 1.0])
        M.Model.flush_request_demand(s, 0)
        short = s._want_counts[0].clone()
        s._req_counts[0] = torch.tensor([1000.0, 1000.0, 1000.0, 1000.0])
        M.Model.flush_request_demand(s, 0)
        delta = s._want_counts[0] - short
        self.assertAlmostEqual(float(delta.sum()), 1.0, places=12)
        self.assertTrue(torch.allclose(delta, short))

    def test_decay_is_per_request(self):
        s = _stub()
        s._req_counts[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        M.Model.flush_request_demand(s, 0)
        first = float(s._want_counts[0, 0])
        s._req_counts[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        M.Model.flush_request_demand(s, 2.0)        # H=2 -> decay 0.5**(1/2)
        expected = first * (0.5 ** 0.5) + 1.0
        self.assertAlmostEqual(float(s._want_counts[0, 0]), expected, places=12)

    def test_reset_clears_only_the_request_buffer(self):
        s = _stub()
        s._want_counts[0, 0] = 5.0
        s._req_counts[0, 0] = 7.0
        M.Model.reset_request_demand(s)
        self.assertEqual(float(s._req_counts.sum()), 0.0)
        self.assertEqual(float(s._want_counts[0, 0]), 5.0)

    def test_slot_mode_flush_is_a_noop(self):
        M.PRUNE_UNIT_REQUEST = False
        s = _stub()
        s._req_counts[0] = torch.tensor([3.0, 1.0, 0.0, 0.0])
        M.Model.flush_request_demand(s, 50)
        self.assertEqual(float(s._want_counts.sum()), 0.0)
        self.assertEqual(float(s._req_counts.sum()), 4.0)   # untouched


if __name__ == "__main__":
    unittest.main()
