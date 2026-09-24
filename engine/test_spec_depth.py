"""DSV41_BLOCK_DYNAMIC policy (engine/spec_depth.py) and the depth-carrying control message.

    python3 -m unittest engine.test_spec_depth

CPU only. Step times are the TP2 measurements (107 ms at depth 3, 124 ms at depth 5).
"""
from __future__ import annotations

import unittest

from engine.spec_depth import DepthPolicy

T3, T5 = 0.107, 0.124


def run(pol, accept_at, steps):
    """Drive the policy like the decode loop: decide, then observe one step at that depth.
    accept_at(depth) -> drafts accepted. Returns the depth sequence."""
    seq = []
    for _ in range(steps):
        d = pol.decide()
        a = accept_at(d)
        pol.observe(d, a, a + 1, T3 if d == 3 else T5)
        seq.append(d)
    return seq


class DepthPolicyTest(unittest.TestCase):
    def test_code_climbs_to_deep(self):
        # code at depth 3 accepts all 3 almost every step; at 5 it takes ~4.65 of 5
        pol = DepthPolicy((3, 5), start=3, interval=60)
        seq = run(pol, lambda d: 3 if d == 3 else 5, 60)
        self.assertEqual(seq[0], 3)
        self.assertEqual(seq[-1], 5)
        self.assertEqual(pol.switches, 1)       # and stays there
        sw = pol.pop_switch()
        self.assertEqual((sw["from"], sw["to"], sw["estimated"]), (3, 5, 5))
        self.assertGreater(sw["rate"][5], sw["rate"][3])
        self.assertIsNone(pol.pop_switch())     # logged once

    def test_prose_drops_to_shallow(self):
        # prose: ~1.1 drafts accepted at either depth -> the deeper step is pure cost
        pol = DepthPolicy((3, 5), start=5, interval=60)
        seq = run(pol, lambda d: 1, 80)
        self.assertEqual(seq[0], 5)
        self.assertEqual(seq[-1], 3)
        self.assertEqual(pol.switches, 1)

    def test_prose_stays_shallow(self):
        pol = DepthPolicy((3, 5), start=3, interval=60)
        seq = run(pol, lambda d: 1, 100)
        self.assertEqual(set(seq), {3})
        self.assertEqual(pol.switches, 0)

    def test_interval_limits_switching(self):
        # alternate between code-like and prose-like every step: no decision before 60 tokens,
        # and at most one per window
        pol = DepthPolicy((3, 5), start=3, interval=60)
        flip = iter(range(10**6))
        seq = run(pol, lambda d: d if next(flip) % 2 else 0, 200)
        changes = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        emitted_per_window = 60
        self.assertLessEqual(changes, sum(d + 1 for d in seq) // emitted_per_window + 1)
        self.assertTrue(all(x == 3 for x in seq[:15]))   # < 60 tokens emitted -> start depth

    def test_counterfactual_is_exact_at_deep(self):
        # at depth 5 accepting exactly 3 gives 4 tokens at either depth; 3 is cheaper -> down
        # ...and having seen that deep steps never get past 3, it must not climb back
        pol = DepthPolicy((3, 5), start=5, interval=60)
        seq = run(pol, lambda d: 3, 100)
        self.assertEqual(seq[-1], 3)
        self.assertEqual(pol.switches, 1)

    def test_measured_times_override_prior(self):
        # if the deep step is measured to cost nothing extra, a saturated shallow window must
        # climb even for a small expected gain (with only the prior 1.16 ratio it would not)
        pol = DepthPolicy((3, 5), start=3, interval=10, extra=0.5)
        pol.step_s = {3: 0.1, 5: 0.1}
        for _ in range(30):
            d = pol.decide()
            pol.observe(d, d, d + 1, 0.1)       # equal step times, all drafts accepted
        self.assertEqual(pol.depth, 5)
        # and if the deep step is 3x the cost, it must not
        pol = DepthPolicy((3, 5), start=3, interval=10, extra=1.5)
        pol.step_s = {3: 0.1, 5: 0.3}
        for _ in range(30):
            d = pol.decide()
            pol.observe(d, d, d + 1, 0.1 if d == 3 else 0.3)
        self.assertEqual(pol.depth, 3)

    def test_outlier_step_time_ignored(self):
        pol = DepthPolicy((3, 5), start=3, interval=10)
        pol.observe(3, 1, 2, 5.0)                # a slow FIRST step (e.g. an untagged capture)
        self.assertAlmostEqual(pol.step_s[3], 5.0)
        pol.observe(3, 1, 2, 0.1)
        self.assertAlmostEqual(pol.step_s[3], 0.1)   # min while under MIN_SAMPLES
        for _ in range(5):
            pol.observe(3, 1, 2, 0.1)
        pol.observe(3, 1, 2, 4.0)
        self.assertAlmostEqual(pol.step_s[3], 0.1)   # median once there are enough

    def test_slow_start_does_not_trap_deep(self):
        # The 2026-09-23 serving failure: prose, the first shallow steps slow (graph captures,
        # 2 s), then 0.105 s; deep steps 0.124 s. The policy may probe depth 5, but must come
        # back to 3 within a couple of windows and stay there.
        pol = DepthPolicy((3, 5), start=3, interval=60)
        seq, i = [], 0
        for _ in range(400):
            d = pol.decide()
            a = [0, 3, 1, 3, 0, 3, 2, 3][i % 8] if d == 3 else [0, 3, 1, 4, 0, 3, 2, 3][i % 8]
            t = (2.0 if i < 3 else 0.105) if d == 3 else 0.124
            pol.observe(d, a, a + 1, t)
            seq.append(d)
            i += 1
        self.assertEqual(seq[-1], 3)
        self.assertLessEqual(sum(1 for d in seq[-200:] if d == 5), 60)

    def test_pinned_and_reset(self):
        pol = DepthPolicy((3, 5), start=3, interval=10)
        pol.pinned = 5
        pol.reset_request()
        self.assertEqual(run(pol, lambda d: 0, 30), [5] * 30)
        pol.pinned = None
        pol.reset_request()
        self.assertEqual(pol.depth, 3)
        self.assertEqual(pol.switches, 0)
        self.assertIsNotNone(pol.step_s[5])      # step times survive the request

    def test_bad_config(self):
        with self.assertRaises(ValueError):
            DepthPolicy((3, 5), start=4)


class ControlMessageTest(unittest.TestCase):
    def test_single_rank_carries_value(self):
        from engine.dist import EPDistributed
        ep = EPDistributed(rank=0, world_size=1)
        self.assertTrue(ep.control(True, 5))
        self.assertEqual(ep.control_value, 5)
        self.assertFalse(ep.control(False))
        self.assertEqual(ep.control_value, 0)


if __name__ == "__main__":
    unittest.main()
