"""CPU tests for confidence decisions; the two-node gate checks graph/rank parity."""
import math
import os
import unittest
from unittest.mock import patch

from engine.spec_depth import ConfidenceDepthPolicy, confidence_depths


class ConfidenceDepthTest(unittest.TestCase):
    def test_selects_all_three_depths(self):
        p = ConfidenceDepthPolicy()
        self.assertEqual(p.choose([-1000] * 5), 1)
        self.assertEqual(p.choose([1000] * 3 + [-1000] * 2), 3)
        self.assertEqual(p.choose([1000] * 5), 5)

    def test_cost_not_acceptance_alone(self):
        p = ConfidenceDepthPolicy()
        # All proposals are predicted certain, but the deeper verifier is very slow.
        p.step_s = {1: .1, 3: 1., 5: 2.}
        self.assertEqual(p.choose([1000] * 5), 1)

    def test_expected_survival_and_no_outcome_oracle(self):
        p = ConfidenceDepthPolicy()
        probs = [.9, .5, .8, .3, .7]
        logits = [math.log(x / (1 - x)) for x in probs]
        expected, survival, rates = 1., 1., {}
        for k, prob in enumerate(probs, 1):
            survival *= prob
            expected += survival
            if k in p.depths:
                rates[k] = expected / p.priors[k]
        target = max(rates, key=rates.get)
        self.assertEqual(p.choose(logits), target)
        for accepted in (0, 5):
            p.observe(5, accepted, accepted + 1, None)
            self.assertEqual(p.choose(logits), target)

    def test_invalid_logits_fall_back(self):
        p = ConfidenceDepthPolicy()
        for values in ([0] * 4, [0] * 6, [math.nan] * 5, [math.inf] * 5):
            self.assertEqual(p.choose(values), 3)
        self.assertEqual(p.invalid_confidence, 4)

    def test_capture_and_invalid_timings_are_excluded(self):
        p = ConfidenceDepthPolicy()
        for t in (None, 0, -1, math.nan, math.inf):
            p.observe(3, 2, 3, t)
        self.assertIsNone(p.step_s[3])
        p.observe(3, 2, 3, .212)
        self.assertAlmostEqual(p._times()[1], .176)
        self.assertAlmostEqual(p._times()[5], .248)
        for t in (.106, .106, .106, 5.):
            p.observe(3, 2, 3, t)
        self.assertAlmostEqual(p.step_s[3], .106)

    def test_reset_keeps_costs_but_clears_request_stats(self):
        p = ConfidenceDepthPolicy()
        p.choose([1000] * 5)
        p.observe(5, 5, 6, .124)
        p.reset_request()
        self.assertEqual(p.steps, {1: 0, 3: 0, 5: 0})
        self.assertEqual(p.switches, 0)
        self.assertEqual(p.step_s[5], .124)
        p.pinned = 1
        self.assertEqual(p.choose([1000] * 5), 1)

    def test_opt_in_and_config_conflicts(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(confidence_depths())
            os.environ['DSV41_BLOCK_CONFIDENCE'] = '1'
            self.assertEqual(confidence_depths(), (1, 3, 5))
            self.assertEqual(confidence_depths((3, 5)), (1, 3, 5))
            with self.assertRaises(ValueError):
                confidence_depths((5, 7))
            os.environ['DSV41_BLOCK'] = '3'
            with self.assertRaises(ValueError):
                confidence_depths()
            os.environ['DSV41_BLOCK_CONFIDENCE'] = 'yes'
            with self.assertRaises(ValueError):
                confidence_depths()


if __name__ == '__main__':
    unittest.main()
