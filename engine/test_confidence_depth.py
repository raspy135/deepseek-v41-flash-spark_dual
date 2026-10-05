"""CPU tests for confidence decisions; the two-node gate checks graph/rank parity."""
import math
import os
import unittest
from collections import defaultdict
from functools import lru_cache
from itertools import product
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

    def test_sampled_selects_all_three_depths(self):
        p = ConfidenceDepthPolicy()
        self.assertEqual(p.choose_sampled([-1000] * 5), 1)
        self.assertEqual(p.choose_sampled([1000] * 3 + [-1000] * 2), 3)
        self.assertEqual(p.choose_sampled([1000] * 5), 5)
        self.assertEqual(p.report()['selection'], 'prefix')
        p.step_s = {1: .1, 3: 1., 5: 2.}
        self.assertEqual(p.choose_sampled([1000] * 5), 1)

    def test_sampled_never_reads_an_excluded_suffix(self):
        p = ConfidenceDepthPolicy()
        for tail in product((-1000, 1000, math.nan), repeat=3):
            self.assertEqual(p.choose_sampled([1000, -1000, *tail]), 1)
        self.assertEqual(p.choose_sampled([1000] * 3 + [-1000, math.nan]), 3)
        self.assertEqual(p.choose_sampled([1000] * 4 + [math.nan]), 5)
        self.assertEqual(p.invalid_confidence, 0)

    def test_sampled_invalid_values_stop_at_guaranteed_width(self):
        p = ConfidenceDepthPolicy()
        self.assertEqual(p.choose_sampled([1000, math.nan, 1000, 1000, 1000]), 1)
        self.assertEqual(p.choose_sampled([1000, 1000, math.nan, 1000, 1000]), 3)
        self.assertEqual(p.invalid_confidence, 2)
        p.pinned = 3
        self.assertEqual(p.choose_sampled([math.nan] * 5), 3)
        p.pinned = 2
        with self.assertRaises(ValueError):
            p.choose_sampled([0] * 5)

    def test_sampled_joint_distribution_at_all_temperatures(self):
        # Enumerate ALL five-token proposal chains and acceptance/rejection outcomes.
        # The confidence at position i depends on proposal i-1, as in DSpark. Thus
        # choosing a whole-block argmax could bias the proposal distribution; checking
        # just the first token would miss it. Roll out the first THREE emitted tokens.
        def tempered(values, temperature, top_p=1.):
            weights = [x ** (1 / temperature) for x in values]
            total = sum(weights)
            probs = [x / total for x in weights]
            cumulative = 0.
            for i in sorted(range(len(probs)), key=lambda j: probs[j], reverse=True):
                value = probs[i]
                if cumulative >= top_p:
                    probs[i] = 0.
                cumulative += value
            total = sum(probs)
            return [x / total for x in probs]

        for temperature, top_p in product((.1, .6, 1., 2.), (.5, .95, 1.)):
            target = [tempered(p, temperature, top_p) for p in ((.25, .75), (.6, .4))]
            proposal = [tempered(q, temperature) for q in ((.75, .25), (.2, .8))]
            seen_depths = set()

            @lru_cache(None)
            def step(initial):
                outcomes = defaultdict(float)
                for drafts in product((0, 1), repeat=5):
                    prev, mass = initial, 1.
                    confidence = []
                    for token in drafts:
                        probability = .5 if prev == 0 else .01
                        confidence.append(math.log(probability / (1 - probability)))
                        mass *= proposal[prev][token]
                        prev = token
                    policy = ConfidenceDepthPolicy()
                    policy.step_s = {1: .088, 3: .106, 5: .110}
                    depth = policy.choose_sampled(confidence)
                    seen_depths.add(depth)
                    prev, accepted = initial, []
                    for token in drafts[:depth]:
                        p, q = target[prev], proposal[prev]
                        accept = min(1., p[token] / q[token])
                        if accept < 1:
                            residual = [max(0., a - b) for a, b in zip(p, q)]
                            total = sum(residual)
                            for bonus, weight in enumerate(residual):
                                outcomes[tuple(accepted + [bonus])] += mass * (1 - accept) * weight / total
                        mass *= accept
                        accepted.append(token)
                        prev = token
                    for bonus, weight in enumerate(target[prev]):
                        outcomes[tuple(accepted + [bonus])] += mass * weight
                return dict(outcomes)

            @lru_cache(None)
            def rollout(prev, count):
                outcomes = defaultdict(float)
                for emitted, mass in step(prev).items():
                    if len(emitted) >= count:
                        outcomes[emitted[:count]] += mass
                    else:
                        for tail, weight in rollout(emitted[-1], count - len(emitted)).items():
                            outcomes[emitted + tail] += mass * weight
                return dict(outcomes)

            with self.subTest(temperature=temperature, top_p=top_p):
                actual = rollout(0, 3)
                self.assertAlmostEqual(sum(actual.values()), 1., places=12)
                for tokens in product((0, 1), repeat=3):
                    prev, expected = 0, 1.
                    for token in tokens:
                        expected *= target[prev][token]
                        prev = token
                    self.assertAlmostEqual(actual.get(tokens, 0.), expected, places=12)
                self.assertEqual(seen_depths, {1, 3, 5})

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
