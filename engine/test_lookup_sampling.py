"""CPU exact-distribution tests for deterministic continuation proposals."""
from collections import defaultdict
from functools import lru_cache
from itertools import product
import unittest

import torch

from engine.lookup_draft import ExactDraftCache, deterministic_draft_probs
from engine.spec_sampling import sample_probs_batch, verify_probabilities


class LookupSamplingTests(unittest.TestCase):
    def test_delta_proposal_rows_and_validation(self):
        drafts = torch.tensor([2, 0, 3], dtype=torch.int32)
        q = deterministic_draft_probs(drafts, 4, dtype=torch.float64)
        self.assertEqual(q.dtype, torch.float64)
        self.assertEqual(q.tolist(), [[0., 0., 1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]])
        self.assertTrue(torch.equal(q.sum(1), torch.ones(3, dtype=q.dtype)))
        for ids, vocab in ((torch.tensor([], dtype=torch.int64), 4),
                           (drafts.reshape(1, 3), 4), (drafts.float(), 4), (drafts, 0)):
            with self.assertRaises(ValueError):
                deterministic_draft_probs(ids, vocab)

    def test_batched_rejection_residual_stops_and_masked_proposals(self):
        drafts = torch.tensor([0, 1, 2])
        q = deterministic_draft_probs(drafts, 4)
        # Grammar/penalty masks are applied to p, never to the deterministic q.
        p = torch.tensor([[.25, .25, .5, 0.], [.2, 0., .8, 0.],
                          [.2, .3, .5, 0.], [0., .25, .75, 0.]])
        result = verify_probabilities(p, q, drafts, torch.tensor([.1, .1, .1, .3]))
        self.assertEqual(result.tolist(), [1, 2, 0, 1, 2])
        # Accepted stop omits the bonus and ignores all later rows.
        result = verify_probabilities(p, q, drafts, torch.tensor([.1, .1, .1, .3]), {0})
        self.assertEqual(result.tolist(), [1, -1, 0, 1, 2])
        # Fully accepted continuation draws its bonus from the full target row.
        p[1] = torch.tensor([0., 1., 0., 0.])
        result = verify_probabilities(p, q, drafts, torch.tensor([.1, .1, .1, .3]))
        self.assertEqual(result.tolist(), [3, 2, 0, 1, 2])

    def test_joint_distribution_all_temperatures_nucleus_and_depths(self):
        # Exact enumeration of every acceptance/rejection branch, followed by
        # further lookup rounds: compare the joint first-three-token distribution.
        # Selection and depth depend only on previously settled history. The
        # target varies by preceding token, with an optional grammar-style mask
        # and a presence/frequency-like adjustment applied before nucleus.
        initial_history = (0, 1, 2, 0, 2, 1, 1, 0, 0, 2, 2, 1, 0, 1)
        base_logits = ((-.7, .5, -.1), (.8, -.3, .2), (.1, -.6, .7))
        for temperature, top_p, altered in product((.1, .6, 1., 2.), (.5, .95, 1.), (False, True)):
            @lru_cache(None)
            def target(history):
                row = torch.tensor(base_logits[history[-1]], dtype=torch.float64)
                if altered:
                    row -= torch.tensor([.1 + .02 * history.count(i) for i in range(3)], dtype=row.dtype)
                    row[(history[-1] + 1) % 3] = -torch.inf
                return tuple(sample_probs_batch(row[None], temperature, top_p)[0].tolist())

            for depth in (1, 3, 5, 'history'):
                @lru_cache(None)
                def step(history):
                    cache = ExactDraftCache(history, min_match=2)
                    # A policy may change width at each step using settled history,
                    # but it cannot inspect the current target logits to select it.
                    selected_depth = (1, 3, 5)[history[-1]] if depth == 'history' else depth
                    proposal = cache.propose(selected_depth)
                    if proposal is None:
                        return {(token,): prob for token, prob in enumerate(target(history)) if prob}
                    q = deterministic_draft_probs(torch.tensor(proposal), 3, dtype=torch.float64)
                    outcomes = defaultdict(float)
                    mass, accepted = 1., ()
                    for i, token in enumerate(proposal):
                        p = target(history + accepted)
                        accept = p[token] / float(q[i, token])
                        residual = [max(0., p[j] - float(q[i, j])) for j in range(3)]
                        total = sum(residual)
                        if total:
                            for bonus, weight in enumerate(residual):
                                outcomes[accepted + (bonus,)] += mass * (1 - accept) * weight / total
                        mass *= accept
                        accepted += (token,)
                    for bonus, weight in enumerate(target(history + accepted)):
                        outcomes[accepted + (bonus,)] += mass * weight
                    return dict(outcomes)

                @lru_cache(None)
                def rollout(history, count):
                    outcomes = defaultdict(float)
                    for emitted, mass in step(history).items():
                        if len(emitted) >= count:
                            outcomes[emitted[:count]] += mass
                        else:
                            for tail, weight in rollout(history + emitted, count - len(emitted)).items():
                                outcomes[emitted + tail] += mass * weight
                    return dict(outcomes)

                with self.subTest(temperature=temperature, top_p=top_p, altered=altered, depth=depth):
                    actual = rollout(initial_history, 3)
                    self.assertAlmostEqual(sum(actual.values()), 1., places=12)
                    for tokens in product(range(3), repeat=3):
                        history, expected = initial_history, 1.
                        for token in tokens:
                            expected *= target(history)[token]
                            history += (token,)
                        self.assertAlmostEqual(actual.get(tokens, 0.), expected, places=12)


if __name__ == '__main__':
    unittest.main()
