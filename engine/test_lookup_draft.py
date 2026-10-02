import unittest

from engine.lookup_draft import ExactDraftCache


class TestExactDraftCache(unittest.TestCase):
    def test_known_continuation(self):
        cache = ExactDraftCache([1, 2, 3, 4, 8, 9, 1, 2, 3, 4], min_match=4)
        self.assertEqual(cache.propose(2), [8, 9])
        cache.record_accept(2)
        self.assertEqual(cache.report()["accepted_tokens"], 2)
        self.assertEqual(cache.report()["full_accepts"], 1)

    def test_requires_complete_continuation(self):
        cache = ExactDraftCache([1, 2, 3, 4, 8, 1, 2, 3, 4], min_match=4)
        self.assertIsNone(cache.propose(6))
        self.assertEqual(cache.propose(1), [8])

    def test_extend_indexes_only_settled_continuations(self):
        cache = ExactDraftCache([5, 6, 7, 8, 9], min_match=4)
        cache.extend([5, 6, 7, 8])
        self.assertEqual(cache.propose(1), [9])
        cache.extend([9])
        self.assertEqual(cache.propose(1), [5])

    def test_longest_context_wins(self):
        # Both occurrences end in 2,3,4. The second also shares the preceding
        # 1 with the current suffix, so its continuation 8 wins over 7.
        history = [0, 2, 3, 4, 7, 1, 2, 3, 4, 8, 1, 2, 3, 4]
        cache = ExactDraftCache(history, min_match=3, max_match=8)
        self.assertEqual(cache.propose(1), [8])
        self.assertEqual(cache.report()["mean_match_tokens"], 4)

    def test_occurrence_storage_is_bounded(self):
        cache = ExactDraftCache([1, 1, 1, 1, 1, 1, 1], min_match=2, max_candidates=2)
        self.assertTrue(all(len(v) <= 2 for v in cache.index.values()))

    def test_large_minimum_extends_default_match_limit(self):
        cache = ExactDraftCache(list(range(70)) * 2, min_match=70)
        self.assertEqual(cache.max_match, 70)

    def test_validation(self):
        with self.assertRaises(ValueError):
            ExactDraftCache([], min_match=1)
        with self.assertRaises(ValueError):
            ExactDraftCache([], min_match=2, max_candidates=0)


if __name__ == "__main__":
    unittest.main()
