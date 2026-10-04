"""Byte parity, eviction, concurrent prefetch, bounds and retained outputs."""
from concurrent.futures import ThreadPoolExecutor
import unittest
import os
from unittest.mock import patch

import numpy as np

from engine.engram_cache import PackedRowCache, cache_budget_mb


class RowCacheTest(unittest.TestCase):
    def test_per_rank_budget(self):
        with patch.dict(os.environ, {'DSV41_ENGRAM_CACHE_MB': '2048'}, clear=True):
            self.assertEqual(cache_budget_mb(0), 2048)
            self.assertEqual(cache_budget_mb(1), 2048)
            os.environ['DSV41_ENGRAM_CACHE_MB_PEER'] = '4096'
            self.assertEqual(cache_budget_mb(0), 2048)
            self.assertEqual(cache_budget_mb(1), 4096)
            os.environ['DSV41_ENGRAM_CACHE_MB_PEER'] = '-1'
            with self.assertRaises(ValueError): cache_budget_mb(0)

    def setUp(self):
        self.rows = np.random.default_rng(42).integers(0, 256, (4096, 264), dtype=np.uint8)

    def test_hits_and_budget(self):
        c = PackedRowCache(272 * 1024 + 13, len(self.rows))
        seen = []
        def read(ids):
            seen.extend(ids.tolist())
            return self.rows[ids]
        ids = np.array([1, 7, 21], np.int64)
        first = c.gather(ids, read)
        np.testing.assert_array_equal(first, self.rows[ids])
        np.testing.assert_array_equal(c.gather(ids[::-1].copy(), read), self.rows[ids[::-1]])
        self.assertEqual(seen, ids.tolist())
        self.assertEqual(c.report()['hits'], 3)
        self.assertEqual(c.report()['filled_rows'], 3)
        self.assertEqual(c.report()['evictions'], 0)
        self.assertEqual(c.report()['allocated_bytes'], 272 * 1024)
        c.clear()
        self.assertEqual(c.report()['filled_rows'], 0)
        c.gather(ids, read)
        self.assertEqual(seen, ids.tolist() * 2)
        np.testing.assert_array_equal(first, self.rows[ids])

    def test_collisions_duplicates_order_and_lifetime(self):
        c = PackedRowCache(272 * 3, len(self.rows))
        for ids in ([1, 4, 7, 1, 2, 5], [7, 4, 1, 5, 2], [], [4095], [0]):
            ids = np.array(ids, np.int64)
            for _ in range(2):
                got = c.gather(ids, lambda ix: self.rows[ix])
                np.testing.assert_array_equal(got, self.rows[ids])
            saved = got.copy()
            c.gather(np.arange(300, dtype=np.int64), lambda ix: self.rows[ix])
            np.testing.assert_array_equal(got, saved)
        self.assertEqual(c.report()['filled_rows'], 3)
        self.assertGreater(c.report()['evictions'], 0)

    def test_parallel_prefetch(self):
        c = PackedRowCache(272 * 17, len(self.rows))
        batches = [np.random.default_rng(i).integers(0, 4096, 300, dtype=np.int64) for i in range(40)]
        with ThreadPoolExecutor(8) as pool:
            outputs = list(pool.map(lambda ids: c.gather(ids, lambda ix: self.rows[ix]), batches))
        for ids, out in zip(batches, outputs):
            np.testing.assert_array_equal(out, self.rows[ids])
        r = c.report()
        self.assertEqual(r['hits'] + r['misses'], 12000)

    def test_bounds_before_reads_or_mutation(self):
        c = PackedRowCache(272, len(self.rows))
        def forbidden(ids):
            self.fail('reader called for invalid input')
        for ids in (np.array([-1], np.int64), np.array([4096], np.int64)):
            with self.assertRaises(IndexError): c.gather(ids, forbidden)
        for ids in (np.arange(4, dtype=np.int32), np.arange(8)[::2], np.zeros((1, 1), np.int64)):
            with self.assertRaises(ValueError): c.gather(ids, forbidden)
        self.assertEqual(c.report()['calls'], 0)


if __name__ == '__main__':
    unittest.main()
