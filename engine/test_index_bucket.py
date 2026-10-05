"""CPU-only checks for decode index buckets and wider sparse attention selection."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.fastdecode import FastDecoder, _index_bucket, _index_cache_rows


class IndexBucketTest(unittest.TestCase):
    def test_bucket_boundaries(self):
        cases = [
            (1, 262_144, 4_096),
            (4_096, 262_144, 4_096),
            (4_097, 262_144, 8_192),
            (14_772, 262_144, 16_384),
            (262_144, 262_144, 262_144),
            (90_000, 100_000, 100_000),
            (2_048, 2_048, 2_048),
        ]
        for used, max_seq, expected in cases:
            with self.subTest(used=used, max_seq=max_seq):
                self.assertEqual(_index_bucket(used, max_seq), expected)

    def test_bucket_rejects_out_of_range(self):
        for used in (0, -1, 262_145):
            with self.subTest(used=used), self.assertRaises(ValueError):
                _index_bucket(used, 262_144)

    def test_cache_rows_follow_ratio_and_allocation_cap(self):
        self.assertEqual(_index_cache_rows(16_384, 1, 262_145), 16_384)
        self.assertEqual(_index_cache_rows(16_384, 2, 131_073), 8_192)
        self.assertEqual(_index_cache_rows(100_000, 2, 40_001), 40_001)
        self.assertEqual(_index_cache_rows(9, 2, 100), 5)

    def test_wider_topk_keeps_causality_and_pads_small_allocations(self):
        # Real decode indexer, with only the learned projections/RoPE stubbed out.
        # Cover both a cache smaller than top-k and selection from a larger cache.
        for rows, visible in ((512, 37), (2048, 1536)):
            with self.subTest(rows=rows):
                args = SimpleNamespace(index_topk=1024, index_n_heads=1,
                                       index_head_dim=1, candidate_source_layer=20)
                decoder = SimpleNamespace(
                    a=args, W=SimpleNamespace(indexers={2: SimpleNamespace(wq_b=None, weights_proj=None)}),
                    m=SimpleNamespace(freqs_c=torch.zeros(1)), c=SimpleNamespace(max_seq=rows * 2),
                    _index_cpos=torch.arange(rows), _step_memo=lambda key, fn: fn(),
                    _rope=lambda q, freq: q)
                state = {'ratio': 2, 'ik': torch.arange(rows, dtype=torch.float32)[:, None]}
                with patch('engine.fastdecode.R.qlinear', return_value=torch.ones(1, 1)), \
                     patch('engine.fastdecode.R.mm', return_value=torch.ones(1, 1)):
                    result = FastDecoder._indexer(decoder, torch.ones(1, 1), None, 2,
                                                 torch.zeros(1, dtype=torch.long),
                                                 torch.tensor([visible]), state)
                self.assertEqual(tuple(result.shape), (1, 1024))
                self.assertTrue(torch.equal(result[result >= 0],
                                            torch.arange(max(0, visible - 1024), visible)))
                self.assertEqual(int((result == -1).sum()), 1024 - min(1024, visible))


if __name__ == "__main__":
    unittest.main()
