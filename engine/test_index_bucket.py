"""CPU-only checks for the fast decoder's context-bucket sizing."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.fastdecode import _index_bucket, _index_cache_rows


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


if __name__ == "__main__":
    unittest.main()
