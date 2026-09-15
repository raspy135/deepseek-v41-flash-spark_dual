"""Dense sharding layout/storage checks independent of GPU arithmetic."""
from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch

import torch
from engine.tensor_parallel import shard, shard_attention, RowParallelWeight, OutputParallelWeight
from engine.v41_engine import R


def fp8(n, k):
    return R.FP8Weight(torch.randn(n, k).to(torch.float8_e4m3fn),
                       torch.full((n//32, k//32), 127, dtype=torch.uint8))


class DenseTPTests(unittest.TestCase):
    def test_fp8_row_shard_releases_full_backing_storage(self):
        w = fp8(256, 128)
        local = shard(w, 0, 1, 2)
        self.assertEqual(local.w.untyped_storage().nbytes(), 128 * 128)
        self.assertNotEqual(local.w.untyped_storage().data_ptr(), w.w.untyped_storage().data_ptr())
        self.assertTrue(torch.equal(local.w.view(torch.uint8), w.w[128:].view(torch.uint8)))

    @patch.dict(os.environ, {'DSV41_TP_LINEAR_LAYOUT': 'intermediate'})
    def test_attention_shards_head_groups_and_output_columns_together(self):
        a = SimpleNamespace(n_heads=8, o_groups=4)
        source_q, source_o = fp8(512, 128), fp8(128, 256)
        source_grouped = R.FP8GroupedWeight(torch.randn(256, 128).to(torch.float8_e4m3fn),
                                            torch.full((8, 4), 127, dtype=torch.uint8), 4, 64)
        for rank in (0, 1):
            w = SimpleNamespace(wq_b=source_q, attn_sink=torch.arange(8.),
                                wo_a=source_grouped, wo_b=source_o)
            shard_attention(w, a, rank, 2)
            self.assertEqual((w.tp_heads, w.tp_groups), (4, 2))
            self.assertEqual(w.wq_b.shape, (256, 128))
            self.assertEqual(w.wo_a.shape, (2, 64, 128))
            self.assertIsInstance(w.wo_b, RowParallelWeight)
            self.assertEqual(w.wo_b.shape, (128, 128))
            self.assertTrue(torch.equal(w.attn_sink, torch.arange(rank*4., (rank+1)*4.)))
            self.assertEqual(w.wo_a.w.untyped_storage().nbytes(), 128 * 128)

    @patch.dict(os.environ, {'DSV41_TP_LINEAR_LAYOUT': 'output'})
    def test_output_parallel_attention_keeps_complete_down_dot_products(self):
        a = SimpleNamespace(n_heads=8, o_groups=4)
        source = fp8(128, 256)
        local_weights = []
        for rank in (0, 1):
            w = SimpleNamespace(wq_b=fp8(512, 128), attn_sink=torch.arange(8.),
                                wo_a=torch.randn(4, 64, 128), wo_b=source)
            shard_attention(w, a, rank, 2)
            self.assertIsInstance(w.wo_b, OutputParallelWeight)
            self.assertEqual(w.wo_b.shape, (128, 128))
            self.assertEqual(w.wo_b.local.shape, (64, 256))
            local_weights.append(w.wo_b.local.w.view(torch.uint8))
        self.assertTrue(torch.equal(torch.cat(local_weights), source.w.view(torch.uint8)))


if __name__ == '__main__':
    unittest.main()
