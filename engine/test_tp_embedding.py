"""Embedding storage and exact concatenation, without a GPU/process group."""
import unittest
from unittest.mock import patch

import torch
from engine.tensor_parallel import FeatureParallelEmbedding, shard


class EmbeddingTPTests(unittest.TestCase):
    def test_lookup_and_half_storage(self):
        weight = torch.randn(19, 32, dtype=torch.bfloat16)
        weight[0, 0] = -0.0
        for rank in (0, 1):
            part = shard(weight, 1, rank, 2)
            self.assertEqual(part.untyped_storage().nbytes(), weight.numel())
            embedding = FeatureParallelEmbedding(part, 2)
            for ids in (torch.tensor(0), torch.tensor([0, 18, 0, 3]),
                        torch.tensor([[0, 3], [8, 2]])):
                expected = weight[ids]
                def gather(output, local):
                    torch.testing.assert_close(local, expected.chunk(2, dim=-1)[rank].reshape(-1, 16))
                    output.copy_(torch.cat([x.reshape(-1, 16) for x in expected.chunk(2, dim=-1)]))
                with patch('engine.tensor_parallel.dist.all_gather_into_tensor', gather):
                    actual = embedding[ids]
                self.assertTrue(torch.equal(actual.view(torch.int16), expected.view(torch.int16)))

    def test_reject_invalid_world(self):
        with self.assertRaises(ValueError):
            FeatureParallelEmbedding(torch.empty(4, 8), 1)


if __name__ == '__main__':
    unittest.main()
