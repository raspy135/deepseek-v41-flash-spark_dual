"""Packed TP slicing and adaptive ownership checks, on CPU."""
import os
import unittest
from unittest.mock import patch

import torch

from engine.dist import EPDistributed
from engine.v41_engine import V41Engine
from engine.test_prefill_swap_routes import make
from tools.fp4_moe import ExpertArena, INTER, DIM, KB1, SG1, KB2, SG2


class TPExpertTests(unittest.TestCase):
    def test_packed_shards_reassemble_exactly(self):
        shapes = ((INTER, KB1), (INTER, SG1), (DIM, KB2), (DIM, SG2),
                  (INTER, KB1), (INTER, SG1))
        full = [torch.randint(0, 256, shape, dtype=torch.uint8) for shape in shapes]
        arenas = [ExpertArena(1, 'cpu', r, 2) for r in (0, 1)]
        for a in arenas:
            a.load_slot(0, *full)
        for name, original, dim in zip(('w1', 's1', 'w2', 's2', 'w3', 's3'), full, (0, 0, 1, 1, 0, 0)):
            combined = torch.cat([getattr(a, name)[0] for a in arenas], dim=dim)
            self.assertTrue(torch.equal(combined, original), name)
        self.assertEqual(arenas[0].bytes_per_slot * 2, sum(t.numel() for t in full))

    @patch.dict(os.environ, {'DSV41_TP_EXPERTS': '1'})
    def test_both_ranks_own_every_expert(self):
        for rank in (0, 1):
            ep = EPDistributed(rank, 2)
            self.assertTrue(all(ep.owns(0, e) for e in range(384)))
            self.assertTrue(ep.owned_mask(torch.arange(384)).all())

    @patch('torch.cuda.is_available', return_value=False)
    def test_adaptation_loads_both_halves_even_across_old_parity(self, _cuda):
        # Each TP rank's fixture has the same residents (the old EP-even subset).
        pair = [make(0), make(0)]
        for rank, e in enumerate(pair):
            e.ep.rank, e.ep.tensor_parallel = rank, True
            V41Engine.apply_swaps(e, [(0, 0, 9, 1.0)])
            self.assertIn((0, 9), e.store.lru)
            self.assertNotIn((0, 0), e.store.lru)
            ids, slots = e.model.prefill_routes[0]
            self.assertTrue(torch.equal(slots[ids.long()], e.fast.lut[0]))
        self.assertTrue(torch.equal(pair[0].fast.lut, pair[1].fast.lut))


if __name__ == '__main__':
    unittest.main()
