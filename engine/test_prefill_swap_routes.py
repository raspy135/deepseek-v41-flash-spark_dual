"""Adaptive swaps must preserve exactly one owner in prefill as well as decode."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from engine.v41_engine import V41Engine


def make(rank):
    owned = list(range(rank, 8, 2))
    lru = {(0, e): i for i, e in enumerate(owned)}
    null = 20
    ids = torch.full((16,), len(owned), dtype=torch.int32)
    lut = torch.full((1, 16), null, dtype=torch.int32)
    for e in owned:
        ids[e] = lru[0, e]
        lut[0, e] = lru[0, e]
    slots = torch.tensor(list(range(len(owned))) + [null], dtype=torch.int32)
    mask = torch.zeros(16, dtype=torch.bool)
    mask[:8] = True
    store = NS(lru=lru, slot_key={}, null_slot=null, _load_into_slot=lambda key, slot: None)
    model = NS(prefill_routes={0: (ids, slots)},
               prune_miss_report=lambda: ({}, torch.zeros(1), None))
    return NS(store=store, ep=NS(world=2, rank=rank), model_prune_mask={0: mask},
              fast=NS(lut=lut), model=model, expert_generation=0)


class TestPrefillSwapRoutes(unittest.TestCase):
    @patch('torch.cuda.is_available', return_value=False)
    def test_repeated_swaps_match_lut_and_single_owner(self, _cuda):
        pair = [make(r) for r in (0, 1)]
        addresses = [(e.fast.lut.data_ptr(), *(t.data_ptr() for t in
                     e.model.prefill_routes[0])) for e in pair]
        slot_maps = [e.model.prefill_routes[0][1].clone() for e in pair]
        # Promote, swap a promoted expert again, then restore the initial set.
        for replacements in (((0, 8), (1, 9)), ((8, 10), (9, 11)),
                             ((10, 0), (11, 1)), ((2, 12), (3, 13))):
            plan = [(0, old, new, 1.0) for old, new in replacements]
            for rank, eng in enumerate(pair):
                V41Engine.apply_swaps(eng, plan)
                ids, slots = eng.model.prefill_routes[0]
                self.assertTrue(torch.equal(slots[ids.long()], eng.fast.lut[0]))
                self.assertTrue(torch.equal(slots, slot_maps[rank]))
                self.assertEqual(addresses[rank], (eng.fast.lut.data_ptr(),
                                 ids.data_ptr(), slots.data_ptr()))
            self.assertTrue(torch.equal(pair[0].model_prune_mask[0],
                                        pair[1].model_prune_mask[0]))
            for expert in range(16):
                owners = sum(int(e.model.prefill_routes[0][1][
                    e.model.prefill_routes[0][0][expert]]) != e.store.null_slot for e in pair)
                self.assertEqual(owners, int(pair[0].model_prune_mask[0][expert]))

    @patch('torch.cuda.is_available', return_value=False)
    def test_without_fixed_routes(self, _cuda):
        eng = make(0)
        del eng.model.prefill_routes
        V41Engine.apply_swaps(eng, [(0, 0, 8, 1.0)])
        self.assertEqual(int(eng.fast.lut[0, 8]), 0)


if __name__ == '__main__':
    unittest.main()
