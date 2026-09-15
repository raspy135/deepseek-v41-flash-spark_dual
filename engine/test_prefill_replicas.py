"""CPU checks for replica planning, unique execution, and untouched original state."""
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
from engine.prefill_replicas import plan_replicas, routing_tables, PrefillReplicas


class TestReplicaPlan(unittest.TestCase):
    def test_balanced_and_empty(self):
        for counts in (np.zeros((2, 8)), np.ones((2, 8))):
            self.assertEqual(plan_replicas(counts, np.ones_like(counts), 53)['moves'], [])

    def test_global_capacity_and_monotone_benefit(self):
        counts = np.tile([30, 5, 20, 5, 10, 5, 10, 5], (40, 1))
        plan = plan_replicas(counts, np.ones_like(counts), 3)
        self.assertLessEqual(len(plan['moves']), 6)  # per-rank capacity, NOT per layer
        self.assertLess(plan['predicted_max_after'], plan['predicted_max_before'])
        for dst in (0, 1):
            slots = [r['slot'] for r in plan['moves'] if r['dst'] == dst]
            self.assertLessEqual(len(slots), 3)
            self.assertEqual(slots, list(range(len(slots))))

    def test_unique_execution_and_restore(self):
        counts = np.array([[30, 5, 20, 5, 10, 5, 10, 5], [5, 30, 5, 20, 5, 10, 5, 10]])
        keep = np.ones_like(counts, dtype=bool)
        plan = plan_replicas(counts, keep, 3)
        lrus = [{(l, e): l*4+e//2 for l in range(2) for e in range(r, 8, 2)} for r in range(2)]
        original = [dict(d) for d in lrus]
        tables = [routing_tables(lrus[r], plan['moves'], r, 8, [9, 10, 11], 2, 8, keep)
                  for r in range(2)]
        for l in range(2):
            for e in range(8):
                self.assertEqual(sum(t[0][l, e] != 8 for t in tables), 1)
                for lut, routes in tables:
                    ids, slots = routes[l]
                    self.assertEqual(slots[ids[e]], lut[l, e])
        self.assertEqual(lrus, original)
        model = SimpleNamespace(prefill_replica_lut=tables[0][0], prefill_replica_routes=tables[0][1],
                                prefill_replica_counts='probe', slot_lut='decode', prefill_routes='original')
        PrefillReplicas(SimpleNamespace(model=model)).clear()
        self.assertEqual(model.slot_lut, 'decode')
        self.assertEqual(model.prefill_routes, 'original')
        self.assertIsNone(model.prefill_replica_lut)
        self.assertIsNone(model.prefill_replica_counts)

    def test_invalid_plan_rejected(self):
        keep = np.ones((1, 4), bool)
        lru = {(0, 0): 0, (0, 2): 1}
        good = dict(layer=0, expert=0, dst=1, slot=0)
        for rows in ([dict(good, dst=0)], [good, good], [dict(good, slot=2)],
                     [dict(good, expert=4)]):
            with self.assertRaises(ValueError):
                routing_tables(lru, rows, 0, 2, [3], 1, 4, keep)

    def test_wanted_counts_are_not_actual_counts(self):
        with self.assertRaises(ValueError):
            plan_replicas([[1, 1]], [[True, False]], 1)
        with self.assertRaises(ValueError):
            plan_replicas([[float('nan'), 1]], [[True, True]], 1)

    def test_activation_budget_and_load_failure(self):
        import torch
        for mode in ('success', 'expired', 'load_failure', 'peer_failure'):
            with self.subTest(mode=mode), ThreadPoolExecutor(2) as pool:
                loaded = []
                def load(key, slot):
                    if mode == 'load_failure':
                        raise OSError('simulated read failure')
                    loaded.append((key, slot))
                store = SimpleNamespace(lru={(0, e): e//2 for e in range(0, 384, 2)},
                    null_slot=192, replica_slots=[193, 194], pool=pool,
                    _load_into_slot=load,
                    _shard=lambda name: SimpleNamespace(expert_runs=lambda prefix: None))
                model = SimpleNamespace(slot_lut='original decode LUT')
                engine = SimpleNamespace(model=model, store=store, device='cpu',
                    args=SimpleNamespace(n_layers=1), model_prune_mask={0: torch.ones(384, dtype=torch.bool)},
                    ep=SimpleNamespace(rank=0, broadcast_obj=lambda payload: payload))
                controller = PrefillReplicas(engine)
                controller.begin(True)
                model.prefill_replica_counts[0, [0, 1, 3]] = torch.tensor([10, 50, 50], dtype=torch.int32)
                controller._status = lambda started, failed=False: [
                    501 if mode == 'expired' else 10, float(failed or mode == 'peer_failure')]
                controller.finish_probe()
                if mode == 'success':
                    self.assertTrue(loaded)
                    self.assertGreater(controller.stats['loaded'], 0)
                    self.assertEqual(int(model.prefill_replica_lut[0, 1]), 193)
                else:
                    self.assertIsNone(model.prefill_replica_lut)
                    self.assertEqual(controller.stats['loaded'], 0)
                self.assertEqual(model.slot_lut, 'original decode LUT')
                self.assertEqual(store.lru, {(0, e): e//2 for e in range(0, 384, 2)})
                controller.clear()
                self.assertIsNone(model.prefill_replica_lut)


if __name__ == '__main__':
    unittest.main()
