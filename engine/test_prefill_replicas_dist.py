"""Two-process CPU/Gloo validation of replica activation and failure coordination.

No model weights or CUDA; this verifies collective order, not kernel quality or speed.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from engine.prefill_replicas import PrefillReplicas


def worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        def broadcast(payload):
            obj = [payload]
            dist.broadcast_object_list(obj, src=0)
            return obj[0]
        for mode in ('success', 'load_failure', 'expired'):
            with ThreadPoolExecutor(2) as pool:
                def load(key, slot):
                    if mode == 'load_failure' and rank == 0:
                        raise OSError('simulated rank-0 failure')
                lru = {(0, e): e//2 for e in range(rank, 384, 2)}
                store = SimpleNamespace(lru=lru, null_slot=192, replica_slots=[193, 194],
                    pool=pool, _load_into_slot=load,
                    _shard=lambda name: SimpleNamespace(expert_runs=lambda prefix: None))
                model = SimpleNamespace(slot_lut='original')
                engine = SimpleNamespace(model=model, store=store, device='cpu',
                    args=SimpleNamespace(n_layers=1), model_prune_mask={0: torch.ones(384, dtype=torch.bool)},
                    ep=SimpleNamespace(rank=rank, broadcast_obj=broadcast))
                controller = PrefillReplicas(engine, budget_ms=0 if mode == 'expired' else 500)
                controller.begin(True)
                model.prefill_replica_counts[0, [0, 1, 3]] = torch.tensor([10, 50, 50], dtype=torch.int32)
                controller.finish_probe()
                assert model.slot_lut == 'original'
                assert store.lru == lru
                if mode == 'success':
                    assert controller.stats['loaded'] > 0, controller.stats
                    executions = (model.prefill_replica_lut != 192).to(torch.int32)
                    dist.all_reduce(executions)
                    assert torch.equal(executions, torch.ones_like(executions))
                else:
                    assert controller.stats['loaded'] == 0, controller.stats
                    assert model.prefill_replica_lut is None
                controller.clear()
                assert model.prefill_replica_lut is None
    finally:
        dist.destroy_process_group()


class TestDistributedReplicas(unittest.TestCase):
    def test_two_rank_activation_and_fallback(self):
        with tempfile.TemporaryDirectory(prefix='prefill-replica-test-') as directory:
            mp.spawn(worker, args=(directory + '/rendezvous',), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
