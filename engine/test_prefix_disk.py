"""CPU persistence and fail-closed tests; no private prompts or checkpoint needed."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import uuid

import torch

from engine.prefix_disk import PrefixDisk
from engine.prefix_persistence import PersistentPrefixes


def payload(tokens=(1, 2, 3, 4, 5, 6)):
    snapshots = {}
    for n in (2, 4, 6):
        w = min(n, 4)
        snapshots[n] = {
            'ids': tuple(tokens[:n]), 'route': 'route-a',
            'slots': torch.arange(n-w, n) % 8,
            'win': {0: torch.ones(w, 2, dtype=torch.bfloat16)},
            'pending': {0: None},
            'rep': {'h': torch.ones(w, 2, 3, dtype=torch.bfloat16),
                    'pre_mix': torch.ones(w, 2), 'topk': torch.zeros(w, 1, dtype=torch.int32),
                    'cand': None}}
    return {'ids': tuple(tokens), 'snapshots': snapshots,
            'ckv': {0: torch.arange(6).view(3, 2).to(torch.bfloat16)},
            'ik': {0: torch.arange(3).view(3, 1).to(torch.bfloat16)}}


class PrefixDiskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.disk = PrefixDisk(self.tmp.name, 'model-a', 1000000)

    def save(self, value=None):
        bid = uuid.uuid4().hex
        self.assertTrue(self.disk.save(bid, value or payload())['saved'])
        return bid

    def test_longest_boundary_and_restart(self):
        bid = self.save()
        disk = PrefixDisk(self.tmp.name, 'model-a', 1000000)
        self.assertEqual(disk.candidates([1, 2, 3, 4, 99]), [(bid, 4), (bid, 2)])
        restored = disk.load((bid, 4), [1, 2, 3, 4, 99])
        self.assertTrue(torch.equal(restored['ckv'][0], payload()['ckv'][0]))
        self.assertEqual(os.stat(Path(self.tmp.name) / (bid + '.pt')).st_mode & 0o777, 0o600)

    def test_multiple_prompts_and_exact_tokens(self):
        a = self.save()
        b = self.save(payload((7, 8, 9, 10, 11, 12)))
        self.assertEqual(self.disk.candidates([1, 2, 3, 4, 5, 6])[0], (a, 6))
        self.assertEqual(self.disk.candidates([7, 8, 9, 10, 11, 12])[0], (b, 6))
        self.assertIsNone(self.disk.load((a, 6), [7, 8, 9, 10, 11, 12]))

    def test_namespace_and_strict_routing(self):
        self.save()
        self.assertEqual(PrefixDisk(self.tmp.name, 'model-b', 1000000).candidates([1, 2, 3, 4]), [])
        self.assertEqual(self.disk.candidates([1, 2, 3, 4], 'route-b'), [])
        self.assertEqual(len(self.disk.candidates([1, 2, 3, 4], 'route-a')), 2)

    def test_corrupt_and_missing_blob_are_misses(self):
        bid = self.save()
        path = Path(self.tmp.name) / (bid + '.pt')
        with path.open('r+b') as f:
            f.seek(100)
            f.write(b'bad')
        self.assertIsNone(self.disk.load((bid, 4), [1, 2, 3, 4]))
        path.unlink()
        self.assertIsNone(self.disk.load((bid, 4), [1, 2, 3, 4]))

    def test_restart_removes_only_unpublished_cache_artifacts(self):
        saved = self.save()
        orphan = Path(self.tmp.name) / (uuid.uuid4().hex + '.pt')
        temporary = Path(self.tmp.name) / '.writing-interrupted'
        unrelated = Path(self.tmp.name) / 'notes.txt'
        for path in (orphan, temporary, unrelated):
            path.write_bytes(b'test')
        PrefixDisk(self.tmp.name, 'model-a', 1000000)
        self.assertFalse(orphan.exists())
        self.assertFalse(temporary.exists())
        self.assertTrue(unrelated.exists())
        self.assertTrue((Path(self.tmp.name) / (saved + '.pt')).exists())

    def test_budget_eviction_and_oversized_rejection(self):
        a = self.save()
        size = (Path(self.tmp.name) / (a + '.pt')).stat().st_size
        self.disk.budget = size + 100
        b = self.save()
        self.assertFalse((Path(self.tmp.name) / (a + '.pt')).exists())
        self.assertTrue((Path(self.tmp.name) / (b + '.pt')).exists())
        self.disk.budget = 10
        self.assertFalse(self.disk.save(uuid.uuid4().hex, payload())['saved'])

    def adapter(self, peer_options=None, peer_load_ok=True):
        from engine.v41_engine import V41Engine
        p = object.__new__(PersistentPrefixes)
        p.disk, p.strict, p.pending, p.stats, p.log = self.disk, False, None, {}, lambda _: None
        e = object.__new__(V41Engine)
        e.args = SimpleNamespace(compress_ratios={0: 2}, window_size=4,
                                 candidate_source_layer=0, head_dim=2, hc_mult=2, dim=3)
        e.caches = SimpleNamespace(ckv={0: torch.zeros(10, 2, dtype=torch.bfloat16)},
                                   ik={0: torch.zeros(10, 1, dtype=torch.bfloat16)},
                                   win=[torch.zeros(8, 2, dtype=torch.bfloat16)], pending={0: None})
        e.model = SimpleNamespace()
        e.device, e._prefix_route, e.swa_replay, e.replica_slots = 'cpu', 'route-a', True, 0
        e._prefix_cache, e._prefix_snapshots, e.prefix_disk = None, {}, p
        def gather(x):
            if isinstance(x, bool):
                return [x, peer_load_ok]
            return [x, peer_options if peer_options is not None else x]
        e.ep = SimpleNamespace(gather_objects=gather)
        p.engine = e
        return p

    def test_restore_into_existing_allocations(self):
        self.save()
        p = self.adapter()
        before = p.engine.caches.ckv[0].data_ptr()
        self.assertEqual(p.restore([1, 2, 3, 4, 99], 0), 4)
        self.assertEqual(p.engine.caches.ckv[0].data_ptr(), before)
        self.assertTrue(torch.equal(p.engine.caches.ckv[0][:2], payload()['ckv'][0][:2]))
        self.assertEqual(p.engine.caches.len, 4)

    def test_rank_missing_entry_or_failed_load_forces_shared_miss(self):
        self.save()
        self.assertEqual(self.adapter(peer_options=[]).restore([1, 2, 3, 4], 0), 0)
        p = self.adapter(peer_load_ok=False)
        self.assertEqual(p.restore([1, 2, 3, 4], 0), 0)
        self.assertEqual(p.engine.caches.ckv[0].count_nonzero(), 0)

    def test_invalid_shape_rejected_before_gpu_state_changes(self):
        value = payload()
        value['ckv'][0] = torch.ones(3, 3, dtype=torch.bfloat16)
        self.save(value)
        p = self.adapter()
        self.assertEqual(p.restore([1, 2, 3, 4], 0), 0)
        self.assertEqual(p.engine.caches.ckv[0].count_nonzero(), 0)


if __name__ == '__main__':
    unittest.main()
