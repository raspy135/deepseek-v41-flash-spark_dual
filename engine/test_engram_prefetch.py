"""CPU tests for prefill read overlap, layer mapping, and failure cleanup."""
import json
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from engram import EngramReadAhead, EngramTable, prefetch_rows


class Table:
    def __init__(self, barrier, fail=False):
        self.barrier, self.fail = barrier, fail
        self.finished = threading.Event()
        self.caller = threading.get_ident()

    def read_raw(self, hashes):
        try:
            self.barrier.wait(timeout=5)
            if self.fail:
                raise OSError('test read failure')
            return hashes.copy(), None, None
        finally:
            self.finished.set()

    def to_device(self, raw, inv, shape):
        assert threading.get_ident() == self.caller, 'device work moved into worker'
        return raw


class Prefetch(unittest.TestCase):
    def run_case(self, fail=False, abort=False):
        barrier = threading.Barrier(2)
        tables = {1: Table(barrier, fail), 14: Table(barrier)}
        hashes = torch.arange(3 * 2 * 24).reshape(3, 2, 24)
        with ThreadPoolExecutor(2) as pool:
            try:
                with prefetch_rows(tables, pool, hashes, [1, 14]) as rows:
                    if abort:
                        raise ValueError('abort before consuming reads')
                    np.testing.assert_array_equal(rows(1, None), hashes[:, 0].numpy())
                    np.testing.assert_array_equal(rows(14, None), hashes[:, 1].numpy())
            finally:
                # Tested before executor shutdown can implicitly join the workers.
                self.assertTrue(all(t.finished.is_set() for t in tables.values()))

    def test_overlap_and_layer_mapping(self):
        self.run_case()

    def test_forward_failure_joins_reads(self):
        with self.assertRaisesRegex(ValueError, 'abort'):
            self.run_case(abort=True)

    def test_read_failure_propagates_and_joins_peer(self):
        with self.assertRaisesRegex(OSError, 'read failure'):
            self.run_case(fail=True)


class ReadAhead(unittest.TestCase):
    class Table:
        def read_raw(self, hashes):
            return hashes.copy(), None, None

        def to_device(self, raw, _inv, _shape):
            return raw

    def test_irregular_image_spans_use_exact_boundaries(self):
        # The 140-token middle span is the shape from the small-image failure. The old scheduler
        # sliced 2048 rows at start=2048 and handed them to a 140-token forward.
        spans = [(0, 2048), (2048, 2188), (2188, 4236), (4236, 5000)]
        hashes = np.arange(5000 * 2 * 24).reshape(5000, 2, 24)
        tables = {1: self.Table(), 14: self.Table()}
        with ThreadPoolExecutor(4) as pool:
            ra = EngramReadAhead(tables, pool, [1, 14], hashes, spans, depth=2)
            try:
                for s, e in spans:
                    rows = ra.rows_for(s)
                    for li, layer in enumerate((1, 14)):
                        got = rows(layer, torch.from_numpy(hashes[s:e, li]))
                        np.testing.assert_array_equal(got, hashes[s:e, li])
                    ra.done(s)
            finally:
                ra.close()

    def test_forward_length_mismatch_is_loud(self):
        hashes = np.zeros((20, 1, 24), dtype=np.int64)
        with ThreadPoolExecutor(1) as pool:
            ra = EngramReadAhead({1: self.Table()}, pool, [1], hashes,
                                  [(0, 7), (7, 20)], depth=1)
            try:
                rows = ra.rows_for(0)
                with self.assertRaisesRegex(RuntimeError, 'has 7 rows, forward requested 8'):
                    rows(1, torch.from_numpy(hashes[:8, 0]))
            finally:
                ra.close()


class RealRows(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('MODEL_DIR'), 'set MODEL_DIR for checkpoint row parity')
    def test_checkpoint_rows(self):
        root = os.environ['MODEL_DIR']
        with open(os.path.join(root, 'model.safetensors.index.json')) as f:
            index = json.load(f)
        tables = {L: EngramTable(root, index, L, 'cpu', threads=4) for L in (1, 14)}
        try:
            with ThreadPoolExecutor(2) as pool:
                for tokens in (17, 128, 513):
                    hashes = torch.randint(0, 1024, (tokens, 2, 24))
                    expected = {L: tables[L].rows(hashes[:, li]) for li, L in enumerate(tables)}
                    for table in tables.values():
                        table.cache.clear()
                    with prefetch_rows(tables, pool, hashes, [1, 14]) as rows:
                        for L in tables:
                            self.assertTrue(torch.equal(expected[L], rows(L, None)))
        finally:
            for table in tables.values():
                table.pool.shutdown()
                os.close(table.fd)


if __name__ == '__main__':
    unittest.main()
