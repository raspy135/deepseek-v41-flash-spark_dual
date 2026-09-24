"""Native gather parity, bounds, concurrency and buffer lifetime. CPU-only."""
from concurrent.futures import ThreadPoolExecutor
import gc
import sys
import unittest
sys.path[:0] = ['/app', '/app/tools']
import numpy as np
import torch
from engine.engram_native import NativeGather, RowBuffers


class NativeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(721)
        self.w = rng.integers(0, 256, (4096, 256), dtype=np.uint8)
        self.s = rng.integers(0, 256, (4096, 8), dtype=np.uint8)

    def test_byte_parity_sizes_duplicates_and_order(self):
        rng = np.random.default_rng(938)
        for workers in (1, 8, 64):
            with NativeGather(self.w, self.s, workers) as native:
                for n in (0, 1, 8, 96, 144, 384, 513):
                    ids = rng.integers(0, len(self.w), n, dtype=np.int64)
                    expected = np.concatenate((self.w[ids], self.s[ids]), axis=1)
                    np.testing.assert_array_equal(native.gather(ids), expected)

    def test_bad_ids_and_layout(self):
        with NativeGather(self.w, self.s, 4) as native:
            for ids in (np.array([-1]), np.array([4096])):
                with self.assertRaises(IndexError): native.gather(ids)
            for ids in (np.arange(4, dtype=np.int32), np.arange(8)[::2], np.zeros((1, 1), np.int64)):
                with self.assertRaises(ValueError): native.gather(ids)

    def test_concurrent_results_keep_their_storage(self):
        with NativeGather(self.w, self.s, 8) as native, ThreadPoolExecutor(8) as pool:
            ids = [np.arange(i, i + 96, dtype=np.int64) for i in range(20)]
            outputs = list(pool.map(native.gather, ids))  # Exhausts four reusable buffers.
            for index, out in zip(ids, outputs):
                np.testing.assert_array_equal(out[:, :256], self.w[index])
                np.testing.assert_array_equal(out[:, 256:], self.s[index])
            for a, b in zip(outputs, outputs[1:]): self.assertFalse(np.shares_memory(a, b))

    def test_lease_survives_numpy_and_torch_views(self):
        buffers = RowBuffers(rows=96, slots=1)
        first = buffers.acquire(96)
        first.fill(17)
        address = first.ctypes.data
        view = first[3:10]
        tensor = torch.from_numpy(view)
        del first, view
        gc.collect()
        self.assertEqual(len(buffers.free), 0)
        second = buffers.acquire(96)
        second.fill(91)
        self.assertNotEqual(second.ctypes.data, address)
        self.assertTrue(torch.all(tensor == 17))
        del tensor
        gc.collect()
        self.assertEqual(len(buffers.free), 1)
        third = buffers.acquire(96)
        self.assertEqual(third.ctypes.data, address)

    def test_close_is_idempotent(self):
        native = NativeGather(self.w, self.s, 4)
        native.close(); native.close()
        with self.assertRaises(RuntimeError): native.gather(np.array([1], np.int64))

    def test_large_gather_keeps_python_fallback(self):
        from engine.engram import EngramTable
        table = EngramTable.__new__(EngramTable)
        table.w_mm, table.s_mm = self.w, self.s
        table.gather_threads, table.gather_min_parallel = 4, 32
        with NativeGather(self.w, self.s, 4) as native, ThreadPoolExecutor(4) as pool:
            table.native_gather, table.pool = native, pool
            calls = []
            original = native.gather
            def counted(ids):
                calls.append(len(ids))
                return original(ids)
            native.gather = counted
            for n in (0, 96, 384, 385, 1024):
                ids = np.arange(n, dtype=np.int64)
                got = table._gather_rows(ids)
                np.testing.assert_array_equal(got, np.concatenate((self.w[ids], self.s[ids]), axis=1))
            self.assertEqual(calls, [0, 96, 384])


if __name__ == '__main__': unittest.main()
