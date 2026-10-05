"""Regression: L2 prefetch reads raw FP8/FP4/BF16 storage, including masked tails and graphs.

Run with the serving CUDA runtime: python tools/test_l2pf_cuda.py.
"""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import l2pf


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class PrefetchTest(unittest.TestCase):
    def setUp(self):
        self.stream = torch.cuda.Stream()

    def weight(self, dtype, n=4101):
        size = torch.empty((), dtype=dtype).element_size()
        # Include every byte pattern, including FP8 NaNs: prefetch must not decode them.
        storage = (torch.arange(n * size, device="cuda", dtype=torch.int32) % 256).to(torch.uint8)
        return storage.view(dtype)

    def check_sink(self, weight, take):
        storage = weight.reshape(-1).view(torch.uint8)
        sums = [storage[i:min(i + 4096, take)].to(torch.float32).sum()
                for i in range(0, take, 4096)]
        self.assertTrue(torch.equal(l2pf._SINK[:len(sums)], torch.stack(sums)))

    def test_dtypes_and_partial_byte_budgets(self):
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16,
                      torch.float32, torch.int8, torch.uint8, torch.int32):
            weight = self.weight(dtype)
            before = weight.view(torch.uint8).clone()
            for budget in (1, 4095, 4096, 4097, before.numel() + 17):
                with self.subTest(dtype=dtype, budget=budget):
                    self.stream.wait_stream(torch.cuda.current_stream())
                    take = min(budget, before.numel())
                    self.assertEqual(l2pf.touch(self.stream, [weight], budget), take)
                    self.stream.synchronize()
                    self.check_sink(weight, take)
                    self.assertTrue(torch.equal(weight.view(torch.uint8), before))

    def test_wrappers_share_one_budget(self):
        first = self.weight(torch.float8_e4m3fn, 7)
        second = self.weight(torch.bfloat16, 17)
        weights = [None, torch.empty(0, device="cuda"),
                   SimpleNamespace(local=SimpleNamespace(w=first)), SimpleNamespace(w=second)]
        self.stream.wait_stream(torch.cuda.current_stream())
        self.assertEqual(l2pf.touch(self.stream, weights, 16), 16)
        self.stream.synchronize()
        self.check_sink(second, 9)
        self.assertEqual(l2pf.touch(self.stream, weights, 0), 0)

    def test_fp8_graph_capture_and_replay(self):
        weight = self.weight(torch.float8_e4m3fn)
        before = weight.view(torch.uint8).clone()
        self.stream.wait_stream(torch.cuda.current_stream())
        l2pf.touch(self.stream, [weight], 4097)  # Compile and allocate the sink before capture.
        self.stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            main = torch.cuda.current_stream()
            self.stream.wait_stream(main)
            l2pf.touch(self.stream, [weight], 4097)
            main.wait_stream(self.stream)
        graph.replay()
        torch.cuda.synchronize()
        self.check_sink(weight, 4097)
        self.assertTrue(torch.equal(weight.view(torch.uint8), before))


if __name__ == "__main__":
    unittest.main()
