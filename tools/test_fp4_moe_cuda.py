"""Synthetic parity and graph-capture gate for the opt-in native CUDA decode kernel."""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp4_moe as K  # noqa: E402


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() >= (12, 1),
                     "native FP4 CUDA kernel requires sm_121a")
class NativeDecodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.arena = K.ExpertArena(1, "cuda")
        except torch.AcceleratorError as error:
            if "out of memory" in str(error).lower():
                raise unittest.SkipTest("GPU memory is occupied; native FP4 test needs one 18.8 MB arena") from error
            raise
        gen = torch.Generator().manual_seed(20260922)
        for tensor in (cls.arena.w1, cls.arena.w3, cls.arena.w2):
            tensor.copy_(torch.randint(0, 256, tensor.shape, dtype=torch.uint8, generator=gen))
        for tensor in (cls.arena.s1, cls.arena.s3, cls.arena.s2):
            tensor.copy_(torch.randint(124, 129, tensor.shape, dtype=torch.uint8, generator=gen))
        # Compile/load outside graph capture and before the comparisons.
        import fp4_moe_cuda
        fp4_moe_cuda._lib()

    @staticmethod
    def _run(tokens, topk=1):
        gen = torch.Generator().manual_seed(1234 + tokens)
        x = torch.randn((tokens, K.DIM), generator=gen).bfloat16().cuda()
        slots = torch.zeros((tokens, topk), dtype=torch.int32, device="cuda")
        weight = torch.full((tokens, topk), 1.0 / topk, dtype=torch.float32, device="cuda")
        K.CUDA_DECODE = False
        expected = K.moe_forward(x, slots, weight, NativeDecodeTest.arena, out_dtype=torch.float32)
        K.CUDA_DECODE = True
        actual = K.moe_forward(x, slots, weight, NativeDecodeTest.arena, out_dtype=torch.float32)
        torch.cuda.synchronize()
        return x, slots, weight, expected, actual

    def test_parity_one_and_six_pairs(self):
        for tokens, topk in ((1, 1), (6, 1), (16, 1), (2, 6), (8, 2), (4, 3), (4, 4)):
            with self.subTest(tokens=tokens, topk=topk):
                _, _, _, expected, actual = self._run(tokens, topk)
                rel = float((actual - expected).norm() / expected.norm())
                self.assertLess(rel, 1e-3)

    def test_router_matches_stable_slot_order(self):
        import fp4_moe_cuda as CUDA
        slots = torch.tensor([[4, 1, 7, 4, 2, 1], [3, 7, 0, 3, 2, 5]],
                             dtype=torch.int32, device="cuda")
        expected_slot, expected_pair, _ = K.build_routing_small(slots, 16)
        actual_slot, actual_pair, actual_route, _ = CUDA.build_routing_small(slots, 16, 6)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(actual_slot, expected_slot))
        self.assertTrue(torch.equal(actual_pair, expected_pair))
        self.assertTrue(torch.equal(torch.where(actual_route >= 0, actual_route & 0xffff, -1),
                                    expected_pair))
        expected_token = torch.where(expected_pair >= 0, expected_pair // 6, -1)
        actual_token = actual_route >> 16
        self.assertTrue(torch.equal(actual_token, expected_token))

    def test_null_routes_skip_uninitialized_intermediate(self):
        gen = torch.Generator(device="cuda").manual_seed(9327)
        x = torch.randn((6, K.DIM), generator=gen, device="cuda", dtype=torch.bfloat16)
        weight = torch.full((6, 6), 1 / 6, device="cuda")
        for all_null in (False, True):
            slots = torch.ones((6, 6), device="cuda", dtype=torch.int32)
            if not all_null:
                slots[:, 0] = 0
            results = []
            for native in (False, True):
                K.CUDA_DECODE = native
                results.append(K.moe_forward(x, slots, weight, self.arena,
                                             out_dtype=torch.float32, slots_repeat=True,
                                             null_slot=1))
            rel = float((results[1] - results[0]).norm() / results[0].norm().clamp_min(1e-30))
            self.assertLess(rel, 1e-3)
            self.assertTrue(torch.isfinite(results[1]).all())
            if all_null:
                self.assertEqual(int(torch.count_nonzero(results[1])), 0)
            else:
                without_null = K.moe_forward(x, slots[:, :1].contiguous(),
                                             weight[:, :1].contiguous(), self.arena,
                                             out_dtype=torch.float32)
                self.assertTrue(torch.equal(results[1], without_null))

    def test_router_splits_long_expert_runs(self):
        import fp4_moe_cuda as CUDA
        slots = torch.zeros((64, 1), device="cuda", dtype=torch.int32)
        blocks, pairs, routes, _ = CUDA.build_routing_small(slots, 16, 1)
        self.assertTrue(torch.equal(blocks[:4], torch.zeros_like(blocks[:4])))
        self.assertTrue(torch.all(blocks[4:] == -1))
        expected = torch.arange(64, device="cuda", dtype=torch.int32)
        self.assertTrue(torch.equal(pairs[:64], expected))
        self.assertTrue(torch.equal(routes[:64] >> 16, expected))
        self.assertTrue(torch.all(pairs[64:] == -1))

    def test_down_partial_scatter_layout_and_tail(self):
        import fp4_moe_cuda as CUDA
        # 258 output columns exercises an incomplete output-row block and local_n=129.
        n, k, topk, ntok = 258, 128, 3, 2
        slots = torch.zeros((ntok, topk), device="cuda", dtype=torch.int32)
        block_slot, block_pair, _, nb = CUDA.build_routing_small(slots, 16, topk)
        h = torch.randn((ntok * topk, k), device="cuda", dtype=torch.bfloat16)
        w = torch.randint(0, 256, (1, n, k // 2), device="cuda", dtype=torch.uint8)
        s = torch.full((1, n, k // 32), 126, device="cuda", dtype=torch.uint8)
        full = torch.empty((ntok * topk, n), device="cuda")
        scatter, rounded = torch.empty_like(full), torch.empty_like(full)
        for out, partial, world in ((full, True, 1), (scatter, True, 2), (rounded, False, 1)):
            CUDA.down(h, w, s, out, block_slot, block_pair, topk, n, k, ntok, nb, -1,
                      partial, world)
        expected = full.view(topk, ntok, 2, n // 2).permute(2, 0, 1, 3).contiguous()
        self.assertTrue(torch.equal(scatter.flatten(), expected.flatten()))
        self.assertTrue(torch.equal(rounded, full.bfloat16().float()))

    def test_cuda_graph_replay(self):
        x, slots, weight, _, eager = self._run(1)
        K.CUDA_DECODE = True
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = K.moe_forward(x, slots, weight, self.arena, out_dtype=torch.float32)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(captured, eager))

    @classmethod
    def tearDownClass(cls):
        K.CUDA_DECODE = os.environ.get("DSV41_FP4_CUDA", "1") == "1"


if __name__ == "__main__":
    unittest.main()
