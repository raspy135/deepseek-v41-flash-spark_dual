"""Bit-exactness gate for decode fp8 scheduling and merged projections (synthetic weights).

Narrower BLOCK_N and one launch over stacked weights are both claimed to leave every output
element unchanged. That is a property of this GPU and Triton build, so it is tested here at the
served model's TP-local shapes, not assumed. The full-engine check is
tools/bench_decode_projection_tp.py.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp8_linear as F8  # noqa: E402
import v41_ref as R  # noqa: E402

# (name, N, K) at the TP2 local shapes of DeepSeek-V4.1-Flash, plus the unsharded shared expert.
SHAPES = (("wq_a", 1280, 5120), ("wkv", 512, 5120), ("sh_w13_local", 1152, 5120),
          ("sh_w13_full", 2304, 5120), ("sh_w2_local", 5120, 1152), ("wq_b_local", 16384, 1280),
          ("wo_b_local", 2560, 8192))
ROWS = (1, 4, 6, 10, 16)


def _weight(n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return F8.quantize_to_fp8(torch.randn(n, k, generator=gen, device="cuda") * 0.02)


def _x(m, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(m, k, generator=gen, device="cuda") * 0.5).to(torch.bfloat16)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class DecodeTileTest(unittest.TestCase):
    def setUp(self):
        self._policy = (F8.DECODE_BLOCK_N, F8.DECODE_WARPS)

    def tearDown(self):
        F8.DECODE_BLOCK_N, F8.DECODE_WARPS = self._policy

    def _run(self, fn, block_n, warps):
        F8.DECODE_BLOCK_N, F8.DECODE_WARPS = block_n, warps
        return fn()

    def test_block_n_is_exact(self):
        for i, (name, n, k) in enumerate(SHAPES):
            w = _weight(n, k, 100 + i)
            for m in ROWS:
                x = _x(m, k, 200 + m)
                for out_dtype in (torch.bfloat16, torch.float32):
                    ref = self._run(lambda: F8.fp8_linear(x, w, out_dtype), "128", 4)
                    for bn in ("64", "32", "16", "auto"):
                        for warps in (2, 4):
                            got = self._run(lambda: F8.fp8_linear(x, w, out_dtype), bn, warps)
                            with self.subTest(name=name, m=m, bn=bn, warps=warps, dtype=out_dtype):
                                self.assertTrue(torch.equal(got, ref),
                                                float((got.float() - ref.float()).abs().max()))

    def test_grouped_block_n_is_exact(self):
        # wo_a under TP2: 4 local groups of 1024 rows over K = 64 * 512 / 8.
        g, r, k = 4, 1024, 4096
        base = _weight(g * r, k, 7)
        w = F8.FP8GroupedWeight(base.w, base.s, g, r)
        for m in ROWS:
            x = _x(m * g, k, 300 + m).view(m, g, k)
            ref = self._run(lambda: F8.fp8_grouped_linear(x, w), "128", 4)
            for bn in ("64", "32", "16", "auto"):
                got = self._run(lambda: F8.fp8_grouped_linear(x, w), bn, 4)
                with self.subTest(m=m, bn=bn):
                    self.assertTrue(torch.equal(got, ref))

    def test_auto_policy(self):
        dev = torch.device("cuda")
        sms = F8._sm_count(dev)
        for n in (512, 1152, 1280, 1792, 2304, 2560, 16384):
            bn = F8.decode_block_n(n, 1, dev, "auto")
            self.assertIn(bn, (16, 32, 64, 128))
            if bn != 16:
                self.assertGreaterEqual(-(-n // bn), sms)
            if bn != 128:
                self.assertLess(-(-n // (bn * 2)), sms)

    def test_concat_rows_views_share_storage(self):
        a, b = _weight(1280, 5120, 1), _weight(512, 5120, 2)
        merged, (va, vb) = F8.concat_rows(a, b)
        self.assertEqual(merged.shape, (1792, 5120))
        self.assertEqual(va.w.data_ptr(), merged.w.data_ptr())
        self.assertEqual(vb.w.data_ptr(), merged.w[1280:].data_ptr())
        self.assertTrue(torch.equal(va.w.view(torch.uint8), a.w.view(torch.uint8)))
        self.assertTrue(torch.equal(vb.s, b.s))
        self.assertIsNone(F8.concat_rows(a, _weight(512, 4096, 3)))    # K differs
        self.assertIsNone(F8.concat_rows(a, a.dequant()))                # not an FP8Weight

    def test_merged_projection_is_exact(self):
        pairs = (("wq_a+wkv", (1280, 5120), (512, 5120)),
                 ("sh_w1+w3 local", (1152, 5120), (1152, 5120)),
                 ("sh_w1+w3 full", (2304, 5120), (2304, 5120)))
        for i, (name, sa, sb) in enumerate(pairs):
            a, b = _weight(*sa, 10 + i), _weight(*sb, 20 + i)
            merged, _ = F8.concat_rows(a, b)
            for bn in ("128", "auto"):
                for m in ROWS:
                    x = _x(m, sa[1], 400 + m)
                    F8.DECODE_BLOCK_N = bn
                    ya, yb = R.qlinear(x, a), R.qlinear(x, b)
                    y = R.qlinear(x, merged)
                    with self.subTest(name=name, m=m, bn=bn):
                        self.assertTrue(torch.equal(y[:, :a.N].contiguous(), ya))
                        self.assertTrue(torch.equal(y[:, a.N:].contiguous(), yb))
                        # the downstream norm sees the same tensor as the split path
                        g = torch.ones(a.N, device="cuda", dtype=torch.bfloat16)
                        self.assertTrue(torch.equal(R.rmsnorm(y[:, :a.N].contiguous(), g, 1e-20),
                                                    R.rmsnorm(ya, g, 1e-20)))

    def test_merged_shared_expert_is_exact(self):
        for local in (1152, 2304):
            w1, w3 = _weight(local, 5120, 31), _weight(local, 5120, 32)
            w2 = _weight(5120, local, 33)
            merged, _ = F8.concat_rows(w1, w3)
            for m in ROWS:
                x = _x(m, 5120, 500 + m)
                ref = R.expert_ffn(x, w1, w2, w3, 10.0)
                got = R.expert_ffn(x, w1, w2, w3, 10.0, w13=merged)
                with self.subTest(local=local, m=m):
                    self.assertTrue(torch.equal(got, ref))

    def test_merge_decode_projections(self):
        import types
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from engine.fastdecode import merge_decode_projections
        blk = types.SimpleNamespace(wq_a=_weight(1280, 5120, 51), wkv=_weight(512, 5120, 52),
                                    sh_w1=_weight(1152, 5120, 53), sh_w3=_weight(1152, 5120, 54))
        wq_a = blk.wq_a.w.clone()
        tp = types.SimpleNamespace(wq_a=_weight(1280, 5120, 55), wkv=_weight(512, 5120, 56),
                                   sh_w1=types.SimpleNamespace(tp_linear=None), sh_w3=_weight(1152, 5120, 57))
        self.assertEqual(merge_decode_projections([blk, tp]), 3)   # tp's wrapped shared expert is skipped
        self.assertEqual(merge_decode_projections([blk, tp]), 0)   # idempotent
        self.assertIsNone(getattr(tp, "_sh_w13", None))
        self.assertEqual(blk.wq_a.w.data_ptr(), blk._wqkv_a.w.data_ptr())
        self.assertEqual(blk.sh_w3.w.data_ptr(), blk._sh_w13.w[1152:].data_ptr())
        self.assertTrue(torch.equal(blk.wq_a.w.view(torch.uint8), wq_a.view(torch.uint8)))

    def test_graph_capture(self):
        a, b = _weight(1280, 5120, 41), _weight(512, 5120, 42)
        merged, _ = F8.concat_rows(a, b)
        x = _x(4, 5120, 43)
        F8.DECODE_BLOCK_N = "auto"
        eager = R.qlinear(x, merged)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            R.qlinear(x, merged)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = R.qlinear(x, merged)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, eager))


if __name__ == "__main__":
    unittest.main()
