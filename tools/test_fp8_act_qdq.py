"""Exactness gate for DSV41_FP8_ACT_QDQ_FUSED: activation quantization inside the fp8 GEMM.

The in-kernel quantization must reproduce v41_ref.act_qdq_fp8 bit for bit. An identity weight
makes the GEMM output the quantized activation itself, so it is compared element by element on
inputs chosen to hit the edges: amax exactly on and one ulp above a power-of-two boundary (where
an approximate log2 or division moves ceil()), groups below the 1e-4 floor, all-zero groups,
e4m3 rounding ties, negative zero, and a wide exponent range. Then real-shape GEMMs, merged
weights, the shared expert and both TP wrappers (with a stub collective) are compared with the
flag on and off.
"""
from __future__ import annotations

import os
import sys
import types
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fp8_linear as F8  # noqa: E402
import v41_ref as R  # noqa: E402

ROWS = (1, 4, 6, 10, 16)


def _identity(k):
    w = torch.eye(k, device="cuda").to(torch.float8_e4m3fn)
    s = torch.full((k // 32, k // 32), 127, dtype=torch.uint8, device="cuda")
    return F8.FP8Weight(w, s)


def _hard_x(m, k, seed):
    """[m, k] bf16 whose 32-wide groups cover the quantizer's edge cases."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    groups = m * k // 32
    x = torch.randn(groups, 32, generator=gen, device="cuda")
    exps = torch.randint(-40, 20, (groups, 1), generator=gen, device="cuda").float()
    x = x * torch.exp2(exps)
    kind = torch.randint(0, 8, (groups,), generator=gen, device="cuda")
    boundary = 448.0 * torch.exp2(torch.randint(-20, 10, (groups,), generator=gen, device="cuda").float())
    x = x.to(torch.bfloat16).float()
    amax = x.abs().amax(1).clamp_min(1e-30)
    x_scaled = x / amax[:, None]                                            # max |.| == 1
    on = (x_scaled * boundary[:, None]).to(torch.bfloat16).float()          # amax exactly 448 * 2^j
    above = torch.nextafter(on.to(torch.bfloat16), torch.tensor(float("inf"), dtype=torch.bfloat16,
                            device="cuda")).float()                          # one bf16 ulp above
    x = torch.where((kind == 0)[:, None], on, x)
    x = torch.where((kind == 1)[:, None], above, x)
    x = torch.where((kind == 2)[:, None], x_scaled * 5e-5, x)               # under the 1e-4 floor
    x = torch.where((kind == 3)[:, None], torch.zeros_like(x), x)
    # e4m3 ties: with amax 448 the scale is 1. Subnormal codes are 2^-9 apart, so odd multiples
    # of 2^-10 are midpoints; in [1, 2) codes are 1/8 apart, so 1 + (2j+1)/16 are midpoints.
    ties = (torch.randint(0, 64, (groups, 32), generator=gen, device="cuda") * 2 + 1).float()
    tie = torch.where(torch.rand(groups, 32, generator=gen, device="cuda") < 0.5,
                      ties / 1024.0, 1.0 + ((ties % 16) // 2 * 2 + 1) / 16.0)
    tie[:, 0] = 448.0
    x = torch.where((kind == 4)[:, None], tie * torch.sign(torch.randn(groups, 32, generator=gen,
                                                                      device="cuda")), x)
    x = torch.where((kind == 5)[:, None], -torch.zeros_like(x), x)
    return x.reshape(m, k).to(torch.bfloat16)


class _StubDist:
    """Two-rank collectives with a synthetic peer that behaves like a real one: its shard is
    independent of ours, and it runs the same code path we do -- so on the unfused path it
    sends act_qdq_fp8(P), on the fused path the raw P. (A peer derived from our input by
    scaling is not equivalent: quantization is not linear.)"""

    def __init__(self, input_width):
        self.input_width = input_width   # OutputParallelWeight gathers inputs, then outputs

    def all_gather_into_tensor(self, out, inp):
        half = out.shape[0] // 2
        out[:half].copy_(inp)
        if inp.shape[1] != self.input_width:
            out[half:].copy_(inp)            # the output gather: any fixed peer output will do
            return
        peer = _hard_x(inp.shape[0], inp.shape[1], int(inp.numel()) + 1)
        out[half:].copy_(peer if R.ACT_QDQ_FUSED else R.act_qdq_fp8(peer))

    def all_reduce(self, t):
        t.mul_(2.0)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class ActQdqFusedTest(unittest.TestCase):
    def setUp(self):
        self._saved = (R.ACT_QDQ_FUSED, F8.DECODE_BLOCK_N)

    def tearDown(self):
        R.ACT_QDQ_FUSED, F8.DECODE_BLOCK_N = self._saved

    def _weight(self, n, k, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return F8.quantize_to_fp8(torch.randn(n, k, generator=gen, device="cuda") * 0.02)

    def _both(self, fn):
        R.ACT_QDQ_FUSED = False
        ref = fn()
        R.ACT_QDQ_FUSED = True
        got = fn()
        return ref, got

    def test_quantizer_matches_elementwise(self):
        k = 512
        w = _identity(k)
        for seed in range(20):
            for m in ROWS:
                x = _hard_x(m, k, 1000 * seed + m)
                # +0.0: the identity GEMM starts its accumulator at +0, and +0 + (-0) = +0, so a
                # -0 activation cannot survive it. That is a property of this probe, not of the
                # quantizer; in serving both arms go through the same GEMM.
                ref = R.act_qdq_fp8(x).float() + 0.0
                for bn in ("128", "auto"):
                    F8.DECODE_BLOCK_N = bn
                    got = F8.fp8_linear(x, w, out_dtype=torch.float32, act_qdq=True) + 0.0
                    with self.subTest(seed=seed, m=m, bn=bn):
                        bad = (got.view(torch.int32) != ref.view(torch.int32)).sum().item()
                        diff = got.view(torch.int32) != ref.view(torch.int32)
                        self.assertEqual(bad, 0, f"{bad} activations differ, e.g. got "
                                         f"{got[diff][:4].tolist()} ref {ref[diff][:4].tolist()}")

    def test_real_shapes(self):
        shapes = ((1280, 5120), (512, 5120), (1792, 5120), (1152, 5120), (2304, 5120),
                  (16384, 1280), (2560, 8192), (5120, 1152))
        for i, (n, k) in enumerate(shapes):
            w = self._weight(n, k, i)
            for m in ROWS:
                x = _hard_x(m, k, 50 + m) if i % 2 else (torch.randn(m, k, device="cuda") * 0.5).to(torch.bfloat16)
                for bn in ("128", "auto"):
                    F8.DECODE_BLOCK_N = bn
                    ref, got = self._both(lambda: R.qlinear(x, w))
                    with self.subTest(n=n, k=k, m=m, bn=bn):
                        self.assertTrue(torch.equal(got, ref))

    def test_dispatch_gates(self):
        w = self._weight(512, 5120, 3)
        R.ACT_QDQ_FUSED = True
        bf = torch.randn(4, 5120, device="cuda").to(torch.bfloat16)
        self.assertTrue(R._act_qdq_fusable(bf, w))
        self.assertFalse(R._act_qdq_fusable(bf.float(), w))                       # fp32 input
        self.assertFalse(R._act_qdq_fusable(torch.randn(17, 5120, device="cuda").to(torch.bfloat16), w))
        self.assertFalse(R._act_qdq_fusable(bf, w.dequant()))                      # bf16 weight
        saved = R.act_qdq_fp8
        try:
            R.act_qdq_fp8 = lambda x, block=32: x.to(torch.bfloat16)               # engine act-quant off
            self.assertFalse(R._act_qdq_fusable(bf, w))
        finally:
            R.act_qdq_fp8 = saved
        with self.assertRaises(ValueError):
            F8.fp8_linear(bf.float(), w, act_qdq=True)
        # prefill-sized input keeps the torch spelling: fused and unfused agree trivially
        x = torch.randn(64, 5120, device="cuda").to(torch.bfloat16)
        ref, got = self._both(lambda: R.qlinear(x, w))
        self.assertTrue(torch.equal(got, ref))

    def test_merged_and_shared_expert(self):
        w1, w3 = self._weight(1152, 5120, 11), self._weight(1152, 5120, 12)
        w2 = self._weight(5120, 1152, 13)
        both, _ = F8.concat_rows(w1, w3)
        for m in ROWS:
            x = (torch.randn(m, 5120, device="cuda") * 0.5).to(torch.bfloat16)
            R.ACT_QDQ_FUSED = False
            ref = R.expert_ffn(x, w1, w2, w3, 10.0)
            R.ACT_QDQ_FUSED = True
            for w13 in (None, both):
                got = R.expert_ffn(x, w1, w2, w3, 10.0, w13=w13)
                with self.subTest(m=m, merged=w13 is not None):
                    self.assertTrue(torch.equal(got, ref))

    def test_tp_wrappers(self):
        import engine.tensor_parallel as TP
        saved = TP.dist
        TP.dist = _StubDist(input_width=4096)
        try:
            row = TP.RowParallelWeight(self._weight(5120, 1152, 21))          # shared w2, K shard
            # wo_b under TP2: local [2560, 8192]; each rank passes its 4096-wide half of the input
            out = TP.OutputParallelWeight(self._weight(2560, 8192, 22), 2)
            for m in ROWS:
                for wrapper, k in ((row, 1152), (out, 4096)):
                    x = _hard_x(m, k, 70 + m)
                    R.ACT_QDQ_FUSED = True
                    self.assertTrue(R._act_qdq_fusable(x, wrapper))
                    ref, got = self._both(lambda: R.qlinear(x, wrapper))
                    with self.subTest(m=m, wrapper=type(wrapper).__name__):
                        self.assertTrue(torch.equal(got, ref))
        finally:
            TP.dist = saved

    def test_graph_capture(self):
        w = self._weight(1792, 5120, 31)
        x = (torch.randn(4, 5120, device="cuda") * 0.5).to(torch.bfloat16)
        R.ACT_QDQ_FUSED = True
        eager = R.qlinear(x, w)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            R.qlinear(x, w)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = R.qlinear(x, w)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, eager))


if __name__ == "__main__":
    unittest.main()
