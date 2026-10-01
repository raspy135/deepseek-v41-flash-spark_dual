"""The drafter's own cheaper head (DSV41_DRAFT_HEAD_FMT) must not touch the verifier's.

    python3 tools/test_draft_head.py        (GPU)

A random bf16 head stands in for head.weight. The test pins the wiring that matters:

  * the decision table -- off / bf16 / already-quantized / FP32 reference reuse the verifier's
    head; fp8 and fp4 build a second, smaller object only when they are cheaper;
  * building the draft head leaves the verifier's tensor bit-identical (`_final` keeps reading
    it, so the accepted tokens cannot move);
  * `head_logits` runs on the draft head at the draft's row count and returns fp32.

Whether the quantized draft moves the accepted length is an end-to-end question; that is what the
two-node A/B answers, not this.
"""
from __future__ import annotations

import os
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import v41_ref as R  # noqa: E402

DEV = "cuda"
VOCAB, K = 1024, 512


def _head():
    g = torch.Generator(device="cpu").manual_seed(11)
    # a small spread so the fp8/fp4 scales are not all one bucket
    return (torch.randn(VOCAB, K, generator=g) * 0.05).bfloat16().to(DEV)


class DraftHead(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.get(k) for k in ("DSV41_HEAD_FMT", "DSV41_DRAFT_HEAD_FMT", "DSV41_HEAD_FP32")}
        for k in self._env:
            os.environ.pop(k, None)
        if R.quantize_to_fp8 is None or R.quantize_to_fp4 is None:
            self.skipTest("tools/fp8_linear.py or tools/fp4_linear.py not importable")

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_decision_table(self):
        head = R.make_head(_head())  # HEAD_FMT default bf16 -> a plain bf16 tensor
        self.assertIsInstance(head, torch.Tensor)

        os.environ["DSV41_DRAFT_HEAD_FMT"] = "off"
        self.assertIsNone(R.make_draft_head(head))
        os.environ["DSV41_DRAFT_HEAD_FMT"] = "bf16"
        self.assertIsNone(R.make_draft_head(head), "a bf16 draft head is not cheaper than the bf16 verifier head")
        os.environ["DSV41_DRAFT_HEAD_FMT"] = "fp8"
        self.assertIsInstance(R.make_draft_head(head), R.FP8Weight)
        os.environ["DSV41_DRAFT_HEAD_FMT"] = "fp4"
        self.assertIsInstance(R.make_draft_head(head), R.FP4Weight)

        # a quantized main head has no cheap bf16 source to copy from
        os.environ["DSV41_DRAFT_HEAD_FMT"] = "fp8"
        os.environ["DSV41_HEAD_FMT"] = "fp8"
        self.assertIsNone(R.make_draft_head(R.make_head(_head())))

        # fp4 verifier, fp8 draft would be bigger, so it must be refused
        os.environ["DSV41_DRAFT_HEAD_FMT"] = "fp8"
        os.environ["DSV41_HEAD_FMT"] = "fp4"
        self.assertIsNone(R.make_draft_head(R.make_head(_head())))

        # the FP32 reference path is left alone
        os.environ.pop("DSV41_HEAD_FMT")
        os.environ["DSV41_DRAFT_HEAD_FMT"] = "fp8"
        os.environ["DSV41_HEAD_FP32"] = "1"
        self.assertIsNone(R.make_draft_head(R.make_head(_head())))

    def test_verifier_tensor_untouched(self):
        src = _head()
        before = src.clone()
        os.environ["DSV41_DRAFT_HEAD_FMT"] = "fp8"
        verifier = R.make_head(src)
        self.assertIs(verifier, src, "bf16 make_head should not copy")
        draft = R.make_draft_head(verifier)
        self.assertTrue(torch.equal(verifier, before), "building the draft head mutated the verifier")
        self.assertIsNot(draft, verifier)
        # e4m3 codes are one byte per element against bf16's two
        self.assertLess(draft.w.element_size(), verifier.element_size())

    def test_head_logits_on_draft_head(self):
        """The DSpark graph runs the head at T_DRAFT rows; the quantized kernel must serve it."""
        x = torch.randn(5, K).bfloat16().to(DEV)  # T_DRAFT = 5
        ref = R.head_logits(x, R.make_head(_head()))
        os.environ["DSV41_DRAFT_HEAD_FMT"] = "fp8"
        draft = R.make_draft_head(R.make_head(_head()))
        got = R.head_logits(x, draft)
        self.assertEqual(got.shape, (x.size(0), VOCAB))
        self.assertEqual(got.dtype, torch.float32)
        self.assertTrue(torch.isfinite(got).all())
        # a quantized head is a different function, but it must still point the same way
        cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0)
        self.assertGreater(float(cos), 0.8, cos)

    def test_bad_format_rejected(self):
        os.environ["DSV41_DRAFT_HEAD_FMT"] = "int8"
        with self.assertRaises(ValueError):
            R.draft_head_fmt()


if __name__ == "__main__":
    unittest.main(verbosity=2)
