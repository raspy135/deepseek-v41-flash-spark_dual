"""DSV41_ABLIT_WOB: the abliteration overlay loader (engine/ablit.py) on a synthetic overlay.

The real overlay (`drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed`,
`wo_b_l10_35.safetensors`, ~1.1 GB, gated) holds the native checkpoint's 52 `layers.{10..35}.attn.wo_b`
tensors. This builds the same file synthetically and checks that the loader substitutes exactly those,
passes everything else through, and refuses a file whose key set is not exactly the 52.

    python3 -m unittest tools.test_ablit        (CPU)
"""
from __future__ import annotations

import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
from engine import ablit  # noqa: E402

# the native checkpoint's wo_b storage: fp8 e4m3 [5120, 8192] + e8m0 block scales [160, 256]
SHAPES = {"weight": (5120, 8192), "scale": (160, 256)}


def make_overlay(path, drop=None, extra=False):
    tens = {}
    for name in ablit.expected():
        if name == drop:
            continue
        shape = SHAPES[name.rsplit(".", 1)[1]]
        tens[name] = (torch.zeros(shape, dtype=torch.float8_e4m3fn)
                      if name.endswith(".weight") else torch.full(shape, 127, dtype=torch.float8_e8m0fnu))
    if extra:
        tens["layers.10.attn.wq_b.weight"] = torch.zeros(1, dtype=torch.float8_e4m3fn)
    save_file(tens, path)


class AblitOverlay(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get("DSV41_ABLIT_WOB")
        os.environ.pop("DSV41_ABLIT_WOB", None)

    def tearDown(self):
        if self._env is None:
            os.environ.pop("DSV41_ABLIT_WOB", None)
        else:
            os.environ["DSV41_ABLIT_WOB"] = self._env

    def test_off_by_default(self):
        self.assertFalse(ablit.enabled())
        self.assertEqual(ablit.digest(), "")
        sentinel = lambda name: torch.zeros(1)
        self.assertIs(ablit.loader(sentinel), sentinel, "off must not wrap the loader")

    def test_substitutes_only_the_overlay(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "wo_b.safetensors")
            make_overlay(p)
            os.environ["DSV41_ABLIT_WOB"] = p
            self.assertTrue(ablit.enabled())
            self.assertEqual(len(ablit.digest()), 16)
            got = {}
            stock = lambda name: torch.full((1,), 9.0)
            g = ablit.loader(stock)
            for name in ablit.expected():
                got[name] = g(name)
            self.assertTrue(torch.equal(got["layers.10.attn.wo_b.weight"],
                                        torch.zeros(SHAPES["weight"], dtype=torch.float8_e4m3fn)))
            self.assertTrue(torch.equal(got["layers.35.attn.wo_b.scale"],
                                        torch.full(SHAPES["scale"], 127, dtype=torch.float8_e8m0fnu)))
            # layers outside 10..35 and every other tensor come from the checkpoint
            self.assertTrue(torch.equal(g("layers.9.attn.wo_b.weight"), stock("x")))
            self.assertTrue(torch.equal(g("layers.36.attn.wo_b.weight"), stock("x")))
            self.assertTrue(torch.equal(g("layers.10.attn.wq_b.weight"), stock("x")))
            self.assertTrue(torch.equal(g("head.weight"), stock("x")))

    def test_rejects_a_wrong_key_set(self):
        stock = lambda name: torch.zeros(1)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "missing.safetensors")
            make_overlay(p, drop="layers.20.attn.wo_b.scale")
            os.environ["DSV41_ABLIT_WOB"] = p
            with self.assertRaisesRegex(RuntimeError, "not a native wo_b overlay"):
                ablit.loader(stock)
            q = os.path.join(d, "extra.safetensors")
            make_overlay(q, extra=True)
            os.environ["DSV41_ABLIT_WOB"] = q
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                ablit.loader(stock)

    def test_missing_file_is_fatal(self):
        os.environ["DSV41_ABLIT_WOB"] = "/nonexistent/wo_b.safetensors"
        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            ablit.loader(lambda name: torch.zeros(1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
