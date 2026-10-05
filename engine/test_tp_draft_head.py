"""TP draft wiring and byte accounting without loading weights or requiring CUDA."""
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from engine.tensor_parallel import (VocabParallelHead, draft_head_bytes,
                                   make_tp_draft_head, tp_draft_head_enabled)


class TPDraftHeadTests(unittest.TestCase):
    def setUp(self):
        self.local = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64).bfloat16()
        self.head = VocabParallelHead(self.local, 2)

    @patch.dict(os.environ, {}, clear=True)
    def test_default_off_never_quantizes(self):
        with patch.dict(sys.modules, {'v41_ref': SimpleNamespace(
                make_draft_head=lambda _: self.fail('default-off must not quantize'))}):
            self.assertFalse(tp_draft_head_enabled())
            self.assertIsNone(make_tp_draft_head(self.head))
            self.assertEqual(draft_head_bytes(None), 0)

    @patch.dict(os.environ, {'DSV41_TP_DRAFT_HEAD': '1'})
    def test_wraps_only_separate_local_shard_without_mutating_verifier(self):
        before = self.local.clone()
        quant = SimpleNamespace(w=torch.zeros(32, 64, dtype=torch.uint8),
                                s=torch.zeros(1, 2, dtype=torch.uint8), shape=(32, 64))
        seen = []
        def build(local):
            seen.append(local)
            return quant
        with patch.dict(sys.modules, {'v41_ref': SimpleNamespace(make_draft_head=build)}):
            draft = make_tp_draft_head(self.head)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], self.local)
        self.assertIs(self.head.local, self.local)
        self.assertTrue(torch.equal(self.local.view(torch.int16), before.view(torch.int16)))
        self.assertIs(draft.local, quant)
        self.assertEqual((draft.world, draft.shape), (2, (64, 64)))
        self.assertEqual(draft_head_bytes(draft), 32 * 64 + 2)
        self.assertEqual(draft_head_bytes(self.head), 32 * 64 * 2)

    @patch.dict(os.environ, {'DSV41_TP_DRAFT_HEAD': '1'})
    def test_load_mtp_false_never_quantizes(self):
        with patch.dict(sys.modules, {'v41_ref': SimpleNamespace(
                make_draft_head=lambda _: self.fail('non-draft diagnostics must not allocate'))}):
            self.assertIsNone(make_tp_draft_head(self.head, load_mtp=False))

    @patch.dict(os.environ, {'DSV41_TP_DRAFT_HEAD': '1'})
    def test_reuses_verifier_when_format_builder_declines(self):
        with patch.dict(sys.modules, {'v41_ref': SimpleNamespace(make_draft_head=lambda _: None)}):
            self.assertIsNone(make_tp_draft_head(self.head))

    def test_rejects_invalid_enable_setting(self):
        for value in ('yes', 'true', '2', ''):
            with self.subTest(value=value), patch.dict(os.environ, {'DSV41_TP_DRAFT_HEAD': value}):
                with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                    make_tp_draft_head(self.head)

    @patch.dict(os.environ, {'DSV41_TP_DRAFT_HEAD': '1'})
    def test_rejects_non_tp_verifier(self):
        with self.assertRaisesRegex(TypeError, 'VocabParallelHead'):
            make_tp_draft_head(self.local)


if __name__ == '__main__':
    unittest.main()
