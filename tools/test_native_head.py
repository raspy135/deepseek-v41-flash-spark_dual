"""Config/dispatch checks and the explicit-layout head arithmetic regression.

The CUDA gate uses tiny synthetic weights only, no checkpoint or throughput
benchmark. The cancellation case distinguishes native kWidth2 from the former
byte-derived kWidth4 despite bit-exact reconstructed BF16 weights.
"""
import ast
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import v41_ref as R
from engine.native_head import head_kernel, make_packed_head, validate_config


@triton.jit
def _native_raw(X, W, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                BN: tl.constexpr = 32, BK: tl.constexpr = 128):
    """Existing native TC geometry, retaining FP32 before its final BF16 round."""
    ns = tl.program_id(0)*BN + tl.arange(0, BN)
    part = tl.program_id(1)
    ms, ks = tl.arange(0, 16), tl.arange(0, BK)
    acc = tl.zeros((16, BN), tl.float32)
    for start in range(part*K, tl.minimum((part+1)*K, K), BK):
        x = tl.load(X + ms[:, None]*K + start + ks[None, :],
                    (ms[:, None] < M) & (start+ks[None, :] < K), 0)
        w = tl.load(W + ns[None, :]*K + start + ks[:, None],
                    (ns[None, :] < N) & (start+ks[:, None] < K), 0)
        acc = tl.dot(x, w, acc)
    tl.store(OUT + ms[:, None]*N + ns[None, :], acc,
              (ms[:, None] < M) & (ns[None, :] < N))


class NativeHeadCPU(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {
            'DSV41_HEAD_KERNEL': 'off', 'DSV41_HEAD_FMT': 'bf16',
            'DSV41_HEAD_FP32': '0', 'DSV41_DRAFT_HEAD_FMT': 'off',
            'DSV41_TP_DRAFT_HEAD': '0',
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_default_off_does_not_pack_or_import_gluon(self):
        os.environ.pop('DSV41_HEAD_KERNEL')
        weight = torch.randn(7, 128).bfloat16()
        with mock.patch('engine.native_head.make_packed_head') as pack:
            with mock.patch.dict(sys.modules, {'head_packed_gluon': None}):
                self.assertIs(R.make_head(weight), weight)
        pack.assert_not_called()
        self.assertEqual(head_kernel(), 'off')

    def test_config_rejects_unsupported_combinations_before_pack(self):
        os.environ['DSV41_HEAD_KERNEL'] = ' PACKED '
        for draft in ('off', 'bf16'):
            os.environ['DSV41_DRAFT_HEAD_FMT'] = draft
            self.assertEqual(validate_config(), 'packed')
        for key, value in (('DSV41_HEAD_FMT', 'fp8'), ('DSV41_HEAD_FMT', 'fp4'),
                           ('DSV41_HEAD_FP32', '1'), ('DSV41_DRAFT_HEAD_FMT', 'fp8')):
            with self.subTest(key=key, value=value):
                with mock.patch.dict(os.environ, {key: value}):
                    with mock.patch('engine.native_head.make_packed_head') as pack:
                        with self.assertRaises(ValueError): R.make_head(torch.ones(3, 128))
                    pack.assert_not_called()
        os.environ['DSV41_HEAD_KERNEL'] = 'unknown'
        with self.assertRaises(ValueError): validate_config()

    def test_load_owns_only_packed_storage_and_reuses_main_for_draft(self):
        from engine.tensor_parallel import VocabParallelHead, draft_head_bytes, make_tp_draft_head
        os.environ['DSV41_HEAD_KERNEL'] = 'packed'
        weight = torch.randn(7, 256).bfloat16()
        original = weight.view(torch.int16).clone()
        packed = R.make_head(weight)
        self.assertTrue(hasattr(packed, 'native_logits'))
        self.assertLess(packed.stored_bytes, weight.numel()*weight.element_size())
        self.assertFalse(any(value is weight for value in vars(packed).values()))
        weight.zero_()
        self.assertTrue(torch.equal(packed.dequant().view(torch.int16), original))
        self.assertIsNone(R.make_draft_head(packed))
        os.environ['DSV41_TP_DRAFT_HEAD'] = '1'
        tp = VocabParallelHead(packed, 2)
        self.assertIsNone(make_tp_draft_head(tp))
        self.assertEqual(draft_head_bytes(tp), packed.stored_bytes)

    def test_head_dense_and_mm_dispatch_preserve_output_dtypes(self):
        class Head:
            def native_logits(self, x):
                return torch.full((*x.shape[:-1], 7), .375, dtype=torch.float32)
        x = torch.zeros(2, 3, 128, dtype=torch.bfloat16)
        self.assertEqual(R.head_logits(x, Head()).dtype, torch.float32)
        for fn in (R.dense, R.mm):
            value = fn(x, Head())
            self.assertEqual(value.dtype, torch.bfloat16)
            self.assertEqual(value.shape, (2, 3, 7))
            self.assertTrue(torch.equal(value, torch.full_like(value, .375)))

    def test_empty_leading_dimensions_scalar_and_width_validation(self):
        packed = make_packed_head(torch.ones(7, 128, dtype=torch.bfloat16))
        for shape in ((0, 128), (2, 0, 128), (0, 2, 128)):
            with self.subTest(shape=shape):
                out = packed.native_logits(torch.empty(shape))
                self.assertEqual(out.shape, (*shape[:-1], 7))
                self.assertEqual(out.dtype, torch.float32)
        for bad in (torch.tensor(1.), torch.ones(3, 127)):
            with self.assertRaises(ValueError): packed.native_logits(bad)

    def test_three_dimensional_prefill_uses_bounded_dequant_dispatch(self):
        torch.manual_seed(1086)
        weight = torch.randn(7, 128).bfloat16()
        packed = make_packed_head(weight)
        x = torch.randn(2, 20, 128).transpose(0, 1)
        with mock.patch.object(packed, 'dequant_rows', wraps=packed.dequant_rows) as decode:
            out = packed.native_logits(x)
        decode.assert_called_once_with(0, 7)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.shape, (20, 2, 7))
        self.assertTrue(torch.equal(out, F.linear(x.bfloat16(), weight).float()))

    def test_pair_boot_guard_pins_kernel_and_version(self):
        tree = ast.parse((HERE.parent/'engine/v41_engine.py').read_text())
        configs = [node for node in ast.walk(tree) if isinstance(node, ast.Dict)
                   and any(isinstance(key, ast.Constant) and key.value == 'head_kernel_version'
                           for key in node.keys)]
        self.assertEqual(len(configs), 1)
        fields = {key.value: value for key, value in zip(configs[0].keys, configs[0].values)
                  if isinstance(key, ast.Constant)}
        self.assertIsInstance(fields['head_kernel'], ast.Call)
        self.assertEqual(fields['head_kernel'].func.id, 'head_kernel')
        self.assertEqual(fields['head_kernel_version'].id, 'HEAD_KERNEL_VERSION')


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class NativeHeadCUDA(unittest.TestCase):
    def setUp(self):
        self.old_precision = (torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_tf32)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(self.restore_precision)

    def restore_precision(self):
        (torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
         torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
         torch.backends.cuda.matmul.allow_tf32) = self.old_precision

    def test_explicit_geometry_cancellation_matches_native_and_cublas(self):
        from head_packed_gluon import project
        from head_packed_tiles import project as inferred_project
        pattern = torch.tensor([2**25, 1, -(2**25), 1]+[0]*12,
                               device='cuda', dtype=torch.bfloat16)
        weight = pattern.repeat(64, 320)
        x = torch.ones(16, 5120, device='cuda', dtype=torch.bfloat16)
        packed = make_packed_head(weight)
        expected = torch.full((16, 64), 640., device='cuda')
        self.assertTrue(torch.equal(packed.dequant().view(torch.int16), weight.view(torch.int16)))
        self.assertTrue(torch.equal(F.linear(x, weight).float(), expected))
        self.assertTrue(torch.equal(project(x, packed, bn=32, raw=True), expected))
        self.assertTrue(torch.equal(packed.native_logits(x), expected))
        self.assertTrue(torch.equal(inferred_project(x, packed), torch.zeros_like(expected)),
                        'negative control no longer exposes the known layout error')

    def test_raw_fp32_odd_rows_vocabulary_tail_and_captured_replay(self):
        from head_packed_gluon import project
        from head_native import project as native_project
        torch.manual_seed(915)
        weight = (torch.randn(39, 5120, device='cuda', dtype=torch.bfloat16)*.02).contiguous()
        x = torch.randn(16, 5120, device='cuda', dtype=torch.bfloat16)
        packed = make_packed_head(weight)
        expected = torch.empty(16, 39, device='cuda')
        _native_raw[(triton.cdiv(39,32),1)](x, weight, expected, 16, 39, 5120,
                                          num_warps=4, num_stages=3)
        rounded = native_project(x, weight, bn=32)
        self.assertTrue(torch.equal(expected.bfloat16().float(), rounded))
        for rows in (1, 2, 3, 4, 5, 6, 16):
            with self.subTest(rows=rows):
                self.assertTrue(torch.equal(project(x[:rows], packed, bn=32, raw=True), expected[:rows]))
                self.assertTrue(torch.equal(packed.native_logits(x[:rows]), rounded[:rows]))
        shaped = x[:6].reshape(2, 3, 5120).transpose(0,1)
        self.assertTrue(torch.equal(packed.native_logits(shaped), rounded[:6].reshape(2,3,39).transpose(0,1)))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = packed.native_logits(x[:5])
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay(); torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, rounded[:5]))
        x[:5].zero_()
        graph.replay(); torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))


if __name__ == '__main__':
    unittest.main()
