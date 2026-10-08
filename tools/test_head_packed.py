"""GPU qualification of lossless BF16 head bits and captured projection.

Run before any real-weight timing: python tools/test_head_packed.py.
"""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from head_packed import PackedHead, project
from head_native import project as native_project


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class PackedHeadTests(unittest.TestCase):
    def test_every_bf16_bit_pattern_including_nonfinite_and_signed_zero(self):
        codes = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16).reshape(16, 4096)
        weight = codes.view(torch.bfloat16)
        packed = PackedHead(weight)
        self.assertTrue(torch.equal(codes, packed.dequant().view(torch.int16)))
        self.assertEqual(packed.escape_groups, 0)

    def test_group_boundary_delta_fifteen_and_escape_sixteen(self):
        # Three exponent groups: delta15 fits, delta16 escapes, and zero/Inf
        # need an escape. Preserve signs and all mantissas at the same time.
        exponent = torch.full((3, 128), 128, device='cuda', dtype=torch.int32)
        exponent[0, 0] = 113
        exponent[1, 0] = 112
        exponent[2, 0] = 0
        exponent[2, 1] = 255
        low = torch.arange(384, device='cuda', dtype=torch.int32).reshape(3, 128) & 255
        codes = ((exponent << 7) | (low & 127) | ((low & 128) << 8)).to(torch.int16)
        packed = PackedHead(codes.view(torch.bfloat16))
        self.assertEqual(packed.escape_groups, 2)
        self.assertTrue(torch.equal(codes, packed.dequant().view(torch.int16)))

    def test_projection_matches_native_tc_and_is_graph_safe(self):
        torch.manual_seed(108)
        weight = (torch.randn(128, 5120, device='cuda')*.02).to(torch.bfloat16)
        original = weight.view(torch.int16).clone()
        packed = PackedHead(weight)
        self.assertTrue(torch.equal(original, packed.dequant().view(torch.int16)))
        x = torch.randn(16, 5120, device='cuda', dtype=torch.bfloat16)
        reference = native_project(x, weight)
        for rows in (1, 2, 4, 6, 16):
            self.assertTrue(torch.equal(project(x[:rows], packed), reference[:rows]))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = project(x[:4], packed)
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(result, reference[:4]))
        self.assertTrue(torch.equal(weight.view(torch.int16), original))

    def test_requires_prepare_outside_capture(self):
        weight = torch.empty(16, 128, device='cuda', dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            with self.assertRaisesRegex(RuntimeError, 'outside graph capture'):
                PackedHead(weight)


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    unittest.main()
