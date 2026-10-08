"""Tiled BF16 layout checks; CPU math first, optional CUDA bit/projection gate."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from head_packed_tiles import PackedHead, project


class TiledHeadCPU(unittest.TestCase):
    def test_all_65536_bf16_bit_patterns(self):
        codes = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(16, 4096)
        packed = PackedHead(codes.view(torch.bfloat16))
        self.assertTrue(torch.equal(codes, packed.dequant().view(torch.int16)))
        self.assertEqual(packed.escape_groups, 0)
        self.assertEqual(packed.low.shape, (32, 16, 128))
        self.assertEqual(packed.delta.shape, (32, 16, 64))
        self.assertEqual(packed.header.shape, (32, 16))

    def test_exponent_escapes_follow_tiled_group_order(self):
        exponent = torch.full((3, 2, 128), 128, dtype=torch.int32)
        exponent[0, 0, 0] = 113  # delta15 fits
        exponent[0, 1, 0] = 112  # delta16 escapes
        exponent[1, 0, 0] = 0; exponent[1, 0, 1] = 255
        exponent[2, 1, 0] = 0
        low = torch.arange(768, dtype=torch.int32).reshape(3, 2, 128) & 255
        codes = ((exponent << 7) | (low & 127) | ((low & 128) << 8)).reshape(3, 256).to(torch.int16)
        packed = PackedHead(codes.view(torch.bfloat16))
        self.assertEqual(packed.escape_groups, 3)
        self.assertEqual((packed.header >> 8).tolist(), [[0, 1, 0], [2, 0, 3]])
        self.assertTrue(torch.equal(codes, packed.dequant().view(torch.int16)))
        self.assertTrue(torch.equal(codes[1:3], packed.dequant_rows(1, 3).view(torch.int16)))

    def test_non_aligned_vocabulary_and_every_low_byte_address(self):
        generator = torch.Generator().manual_seed(809)
        codes = torch.randint(-32768, 32768, (17, 256), generator=generator, dtype=torch.int16)
        packed = PackedHead(codes.view(torch.bfloat16))
        row = torch.arange(17)[:, None]
        ks = torch.arange(256)[None, :]
        offset = ((ks // 128) * 17 + row) * 128 + ks % 128
        expected = ((codes.to(torch.int32) & 127) | ((codes.to(torch.int32) >> 8) & 128)).to(torch.uint8)
        self.assertTrue(torch.equal(packed.low.flatten()[offset], expected))
        self.assertTrue(torch.equal(codes, packed.dequant().view(torch.int16)))
        self.assertEqual(packed.dequant_rows(17, 17).shape, (0, 256))

    def test_original_is_not_retained_or_aliased(self):
        weight = torch.randn(7, 256).bfloat16()
        codes = weight.view(torch.int16).clone()
        packed = PackedHead(weight)
        self.assertFalse(any(value is weight for value in vars(packed).values()))
        self.assertFalse(any(isinstance(value, torch.Tensor) and value.data_ptr() == weight.data_ptr()
                             for value in vars(packed).values()))
        weight.zero_()
        self.assertTrue(torch.equal(codes, packed.dequant().view(torch.int16)))

    def test_word_pointer_and_shift_math_preserves_every_code(self):
        codes = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(16, 4096)
        packed = PackedHead(codes.view(torch.bfloat16))
        row, ks = torch.arange(16)[:, None], torch.arange(4096)[None, :]
        group = (ks // 128) * 16 + row
        low = packed.low.view(torch.uint16).flatten().to(torch.int32)[group * 64 + (ks % 128) // 2]
        low = (low >> ((ks & 1) * 8)) & 255
        delta = packed.delta.view(torch.uint16).flatten().to(torch.int32)[group * 32 + (ks % 128) // 4]
        delta = (delta >> ((ks & 3) * 4)) & 15
        header = packed.header.flatten()[group]
        exponent = (header & 255) - delta
        actual = ((low & 127) | ((low & 128) << 8) | (exponent << 7)).to(torch.int16)
        self.assertTrue(torch.equal(actual, codes))
        # Nonzero escapes use the same two-byte load and parity shift.
        escaped = torch.arange(256, dtype=torch.uint8).reshape(2, 128)
        ep = escaped.view(torch.uint16).to(torch.int32)
        at = torch.arange(128)
        self.assertTrue(torch.equal(((ep[:, at // 2] >> ((at & 1) * 8)) & 255).byte(), escaped))


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class TiledHeadCUDA(unittest.TestCase):
    def test_all_65536_gpu_bf16_bits(self):
        codes = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16).reshape(16, 4096)
        packed = PackedHead(codes.view(torch.bfloat16))
        self.assertTrue(torch.equal(codes, packed.dequant().view(torch.int16)))
        self.assertEqual(packed.escape_groups, 0)

    def test_full_bits_escapes_and_chunk_decode(self):
        generator = torch.Generator().manual_seed(819)
        cpu = torch.randint(-32768, 32768, (17, 512), generator=generator, dtype=torch.int16)
        weight = cpu.view(torch.bfloat16).cuda()
        packed = PackedHead(weight)
        self.assertTrue(torch.equal(weight.view(torch.int16), packed.dequant().view(torch.int16)))
        for first, last in ((0, 1), (1, 7), (7, 17), (17, 17)):
            self.assertTrue(torch.equal(weight[first:last].view(torch.int16), packed.dequant_rows(first, last).view(torch.int16)))
        expected = PackedHead(cpu.view(torch.bfloat16))
        for name in ('low', 'delta', 'header', 'escape'):
            self.assertTrue(torch.equal(getattr(packed, name).cpu(), getattr(expected, name)))

    def test_project_row_invariance_and_graph(self):
        from head_native import project as native_project
        torch.manual_seed(821)
        weight = (torch.randn(256, 5120, device='cuda') * .02).bfloat16()
        x = torch.randn(16, 5120, device='cuda', dtype=torch.bfloat16)
        packed = PackedHead(weight)
        reference = native_project(x, weight)
        full = project(x, packed)
        for rows in (1, 2, 4, 6, 16):
            actual = project(x[:rows], packed)
            self.assertTrue(torch.equal(actual, full[:rows]))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = project(x[:4], packed)
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay(); torch.cuda.synchronize()
        self.assertTrue(torch.equal(output, full[:4]))

    def test_word_load_cancellation_then_native_row_and_graph_equivalence(self):
        from head_native import project as native_project
        pattern = torch.zeros(16, device='cuda', dtype=torch.bfloat16)
        pattern[:4] = torch.tensor([2**25, 1, -2**25, 1], device='cuda', dtype=torch.bfloat16)
        weight = pattern.repeat(64, 320)
        x = torch.ones(16, 5120, device='cuda', dtype=torch.bfloat16)
        packed = PackedHead(weight)
        reference = native_project(x, weight)
        word = project(x, packed, word_loads=True)
        byte = project(x, packed)
        print('TILED_WORD_CANCELLATION', {'native': float(reference[0, 0]),
            'byte': float(byte[0, 0]), 'word': float(word[0, 0])}, flush=True)
        self.assertTrue(torch.equal(reference, torch.full_like(reference, 640)))
        self.assertTrue(torch.equal(word, reference), 'word-load dot still changes native cancellation')
        torch.manual_seed(825)
        weight = (torch.randn(256, 5120, device='cuda') * .02).bfloat16()
        x = torch.randn(16, 5120, device='cuda', dtype=torch.bfloat16)
        packed = PackedHead(weight)
        reference = native_project(x, weight)
        for rows in (1, 2, 4, 6, 16):
            self.assertTrue(torch.equal(project(x[:rows], packed, word_loads=True), reference[:rows]))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = project(x[:4], packed, word_loads=True)
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay(); torch.cuda.synchronize()
        self.assertTrue(torch.equal(output, reference[:4]))


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    unittest.main()
