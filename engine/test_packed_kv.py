"""CUDA storage equivalence and graph replay tests; no model weights required."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import v41_ref as R
from engine.packed_kv import write, gather


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class PackedKVTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def check(self, x):
        packed = torch.zeros(x.shape[0] + 7, 36, dtype=torch.int64, device='cuda')
        ids = torch.arange(x.shape[0], device='cuda') + 3
        write(packed, x, ids)
        expected = R.fp4_qdq(x, 16, 'e4m3')
        actual = gather(packed, ids)
        self.assertTrue(torch.equal(actual.view(torch.int16), expected.view(torch.int16)),
                        f'bit mismatches: {(actual.view(torch.int16) != expected.view(torch.int16)).sum().item()}')
        indexed = ids[torch.randint(len(ids), (4, 512), device='cuda')]
        self.assertTrue(torch.equal(gather(packed, indexed), expected[indexed - 3]))
        write(packed, x, 3)
        self.assertTrue(torch.equal(gather(packed, ids).view(torch.int16), expected.view(torch.int16)))
        self.assertEqual(packed.element_size() * packed.shape[1], 288)

    def test_random_scales(self):
        for scale in (1e-7, .01, 1., 16., 100.):
            self.check((torch.randn(257, 512, device='cuda') * scale).to(torch.bfloat16))

    def test_midpoints_zeros_and_scale_rounding(self):
        x = torch.tensor([0., -.0, .25, -.25, .75, -.75, 1.25, -1.25,
                          1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 5., 6.],
                         device='cuda', dtype=torch.bfloat16).repeat(256, 32)
        x *= torch.logspace(-6, 2, 256, device='cuda')[:, None]
        self.check(x)

    def test_graph_dynamic_positions(self):
        x = torch.randn(4, 512, device='cuda', dtype=torch.bfloat16)
        packed = torch.zeros(64, 36, dtype=torch.int64, device='cuda')
        ids = torch.arange(4, device='cuda')
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            write(packed, x, ids)
            gather(packed, ids)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            write(packed, x, ids)
            result = gather(packed, ids)
        for n in (0, 9, 27):
            x.normal_()
            ids.copy_(torch.arange(4, device='cuda') + n)
            graph.replay()
            self.assertTrue(torch.equal(result.view(torch.int16),
                                       R.fp4_qdq(x, 16, 'e4m3').view(torch.int16)))


if __name__ == '__main__':
    unittest.main()
