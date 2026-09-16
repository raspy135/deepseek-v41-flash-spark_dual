"""Check HC padding selection and single-token/verify row invariance on CUDA."""
import sys
sys.path[:0] = ['/app', '/app/tools']
import torch
import torch.nn.functional as F
import v41_ref as R


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    generator = torch.Generator(device='cuda').manual_seed(20260916)
    R.MM_TILE = 16
    w = torch.randn(24, 20480, device='cuda', generator=generator) * .01
    x = torch.randn(32, 20480, device='cuda', generator=generator, dtype=torch.bfloat16).float()
    tests = 0
    for tile in (16, 32):
        R.HC_MM_TILE = tile
        full = R.mm(x[:16], w)
        for rows in (1, 2, 4, 8, 16):
            actual = R.mm(x[:rows], w)
            expected = F.linear(F.pad(x[:rows], (0, 0, 0, tile-rows)), w)[:rows]
            assert torch.equal(actual, expected), (tile, rows, 'padding')
            assert torch.equal(actual, full[:rows]), (tile, rows, 'row invariance')
            tests += 1
        for rows in (17, 32):
            assert torch.equal(R.mm(x[:rows], w), F.linear(x[:rows], w))
            tests += 1
        # Expert router/non-HC projections keep the original fixed-16 shape.
        other = w[:12]
        assert torch.equal(R.mm(x[:4], other), F.linear(F.pad(x[:4], (0, 0, 0, 12)), other)[:4])
        tests += 1
    print(f'HC_MM_PASS {tests} checks')


if __name__ == '__main__':
    main()
