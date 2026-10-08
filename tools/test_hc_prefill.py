"""hc_prefill.hc_front against the torch path it replaces: error vs an FP64 reference, row
invariance (a row's bits do not depend on the chunk it is in), and timing at a 2,048-row chunk.

    python tools/test_hc_prefill.py
"""
import sys
import time

sys.path[:0] = ['/app', '/app/tools', __file__.rsplit('/', 1)[0]]
import torch
import v41_ref as R
from hc_prefill import hc_front

EPS = 1e-6


def old(x, w):
    xf = x.float()
    return R.mm(xf, w) * R.rms_rsqrt(xf, EPS)


def bench(fn, n=20):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def main():
    R.MM_TILE = 16
    g = torch.Generator(device='cuda').manual_seed(0)
    ok = True
    for T in (17, 128, 512, 2048):
        x = (torch.randn(T, 20480, generator=g, device='cuda') * 3).bfloat16()
        w = torch.randn(24, 20480, generator=g, device='cuda') * 0.01
        ref = (x.double() @ w.double().t()) * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + EPS)
        a, b = old(x, w), hc_front(x, w, EPS)
        ea = ((a.double() - ref).abs().max() / ref.abs().max()).item()
        eb = ((b.double() - ref).abs().max() / ref.abs().max()).item()
        # row invariance: the last 17 rows alone, and the whole call shifted by 5 rows
        inv = torch.equal(hc_front(x[-17:], w, EPS), b[-17:]) and torch.equal(hc_front(x[5:], w, EPS), b[5:])
        print(f'T={T:5d} max rel err vs fp64: torch {ea:.2e}  fused {eb:.2e}  '
              f'fused vs torch {((a - b).abs().max() / a.abs().max()).item():.2e}  row-invariant={inv}')
        ok &= inv and eb <= max(4 * ea, 1e-6)
    x = (torch.randn(2048, 20480, generator=g, device='cuda') * 3).bfloat16()
    w = torch.randn(24, 20480, generator=g, device='cuda') * 0.01
    for bm, bk, nw in ((64, 64, 4), (32, 64, 4), (64, 32, 4), (128, 64, 8), (64, 64, 8)):
        try:
            t = bench(lambda: hc_front(x, w, EPS, block_m=bm, block_k=bk, num_warps=nw))
        except Exception as ex:  # noqa: BLE001 - shared-memory limits differ per GPU
            print(f'BLOCK_M={bm} BLOCK_K={bk} warps={nw}: {type(ex).__name__}')
            continue
        same = torch.equal(hc_front(x, w, EPS, block_m=bm, block_k=bk, num_warps=nw), hc_front(x, w, EPS))
        print(f'BLOCK_M={bm} BLOCK_K={bk} warps={nw}: {t:.3f} ms  same-as-default={same}')
    print(f'torch path: {bench(lambda: old(x, w)):.3f} ms')
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
