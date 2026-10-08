"""indexer_prefill.index_scores against Model._indexer's torch loop: score agreement, top-512
selection overlap, row invariance, timing.

    python tools/test_indexer_prefill.py
"""
import sys
import time

sys.path[:0] = ['/app', '/app/tools', __file__.rsplit('/', 1)[0]]
import torch
from indexer_prefill import index_scores, index_scores_ref


def bench(fn, n=5):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def main():
    g = torch.Generator(device='cuda').manual_seed(0)
    ok = True
    for T, N in ((64, 512), (300, 1536), (2048, 4096), (2048, 16384)):
        q = (torch.randn(T, 32, 128, generator=g, device='cuda') * 0.5).bfloat16()
        k = (torch.randn(N, 128, generator=g, device='cuda') * 0.5).bfloat16()
        w = torch.randn(T, 32, generator=g, device='cuda') * (128 ** -0.5 * 32 ** -0.5)
        a, b = index_scores_ref(q, k, w), index_scores(q, k, w)
        same = (a == b).float().mean().item()
        ta, tb = a.float().topk(512, -1).indices.sort(-1).values, b.float().topk(512, -1).indices.sort(-1).values
        overlap = sum(len(set(x.tolist()) & set(y.tolist())) for x, y in zip(ta, tb)) / ta.numel()
        inv = torch.equal(index_scores(q[7:], k, w[7:]), b[7:])
        print(f'T={T} N={N}: identical scores {same * 100:.3f}%, max |diff| {(a.float() - b.float()).abs().max().item():.3e}, '
              f'top-512 overlap {overlap * 100:.3f}%, row-invariant={inv}', flush=True)
        ok &= inv and overlap > 0.995
        if T == 2048:
            print(f'   torch loop {bench(lambda: index_scores_ref(q, k, w)):.2f} ms, fused {bench(lambda: index_scores(q, k, w)):.2f} ms')
            for bm, bn, nw in ((64, 64, 4), (64, 128, 4), (128, 64, 8), (32, 64, 4), (64, 64, 8)):
                print(f'   BM={bm} BN={bn} warps={nw}: {bench(lambda: index_scores(q, k, w, bm=bm, bn=bn, num_warps=nw)):.2f} ms, '
                      f'same={torch.equal(index_scores(q, k, w, bm=bm, bn=bn, num_warps=nw), b)}')
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
