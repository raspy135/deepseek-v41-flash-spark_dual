"""prefill_attention_indexed must give the bits of gather + decode_attention(split=1).

GPU only. Random data at the prefill shapes (32 local heads, d = 512, a 128-row window, 512
selected compressed rows), packed and BF16 caches, -1 selections and window positions, partial
masks and fully masked rows. Also times both paths at a 2,048-row chunk.

    python tools/test_prefill_attn.py
"""
import sys
import time

sys.path[:0] = ['/app', '/app/tools', __file__.rsplit('/', 1)[0], __file__.rsplit('/', 2)[0]]
import torch
from decode_attn import decode_attention
from prefill_attn import prefill_attention_indexed
from engine import packed_kv

RING = 4096


def case(T, n_cache, packed, seed, bh=16, bn=32, H=32, D=512, NW=128, NC=512, start=None):
    g = torch.Generator(device='cuda').manual_seed(seed)
    dev = 'cuda'
    q = (torch.randn(T, H, D, generator=g, device=dev) * 2).bfloat16()
    ring = torch.randn(RING, D, generator=g, device=dev).bfloat16()
    start = n_cache * 4 if start is None else start
    pos = start + torch.arange(T, device=dev)
    wpos = pos[:, None] - torch.arange(NW - 1, -1, -1, device=dev)[None, :]
    wpos = torch.where(wpos >= 0, wpos, torch.full_like(wpos, -1))
    vals = (torch.randn(n_cache, D, generator=g, device=dev) * 0.7).bfloat16()
    if packed:
        cache = torch.zeros(n_cache, D // 16 + D // 128, dtype=torch.int64, device=dev)
        packed_kv.write(cache, vals.contiguous(), 0)
    else:
        cache = vals
    cidx = torch.randint(0, n_cache, (T, NC), generator=g, device=dev)
    cidx = cidx.sort(dim=-1).values
    cidx[torch.rand(T, NC, generator=g, device=dev) < 0.1] = -1        # unselected slots
    mask = torch.cat([wpos >= 0, cidx >= 0], dim=1)
    mask[torch.rand(T, NW + NC, generator=g, device=dev) < 0.05] = False
    mask[0] = False                                                     # a fully masked row
    sink = torch.randn(H, generator=g, device=dev)
    scale = D ** -0.5
    wkv = ring[wpos.clamp_min(0) % RING]
    rows = packed_kv.gather(cache, cidx.clamp_min(0))
    ref = decode_attention(q, wkv, rows, mask, sink, scale, block_h=bh, block_n=bn, split=1)
    got = prefill_attention_indexed(q, ring, wpos, cache, cidx, mask, sink, scale, block_h=bh, block_n=bn)
    same = torch.equal(ref.view(torch.int16), got.view(torch.int16))
    # window-only layers (no compressed rows)
    ref_w = decode_attention(q, wkv, None, mask[:, :NW].contiguous(), sink, scale, block_h=bh, block_n=bn, split=1)
    got_w = prefill_attention_indexed(q, ring, wpos, None, None, mask[:, :NW], sink, scale, block_h=bh, block_n=bn)
    same_w = torch.equal(ref_w.view(torch.int16), got_w.view(torch.int16))
    return same and same_w, (q, ring, wpos, cache, cidx, mask, sink, scale, wkv, rows)


def bench(fn, n=10):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def main():
    ok = True
    for T in (17, 128, 512, 2048):
        for packed in (True, False):
            for seed in (1, 2):
                same, _ = case(T, 9000, packed, seed)
                print(f'T={T:5d} packed={packed} seed={seed} identical={same}', flush=True)
                ok &= same
    same, _ = case(300, 200, True, 3, start=100)  # positions below the window: -1 window slots
    print(f'short-history identical={same}')
    ok &= same
    _, (q, ring, wpos, cache, cidx, mask, sink, scale, wkv, rows) = case(2048, 9000, True, 7)
    t_old = bench(lambda: decode_attention(q, ring[wpos.clamp_min(0) % RING],
                                           packed_kv.gather(cache, cidx.clamp_min(0)), mask, sink, scale,
                                           block_h=16, block_n=32, split=1))
    t_attn = bench(lambda: decode_attention(q, wkv, rows, mask, sink, scale, block_h=16, block_n=32, split=1))
    t_new = bench(lambda: prefill_attention_indexed(q, ring, wpos, cache, cidx, mask, sink, scale))
    print(f'T=2048 packed: gathers+attention {t_old:.2f} ms (attention alone {t_attn:.2f}), indexed {t_new:.2f} ms')
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
