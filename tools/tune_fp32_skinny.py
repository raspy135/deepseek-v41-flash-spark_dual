"""Sweep the skinny fp32 split-K kernel's launch config on the real HC shape (M=6, N=24, K=20480).

    python3 tools/tune_fp32_skinny.py [layer] [M]     (GPU; MODEL_DIR)

Times each config with CUDA-graph replay and reports achieved GB/s, the max difference from the
serving padded-cuBLAS result, and whether the kernel replays bit-identically. Picks a winner for the
defaults in tools/fp32_skinny.py; it does not touch the engine.

Configs that make the single-program reduce load too much are kept in the sweep so the cost is
visible, not filtered out.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import v41_ref as R  # noqa: E402
from fp32_skinny import BLOCK_K, BLOCK_MP, TARGET_CTAS, skinny_linear  # noqa: E402

MD = os.path.expanduser(os.environ.get("MODEL_DIR", "~/models/DeepSeek-V4.1-Flash"))


def per_call_ms(fn, reps=100):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(5):
        for _ in range(reps):
            graph.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / (5 * reps), out


def main(layer, m):
    torch.backends.cuda.matmul.allow_tf32 = False
    R.MM_TILE, R.HC_MM_TILE = 16, 32
    index = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(MD, index[f"layers.{layer}.hc_attn_fn"]), "pt", device="cpu") as f:
        w = f.get_tensor(f"layers.{layer}.hc_attn_fn").cuda().float()
    n, k = w.shape
    x = (torch.randn(m, k, device="cuda") * 0.3).bfloat16().float()
    ref = R.mm(x, w)
    gb = (n * k + x.numel()) * 4 / 1e9
    print(f"HC {n}x{k}, M={m}; serving R.mm padded {per_call_ms(lambda: R.mm(x, w))[0] * 1e3:.1f} us")

    grid = list(itertools.product(("ieee", "tf32x3", "tf32"),  # tl.dot input precision
                                  (64, 128),                # block_k
                                  (48, 96),                 # target_ctas
                                  (4, 8),                   # num_warps
                                  (3, 4),                   # num_stages
                                  (8,)))                    # reduce_warps
    results = []
    for prec, bk, tc, nw, ns, rw in grid:
        try:
            fn = lambda: skinny_linear(x, w, block_k=bk, target_ctas=tc, num_warps=nw,
                                       num_stages=ns, reduce_warps=rw, prec=prec)
            ms, out = per_call_ms(fn)
            diff = float((out - ref).abs().max())
            det = bool(torch.equal(out, fn()))
            results.append((ms, prec, bk, tc, nw, ns, gb / ms * 1e3, diff, det))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip prec={prec} bk={bk} tc={tc} nw={nw} ns={ns}: {type(exc).__name__}")
    results.sort()
    print(f"\n{'us':>7} {'GB/s':>6}  prec    block_k  target_ctas  warps  stages  maxdiff  det")
    for ms, prec, bk, tc, nw, ns, gbs, diff, det in results[:20]:
        print(f"{ms * 1e3:7.1f} {gbs:6.1f}  {prec:6s}  {bk:7d}  {tc:11d}  {nw:5d}  {ns:6d}  "
              f"{diff:.2e}  {det}")
    dms, *dr = next((r for r in results if r[1:5] == ("ieee", BLOCK_K, TARGET_CTAS, 4)), results[0])
    print(f"\ndefault (ieee bk={BLOCK_K} tc={TARGET_CTAS} nw=4 ns=3): {dms * 1e3:.1f} us, "
          f"best {results[0][0] * 1e3:.1f} us ({results[0][1:6]})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0, int(sys.argv[2]) if len(sys.argv) > 2 else 6)
