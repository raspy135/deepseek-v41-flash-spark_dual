"""The skinny fp32 HC GEMM on cuBLAS vs tools/fp32_skinny.py, single GPU, real weights.

Two Hyper-Connection mix projections per layer are [6, 20480] x [20480, 24] fp32 (1.97 MB weight).
N=24 is a single tile, so cuBLAS parallelises only over K and is latency-bound; fp32_skinny splits K
across programs instead. It measured ~3x on 2026-09-11 and was then removed from the decode path by
b79092a ("fix quality issue") because the graphed verifier has to reproduce Model.forward's logits
and the kernel's ~4e-7 difference does not, and that flips borderline routing.

This re-measures the speed and the delta so the decision can be re-made without a two-node gate. It
reports counts and exactness only; it does not judge quality.

    python3 tools/bench_fp32_skinny.py [layer ...]     (GPU; MODEL_DIR)
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import v41_ref as R  # noqa: E402
from fp32_skinny import BLOCK_MP, skinny_linear, wins  # noqa: E402

MD = os.path.expanduser(os.environ.get("MODEL_DIR", "~/models/DeepSeek-V4.1-Flash"))


def per_call_ms(fn, reps=100):
    """CUDA-graph replay time per call, and the captured output (same buffer every replay)."""
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


def main(layers):
    torch.backends.cuda.matmul.allow_tf32 = False
    R.MM_TILE, R.HC_MM_TILE = 16, 32  # serving spelling (engine/model.py)
    index = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def get(n):
        f = index[n]
        if f not in handles:
            handles[f] = safe_open(os.path.join(MD, f), "pt", device="cpu")
        return handles[f].get_tensor(n)

    print(f"MM_TILE={R.MM_TILE} HC_MM_TILE={R.HC_MM_TILE}  skinny BLOCK_MP={BLOCK_MP} "
          f"wins(N<=64)={wins(24)}")
    print(f"{'weight':26} {'N x K':>12} {'candidate':>16} {'us':>8} {'GB/s':>7} {'TFLOP/s':>8} "
          f"{'maxdiff':>10} {'exact':>6}")
    for L in layers:
        for name in (f"layers.{L}.hc_attn_fn", f"layers.{L}.hc_ffn_fn", f"layers.{L}.ffn.gate.weight"):
            w = get(name).cuda().float()
            n, k = w.shape
            x = (torch.randn(6, k, device="cuda") * 0.3).bfloat16().float()
            ref = R.mm(x, w)
            gb = (n * k + x.numel()) * 4 / 1e9
            flops = 2.0 * 6 * n * k
            for label, fn in (("F.linear", lambda: F.linear(x, w)),
                              ("R.mm padded", lambda: R.mm(x, w)),
                              (f"skinny{'*' if wins(n) else ''}",
                               (lambda: skinny_linear(x, w)) if wins(n) else None)):
                if fn is None:
                    print(f"{name:26} {n:6d}x{k:<6d} {label:>16} {'-- not the kernel shape':>26}")
                    continue
                ms, out = per_call_ms(fn)
                md = float((out - ref).abs().max())
                rel = float((out - ref).norm() / ref.norm())
                print(f"{name:26} {n:6d}x{k:<6d} {label:>16} {ms * 1e3:8.1f} {gb / ms * 1e3:7.1f} "
                      f"{flops / ms / 1e9:8.2f} {md:10.2e} {str(bool(torch.equal(out, ref))):>6}"
                      f"  rel {rel:.1e}")
            del w, x, ref

    # row-count invariance of the serving spelling, which is the invariant b79092a is protecting
    w = get(f"layers.{layers[0]}.hc_attn_fn").cuda().float()
    x = (torch.randn(6, w.size(1), device="cuda") * 0.3).bfloat16().float()
    full = R.mm(x, w)
    inv = all(torch.equal(R.mm(x[:m], w), full[:m]) for m in (1, 2, 4, 6))
    sk_full = skinny_linear(x, w)          # skinny asserts M <= BLOCK_MP (8)
    sk_inv = all(torch.equal(skinny_linear(x[:m], w), sk_full[:m]) for m in (1, 2, 4, 6))
    print(f"\nrow invariance  R.mm padded: {inv}   skinny: {sk_inv}")
    print("skinny* = wins(N) says this shape should use the kernel")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [0, 20])
