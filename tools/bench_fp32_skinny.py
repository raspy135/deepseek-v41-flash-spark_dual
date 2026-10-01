"""The skinny fp32 HC GEMM on cuBLAS vs tools/fp32_skinny.py, single GPU, real weights.

Two Hyper-Connection mix projections per layer are [6, 20480] x [20480, 24] fp32 (1.97 MB weight).
N=24 is a single tile, so cuBLAS parallelises only over K and is latency-bound; fp32_skinny splits K
across programs instead. It measured ~3x on 2026-09-11 and was then removed from the decode path by
b79092a ("fix quality issue") because the graphed verifier has to reproduce Model.forward's logits
and the kernel's difference does not, and that flips borderline routing.

**The timing cycles over a different layer's weight on every call.** A decode step reads 40 layers'
worth (158 MB), so the weight is L2-cold there; timing one weight repeatedly reports 8.2 us where
the engine sees 13.4. (This file made exactly that mistake once -- see the L2-hot pitfall in
docs/decode-projection-fusion.md. Do not time a 2 MB weight in a loop.)

    python3 tools/bench_fp32_skinny.py [layers]     (GPU; MODEL_DIR)

Counts and exactness only; it does not judge quality.
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
CANDIDATES = (
    ("F.linear", lambda x, w: F.linear(x, w)),
    ("R.mm padded", lambda x, w: R.mm(x, w)),
    ("skinny ieee", lambda x, w: skinny_linear(x, w, prec="ieee")),
    ("skinny tf32x3", lambda x, w: skinny_linear(x, w, prec="tf32x3")),
)


def cold_ms(fn, pairs, reps=1):
    """Per-call ms with a different (x, w) every call, so the 1.97 MB weight is L2-cold."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for x, w in pairs:
            fn(x, w)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(5):
        for _ in range(reps):
            graph.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / (5 * reps * len(pairs))


def main(layers):
    torch.backends.cuda.matmul.allow_tf32 = False
    R.MM_TILE, R.HC_MM_TILE = 16, 32
    index = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def get(n):
        f = index[n]
        if f not in handles:
            handles[f] = safe_open(os.path.join(MD, f), "pt", device="cpu")
        return handles[f].get_tensor(n)

    hc = [(f"hc_attn_fn", get(f"layers.{L}.hc_attn_fn").cuda().float(),
           (torch.randn(6, 20480, device="cuda") * 0.3).bfloat16().float()) for L in layers]
    print(f"HC 24x20480, M=6, L2-cold over {len(hc)} layers; skinny wins(N<=64)={wins(24)}")
    print(f"{'candidate':16} {'us':>8} {'GB/s':>7} {'TFLOP/s':>8} {'maxdiff':>10} {'exact':>6}")
    _, w0, x0 = hc[0]
    ref = R.mm(x0, w0)
    gb = (24 * 20480 + x0.numel()) * 4 / 1e9
    flops = 2.0 * 6 * 24 * 20480
    for label, fn in CANDIDATES:
        ms = cold_ms(fn, [(x, w) for _, w, x in hc])
        out = fn(x0, w0)
        md = float((out - ref).abs().max())
        rel = float((out - ref).norm() / ref.norm())
        print(f"{label:16} {ms * 1e3:8.1f} {gb / ms * 1e3:7.1f} {flops / ms / 1e9:8.2f} {md:10.2e} "
              f"{str(bool(torch.equal(out, ref))):>6}  rel {rel:.1e}")

    rg = get(f"layers.{layers[0]}.ffn.gate.weight").cuda().float()
    xr = (torch.randn(6, rg.size(1), device="cuda") * 0.3).bfloat16().float()
    ms = cold_ms(lambda x, w: R.mm(x, w), [(xr, rg)])
    print(f"\nrouter gate 384x5120: R.mm padded {ms * 1e3:.1f} us, F.linear "
          f"{cold_ms(lambda x, w: F.linear(x, w), [(xr, rg)]) * 1e3:.1f} us "
          f"(skinny is not defined for N>64)")

    w = hc[0][1]
    x6 = (torch.randn(6, 20480, device="cuda") * 0.3).bfloat16().float()
    inv = all(torch.equal(R.mm(x6[:m], w), R.mm(x6, w)[:m]) for m in (1, 2, 4, 6))
    for prec in ("ieee", "tf32x3"):
        full = skinny_linear(x6, w, prec=prec)
        inv &= all(torch.equal(skinny_linear(x6[:m], w, prec=prec), full[:m]) for m in (1, 2, 4, 6))
    print(f"row invariance (R.mm and both skinny precisions): {inv}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or list(range(16)))
