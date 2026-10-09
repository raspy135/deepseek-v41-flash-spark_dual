"""engine/exl3_dense.py against the reference decoder, on a real dense pack (one rank).

    python3 tools/test_exl3_dense.py [--pack /models/exl3-packs/exl3-dense-r0of2.safetensors]

Per matrix: Exl3Matrix(x) vs x @ exl3_ref.dequantize(...) in fp64 (the kernel rounds the rotated
input to fp16, so ~1e-3 relative is expected; the bar is the routed tests' 2e-2), row invariance
(a row's bits do not depend on how many rows share the call: decode's 1-row step vs its verify
block), and the shared-expert arena slot's stored widths. Layers: 3 (5-bit), the pack's 6-bit
wq_a/wkv layer, and 20 (ablit term when present).
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import exl3_ref as R  # noqa: E402
from engine.exl3_dense import Exl3Dense, MAX_ROWS  # noqa: E402

fails = []


def ok(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'} {msg}")
    if not cond:
        fails.append(msg)


def ref_matrix(f, key, bits):
    W = R.dequantize(f.get_tensor(key + ".trellis").cuda(), f.get_tensor(key + ".suh").cuda(),
                     f.get_tensor(key + ".svh").cuda(), float(bits), "mul1")
    return W.double()                     # [K, N]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="/models/exl3-packs/exl3-dense-r0of2.safetensors")
    a = ap.parse_args()
    from safetensors import safe_open
    f = safe_open(a.pack, "pt")
    meta = f.metadata()
    bits = json.loads(meta["bits"])
    rank, world = int(meta["rank"]), int(meta["world"])
    d = Exl3Dense(a.pack, "cuda", rank, world, 40)
    print(f"pack rank {rank}/{world}, {d.bytes / 2**30:.2f} GiB on the GPU")
    six = [L for L in range(40) if int(bits[f"layers.{L}.attn.wq_a"]) == 6]
    layers = sorted({3, 20, *six[:1]})
    g = torch.Generator(device="cuda").manual_seed(0)
    for L in layers:
        x3 = d.layers[L]
        for name, m, key, G in (("wq_a", x3.wq_a, f"layers.{L}.attn.wq_a", 1),
                                ("wkv", x3.wkv, f"layers.{L}.attn.wkv", 1),
                                ("wq_b", x3.wq_b, f"layers.{L}.attn.wq_b", 1),
                                ("wo_a", x3.wo_a, f"layers.{L}.attn.wo_a", 4),
                                ("wo_b", x3.wo_b.local, f"layers.{L}.attn.wo_b", 1)):
            T = 5
            x = (torch.randn(T, G, m.K, generator=g, device="cuda") * 0.5).bfloat16()
            got = m(x).double().view(T, G, m.N)
            keys = [key] if G == 1 else [f"{key}.{i}" for i in range(G)]
            ref = torch.stack([x[:, i].double() @ ref_matrix(f, k, bits[k]) for i, k in enumerate(keys)], 1)
            rel = float((got - ref).norm() / ref.norm())
            ok(rel < 2e-2, f"L{L} {name} ({bits[keys[0]]}-bit, K={m.K} N={m.N} G={G}) rel L2 {rel:.2e}")
            one = m(x[:1])
            many = m(torch.cat([x[:1], x[1:]]))[:1]
            wide = m(torch.cat([x, x, x])[:MAX_ROWS])[:1]
            ok(torch.equal(one, many) and torch.equal(one, wide), f"L{L} {name} row-invariant (T=1, 5, {MAX_ROWS})")
        if x3.wo_b.ablit is not None:
            u, v = x3.wo_b.ablit
            ok(u.numel() == 5120 // world and v.numel() == 8192, f"L{L} ablit term shapes {tuple(u.shape)} {tuple(v.shape)}")
    sa = d.shared_arena
    for L in range(40):
        b13, b2 = int(bits[f"layers.{L}.shared.w1"]), int(bits[f"layers.{L}.shared.w2"])
        if not (int(sa.bits_gpu[L]) == 2 * b13 and int(sa.bits2_gpu[L]) == 2 * b2):
            ok(False, f"L{L} shared slot widths {int(sa.bits_gpu[L])}/{int(sa.bits2_gpu[L])} vs pack {b13}/{b2}")
        for t, w in (("t1", "w1"), ("t3", "w3"), ("t2", "w2")):
            src = f.get_tensor(f"layers.{L}.shared.{w}.trellis")
            if not torch.equal(getattr(sa, t)[L].reshape(-1)[:src.numel()].cpu(), src.reshape(-1)):
                ok(False, f"L{L} shared {w} trellis != pack")
    mixed = [L for L in range(40) if bits[f"layers.{L}.shared.w1"] != bits[f"layers.{L}.shared.w2"]]
    ok(True, f"shared arena: 40 slots checked against the pack (mixed-width layers {mixed})")
    print(f"\n{'FAIL: ' + str(len(fails)) if fails else 'all checks passed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
