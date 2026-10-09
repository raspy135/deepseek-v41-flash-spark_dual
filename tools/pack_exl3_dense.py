"""Offline: one TP rank's EXL3 attention + shared-expert matrices, sliced the way the engine shards FP8.

    python tools/pack_exl3_dense.py --source ~/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw \\
        --rank 0 --world 2 --out ~/models/exl3-packs/exl3-dense-r0of2.safetensors \\
        [--ablit ~/models/dsv41-wo-b-ablit/wo_b_l10_35.safetensors --native ~/models/DeepSeek-V4.1-Flash]

The decode graph reads these instead of the FP8 weights (DSV41_EXL3_DENSE, engine/exl3_dense.py);
prefill keeps FP8. Slices follow engine/tensor_parallel.py exactly:

  wq_a, wkv          replicated (whole)
  wq_b               output rows = heads: N tiles [rank*N/2/16, +N/2/16), svh slice, suh whole
  wo_a.slice.g       this rank's groups: g in [rank*4, rank*4 + 4), whole
  wo_b               OutputParallel output rows: N tiles of 5120, svh slice, suh whole (K 8192)
  shared w1 / w3     output columns 1152 of 2304: N tiles [rank*72, +72), svh slice, suh whole
  shared w2          OutputParallel output rows 2560 of 5120: N tiles [rank*160, +160)

EXL3 rotates in blocks of 128 along K and N, and every split above is on a 128 boundary, so a
rank's matrix is an exact slice (the routed experts' argument, docs/exl3-plan.md).

--ablit: the abliteration overlay replaces wo_b in layers 10-35 with an FP8 matrix; the EXL3
checkpoint is the base model. Measured 2026-10-08 (layer 20): overlay - base is rank-1 to FP8
rounding (singular values 6.94, then 0.48, 0.21, 0.20, ... flat). The pack stores that term,
u = sigma_1 * u_1 (this rank's output rows) and v = v_1 (all 8192 inputs), and the decode
projection adds u * (v . x) after the EXL3 wo_b.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exl3_ref as R  # noqa: E402
from pack_exl3_experts import SourceCheckpoint  # noqa: E402

N_LAYERS = 40
O_GROUPS = 8
ABLIT_LAYERS = range(10, 36)
FORMAT = "dsv41-exl3-dense-v1"


def bits_of(trellis_shape) -> int:
    return int(trellis_shape[-1]) // 16


def slice_matrix(src: SourceCheckpoint, name: str, n_lo: int, n_cols: int | None):
    """trellis [KT, NT, 16*bits] int16 -> the N columns [n_lo, n_lo + n_cols) (None = whole)."""
    shape = src.shape(f"{name}.trellis")
    mul = int(src.array(f"{name}.mul1").reshape(-1)[0]) & 0xFFFFFFFF
    if mul != R.MUL1_MUL:
        raise SystemExit(f"{name}: mul1 marker {mul:#x}, kernel decodes {R.MUL1_MUL:#x} only")
    if n_cols is None:
        t = src.array(f"{name}.trellis")
        svh = src.array(f"{name}.svh")
    else:
        assert n_lo % 128 == 0 and n_cols % 128 == 0, (name, n_lo, n_cols)
        t = src.array(f"{name}.trellis", (slice(None), slice(n_lo // 16, (n_lo + n_cols) // 16), slice(None)))
        svh = src.array(f"{name}.svh", (slice(n_lo, n_lo + n_cols),))
    suh = src.array(f"{name}.suh")
    return {"trellis": t, "suh": suh, "svh": svh}, bits_of(shape)


def ablit_terms(native_dir: str, overlay: str, rank: int, world: int):
    """{layer: (u [5120/world] fp32, v [8192] fp32)}: the rank-1 term of overlay - base wo_b."""
    import torch
    from safetensors import safe_open
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    index = json.load(open(os.path.join(native_dir, "model.safetensors.index.json")))["weight_map"]

    def deq(w, s):
        w = w.to(dev).float()
        s = s.to(dev).float()
        return w * s.repeat_interleave(32, 0).repeat_interleave(32, 1)

    out, stats = {}, {}
    ov = safe_open(overlay, "pt")
    for L in ABLIT_LAYERS:
        name = f"layers.{L}.attn.wo_b"
        base = safe_open(os.path.join(native_dir, index[f"{name}.weight"]), "pt")
        wb = deq(base.get_tensor(f"{name}.weight"), base.get_tensor(f"{name}.scale"))
        wa = deq(ov.get_tensor(f"{name}.weight"), ov.get_tensor(f"{name}.scale"))
        d = wa - wb
        torch.manual_seed(0)
        U, S, V = torch.svd_lowrank(d, q=8, niter=6)
        u = (U[:, 0] * S[0]).float()
        v = V[:, 0].float()
        resid = float((d - torch.outer(u, v)).norm() / d.norm())
        rows = u.numel() // world
        out[L] = (u[rank * rows:(rank + 1) * rows].cpu().numpy(), v.cpu().numpy())
        stats[L] = dict(sigma=[round(float(x), 4) for x in S[:4]], resid=round(resid, 4))
        print(f"  ablit L{L}: sigma {stats[L]['sigma']} rank-1 residual {resid:.3f} of |delta|", flush=True)
    return out, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ablit", default=None)
    ap.add_argument("--native", default=None, help="the native FP8 checkpoint (base wo_b) for --ablit")
    a = ap.parse_args()
    assert a.world == 2 and 0 <= a.rank < 2, "the engine's attention TP is two ranks"
    if a.ablit and not a.native:
        raise SystemExit("--ablit needs --native (the base FP8 wo_b it differs from)")
    from safetensors.numpy import save_file
    src = SourceCheckpoint(a.source)
    r, w = a.rank, a.world
    tensors, meta_bits = {}, {}

    def put(key, parts, bits):
        for k, v in parts.items():
            tensors[f"{key}.{k}"] = np.ascontiguousarray(v)
        meta_bits[key] = bits

    for L in range(N_LAYERS):
        p = f"layers.{L}"
        for name in ("attn.wq_a", "attn.wkv"):
            put(f"{p}.{name}", *slice_matrix(src, f"{p}.{name}", 0, None))
        n = src.shape(f"{p}.attn.wq_b.svh")[0] // w
        put(f"{p}.attn.wq_b", *slice_matrix(src, f"{p}.attn.wq_b", r * n, n))
        for gi, g in enumerate(range(r * O_GROUPS // w, (r + 1) * O_GROUPS // w)):
            put(f"{p}.attn.wo_a.{gi}", *slice_matrix(src, f"{p}.attn.wo_a.slice.{g}", 0, None))
        n = src.shape(f"{p}.attn.wo_b.svh")[0] // w
        put(f"{p}.attn.wo_b", *slice_matrix(src, f"{p}.attn.wo_b", r * n, n))
        for name in ("w1", "w3"):
            n = src.shape(f"{p}.ffn.shared_experts.{name}.svh")[0] // w
            put(f"{p}.shared.{name}", *slice_matrix(src, f"{p}.ffn.shared_experts.{name}", r * n, n))
        n = src.shape(f"{p}.ffn.shared_experts.w2.svh")[0] // w
        put(f"{p}.shared.w2", *slice_matrix(src, f"{p}.ffn.shared_experts.w2", r * n, n))
        print(f"layer {L}: {sum(v.nbytes for k, v in tensors.items() if k.startswith(p + '.')) / 1e6:.1f} MB",
              flush=True)
    ablit_meta = None
    if a.ablit:
        terms, stats = ablit_terms(a.native, a.ablit, r, w)
        for L, (u, v) in terms.items():
            tensors[f"layers.{L}.attn.wo_b.ablit_u"] = u
            tensors[f"layers.{L}.attn.wo_b.ablit_v"] = v
        h = hashlib.sha256()
        with open(a.ablit, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        ablit_meta = dict(file=os.path.basename(a.ablit), sha256=h.hexdigest(), layers=stats)
    meta = {"format": FORMAT, "rank": str(r), "world": str(w),
            "source_sha256": R.manifest_sha256(a.source), "bits": json.dumps(meta_bits),
            "ablit": json.dumps(ablit_meta)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    save_file(tensors, tmp, metadata=meta)
    os.replace(tmp, a.out)
    total = sum(v.nbytes for v in tensors.values())
    print(f"wrote {a.out}: {len(tensors)} tensors, {total / 1e9:.2f} GB, source {meta['source_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
