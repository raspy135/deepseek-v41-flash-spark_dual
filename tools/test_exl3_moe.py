"""P2 gate for EXL3: the arena's rank slice and the reference MoE against the numpy oracle.

Builds a small world=1 pack from the source (so the arena is single-rank, which the reference
path is) and checks each projection against ``exl3_format.dequantize``, then the reference MoE
against a float64 oracle.

Run:  .venv/bin/python tools/test_exl3_moe.py
Env:  EXL3_SOURCE (default ~/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw).

CPU only on purpose: the arena and the reference are checked without touching the GPU, so this
runs beside a live server.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exl3_format as F  # noqa: E402
import exl3_moe as M  # noqa: E402
import exl3_ref as R  # noqa: E402
from pack_exl3_experts import SourceCheckpoint  # noqa: E402

SRC = os.environ.get("EXL3_SOURCE", os.path.expanduser("~/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw"))
CASES = [(0, 0), (18, 0), (22, 0), (39, 0)]      # 3-bit, 2-bit, 2-bit, 3-bit
fails = []


def ok(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'} {msg}")
    if not cond:
        fails.append(msg)


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(b))


def main():
    if not os.path.isdir(SRC):
        print(f"SKIP: {SRC} not present")
        return 0
    src = SourceCheckpoint(SRC)
    all_bits = src.layer_bits()
    tmp = tempfile.mkdtemp(prefix="exl3moe-")
    try:
        pack = os.path.join(tmp, "exl3-mini-w1.bin")
        bits_map = {L: all_bits[L] for L, _ in CASES}
        R.write_pack(pack, bits_map, 0, 1, src.codebook, "test",
                     ((LE, src.record(*LE, 0, 1)) for LE in CASES), n_experts=1)
        rd = R.PackReader(pack)
        print(f"mini pack: {[L for L, _ in CASES]} bits {[bits_map[L] for L, _ in CASES]}, "
              f"header {rd.header_sha256()[:16]}")

        arena = M.Exl3Arena(len(CASES), device="cpu", tp_rank=0, tp_world=1, bits=3.0,
                            codebook=rd.codebook)
        expect = R.record_nbytes(3.0, 0, 1)
        ok(arena.bytes_per_slot == expect,
           f"bytes_per_slot {arena.bytes_per_slot} == 3-bit record {expect} ({expect / 1e6:.2f} MB)")

        for slot, (L, E) in enumerate(CASES):
            bits = bits_map[L]
            arena.load_slot(slot, rd.read_expert(L, E), bits)
            ok(float(arena.bits[slot]) == bits, f"slot {slot} layer {L} expert {E} bits {bits:g}")
            rec = arena.read_slot(slot)
            ok(rec["t1"].shape[-1] == R.tile_words(bits),
               f"slot {slot} trellis cut to {rec['t1'].shape[-1]} words "
               f"(slot width {arena.shapes['t1'][-1]})")

        print("dequant")
        dec = {}
        for slot, (L, E) in enumerate(CASES):
            bits = bits_map[L]
            w1, w2, w3 = arena.dequant_slot(slot)
            dec[slot] = (w1, w2, w3)
            rec = arena.read_slot(slot)
            for name, w, pre, suh, svh in (("w1", w1, "t1", "suh1", "svh1"),
                                           ("w3", w3, "t3", "suh3", "svh3"),
                                           ("w2", w2, "t2", "suh2", "svh2")):
                o = F.dequantize(rec[pre].numpy(), rec[suh].numpy(), rec[svh].numpy(), bits, rd.codebook)
                r = rel(w.numpy(), o)
                ok(r < 1e-6, f"layer {L} expert {E} {name} dequant vs oracle rel {r:.2e} {tuple(w.shape)}")

        print("reference MoE")
        torch.manual_seed(0)
        T = 3
        x = (torch.randn(T, M.DIM, dtype=torch.float32) * 0.1).to(torch.bfloat16)
        slots = torch.randint(0, len(CASES), (T, 6), dtype=torch.int32)
        weights = torch.rand(T, 6, dtype=torch.float32)
        got = M.moe_forward_exl3_ref(x, slots, weights, arena, swiglu_limit=10.0,
                                     out_dtype=torch.float32)
        y = np.zeros((T, M.DIM), dtype=np.float64)
        xf = x.float().numpy().astype(np.float64)
        for i, (L, E) in enumerate(CASES):
            w1, w2, w3 = dec[i]
            w1n, w2n, w3n = w1.double().numpy(), w2.double().numpy(), w3.double().numpy()
            mask = (slots == i).numpy()
            for t in range(T):
                for kk in range(6):
                    if not mask[t, kk]:
                        continue
                    gate = xf[t] @ w1n
                    up = xf[t] @ w3n
                    up = np.clip(up, -10.0, 10.0)
                    gate = np.minimum(gate, 10.0)
                    h = (gate / (1.0 + np.exp(-gate))) * up * float(weights[t, kk])
                    y[t] += h @ w2n
        r = rel(got.numpy(), y)
        ok(r < 3e-2, f"reference MoE vs float64 oracle rel L2 {r:.2e} (bf16 weight/activation rounding)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'FAIL: ' + str(len(fails)) if fails else 'all checks passed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
