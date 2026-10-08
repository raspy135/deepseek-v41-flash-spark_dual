"""P3 gate: the EXL3 CUDA decode kernel vs the torch reference, real experts, decode shapes.

3-bit slots only for now (TF's tile stride is the bit width, so 2-bit needs a separate pool).
Run in the serving image (nvcc):  python3 /app/tools/test_exl3_moe_cuda.py
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exl3_moe as X3  # noqa: E402
import exl3_moe_cuda as XC  # noqa: E402
from pack_exl3_experts import SourceCheckpoint  # noqa: E402

SRC = os.environ.get("EXL3_SOURCE", os.path.expanduser("~/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw"))
E = int(os.environ.get("EXPERTS", "4"))
fails = []


def ok(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'} {msg}")
    if not cond:
        fails.append(msg)


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    src = SourceCheckpoint(SRC)
    bits = src.layer_bits()
    l3 = [L for L in range(40) if bits[L] == 3][: max(1, E // 2)]
    l2 = [L for L in range(40) if bits[L] == 2][: E - len(l3)]
    layers = l3 + l2                       # mixed 3-bit and 2-bit slots
    arena = X3.Exl3Arena(E, device="cuda", tp_rank=0, tp_world=1, bits=3.0)
    for i, L in enumerate(layers):
        arena.load_slot(i, src.record(L, 0, 0, 1), bits[L])
    print(f"layers {layers} bits {[bits[L] for L in layers]}")

    torch.manual_seed(0)
    for T in (1, 2, 4, 6):
        x = (torch.randn(T, X3.DIM, dtype=torch.float32, device="cuda") * 0.1).to(torch.bfloat16)
        slots = torch.randint(0, E, (T, 6), dtype=torch.int32, device="cuda")
        w = torch.rand(T, 6, device="cuda")
        ref = X3.moe_forward_exl3_ref(x, slots, w, arena, 10.0, out_dtype=torch.float32)
        got = XC.moe_forward(x, slots, w, arena, 10.0)
        rel = float((got - ref).norm() / ref.norm())
        ok(rel < 2e-2, f"T={T} kernel vs reference rel L2 {rel:.3e}  ({tuple(got.shape)})")

    print(f"\n{'FAIL: ' + str(len(fails)) if fails else 'all checks passed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
