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


def serial_group(pick, slots, maxm=16):
    """The serial scan exl3m_group's kernel replaced: leaders in first-occurrence order, members ascending."""
    uids, members = [], []
    for i, e in enumerate(pick):
        if e in pick[:i]:
            continue
        m = [(j // slots) * 32 + (j % slots) for j in range(i, len(pick)) if pick[j] == e][:maxm]
        uids.append(e)
        members.append(m + [-1] * (maxm - len(m)))
    return uids, members


def check_grouping():
    g = torch.Generator().manual_seed(1)
    for T in (1, 2, 4, 6, 8, 10):
        for pool in (3, 9, 40, 1000):
            P = T * 6
            if P > 64:
                continue
            pick = torch.randint(-1, pool, (P,), generator=g, dtype=torch.int32)
            arena = type("A", (), {"slots": 13522})()
            uids, ucount, members = XC.decode_group(pick.cuda(), T, 6, arena)
            ru, rm = serial_group(pick.tolist(), 6)
            n = int(ucount.item())
            same = (n == len(ru) and uids[:n].tolist() == ru and members[:n].tolist() == rm)
            if not same:
                ok(False, f"grouping T={T} pool={pool}")
                return
    ok(True, "decode grouping == serial scan (T 1-10, pools 3-1000, -1 picks)")


def check_kernels(arena, tag):
    torch.manual_seed(0)
    for T in (1, 2, 4, 6):
        x = (torch.randn(T, X3.DIM, dtype=torch.float32, device="cuda") * 0.1).to(torch.bfloat16)
        slots = torch.randint(0, E, (T, 6), dtype=torch.int32, device="cuda")
        w = torch.rand(T, 6, device="cuda")
        ref = X3.moe_forward_exl3_ref(x, slots, w, arena, 10.0, out_dtype=torch.float32)
        got = XC.moe_forward(x, slots, w, arena, 10.0)
        rel = float((got - ref).norm() / ref.norm())
        ok(rel < 2e-2, f"{tag}: T={T} kernel vs reference rel L2 {rel:.3e}  ({tuple(got.shape)})")
    # prefill shapes go through build_routing + the same pipeline
    for T in (128, 512):
        x = (torch.randn(T, X3.DIM, dtype=torch.float32, device="cuda") * 0.1).to(torch.bfloat16)
        slots = torch.randint(0, E, (T, 6), dtype=torch.int32, device="cuda")
        w = torch.rand(T, 6, device="cuda")
        ref = X3.moe_forward_exl3_ref(x, slots, w, arena, 10.0, out_dtype=torch.float32)
        got = XC.moe_forward_prefill(x, slots, w, arena, 10.0)
        rel = float((got - ref).norm() / ref.norm())
        ok(rel < 2e-2, f"{tag}: prefill T={T} kernel vs reference rel L2 {rel:.3e}  ({tuple(got.shape)})")


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    check_grouping()
    src = SourceCheckpoint(SRC)
    bits = src.layer_bits()
    l3 = [L for L in range(40) if bits[L] == 3][: max(1, E // 2)]
    l2 = [L for L in range(40) if bits[L] == 2][: E - len(l3)]
    layers = l3 + l2                       # mixed 3-bit and 2-bit slots
    # uniform 3-bit-sized slots, then an exact-slot arena (DSV41_EXL3_EXACT_SLOTS): 2-bit experts in
    # the narrow pool [0, len(l2)), 3-bit ones in the wide pool after it
    for narrow in (0, len(l2)):
        arena = X3.Exl3Arena(E, device="cuda", tp_rank=0, tp_world=1, bits=3.0, narrow_slots=narrow)
        order = (l2 + l3) if narrow else layers
        for i, L in enumerate(order):
            arena.load_slot(i, src.record(L, 0, 0, 1), bits[L])
        tag = f"exact slots ({narrow} narrow)" if narrow else "uniform slots"
        print(f"{tag}: layers {order} bits {[bits[L] for L in order]}, {arena.total_bytes / 1e6:.1f} MB")
        if narrow:
            try:
                arena.load_slot(0, src.record(l3[0], 0, 0, 1), bits[l3[0]])
                ok(False, "a narrow slot accepted a 3-bit expert")
            except ValueError:
                ok(True, "a narrow slot refuses a 3-bit expert")
            arena.load_slot(0, src.record(order[0], 0, 0, 1), bits[order[0]])
            back = arena.read_slot(0)["t1"].cpu().numpy()
            ok((back == np.asarray(src.record(order[0], 0, 0, 1)["t1"])).all(), "narrow slot round-trips its trellis")
        check_kernels(arena, tag)
    print(f"\n{'FAIL: ' + str(len(fails)) if fails else 'all checks passed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
