#!/usr/bin/env python3
"""test_moe_null_slot.py -- the decode-routing invariant that expert parallel breaks.

EP2 (engine/dist.py) maps every expert this rank does NOT own onto one shared, zero-filled
"null" arena slot, so its contribution is exactly 0 and the peer's all-reduce supplies the real
value. That makes a decode block aim many (token, k) pairs at a SINGLE slot -- which is the one
thing tools/fp4_moe.py::build_routing_small is not allowed to be given: it hands each distinct
slot exactly one BM-wide block, so past BM pairs the write walks into the next block and
overwrites a legitimate pair index. No error, no bounds fault, just a token quietly losing a real
expert's contribution.

Found in production, not in a test: the dual-Spark pair served fluent prose with roughly one
token in ten mangled ("mixture-of-experualiayer") and a stray <|begin of sentence|> mid-sentence,
while draft acceptance fell from 2.77 to 1.91. With BM=16 and the 6-token verify block, 36 pairs
of which ~18 were remote, the overflow was 2 pairs deep.

moe_forward_reference is the oracle: it loops over unique slots and index_add_s, so it is correct
for any number of pairs per slot.

Run (needs a GPU + a Triton that can compile): python3 tools/test_moe_null_slot.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp4_moe as K  # noqa: E402


def build_arena(n_slots: int, null_slot: int, device="cuda", seed=0):
    """Random FP4 experts, except `null_slot`, which is zeroed exactly as ExpertStore zeroes it."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = K.ExpertArena(n_slots, device)
    for t in (a.w1, a.w3, a.w2):
        t.copy_(torch.randint(0, 256, t.shape, dtype=torch.uint8, generator=g).to(device))
    for t in (a.s1, a.s3, a.s2):
        # UE8M0 exponents around 127 (2^0) keep the dequantised weights O(1).
        t.copy_(torch.randint(124, 130, t.shape, dtype=torch.uint8, generator=g).to(device))
    for t in (a.w1, a.s1, a.w3, a.s3, a.w2, a.s2):
        t[null_slot].zero_()
    return a


def ep2_decode_block(T=6, Kk=6, n_real=8, null_slot=8, device="cuda", seed=0):
    """The shape EP2 actually produces: a T-token verify block whose non-owned pairs (about half,
    by the `expert % 2 == rank` split) all resolve to the null slot."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    experts = torch.stack([torch.randperm(2 * n_real, generator=g)[:Kk] for _ in range(T)])
    owned = (experts % 2) == 0                       # this rank owns the even expert ids
    slots = torch.where(owned, experts // 2, torch.full_like(experts, null_slot))
    w = torch.rand(T, Kk, generator=g)
    x = torch.randn(T, K.DIM, generator=g).to(torch.bfloat16)
    return (x.to(device), slots.to(torch.int32).to(device), w.to(torch.float32).to(device),
            int(owned.numel() - owned.sum()))


def rel_err(got, ref):
    return ((got.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-9)).item()


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); return 0
    dev = "cuda"
    n_slots, null_slot = 9, 8
    arena = build_arena(n_slots, null_slot, dev)
    x, slots, w, n_remote = ep2_decode_block(null_slot=null_slot, device=dev)
    P = slots.numel()
    print(f"decode block: T={slots.shape[0]} K={slots.shape[1]} P={P} pairs, "
          f"{n_remote} of them on the null slot (BM would be {K._pick_bm(P)} without the fix)")

    ref = K.moe_forward_reference(x, slots, w, arena, 10.0).float()

    bad = K.moe_forward(x, slots, w, arena, 10.0).float()                      # old behaviour
    good = K.moe_forward(x, slots, w, arena, 10.0, slots_repeat=True).float()  # fixed
    e_bad, e_good = rel_err(bad, ref), rel_err(good, ref)
    print(f"  slots_repeat=False : rel err vs reference = {e_bad:.4f}")
    print(f"  slots_repeat=True  : rel err vs reference = {e_good:.4f}")

    ok = True
    # The fixed path should differ from the reference only by FP4-kernel-vs-bf16-GEMM noise.
    if e_good > 0.05:
        print(f"FAIL: slots_repeat=True still disagrees with the reference ({e_good:.4f})"); ok = False
    # And the invariant must actually have been violated, or this test proves nothing.
    if n_remote <= K._pick_bm(P):
        print(f"FAIL: only {n_remote} pairs on the null slot; not an overflow, test is not exercising the bug")
        ok = False
    elif e_bad <= e_good * 2:
        print(f"FAIL: the unfixed path was not measurably wrong ({e_bad:.4f} vs {e_good:.4f}) -- "
              f"has build_routing_small changed?"); ok = False

    # The guarded assertion must name the problem rather than let it through silently.
    os.environ["DSV41_CHECK_ROUTING"] = "1"
    try:
        K.moe_forward(x, slots, w, arena, 10.0)
        print("FAIL: DSV41_CHECK_ROUTING=1 did not catch the overflow"); ok = False
    except AssertionError as e:
        print(f"  DSV41_CHECK_ROUTING=1 correctly refuses: {str(e)[:80]}...")
    try:
        K.moe_forward(x, slots, w, arena, 10.0, slots_repeat=True)
    except AssertionError as e:
        print(f"FAIL: the fixed path tripped its own check: {e}"); ok = False
    finally:
        os.environ.pop("DSV41_CHECK_ROUTING", None)

    # A single-box decode block (distinct experts, no null slot) must be untouched by all of this.
    solo = torch.arange(6, dtype=torch.int32, device=dev).reshape(1, 6)
    xs = torch.randn(1, K.DIM, dtype=torch.bfloat16, device=dev)
    ws = torch.rand(1, 6, dtype=torch.float32, device=dev)
    if not torch.equal(K.moe_forward(xs, solo, ws, arena, 10.0),
                       K.moe_forward(xs, solo, ws, arena, 10.0)):
        print("FAIL: single-box path is not deterministic"); ok = False

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
