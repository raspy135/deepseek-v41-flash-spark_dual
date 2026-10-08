"""P2 gate for the EXL3 store: O_DIRECT pack read -> arena slot through ExpertStore's own policy.

Builds a small world=1 pack from the source, then drives `Exl3ExpertStore.resolve` (LRU path) and
compares the resident slot to the source record. Needs CUDA because the store's staging buffers are
pinned; the arena is 12 slots so it is tiny next to a live server.

Run:  .venv/bin/python tools/test_exl3_store.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exl3_ref as R  # noqa: E402
import exl3_moe as M  # noqa: E402
from exl3_store import Exl3ExpertStore  # noqa: E402
from pack_exl3_experts import SourceCheckpoint  # noqa: E402

SRC = os.environ.get("EXL3_SOURCE", os.path.expanduser("~/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw"))
CASES = [(0, 0), (18, 0), (22, 0), (39, 0)]
fails = []


def ok(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'} {msg}")
    if not cond:
        fails.append(msg)


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    if not os.path.isdir(SRC):
        print(f"SKIP: {SRC} not present")
        return 0
    src = SourceCheckpoint(SRC)
    all_bits = src.layer_bits()
    tmp = tempfile.mkdtemp(prefix="exl3store-")
    try:
        pack = os.path.join(tmp, "exl3-mini-w1.bin")
        bits_map = {L: all_bits[L] for L, _ in CASES}
        R.write_pack(pack, bits_map, 0, 1, src.codebook, "test",
                     ((LE, src.record(*LE, 0, 1)) for LE in CASES), n_experts=1)
        rd = R.PackReader(pack)

        arena = M.Exl3Arena(12, device="cuda", tp_rank=0, tp_world=1, bits=3.0, codebook=rd.codebook)
        store = Exl3ExpertStore(pack, arena, n_layers=40, io_threads=2, transient_slots=8)

        # 1. read_expert: the pinned O_DIRECT read and the 9 typed views
        print("read_expert")
        for L, E in CASES:
            got = store.read_expert(L, E)
            want = rd.read_expert(L, E)
            same = all(torch.equal(got[n].cpu(), torch.from_numpy(np.ascontiguousarray(want[n])))
                       for n in R.RECORD_TENSORS)
            ok(same, f"O_DIRECT read layer {L} expert {E} equals the pack record")

        # 2. resolve(): the LRU path copies into the arena and returns slot ids
        print("resolve -> arena")
        for L, E in CASES:
            slots = store.resolve(L, torch.tensor([[E]], dtype=torch.int32), prefill=False)
            slot = int(slots[0, 0].item())
            got = arena.read_slot(slot)
            want = rd.read_expert(L, E)
            same = all(torch.equal(got[n].cpu(), torch.from_numpy(np.ascontiguousarray(want[n])))
                       for n in R.RECORD_TENSORS)
            ok(same, f"resolve layer {L} expert {E} -> slot {slot}, arena equals the record")
        ok(store.stats["misses"] == len(CASES), f"misses {store.stats['misses']} == {len(CASES)}")
        # a second resolve is an LRU hit and must not re-read
        before = store.stats["loads"]
        store.resolve(0, torch.tensor([[0]], dtype=torch.int32), prefill=False)
        ok(store.stats["loads"] == before, "second resolve of a resident expert does not re-read")
        ok(store.stats["hits"] >= 1, "second resolve counted a hit")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'FAIL: ' + str(len(fails)) if fails else 'all checks passed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
