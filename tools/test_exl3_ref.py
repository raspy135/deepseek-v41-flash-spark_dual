"""P1 gate for EXL3: the torch reference decoder vs the numpy oracle, the rank slice, and the pack.

Run:  .venv/bin/python tools/test_exl3_ref.py
Env:  EXL3_SOURCE (default ~/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw)
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
import exl3_ref as R  # noqa: E402
from pack_exl3_experts import SourceCheckpoint  # noqa: E402

MD = os.environ.get("EXL3_SOURCE", os.path.expanduser("~/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw"))
fails = []


def ok(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'} {msg}")
    if not cond:
        fails.append(msg)


def main():
    if not os.path.isdir(MD):
        print(f"SKIP: {MD} not present")
        return 0
    src = SourceCheckpoint(MD)
    bits_map = src.layer_bits()
    print(f"source {MD}: {len(bits_map)} layers, bits "
          f"{sorted(set(bits_map.values()))}, codebook {src.codebook}")

    # 1. the codebook itself
    print("codebook mul1")
    cb_t = R.codebook("mul1").numpy()
    ok(np.array_equal(cb_t, F.codebook("mul1")), "torch codebook == numpy codebook (65536/65536)")

    # 2. unpack bit-exact on real trellis from a 3-bit, a 2-bit and the last layer
    for layer in (0, 18, 39):
        bits = bits_map[layer]
        name = f"layers.{layer}.ffn.experts.0.w1.trellis"
        full = src.array(name)
        tr = np.ascontiguousarray(full[:64])                    # 64 k tiles = K 1024, multiple of 128
        tq = R.unpack(torch.from_numpy(np.array(tr)), bits, "mul1")
        nq = F.unpack(tr, bits, "mul1")
        ok(torch.equal(tq, torch.from_numpy(nq)),
           f"layer {layer} ({bits:g}-bit) unpack bit-exact, {tuple(tq.shape)}")

    # 3. dequantize close to the numpy oracle
    layer, bits = 0, bits_map[0]
    p = f"layers.{layer}.ffn.experts.0."
    tr = np.ascontiguousarray(src.array(p + "w1.trellis")[:64])
    suh = src.array(p + "w1.suh")[:1024]
    svh = src.array(p + "w1.svh")
    dq = R.dequantize(tr, suh, svh, bits, "mul1").numpy()
    dn = F.dequantize(tr, suh, svh, bits, "mul1")
    rel = np.linalg.norm(dq - dn) / np.linalg.norm(dn)
    ok(rel < 1e-12, f"layer {layer} dequantize vs oracle: rel L2 {rel:.2e}")

    # 4. forward close to the numpy oracle (random input, same order)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 1024)).astype(np.float32)
    yt = R.forward(x, tr, suh, svh, bits, "mul1").numpy()
    yn = F.forward(x, tr, suh, svh, bits, "mul1")
    rel = np.linalg.norm(yt - yn) / np.linalg.norm(yn)
    ok(rel < 1e-10, f"forward vs oracle: rel L2 {rel:.2e}")

    # 5. the rank slice equals the slice of the full decode, and the ranks partition N
    print("TP-output rank slices")
    full = torch.from_numpy(np.array(src.array(p + "w1.trellis")[:64]))     # K = 1024, all 144 N tiles
    wq_full = R.unpack(full, bits, "mul1")
    svh_full = torch.from_numpy(np.array(src.array(p + "w1.svh")))
    for rank, world in ((0, 2), (1, 2)):
        rec = src.record(layer, 0, rank, world)
        wq_rank = R.unpack(torch.from_numpy(np.array(rec["t1"][:64])), bits, "mul1")
        n_rank = R.record_shapes(bits, rank, world)["svh1"][0]
        lo = rank * n_rank
        ok(wq_rank.shape == wq_full[:, lo:lo + n_rank].shape
           and torch.equal(wq_rank, wq_full[:, lo:lo + n_rank]),
           f"rank {rank} W_q slice [:, {lo}:{lo + n_rank}] exact")
        ok(np.array_equal(rec["svh1"], svh_full.numpy()[lo:lo + n_rank]),
           f"rank {rank} svh slice exact")
    r0 = src.record(layer, 0, 0, 2)
    r1 = src.record(layer, 0, 1, 2)
    ok(np.array_equal(np.concatenate([r0["svh1"], r1["svh1"]], axis=0), svh_full.numpy()),
       "the two ranks' svh slices rejoin the full svh")

    # 6. the pack round-trips and every record is 4096-aligned
    print("pack round-trip")
    tmp = tempfile.mkdtemp(prefix="exl3pack-")
    try:
        mini_bits = {0: bits_map[0], 18: bits_map[18]}
        out = os.path.join(tmp, "exl3-experts-r0of2.bin")

        def records():
            for L in mini_bits:
                for E in (0, 1):
                    yield (L, E), src.record(L, E, 0, 2)

        R.write_pack(out, mini_bits, 0, 2, "mul1", "deadbeef", records(), n_experts=2)
        rd = R.PackReader(out)
        ok(all(v["off"] % R.ALIGN == 0 for v in rd.header["records"].values()),
           "every record offset is 4096-aligned")
        for L in mini_bits:
            for E in (0, 1):
                back = rd.read_expert(L, E)
                want = src.record(L, E, 0, 2)
                same = all(np.array_equal(back[n], want[n]) for n in R.RECORD_TENSORS)
                ok(same, f"pack round-trip layer {L} expert {E} ({mini_bits[L]:g}-bit), "
                         f"{sum(want[n].nbytes for n in R.RECORD_TENSORS) / 1e6:.2f} MB")
        ok(rd.header_sha256() == R.PackReader(out).header_sha256(), "header sha256 is stable")
        ok(int(rd.header["records"]["0,0"]["bits"]) == mini_bits[0], "header carries per-expert bits")
        # unpack straight from the pack and compare to the source decode
        rec = rd.read_expert(0, 0)
        got = R.unpack(torch.from_numpy(np.array(rec["t1"])), bits_map[0], "mul1")
        want = R.unpack(torch.from_numpy(np.array(src.record(0, 0, 0, 2)["t1"])), bits_map[0], "mul1")
        ok(torch.equal(got, want), "unpack(pack record) == unpack(source slice)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'FAIL: ' + str(len(fails)) if fails else 'all checks passed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
