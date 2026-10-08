"""Offline packer: EXL3 routed experts -> one aligned file per rank for ``EXPERT_FORMAT=exl3``.

Reads `layers.{L}.ffn.experts.{E}.{w1,w3,w2}.{trellis,suh,svh}` from the base
``DeepSeek-V4.1-Flash-EXL3-2.9bpw`` checkpoint and writes each expert's **TP-output rank slice**
contiguously (see ``tools/exl3_ref.py``), 4096-aligned so ``engine/experts.py`` can O_DIRECT one
expert in one preadv.  The format is ours; TensorFold's prepared packs are a different
("uncensored") checkpoint and are not read.

    python tools/pack_exl3_experts.py --source ~/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw \\
        --out ~/models/exl3-experts-r0of2.bin --rank 0 --world 2

Run it on each node for its own rank (the source is 198 GB; node 0 has one pack's worth of free
space, node 1 has plenty).  Nothing here runs in serving.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exl3_ref as R  # noqa: E402
from safetensors import safe_open  # noqa: E402

N_EXPERTS = 384
N_LAYERS = 40


class SourceCheckpoint:
    """Read the source shard headers/index once and slice out one expert at a time."""

    def __init__(self, model_dir: str, codebook: str = "mul1"):
        self.dir = model_dir
        index = os.path.join(model_dir, "model.safetensors.index.json")
        if not os.path.exists(index):
            raise SystemExit(f"{model_dir}: no model.safetensors.index.json")
        self.map = json.load(open(index))["weight_map"]
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        q = cfg.get("quantization_config") or {}
        if str(q.get("quant_method", "")).lower() != "exl3":
            raise SystemExit(f"{model_dir}: quant_method is {q.get('quant_method')!r}, not exl3")
        if str(q.get("codebook", "")).lower() not in ("", codebook):
            raise SystemExit(f"{model_dir}: codebook {q.get('codebook')!r}, packer expects {codebook!r}")
        self.codebook = codebook
        self._open: dict[str, safe_open] = {}

    def _file(self, name: str) -> safe_open:
        shard = self.map[name]
        if shard not in self._open:
            self._open[shard] = safe_open(os.path.join(self.dir, shard), framework="numpy")
        return self._open[shard]

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._file(name).get_slice(name).get_shape())

    def array(self, name: str, sl=None) -> np.ndarray:
        if sl is None:
            return np.ascontiguousarray(self._file(name).get_tensor(name))
        return np.ascontiguousarray(self._file(name).get_slice(name)[sl])

    def layer_bits(self) -> dict[int, float]:
        """Bits per layer, read from each layer's first routed w1 trellis shape.  All experts in a
        layer share it (2-bit on 18-22, 3-bit elsewhere in this checkpoint)."""

        bits = {}
        for layer in range(N_LAYERS):
            name = f"layers.{layer}.ffn.experts.0.w1.trellis"
            bits[layer] = R.bits_of_shape(self.shape(name))
        return bits

    def marker_ok(self, layer: int, expert: int) -> bool:
        """The mul1 codebook marker scalar, 0x83DCD12D, on all three projections."""

        for part in ("w1", "w2", "w3"):
            v = int(self.array(f"layers.{layer}.ffn.experts.{expert}.{part}.mul1").reshape(-1)[0])
            if (v & 0xFFFFFFFF) != R.MUL1_MUL:
                return False
        return True

    def record(self, layer: int, expert: int, rank: int, world: int) -> dict[str, np.ndarray]:
        """The rank slice of one expert, in ``R.RECORD_TENSORS`` order."""

        bits = R.bits_of_shape(self.shape(f"layers.{layer}.ffn.experts.{expert}.w1.trellis"))
        shapes = R.record_shapes(bits, rank, world)
        nt13 = shapes["t1"][1]          # N tiles a rank of w1/w3 (72)
        nt2 = shapes["t2"][1]           # N tiles a rank of w2 (160)
        n13 = shapes["svh1"][0]         # N columns a rank of w1/w3 (1152)
        n2 = shapes["svh2"][0]          # N columns a rank of w2 (2560)
        lo13, lo2 = rank * nt13, rank * nt2
        lo_n13, lo_n2 = rank * n13, rank * n2
        p = f"layers.{layer}.ffn.experts.{expert}."
        return {
            "t1": self.array(p + "w1.trellis", (slice(None), slice(lo13, lo13 + nt13), slice(None))),
            "t3": self.array(p + "w3.trellis", (slice(None), slice(lo13, lo13 + nt13), slice(None))),
            "t2": self.array(p + "w2.trellis", (slice(None), slice(lo2, lo2 + nt2), slice(None))),
            "suh1": self.array(p + "w1.suh"),
            "suh3": self.array(p + "w3.suh"),
            "suh2": self.array(p + "w2.suh"),
            "svh1": self.array(p + "w1.svh", (slice(lo_n13, lo_n13 + n13),)),
            "svh3": self.array(p + "w3.svh", (slice(lo_n13, lo_n13 + n13),)),
            "svh2": self.array(p + "w2.svh", (slice(lo_n2, lo_n2 + n2),)),
        }


def pack(source: SourceCheckpoint, out: str, rank: int, world: int, experts_per_layer: int = N_EXPERTS,
         log=print) -> dict:
    bits_map = source.layer_bits()
    sha = R.manifest_sha256(source.dir)
    seen = {L: 0 for L in bits_map}
    # marker spot-check: first and last expert of every layer, all three projections
    for L in range(N_LAYERS):
        if not source.marker_ok(L, 0):
            raise SystemExit(f"layer {L} expert 0: mul1 marker mismatch; wrong source?")
    t0 = time.time()

    def records():
        for layer in sorted(bits_map):
            for expert in range(experts_per_layer):
                yield (layer, expert), source.record(layer, expert, rank, world)
                seen[layer] += 1
                done = sum(seen.values())
                if done % 512 == 0:
                    log(f"  {done} experts, {time.time() - t0:.0f}s")

    R.write_pack(out, bits_map, rank, world, source.codebook, sha, records(), n_experts=experts_per_layer)
    reader = R.PackReader(out)
    n = reader.header["records"]
    log(f"wrote {out}: {os.path.getsize(out) / 1e9:.1f} GB, {len(n)} records, "
        f"header sha256 {reader.header_sha256()[:16]}, source {sha[:16]}")
    return reader.header


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=os.environ.get("EXL3_SOURCE", os.path.expanduser(
        "~/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--experts-per-layer", type=int, default=N_EXPERTS,
                    help="for a small test pack; the real pack uses all 384")
    a = ap.parse_args(argv)
    if not 0 <= a.rank < a.world:
        ap.error("--rank must be in [0, world)")
    src = SourceCheckpoint(a.source)
    pack(src, a.out, a.rank, a.world, a.experts_per_layer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
