"""EXL3 expert store: the pack reader behind ``engine/experts.py::ExpertStore``.

``Exl3ExpertStore`` keeps ALL of the store's policy -- LRU, transient ring, slot LUT, warm start,
swaps, stats -- and overrides only the two format-specific points:

  * ``_read_leased``: O_DIRECT-read one 4096-aligned record out of the pack into a pinned
    staging buffer and hand back the 9 tensors as views (instead of the 6 safetensors runs);
  * ``_copy_into_slot``: hand those views to ``Exl3Arena.load_slot`` on the copy stream.

One record per expert (already the rank slice), so it is a single preadv, exactly like the FP4
``ShardFile`` spans.  The base class is not forked and the FP4 path is untouched.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "engine")):
    if p not in sys.path:
        sys.path.insert(0, p)

import exl3_ref as R  # noqa: E402
from experts import ALIGN, ExpertStore, _pread_chunk  # noqa: E402

_TORCH_DTYPE = {np.dtype(np.int16): torch.int16, np.dtype(np.float16): torch.float16}


class Exl3ExpertStore(ExpertStore):
    """FP4 store policy, EXL3 pack bytes. `pack_path` is one rank's ``exl3-experts-r?of?.bin``."""

    def __init__(self, pack_path: str, arena, n_layers: int, ep=None, **kwargs):
        # The base __init__ builds the LRU/ring/pools and a staging buffer sized for FP4. It wants an
        # index it never reads on this path, so hand it an empty one and swap the buffer after.
        # ep=None: EXL3 is TP, not EP, so there is no null slot to reserve -- passing the real ep
        # would trip the base's EP null-slot assertion and reserve a slot nothing routes to. The
        # attribute is restored below for callers that ask (model.py checks store.ep.tensor_parallel).
        super().__init__("", {"weight_map": {}}, arena, n_layers, ep=None, **kwargs)
        self.ep = ep
        self.pack_path = pack_path
        self.pack = R.PackReader(pack_path)
        if (self.pack.rank, self.pack.world) != (arena.tp_rank, arena.tp_world):
            raise ValueError(f"{pack_path}: pack is rank {self.pack.rank}/{self.pack.world}, "
                             f"arena is rank {arena.tp_rank}/{arena.tp_world}")
        self.fd = os.open(pack_path, os.O_RDONLY | os.O_DIRECT)
        self.expert_bytes = self.arena.bytes_per_slot
        size = self.expert_bytes + 8 * ALIGN
        self.stage = [torch.empty(size, dtype=torch.uint8, pin_memory=True) for _ in range(self.io_threads)]
        self.stage_mv = [memoryview(b.numpy()) for b in self.stage]

    # -- the read path ---------------------------------------------------------------------------
    def _read_leased(self, layer: int, expert: int, prefix, sink):
        """Read one pack record into a pinned buffer and build its 9 tensor views while leased."""
        meta = self.pack.records.get((layer, expert))
        if meta is None:
            raise KeyError(f"pack has no record for layer {layer} expert {expert}")
        nbytes = int(meta["nbytes"])
        off = self.pack.data0 + int(meta["off"])
        shapes = R.record_shapes(float(meta["bits"]), self.arena.tp_rank, self.arena.tp_world)
        sid = self._lease()
        buf = self.stage[sid]
        try:
            mv = self.stage_mv[sid]
            base = (-buf.data_ptr()) % ALIGN
            n = (nbytes + ALIGN - 1) // ALIGN * ALIGN
            t0 = time.perf_counter()
            _pread_chunk(self.fd, mv[base: base + n], off, n)
            self.stats["bytes_read"] += nbytes
            self.stats["read_s"] += time.perf_counter() - t0
            self.stats["loads"] += 1
            out, cur = {}, base
            for name in R.RECORD_TENSORS:
                shape = tuple(shapes[name])
                nb = int(np.prod(shape)) * np.dtype(R._RECORD_DTYPES[name]).itemsize
                # a typed view over the pinned buffer: int16/fp16, offset even, so .view is legal
                out[name] = buf[cur: cur + nb].view(_TORCH_DTYPE[np.dtype(R._RECORD_DTYPES[name])]).view(shape)
                cur += nb
            return sink(out)
        finally:
            self._release(sid)

    def _copy_into_slot(self, slot, key, views, stream, compute) -> None:
        with torch.cuda.stream(stream):
            stream.wait_stream(compute)
            self.arena.load_slot(slot, views, float(self.pack.records[key]["bits"]), non_blocking=True)

    def read_expert(self, layer: int, expert: int, prefix=None):
        """The 9 rank-slice tensors of one expert (CPU), for tests and the reference path."""
        return self._read_leased(layer, expert, prefix, lambda v: {k: t.clone() for k, t in v.items()})
