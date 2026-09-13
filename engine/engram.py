"""
engram.py -- Engram table rows at serve time: 24 random 264-byte reads per token per engram
layer, straight from the two 101 GB safetensors shards on NVMe (page cache, buffered preadv in a
thread pool; the reads are far too small for O_DIRECT to help). Hash ids come from the reference
`NgramHashState`, which depends on the token ids only.
"""

from __future__ import annotations

import json
import os
import mmap
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager

import numpy as np
import torch


class EngramTable:
    def __init__(self, model_dir: str, index: dict, layer: int, device: str, threads: int = 32):
        wm = index["weight_map"]
        self.path = os.path.join(model_dir, wm[f"layers.{layer}.engram.embed.weight"])
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        base = 8 + n
        w = hdr[f"layers.{layer}.engram.embed.weight"]
        s = hdr[f"layers.{layer}.engram.embed.scale"]
        assert w["shape"][1] == 256 and s["shape"][1] == 8
        self.w_off = base + w["data_offsets"][0]
        self.s_off = base + s["data_offsets"][0]
        self.n_rows = w["shape"][0]
        self.fd = os.open(self.path, os.O_RDONLY)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
        # Row gather over a memmap instead of a Python loop of preads. The loop cost ~12 us/row
        # even fully warm in page cache (two syscalls plus ~10 bytecode ops per row, and the GIL
        # kept the thread pool from helping): 83k rows/s warm, 47k cold. Fancy-indexing a memmap
        # does the whole gather in C and lets page faults do the I/O -- 8.9M rows/s warm, 891k
        # cold, byte-identical output. DSV41_ENGRAM_MMAP=0 restores the pread path.
        self.mmap_rows = os.environ.get("DSV41_ENGRAM_MMAP", "1") == "1"
        self.w_mm = self.s_mm = None
        if self.mmap_rows:
            self.w_mm = np.memmap(self.path, dtype=np.uint8, mode="r",
                                  offset=self.w_off, shape=(self.n_rows, 256))
            self.s_mm = np.memmap(self.path, dtype=np.uint8, mode="r",
                                  offset=self.s_off, shape=(self.n_rows, 8))
            for mm in (self.w_mm, self.s_mm):   # no readahead: every row lands on its own page
                try:
                    mm._mmap.madvise(mmap.MADV_RANDOM)
                except (AttributeError, OSError):
                    pass
        self.gather_threads = int(os.environ.get("DSV41_ENGRAM_GATHER_THREADS", "64"))
        # EP2 row split: both ranks hash the same tokens, so both were reading the SAME rows off
        # their own NVMe -- the work was duplicated, and it is the prefill bottleneck. With the
        # split each rank reads only `uniq % world == rank` into a zero-filled full array and one
        # all-reduce (the halves are disjoint, so SUM is exactly the union) rebuilds it on both.
        # That halves per-box read volume AND halves the footprint each box's page cache has to
        # cover. Default off: a collective that only one rank reaches wedges the pair.
        # Set by the engine after construction; None means single-box.
        self.ep = None
        self.row_split = os.environ.get("DSV41_ENGRAM_ROW_SPLIT", "0") == "1"
        self.pool = ThreadPoolExecutor(max(threads, self.gather_threads if self.mmap_rows else 0))
        self.device = device
        self.stats = {"rows": 0, "seconds": 0.0, "calls": 0}
        # small process-local row cache (exact n-gram repeats inside a conversation hit here)
        self.cache: dict[int, bytes] = {}
        self.cache_max = 200_000
        # Pinned staging for the H2D (see to_device), OFF by default: it does make `to_device`
        # itself ~10x cheaper (0.27 s vs 3.16 s of a 200-token run, because the pageable copy
        # synchronises the stream and absorbs the queued graph work) but it does NOT make the run
        # faster -- measured 12.04-13.57 s of decode against 12.03-12.14 s pageable, i.e. the host
        # simply blocks somewhere else instead. Kept behind the switch rather than deleted.
        self.pinned = os.environ.get("DSV41_ENGRAM_PINNED", "0") == "1"
        self._stage = [None, None]
        self._inv_stage = [None, None]
        self._ev = [None, None]
        self._cur = 0

    def _read_rows(self, ids: np.ndarray) -> np.ndarray:
        out = np.empty((len(ids), 264), np.uint8)
        for i, r in enumerate(ids):
            r = int(r)
            b = self.cache.get(r)
            if b is None:
                b = os.pread(self.fd, 256, self.w_off + r * 256) + os.pread(self.fd, 8, self.s_off + r * 8)
                if len(self.cache) < self.cache_max:
                    self.cache[r] = b
            out[i] = np.frombuffer(b, np.uint8)
        return out

    def _gather_rows(self, ids: np.ndarray) -> np.ndarray:
        """Parallel memmap gather of the unique rows. Page faults do the I/O, so the threads are
        blocked in the kernel rather than fighting over the GIL."""
        out = np.empty((len(ids), 264), np.uint8)
        nt = max(1, self.gather_threads)
        step = max(1024, len(ids) // nt + 1)

        def one(i):
            sl = slice(i, min(i + step, len(ids)))
            rows = ids[sl]
            out[sl, :256] = self.w_mm[rows]
            out[sl, 256:] = self.s_mm[rows]

        starts = range(0, len(ids), step)
        if len(ids) <= step:
            for i in starts:
                one(i)
        else:
            list(self.pool.map(one, starts))
        return out

    def _splitting(self) -> bool:
        """Both ranks must agree, so this reads only constants -- never per-call state."""
        return bool(self.row_split and self.ep is not None and self.ep.active)

    def read_raw(self, hashes_np: np.ndarray):
        """Host-only part (safe in a background thread: no CUDA calls): NVMe reads of the unique rows.
        hashes_np: int64 [T, 24]. Returns (raw uint8 [n, 264], inv, shape).

        Under the EP2 row split this reads only this rank's share; the rest stays zero and
        to_device's all-reduce fills it in."""
        t1 = time.perf_counter()
        flat = hashes_np.reshape(-1)
        uniq, inv = np.unique(flat, return_inverse=True)
        n = len(uniq)
        if self._splitting():
            # np.unique sorts, so both ranks see identical `uniq` and pick disjoint halves.
            mine = (uniq % self.ep.world) == self.ep.rank
            raw = np.zeros((n, 264), np.uint8)
            sel = uniq[mine]
            if len(sel):
                raw[mine] = (self._gather_rows(sel) if self.mmap_rows
                             else self._read_rows(sel))
            self.stats["read_s"] = self.stats.get("read_s", 0.0) + time.perf_counter() - t1
            self.stats["rows"] += int(len(sel)); self.stats["calls"] += 1
            return raw, inv, hashes_np.shape
        if self.mmap_rows:
            raw = self._gather_rows(uniq)
        else:
            chunk = max(8, n // (self.pool._max_workers * 2) + 1)
            parts = list(self.pool.map(self._read_rows, [uniq[i:i + chunk] for i in range(0, n, chunk)]))
            raw = np.concatenate(parts) if parts else np.empty((0, 264), np.uint8)
        self.stats["read_s"] = self.stats.get("read_s", 0.0) + time.perf_counter() - t1
        self.stats["rows"] += int(n); self.stats["calls"] += 1
        return raw, inv, hashes_np.shape

    def to_device(self, raw, inv, shape) -> torch.Tensor:
        """GPU part (main thread): dequantize the rows and expand to [T, 24, 256] float32.

        The raw rows and the row-index vector go through PINNED staging buffers and are copied
        non-blocking. A plain `.to(device)` from pageable numpy memory synchronises the calling
        stream, which inside a decode step means blocking the host until every graph queued so far
        has finished -- exactly the overlap this pipeline exists to avoid. The buffers are reused
        across steps, so the previous step's copy out of them is waited on first (`_ev`); a whole
        step elapses in between, so that wait is free.
        """
        t0 = time.perf_counter()
        n = raw.shape[0]
        if not self.pinned:
            rawt = torch.from_numpy(raw).to(self.device)
            invt = torch.from_numpy(inv.reshape(-1)).to(self.device)
            if self._splitting() and n:
                # Disjoint halves over a zero background: SUM is the union, bit-exact, no
                # overflow. Must run on every rank for the same (chunk, layer) or the pair wedges
                # -- `n` is derived from the hashes, which both ranks compute identically, so the
                # skip-on-empty is taken on both or neither.
                import torch.distributed as _dist
                _dist.all_reduce(rawt, op=_dist.ReduceOp.SUM)
        else:
            assert not self._splitting(), \
                "DSV41_ENGRAM_PINNED and DSV41_ENGRAM_ROW_SPLIT are not wired together"
            i = self._cur
            self._cur ^= 1  # two buffers: the wait is on the copy from two calls ago
            if self._stage[i] is None or self._stage[i].shape[0] < n:
                cap = max(n, 4096)
                self._stage[i] = torch.empty(cap, 264, dtype=torch.uint8, pin_memory=True)
                self._inv_stage[i] = torch.empty(cap * 64, dtype=torch.int64, pin_memory=True)
                self._ev[i] = torch.cuda.Event()
                self._ev[i].record()
            self._ev[i].synchronize()  # the last H2D out of these buffers is done
            self._stage[i][:n].copy_(torch.from_numpy(raw))
            ni = inv.size if hasattr(inv, "size") else len(inv)
            self._inv_stage[i][:ni].copy_(torch.from_numpy(inv.reshape(-1).astype(np.int64)))
            rawt = self._stage[i][:n].to(self.device, non_blocking=True)
            invt = self._inv_stage[i][:ni].to(self.device, non_blocking=True)
        vals = rawt[:, :256].view(torch.float8_e4m3fn).float()
        scales = torch.exp2(rawt[:, 256:].float() - 127.0)
        deq = (vals.unflatten(-1, (8, 32)) * scales.unsqueeze(-1)).flatten(-2)
        out = deq[invt].view(shape[0], shape[1], 256)
        if self.pinned:
            self._ev[i].record()
        self.stats["seconds"] += time.perf_counter() - t0
        return out

    def rows(self, hashes: torch.Tensor) -> torch.Tensor:
        """hashes: int64 [T, 24] -> float32 [T, 24, 256] dequantized rows."""
        t0 = time.perf_counter()
        flat = hashes.reshape(-1).cpu().numpy()
        uniq, inv = np.unique(flat, return_inverse=True)
        n = len(uniq)
        t1 = time.perf_counter()
        if self.mmap_rows:
            raw = self._gather_rows(uniq)
        else:
            chunk = max(8, n // (self.pool._max_workers * 2) + 1)
            parts = list(self.pool.map(self._read_rows, [uniq[i:i + chunk] for i in range(0, n, chunk)]))
            raw = np.concatenate(parts) if parts else np.empty((0, 264), np.uint8)
        self.stats["read_s"] = self.stats.get("read_s", 0.0) + time.perf_counter() - t1
        raw = torch.from_numpy(raw).to(self.device)
        vals = raw[:, :256].view(torch.float8_e4m3fn).float()
        scales = torch.exp2(raw[:, 256:].float() - 127.0)
        deq = (vals.unflatten(-1, (8, 32)) * scales.unsqueeze(-1)).flatten(-2)  # [n, 256]
        out = deq[torch.from_numpy(inv).to(self.device)].view(hashes.shape[0], hashes.shape[1], 256)
        self.stats["rows"] += int(n); self.stats["seconds"] += time.perf_counter() - t0; self.stats["calls"] += 1
        return out


@contextmanager
def prefetch_rows(tables, pool, hashes, layer_ids):
    """Overlap a chunk's host row reads with its GPU layers, joining on every exit."""
    host_hashes = hashes.cpu().numpy()
    futures = {}
    try:
        for li, layer in enumerate(layer_ids):
            futures[layer] = pool.submit(tables[layer].read_raw, host_hashes[:, li, :])

        def rows(layer, _hashes):
            return tables[layer].to_device(*futures[layer].result())

        yield rows
    finally:
        # Workers must finish before the next request resets counters/caches,
        # including when a forward fails before reaching the second table.
        wait(list(futures.values()))


class EngramReadAhead:
    """Cross-chunk read-ahead for the engram tables.

    `prefetch_rows` submits a chunk's reads at the top of that chunk's OWN forward, and
    `engram_layer_ids` starts at layer 1 -- so only layer 0's GPU work, ~25 ms of a ~1.5 s
    chunk, is available to hide what can be ~1 s of random NVMe reads. Prefill over text
    with novel n-grams therefore alternates roughly 1 s busy / 1 s idle, and the effect is
    invisible to any benchmark whose prompt repeats (a repeated template collapses to a
    handful of unique hashes, which the row cache serves for free).

    The hashes depend on the token ids alone, so the whole prompt can be hashed up front and
    chunk k+1's reads submitted while chunk k is still on the GPU. `depth` chunks are kept in
    flight, which both hides the latency and keeps the NVMe queue deep enough to be worth the
    32-thread pool.
    """

    def __init__(self, tables, pool, layer_ids, hashes_np, chunk: int, depth: int = 2):
        self.tables, self.pool, self.layer_ids = tables, pool, list(layer_ids)
        self.hashes, self.chunk = hashes_np, chunk
        self.depth = max(1, depth)
        self.total = hashes_np.shape[0]
        self.futures: dict[int, dict] = {}

    def _submit(self, s: int):
        if s in self.futures or s >= self.total:
            return
        hs = self.hashes[s:s + self.chunk]
        self.futures[s] = {L: self.pool.submit(self.tables[L].read_raw, hs[:, li, :])
                           for li, L in enumerate(self.layer_ids)}

    def rows_for(self, s: int):
        """Ensure chunk `s` and the next `depth-1` chunks are in flight, and return the
        `get_rows(layer, hashes)` callable `Model.forward` expects for chunk `s`."""
        for k in range(self.depth):
            self._submit(s + k * self.chunk)
        futs = self.futures[s]

        def get_rows(layer, _hashes):
            return self.tables[layer].to_device(*futs[layer].result())

        return get_rows

    def done(self, s: int):
        """Release chunk `s`'s futures once its forward has consumed them."""
        self.futures.pop(s, None)

    def close(self):
        # Workers must finish before the next request resets counters/caches, including when a
        # forward raises before reaching the second table.
        for futs in self.futures.values():
            wait(list(futs.values()))
        self.futures.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make_hash_state(model_dir: str, tokenizer, max_seq: int, device: str):
    """The reference NgramHashState (engram.py from the checkpoint's inference/ folder)."""
    sys.path.insert(0, os.path.join(model_dir, "inference"))
    import engram as E  # noqa: E402
    cfg = json.load(open(os.path.join(model_dir, "inference", "config.json")))

    class A:
        engram_layer_ids = tuple(cfg["engram_layer_ids"])
        engram_max_ngram_size = cfg["engram_max_ngram_size"]
        engram_n_heads = cfg["engram_n_heads"]
        engram_vocab_size = cfg["engram_vocab_size"]
        engram_num_embeddings = tuple(cfg["engram_num_embeddings"])
        engram_head_dim = cfg["engram_head_dim"]
        engram_pad_id = cfg["engram_pad_id"]
        engram_compressed_vocab_size = cfg["engram_compressed_vocab_size"]
        max_batch_size = 1
        max_seq_len = max_seq + 16

    layout = E.EngramLayout.from_args(A)
    st = E.NgramHashState(A, layout, tokenizer)
    return st.to(device)
