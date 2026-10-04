"""Bounded, byte-exact host row cache for immutable Engram tables.

Direct mapped: one int64 tag and 264 payload bytes per slot, no per-row Python
objects. Hits are copied under the lock; misses use the existing parallel reader
outside it, preserving read-ahead concurrency. Collisions only cause extra reads.
"""
import threading
import os

import numpy as np


def cache_budget_mb(rank):
    """Both ranks receive the same settings; only capacity is rank-specific."""
    head = int(os.environ.get('DSV41_ENGRAM_CACHE_MB', '0'))
    peer = int(os.environ.get('DSV41_ENGRAM_CACHE_MB_PEER', str(head)))
    if not all(0 <= mb <= 4096 for mb in (head, peer)):
        raise ValueError('Engram cache budgets must be between 0 and 4096 MiB per rank')
    return peer if rank == 1 else head


class PackedRowCache:
    VERSION = 1
    SLOT_BYTES = 272

    def __init__(self, budget_bytes, n_rows):
        self.capacity = int(budget_bytes) // self.SLOT_BYTES
        if self.capacity < 1 or n_rows < 1:
            raise ValueError('row cache needs at least one slot and a nonempty table')
        self.n_rows = n_rows
        self.keys = np.full(self.capacity, -1, dtype=np.int64)
        self.data = np.empty((self.capacity, 264), dtype=np.uint8)
        self.lock = threading.Lock()
        self.enabled = True
        self.hits = self.misses = self.calls = 0
        self.filled = self.evictions = 0

    def gather(self, ids, reader):
        if ids.dtype != np.int64 or ids.ndim != 1 or not ids.flags.c_contiguous:
            raise ValueError('row IDs must be a contiguous int64 vector')
        if len(ids) and (ids.min() < 0 or ids.max() >= self.n_rows):
            raise IndexError('Engram row ID out of bounds')
        # Engram row IDs are already hashed, spread across disjoint head ranges.
        slots = ids % self.capacity
        out = np.empty((len(ids), 264), dtype=np.uint8)
        with self.lock:
            hit = self.keys[slots] == ids
            out[hit] = self.data[slots[hit]]
            nhit = int(hit.sum())
            self.hits += nhit
            self.misses += len(ids) - nhit
            self.calls += 1
        if nhit != len(ids):
            missing = ids[~hit]
            rows = reader(missing)
            out[~hit] = rows
            target = slots[~hit]
            # Only one writer per slot in the indexed assignment. Otherwise
            # NumPy duplicate-index ordering could mismatch a tag and its bytes.
            unique, first = np.unique(target, return_index=True)
            with self.lock:
                previous = self.keys[unique]
                self.filled += int((previous == -1).sum())
                self.evictions += int(((previous != -1) & (previous != missing[first])).sum())
                self.data[unique] = rows[first]
                self.keys[unique] = missing[first]
        return out

    def clear(self):
        # Call only at request boundaries, after read-ahead futures have joined.
        with self.lock:
            self.keys.fill(-1)
            self.hits = self.misses = self.calls = 0
            self.filled = self.evictions = 0

    def report(self):
        with self.lock:
            return dict(enabled=self.enabled, capacity_rows=self.capacity,
                        allocated_bytes=self.keys.nbytes + self.data.nbytes,
                        filled_rows=self.filled, evictions=self.evictions,
                        hits=self.hits, misses=self.misses, calls=self.calls)
