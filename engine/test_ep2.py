"""test_ep2.py -- CPU-only unit tests for the EP2 skeleton (engine/dist.py, experts.py).

Deliberately GPU-free and process-group-light where possible, so they run on a box whose
unified pool is busy (a CUDA context costs ~0.5 GB even to probe). The full two-GPU parity
gates -- spec losslessness across nodes, NLL within +/-0.01 nats, chunk invariance under
the split sum -- are docs/dual-spark-plan.md Phase 2 and need both boxes free.

Run: python3 engine/test_ep2.py
"""

import os
import sys
import tempfile
import unittest
from collections import OrderedDict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

from engine import experts as EX          # noqa: E402
from engine.dist import EPDistributed     # noqa: E402


def bare_store(rank, world, slots=512, transient=100):
    """An ExpertStore with resolve()'s bookkeeping fields but none of the I/O machinery
    (whose __init__ pins host memory -- a CUDA call on this hardware). Tests that need a
    load path monkeypatch _load_into_slot and give it a real ThreadPoolExecutor."""
    st = object.__new__(EX.ExpertStore)
    st.arena = None                            # warm_start's per_slot getattr falls back to EXPERT_BYTES
    ep = EPDistributed(rank=rank, world_size=world)
    st.ep = ep
    st.n_slots = slots - 1                     # last slot is the null slot (no arena here)
    st.null_slot = slots - 1
    st.transient_slots = transient
    st.lru_slots = st.n_slots - transient
    st.lru = OrderedDict()
    st.slot_key = {}
    st.free_lru = list(range(st.lru_slots))
    st.transient_ring = list(range(st.lru_slots, st.n_slots))
    st.transient_index = {s: i for i, s in enumerate(st.transient_ring)}
    st.transient_pos = 0
    st.transient_map = {}
    st.n_experts = 384
    st.stats = dict(EX.ZERO_STATS)
    return st


class Ownership(unittest.TestCase):
    def test_partition_is_exact(self):
        a, b = EPDistributed(rank=0, world_size=2), EPDistributed(rank=1, world_size=2)
        for e in range(384):
            self.assertTrue(a.owns(7, e) ^ b.owns(7, e), f"expert {e} owned by both/neither")

    def test_resolve_maps_remote_to_null_and_counts_nothing(self):
        st = bare_store(rank=0, world=2)
        layer = 3
        for e in (0, 2):                       # rank 0 owns the evens; make them residents
            st.lru[(layer, e)] = e
        ex = torch.tensor([[0, 1, 2, 3], [2, 5, 7, 3]], dtype=torch.int64)
        st.pool = None                         # any real load would explode: none must happen
        slots = st.resolve(layer, ex, prefill=False)
        self.assertEqual(slots.tolist(), [[0, st.null_slot, 2, st.null_slot],
                                          [2, st.null_slot, st.null_slot, st.null_slot]])
        # 0 and 2 are hits; 1,3,5,7 are REMOTE -- not misses, not ring traffic, not reads
        self.assertEqual(st.stats["hits"], 2)
        self.assertEqual(st.stats["misses"] + st.stats["prefill_misses"], 0)
        self.assertEqual(st.stats["remote"], 4)

    def test_null_slot_sharing_passes_the_collision_assert(self):
        # every remote pair legitimately shares one slot; the LRU/ring collision rule must
        # not fire on it (fp4_moe groups pairs by slot, which already allows sharing).
        st = bare_store(rank=0, world=2)
        st.pool = None
        ex = torch.tensor([[1, 3, 5, 7, 9, 11]], dtype=torch.int64)   # six remote, zero owned
        slots = st.resolve(0, ex, prefill=True)
        self.assertTrue((slots == st.null_slot).all())

    def test_warm_start_filters_to_owned_before_capacity_cut(self):
        # the bug this guards: slicing ranked_keys[:lru_slots] BEFORE the ownership filter
        # would load rank 0 only the owned experts of the first global slice -- half the
        # arena's worth -- instead of the top lru_slots OF ITS OWNED HALF.
        from concurrent.futures import ThreadPoolExecutor
        st = bare_store(rank=0, world=2, slots=512, transient=100)
        st.lru_slots = 10
        st.free_lru = list(range(10))
        st.pool = ThreadPoolExecutor(1)
        loaded = []
        st._load_into_slot = lambda key, slot: loaded.append(key)       # pretend
        ranked = [(0, e) for e in range(40)]                             # all layer 0
        st.warm_start(ranked, log=lambda *a: None)
        self.assertEqual(sorted(k[1] for k in loaded), [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
        self.assertTrue(all(e % 2 == 0 for _, e in loaded))
        self.assertEqual(len(loaded), 10)                               # exactly LRU capacity
        st.pool.shutdown()


class Combine(unittest.TestCase):
    """combine() over a real 2-rank gloo group. gloo stands in for NCCL so this runs on a
    busy GPU box; the value semantics (both ranks leave with identical bits) are the same
    guarantee NCCL makes. fp32-only is an interface assertion, not a performance claim."""

    @staticmethod
    def _worker(rank, port):
        import traceback
        os.environ.update(RANK=str(rank), WORLD_SIZE="2", MASTER_ADDR="127.0.0.1",
                          MASTER_PORT=str(port))
        import torch.distributed as dist
        try:
            ep = EPDistributed()                       # reads RANK/WORLD_SIZE from env
            ep.init("cpu")
            full = torch.arange(16, dtype=torch.float32).reshape(2, 8)   # [T=2 tokens, K=8 pairs]
            owned = (torch.arange(8) % 2 == rank).float()                 # parity split, like expert ids
            partial = (full * owned).sum(1)                              # this rank's k-subset sum
            got = ep.combine(partial)
            ref = full.sum(1)
            assert (got - ref).abs().max() < 1e-6, f"combine wrong on rank {rank}: {got} vs {ref}"
            assert ep.broadcast_request("prompt-ids" if rank == 0 else None) == "prompt-ids"
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def test_two_rank_combine_and_request_broadcast(self):
        import socket
        import torch.multiprocessing as mp
        with socket.socket() as s:                     # a free port, not a hardcoded one that
            s.bind(("127.0.0.1", 0))                   # a previous crashed run may have wedged
            port = s.getsockname()[1]
        with tempfile.TemporaryDirectory() as td:
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
            procs = []
            for r in (0, 1):
                p = mp.Process(target=self._worker, args=(r, port))
                p.start()
                procs.append(p)
            for p in procs:
                p.join(60)
                self.assertEqual(p.exitcode, 0, f"gloo rank exited {p.exitcode} (timeout=60s kill)")


class Lockstep(unittest.TestCase):
    """control() -- the decode loop's stop decision, which rank 0 owns.

    The bug this pins down: rank 1 cannot see a text stop sequence, a client disconnect or the
    server's max_tokens re-clamp, so if it evaluates `n_out < max_tokens and tok not in stop_ids`
    itself it keeps stepping after rank 0 has left. The pair then mismatches collectives and the
    damage lands on the NEXT request. Here rank 0 stops after 3 steps for a reason rank 1 has no
    way to know, and rank 1 must stop after exactly 3.
    """

    @staticmethod
    def _worker(rank, port, steps):
        import traceback
        os.environ.update(RANK=str(rank), WORLD_SIZE="2", MASTER_ADDR="127.0.0.1",
                          MASTER_PORT=str(port))
        import torch.distributed as dist
        try:
            ep = EPDistributed()
            ep.init("cpu")
            if rank == 0:
                for _ in range(steps):
                    assert ep.control(True) is True, "rank 0 gets back what it sent"
                # The abort path: GeneratorExit at the pending `yield`. Rank 1 is blocked in the
                # control() of a step that will now never run, and this is what frees it.
                ep.release_peer()
            else:
                n = 0
                # `True` is rank 1's OWN condition -- it still wants to keep going, every time.
                # Every exit here comes from rank 0's answer, which is the whole point.
                while ep.control(True):
                    n += 1
                assert n == steps, f"rank 1 ran {n} steps, rank 0 ran {steps}"
            assert ep.broadcast_request("in-step" if rank == 0 else None) == "in-step", \
                "the pair must still be usable for the next request"
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def test_rank0_owns_the_stop_decision(self):
        import socket
        import torch.multiprocessing as mp
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        procs = []
        for r in (0, 1):
            p = mp.Process(target=self._worker, args=(r, port, 3))
            p.start()
            procs.append(p)
        for p in procs:
            p.join(60)
            self.assertEqual(p.exitcode, 0, f"gloo rank exited {p.exitcode} (timeout=60s kill)")

    def test_inert_at_world_one(self):
        """The single-box path must not gain a collective, a branch or a behaviour change."""
        ep = EPDistributed(rank=0, world_size=1)
        self.assertFalse(ep.active)
        self.assertTrue(ep.control(True))
        self.assertFalse(ep.control(False))
        ep.release_peer()   # no process group exists; must be a no-op, not a crash


if __name__ == "__main__":
    unittest.main(verbosity=2)
