"""Experimental EP2 prefill-only replicas. Original ownership and decode LUT stay intact.

The first chunk measures actual (post-pruning) routing. Whole experts may execute on
the other rank for later chunks. No expert is dropped or computed twice. This changes
the grouping of FP32 sums, so full-model quality checks are required before deployment.
"""
import time
import numpy as np


def plan_replicas(counts, keep, capacity):
    """Greedily reduce predicted per-layer maximum assignment count, not just slot count.

Counts predict later chunks; they do not predict milliseconds or guarantee savings.
Each replica occupies one slot globally on its destination, not one slot per layer.
"""
    counts, keep = np.asarray(counts), np.asarray(keep, dtype=bool)
    if counts.ndim != 2 or counts.shape != keep.shape or capacity < 0:
        raise ValueError('invalid replica planner shape/capacity')
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts[~keep] != 0).any():
        raise ValueError('counts must be finite nonnegative actual kept-expert demand')
    work = np.stack((counts[:, ::2].sum(1), counts[:, 1::2].sum(1)), axis=1).astype(float)
    before = float(work.max(axis=1).sum())
    moves, used = [], np.zeros(2, dtype=np.int32)
    selected = np.zeros_like(keep)
    layers = np.arange(len(counts))
    parity = np.arange(counts.shape[1]) % 2
    while True:
        src = (work[:, 1] > work[:, 0]).astype(np.int32)
        dst = 1 - src
        gain = work.max(axis=1)[:, None] - np.maximum(
            work[layers, src][:, None] - counts, work[layers, dst][:, None] + counts)
        eligible = (keep & ~selected & (counts > 0) & (parity[None, :] == src[:, None])
                    & (used[dst, None] < capacity))
        gain = np.where(eligible, gain, -np.inf)
        if gain.size == 0:
            break
        layer, expert = np.unravel_index(np.argmax(gain), gain.shape)
        improvement = float(gain[layer, expert])
        if improvement <= 0:
            break
        destination = int(dst[layer])
        amount = float(counts[layer, expert])
        moves.append(dict(layer=int(layer), expert=int(expert), dst=destination,
                          slot=int(used[destination]), predicted_gain=improvement))
        used[destination] += 1
        selected[layer, expert] = True
        work[layer, 1-destination] -= amount
        work[layer, destination] += amount
    return dict(moves=moves, predicted_max_before=before,
                predicted_max_after=float(work.max(axis=1).sum()))


def routing_tables(lru, moves, rank, null_slot, replica_slots, n_layers, n_experts, keep):
    """Build private prefill tables without modifying original resident/decode state."""
    keep = np.asarray(keep, dtype=bool)
    seen, destinations = set(), set()
    for row in moves:
        layer, expert, dst, slot = (int(row[k]) for k in ('layer', 'expert', 'dst', 'slot'))
        if (not 0 <= layer < n_layers or not 0 <= expert < n_experts or
                dst != 1 - expert % 2 or not 0 <= slot < len(replica_slots) or
                not keep[layer, expert] or (layer, expert) in seen or (dst, slot) in destinations):
            raise ValueError('invalid/duplicate replica move')
        seen.add((layer, expert))
        destinations.add((dst, slot))
        if expert % 2 == rank and (layer, expert) not in lru:
            raise ValueError('cannot replicate a nonresident source')
    mapping = dict(lru)
    for row in moves:
        key = row['layer'], row['expert']
        if row['dst'] == rank:
            mapping[key] = replica_slots[row['slot']]
        else:
            mapping.pop(key, None)
    lut = np.full((n_layers, n_experts), null_slot, dtype=np.int32)
    routes = {}
    for layer in range(n_layers):
        pairs = sorted((e, s) for (l, e), s in mapping.items() if l == layer)
        ids = np.full(n_experts, len(pairs), dtype=np.int32)
        slots = np.array([s for _, s in pairs] + [null_slot], dtype=np.int32)
        for i, (expert, slot) in enumerate(pairs):
            lut[layer, expert] = slot
            ids[expert] = i
        routes[layer] = ids, slots
    return lut, routes


class PrefillReplicas:
    def __init__(self, engine, budget_ms=500.0, batch=8):
        self.engine = engine
        self.budget_ms = budget_ms
        self.batch = batch
        self.stats = {}

    def begin(self, enabled):
        import torch
        self.clear()
        self.stats = dict(enabled=bool(enabled), loaded=0, load_ms=0.0)
        if enabled:
            e = self.engine
            e.model.prefill_replica_counts = torch.zeros(
                (e.args.n_layers, 384), dtype=torch.int32, device=e.device)

    def clear(self):
        m = self.engine.model
        m.prefill_replica_counts = None
        m.prefill_replica_lut = None
        m.prefill_replica_routes = None

    def _status(self, started, failed=False):
        import torch
        import torch.distributed as dist
        # Every rank reaches every status reduction, including ranks with zero loads.
        status = torch.tensor([(time.perf_counter()-started)*1000, float(failed)],
                              dtype=torch.float32, device=self.engine.device)
        dist.all_reduce(status, op=dist.ReduceOp.MAX)
        return status.tolist()

    def finish_probe(self):
        import torch
        e, m = self.engine, self.engine.model
        counts = m.prefill_replica_counts
        m.prefill_replica_counts = None
        if counts is None:
            return
        started = time.perf_counter()
        plan = None
        preparation_failed = False
        try:
            keep = torch.stack([e.model_prune_mask[l] for l in range(e.args.n_layers)]).cpu().numpy()
        except Exception:
            keep = np.zeros((e.args.n_layers, 384), dtype=bool)
            preparation_failed = True
        if e.ep.rank == 0:
            try:
                plan = plan_replicas(counts.cpu().numpy(), keep, len(e.store.replica_slots))
            except Exception:
                plan = dict(moves=[], planning_failed=True)
        plan = e.ep.broadcast_obj(plan)
        moves = plan['moves']
        # Validate the complete plan on both ranks before submitting any I/O.
        failed = preparation_failed or bool(plan.get('planning_failed'))
        try:
            routing_tables(e.store.lru, moves, e.ep.rank, e.store.null_slot,
                           e.store.replica_slots, e.args.n_layers, 384, keep)
        except Exception:
            failed = True
        elapsed, error = self._status(started, failed)
        accepted = []
        last_wave_ms = 0.0
        for offset in range(0, len(moves), self.batch):
            # Admission budget, not a timeout/cancellation of active disk reads.
            if error or elapsed + last_wave_ms >= self.budget_ms:
                break
            wave = moves[offset:offset+self.batch]
            wave_start = elapsed
            futures = []
            failed = False
            try:
                local = [r for r in wave if r['dst'] == e.ep.rank]
                # Open/cache shard descriptors on this thread, before concurrent reads.
                for r in local:
                    prefix = f"layers.{r['layer']}.ffn.experts.{r['expert']}."
                    e.store._shard(prefix+'w1.weight').expert_runs(prefix)
                for r in local:
                    futures.append(e.store.pool.submit(e.store._load_into_slot,
                                   (r['layer'], r['expert']), e.store.replica_slots[r['slot']]))
            except Exception:
                failed = True
            for future in futures:
                try:
                    future.result()
                except Exception:
                    failed = True
            elapsed, error = self._status(started, failed)
            last_wave_ms = elapsed - wave_start
            if error:
                break
            accepted.extend(wave)
        # No partial activation on failure. Original routing has never been modified.
        lut, routes = None, None
        if not error and accepted:
            try:
                cpu_lut, cpu_routes = routing_tables(e.store.lru, accepted, e.ep.rank,
                    e.store.null_slot, e.store.replica_slots, e.args.n_layers, 384, keep)
                lut = torch.as_tensor(cpu_lut, device=e.device)
                routes = {l: (torch.as_tensor(ids, device=e.device), torch.as_tensor(slots, device=e.device))
                          for l, (ids, slots) in cpu_routes.items()}
            except Exception:
                error = True
        elapsed, error = self._status(started, error)
        if not error:
            m.prefill_replica_lut, m.prefill_replica_routes = lut, routes
        self.stats.update(loaded=0 if error else len(accepted), proposed=len(moves),
                          load_ms=round(elapsed, 3), failed=bool(error),
                          budget_exceeded=elapsed > self.budget_ms,
                          predicted_max_before=plan.get('predicted_max_before'),
                          predicted_max_after=(None if plan.get('predicted_max_before') is None else
                              plan['predicted_max_before'] - (0 if error else sum(r['predicted_gain'] for r in accepted))))
