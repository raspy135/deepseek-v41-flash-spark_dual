"""Lossless two-sequence decode core.

This is deliberately below the HTTP scheduler.  Each lane owns an ordinary
``FastDecoder`` (and therefore its own KV, compressed-index, Engram and rollback
state), while the expensive FFN half of every backbone layer is issued once for
the concatenated 2 x T_VERIFY token rows.  One combined EP all-reduce preserves
collective ordering across the two Sparks.

The class is opt-in scaffolding until its full-model equivalence gate passes; the
single-request serving path does not import or instantiate it.
"""

from __future__ import annotations

import torch

from engine.fastdecode import FastDecoder, T_VERIFY, R
from engine.model import Caches, HC_OPS as _HC_OPS, Model, _hc_post_fused


class Batch2FastDecoder:
    """Run two independent FastDecoder lanes through one batched MoE backbone."""

    def __init__(self, lane0: FastDecoder, lane1: FastDecoder, use_graphs: bool = True):
        if lane0 is lane1:
            raise ValueError("batch-2 lanes must have independent mutable state")
        if lane0.W is not lane1.W or lane0.m.store is not lane1.m.store:
            raise ValueError("batch-2 lanes must share weights and the expert arena")
        if lane0.c is lane1.c:
            raise ValueError("batch-2 lanes must not share KV caches")
        if lane0.a is not lane1.a:
            raise ValueError("batch-2 lanes must use the same model configuration")
        if lane0.lut is None or lane1.lut is None:
            raise ValueError("batch-2 requires the all-resident device expert LUT")
        self.lanes = (lane0, lane1)
        self.a = lane0.a
        self.W = lane0.W
        self.store = lane0.m.store
        self.dev = lane0.dev
        self.use_graphs = use_graphs
        self.pool = None
        self.graphs = {}

    def _layer_b(self, layer: int) -> None:
        """Shared expensive half of a layer; exact per-lane HC state stays separate."""
        a0, a1 = self.lanes
        cfg = self.a
        w = self.W.layers[layer]
        y = torch.cat((a0.y, a1.y), dim=0)
        routing_ids = torch.cat((a0.route_idx, a1.route_idx), dim=0)
        slot_map = a0.lut[layer]
        slots = slot_map[routing_ids]
        route_w = torch.cat((a0.route_w, a1.route_w), dim=0)

        if getattr(self.store, "null_slot", None) is not None:
            out = a0.m.moe_fn(
                y, slots, route_w, self.store.arena, cfg.swiglu_limit,
                out_dtype=torch.float32, slots_repeat=True, null_slot=self.store.null_slot,
                routing_ids=routing_ids, routing_slot_map=slot_map,
            )
            self.store.ep.combine(out)
            out = out.to(torch.bfloat16).float()
        else:
            out = a0.m.moe_fn(
                y, slots, route_w, self.store.arena, cfg.swiglu_limit,
                routing_ids=routing_ids, routing_slot_map=slot_map,
            ).float()

        # The shared expert is identical on both ranks and is evaluated once over the 12 rows.
        out += R.expert_ffn(y, w.sh_w1, w.sh_w2, w.sh_w3, cfg.swiglu_limit).float()
        for lane, part in zip(self.lanes, out.split(T_VERIFY, dim=0)):
            h = (_hc_post_fused(part.to(torch.bfloat16), lane.h,
                                lane.ffn_post, lane.ffn_comb)
                 if _HC_OPS else
                 R.hc_post(part.to(torch.bfloat16), lane.h,
                           lane.ffn_post, lane.ffn_comb))
            lane.h.copy_(h)
            lane.pre_mix.copy_(lane.ffn_pre)

    def _segment(self, lo: int, hi: int, states: tuple[dict, dict]) -> None:
        for layer in range(lo, hi):
            for lane, state in zip(self.lanes, states):
                lane._layer_a(layer, state)
            self._layer_b(layer)
        if hi == self.a.n_layers:
            # Keep final projections lane-local initially.  Combining the two vocabulary heads is
            # a later, independently measurable optimization and is not needed to prove the MoE
            # batching result.
            for lane in self.lanes:
                lane._final()

    def capture(self, key: tuple[int, int, int, int]) -> None:
        if key in self.graphs or not self.use_graphs:
            return
        p0, b0, p1, b1 = key
        states = (
            {"parity": p0, "index_bucket": b0, "ckv": None, "ik": None, "ratio": 0},
            {"parity": p1, "index_bucket": b1, "ckv": None, "ik": None, "ratio": 0},
        )
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        # Compile kernels and allocate library workspaces outside graph capture. Cache writes are
        # append-only and the real replay overwrites these same positions, matching the existing
        # FastDecoder capture contract.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            saved = [(lane.h.clone(), lane.pre_mix.clone()) for lane in self.lanes]
            self._segment(0, self.a.n_layers, states)
            for lane, (h, pre) in zip(self.lanes, saved):
                lane.h.copy_(h)
                lane.pre_mix.copy_(pre)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        states = (
            {"parity": p0, "index_bucket": b0, "ckv": None, "ik": None, "ratio": 0},
            {"parity": p1, "index_bucket": b1, "ckv": None, "ik": None, "ratio": 0},
        )
        bounds = sorted({0, self.a.n_layers} |
                        {layer for layer in self.a.engram_layer_ids
                         if 0 < layer < self.a.n_layers})
        segments = []
        for lo, hi in zip(bounds, bounds[1:]):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                self._segment(lo, hi, states)
            segments.append((lo, graph))
        self.graphs[key] = segments
        torch.cuda.synchronize()

    @staticmethod
    def _load_rows(lane: FastDecoder, rows) -> None:
        if callable(rows):
            rows = rows()
        for layer, value in rows.items():
            if isinstance(value, tuple):
                future, finish = value
                value = finish(*future.result())
            lane.eg_rows[layer].copy_(value)

    def step(self, blocks: tuple[torch.Tensor, torch.Tensor], positions: tuple[int, int],
             engram_rows: tuple[dict, dict]):
        """Execute one speculative verify step for each lane.

        ``engram_rows`` uses the same eager mapping accepted by ``FastDecoder.step``.  The HTTP
        scheduler will later pass boundary futures so host reads overlap the preceding segment.
        """
        buckets = []
        for lane, block, pos in zip(self.lanes, blocks, positions):
            if block.numel() != T_VERIFY or lane.c.len != pos:
                raise ValueError((block.numel(), lane.c.len, pos))
            lane.ids.copy_(block)
            lane.pos.copy_(pos + lane._ar_t)
            lane.prepare_pending_buffers()
            lane.h.copy_(lane.W.embed[lane.ids].unsqueeze(1).expand(-1, self.a.hc_mult, -1))
            lane.pre_mix.copy_(lane._premix0)
            buckets.append(MIN_BUCKET(pos + T_VERIFY, lane.c.max_seq))

        key = (positions[0] % 2, buckets[0], positions[1] % 2, buckets[1])
        if key not in self.graphs and self.use_graphs:
            self.capture(key)
            for lane in self.lanes:
                lane.prepare_pending_buffers()

        # Initial prototype resolves both row sets before replay.  Boundary-overlapped futures are
        # a scheduler optimization; keeping them out of the equivalence gate makes failures local.
        for lane, rows in zip(self.lanes, engram_rows):
            self._load_rows(lane, rows)
        if self.use_graphs:
            for _lo, graph in self.graphs[key]:
                graph.replay()
        else:
            states = (
                {"parity": positions[0] % 2, "index_bucket": buckets[0],
                 "ckv": None, "ik": None, "ratio": 0},
                {"parity": positions[1] % 2, "index_bucket": buckets[1],
                 "ckv": None, "ik": None, "ratio": 0},
            )
            self._segment(0, self.a.n_layers, states)

        for lane, pos in zip(self.lanes, positions):
            for layer in lane.kvl_buf:
                before = lane.c.pending.get(layer)
                lane.c._chunk_inputs[layer] = (
                    pos, lane.kvl_buf[layer], lane.sc_buf[layer], before,
                )
                lane.c.pending[layer] = (None if pos % 2 == 0 else
                                         (lane.kvl_buf[layer][T_VERIFY - 1].clone(),
                                          lane.sc_buf[layer][T_VERIFY - 1].clone()))
            lane.c.len = pos + T_VERIFY
        return tuple((lane.logits, lane.main_hidden) for lane in self.lanes)


def MIN_BUCKET(used_tokens: int, max_seq: int) -> int:
    """Late import avoids making fastdecode's private helper part of this module's API."""
    from engine.fastdecode import _index_bucket
    return _index_bucket(used_tokens, max_seq)


def clone_lane(engine, source: FastDecoder | None = None) -> FastDecoder:
    """Create an independent mutable lane over an engine's shared immutable weights/arena.

    This is intentionally explicit and relatively expensive (~one additional KV cache).  The
    eventual scheduler owns two lanes for its lifetime; constructing a lane per request would erase
    the throughput benefit and fragment the unified-memory pool.
    """
    src = source or engine.fast
    caches = Caches(engine.args, src.c.max_seq, engine.device)
    model = Model(engine.W, engine.store, caches, src.m.moe_fn, act_quant=engine.act_quant)
    model.prune_mask = src.m.prune_mask
    model.slot_lut = src.m.slot_lut
    model.prefill_routes = getattr(src.m, "prefill_routes", None)
    model.hash_state = src.m.hash_state
    model.engram_rows = src.m.engram_rows
    lane = FastDecoder(model, engine, use_graphs=src.use_graphs)
    lane.lut = src.lut
    lane.lut_version = src.lut_version

    for dst, old in zip(caches.win, src.c.win):
        dst.copy_(old)
    for dst, old in zip(caches.mtp_win, src.c.mtp_win):
        dst.copy_(old)
    for layer in caches.ckv:
        caches.ckv[layer].copy_(src.c.ckv[layer])
        caches.ik[layer].copy_(src.c.ik[layer])
        pending = src.c.pending[layer]
        caches.pending[layer] = (None if pending is None else
                                  (pending[0].clone(), pending[1].clone()))
    caches.len = src.c.len
    return lane
