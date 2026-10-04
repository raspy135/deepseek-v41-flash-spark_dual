"""Bounded, position-independent CUDA graphs for prefill feed-forward blocks.

Attention, KV bookkeeping and Engram I/O remain eager. Capture exactly the usual
FFN arithmetic; routing masks, compact IDs and arena addresses are updated in
place by adaptation. No graph is keyed on context position. All ranks must use
the same configuration and call order (guarded by V41Engine at boot).
"""
import os
import time

import torch


def enabled():
    value = os.environ.get('DSV41_PREFILL_GRAPHS', '0')
    if value not in ('0', '1'):
        raise ValueError('DSV41_PREFILL_GRAPHS must be 0 or 1')
    return value == '1'


class PrefillFFNGraphs:
    VERSION = 2

    def __init__(self, model, chunk_size):
        self.model = model
        self.enabled = True
        self.rows = tuple(sorted({chunk_size, model.args.window_size}))
        self.graphs = {}
        self.buffers = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream(device=model.dev)
        self.captures = self.replays = self.fallbacks = 0
        self.capture_s = 0.0

    def eligible(self, h, L, prefill, n_experts):
        m = self.model
        ok = (self.enabled and prefill and h.shape[0] in self.rows
              and L < m.args.n_layers and n_experts == m.args.n_routed_experts
              and m.image_mask is None and not getattr(m, '_prefix_replay_only', False))
        if self.enabled and prefill and not ok:
            self.fallbacks += 1
        return ok

    def _counter_copies(self):
        # Warmup executes routing statistics; capture itself also executes Python
        # bookkeeping. Restore both so each real block is counted exactly once.
        m = self.model
        # The engine allocates these before graph construction when recording is
        # enabled. Do not turn recording statistics on as a side effect of capture.
        if m._want_counts is None:
            return []
        tensors = {}
        for name in ('_want_counts', '_want_mass', '_want_phase', '_miss_tot',
                     '_miss_phase', '_req_counts', '_rec_counts'):
            value = getattr(m, name, None)
            for tensor in value if isinstance(value, list) else [value]:
                if tensor is not None:
                    tensors[id(tensor)] = tensor
        return [(tensor, tensor.clone()) for tensor in tensors.values()]

    @torch.inference_mode()
    def run(self, h, attn_pre, w, L, store, arena, n_experts, retain=True):
        m = self.model
        if m.tap is not None:
            raise RuntimeError('prefill graphs do not support tensor diagnostic hooks')
        key = (h.shape[0], L)
        T = key[0]
        if T not in self.buffers:
            # Shared across layers, overwritten with the output after all input reads.
            # Only encoder/decoder final outputs escape their next attention block.
            self.buffers[T] = (torch.empty_like(h), torch.empty_like(attn_pre))
        hi, ai = self.buffers[T]
        # Preserve caller input if it came directly from a previous borrowed result.
        if h.data_ptr() == hi.data_ptr():
            h = h.clone()
        if attn_pre.data_ptr() == ai.data_ptr():
            attn_pre = attn_pre.clone()
        hi.copy_(h)
        ai.copy_(attn_pre)

        def compute():
            out, pre = m._ffn(hi, ai, w, L, True, store, arena, n_experts)
            hi.copy_(out)
            ai.copy_(pre)

        if key not in self.graphs:
            t0 = time.perf_counter()
            counters = self._counter_copies()
            stats = dict(m.stats)
            current = torch.cuda.current_stream(m.dev)
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                compute()  # compile kernels and initialize collectives outside capture
            current.wait_stream(self.stream)
            torch.cuda.synchronize(m.dev)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool, stream=self.stream):
                compute()
            current.wait_stream(self.stream)
            for tensor, saved in counters:
                tensor.copy_(saved)
            # Warmup overwrote the staging inputs with its output. Restore the real
            # inputs before replay; otherwise first capture applies this FFN twice.
            hi.copy_(h)
            ai.copy_(attn_pre)
            m.stats.clear()
            m.stats.update(stats)
            self.graphs[key] = graph
            self.captures += 1
            self.capture_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        self.graphs[key].replay()
        # CUDA replay skips Python counters. Device demand counters are in the graph.
        m.stats['hits'] = m.stats.get('hits', 0) + T * m.args.n_activated_experts
        if getattr(store, 'null_slot', None) is not None:
            m.stats['ep_calls'] += 1
        self.replays += 1
        out, pre = (hi.clone(), ai.clone()) if retain else (hi, ai)
        m.stats['moe_s'] += time.perf_counter() - t0
        return out, pre

    def report(self):
        return {'enabled': self.enabled, 'scope': 'ffn', 'rows': list(self.rows),
                'graphs': len(self.graphs), 'captures': self.captures,
                'replays': self.replays, 'fallbacks': self.fallbacks,
                'capture_s': round(self.capture_s, 3)}
