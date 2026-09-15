"""Opt-in chunk envelopes; no synchronization until reporting after generation.

Host duration includes any existing blocking calls: it is NOT pure launch overhead.
GPU envelopes include stream idle/collective waits: they are NOT kernel busy time.
Host and CUDA clocks (and clocks on different ranks) must not be subtracted.
"""
from contextlib import contextmanager
import time


class PrefillMoeTiming:
    """Per-call subphase envelopes, not kernel-busy or pure CPU execution time.

    No synchronization until report. Neither host minus GPU time nor vice versa
    is a valid idle-time estimate: the two timelines overlap asynchronously.
    """
    def __init__(self, rank, event_factory=None, clock=time.perf_counter):
        if event_factory is None:
            import torch
            event_factory = lambda: torch.cuda.Event(enable_timing=True)
        self.rank, self.event_factory, self.clock = rank, event_factory, clock
        self.rows = []

    def start(self, layer, tokens):
        self.current = {'layer': layer, 'tokens': tokens, 'marks': []}
        self.rows.append(self.current)
        self.mark('begin')

    def mark(self, name):
        event = self.event_factory()
        event.record()
        self.current['marks'].append((name, self.clock(), event))

    def report(self):
        totals, calls = {}, []
        if self.rows:
            self.rows[-1]['marks'][-1][2].synchronize()
        for row in self.rows:
            stages = {}
            for (_, h0, g0), (name, h1, g1) in zip(row['marks'], row['marks'][1:]):
                host, gpu = (h1 - h0) * 1000, g0.elapsed_time(g1)
                stages[name] = {'host_ms': round(host, 3), 'gpu_envelope_ms': round(gpu, 3)}
                total = totals.setdefault(name, {'host_ms': 0., 'gpu_envelope_ms': 0.})
                total['host_ms'] += host
                total['gpu_envelope_ms'] += gpu
            calls.append({'layer': row['layer'], 'tokens': row['tokens'], 'stages': stages})
        return {'rank': self.rank, 'totals': {k: {a: round(b, 3) for a, b in v.items()}
                for k, v in totals.items()}, 'calls': calls}


class PrefillTiming:
    def __init__(self, rank, event_factory=None, clock=time.perf_counter):
        if event_factory is None:
            import torch
            event_factory = lambda: torch.cuda.Event(enable_timing=True)
        self.rank = rank
        self.event_factory = event_factory
        self.clock = clock
        self.origin = clock()
        self.rows = []

    @contextmanager
    def chunk(self, start, end):
        begin, finish = self.event_factory(), self.event_factory()
        host_start = self.clock()
        begin.record()
        try:
            yield
        finally:
            finish.record()
            self.rows.append((start, end, host_start, self.clock(), begin, finish))

    def report(self):
        if not self.rows:
            return {"rank": self.rank, "chunks": []}
        # Called after generation, never between chunks. Waiting for the last event
        # makes all earlier events on this stream readable without changing dispatch.
        self.rows[-1][5].synchronize()
        rows = []
        previous = None
        for start, end, h0, h1, g0, g1 in self.rows:
            rows.append({
                "start": start, "tokens": end - start,
                "host_start_ms": round((h0 - self.origin) * 1000, 3),
                "host_call_ms": round((h1 - h0) * 1000, 3),
                "host_gap_ms": None if previous is None else round((h0 - previous[0]) * 1000, 3),
                "gpu_envelope_ms": round(g0.elapsed_time(g1), 3),
                "gpu_gap_ms": None if previous is None else round(previous[1].elapsed_time(g0), 3),
            })
            previous = h1, g1
        return {"rank": self.rank, "chunks": rows}
