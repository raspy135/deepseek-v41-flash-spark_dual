"""Diagnostic-only call timers. Thread CPU duration is measured within each call.

CPU includes native CPU work performed by the calling thread, not just Python.
Wall time also includes waits. Nested scopes overlap and must not be summed.
"""
import functools
import statistics
import time


class HostTimers:
    def __init__(self, engine):
        self.engine = engine
        self.originals = []
        self.samples = {}

    def wrap(self, obj, attr, name):
        original = getattr(obj, attr)
        had_local = attr in vars(obj)
        local = vars(obj).get(attr)
        @functools.wraps(original)
        def measured(*args, **kwargs):
            wall = time.perf_counter_ns()
            cpu = time.thread_time_ns()
            try:
                return original(*args, **kwargs)
            finally:
                cpu_ms = (time.thread_time_ns()-cpu)/1e6
                wall_ms = (time.perf_counter_ns()-wall)/1e6
                self.samples.setdefault(name, []).append((cpu_ms, wall_ms))
        self.originals.append((obj, attr, had_local, local))
        setattr(obj, attr, measured)

    def __enter__(self):
        fd = self.engine.fast
        for attr in ('step', 'draft', 'prepare_pending_buffers'):
            self.wrap(fd, attr, attr)
        self.wrap(self.engine.model.c, 'rollback', 'rollback')
        self.wrap(self.engine.ep, 'control', 'control')
        for layer, table in self.engine.tables.items():
            for attr in ('read_raw', 'to_device'):
                self.wrap(table, attr, f'engram/{layer}/{attr}')
        # Both object __getitem__ and hash-state __call__ need class-level hooks;
        # avoid modifying those globally. The encompassing step timer bounds them.
        return self

    def __exit__(self, *exc):
        for obj, attr, had_local, local in reversed(self.originals):
            if had_local: setattr(obj, attr, local)
            else: delattr(obj, attr)

    def report(self):
        return {name: dict(calls=len(values), median_cpu_ms=statistics.median(v[0] for v in values),
                           median_wall_ms=statistics.median(v[1] for v in values),
                           max_cpu_ms=max(v[0] for v in values))
                for name, values in self.samples.items()}
