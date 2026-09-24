"""Opt-in, isolated-engine instrumentation. Never imported by serving.

No prompt, token IDs, activations or stack traces are recorded. CPU intervals
include waits on previously queued GPU work; they are not exclusive CPU costs.
"""
import functools
import json
import os
import threading
import time
from contextlib import contextmanager


class DecodeTimeline:
    def __init__(self, engine):
        self.engine = engine
        self.events = []
        self.originals = []
        self.active = False

    @contextmanager
    def span(self, name):
        if not self.active:
            yield
            return
        start = time.time_ns()
        try:
            yield
        finally:
            self.events.append(dict(name=name, ph='X', cat='decode_host',
                                    ts_ns=start, dur=(time.time_ns()-start)/1000,
                                    pid=os.getpid(), tid=threading.get_native_id()))

    def wrap(self, obj, attr, name, after=None):
        original = getattr(obj, attr)
        had_local = attr in vars(obj)
        local = vars(obj).get(attr)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with self.span(name):
                result = original(*args, **kwargs)
                if after is not None and self.active:
                    after(result)
                return result
        self.originals.append((obj, attr, had_local, local))
        setattr(obj, attr, wrapped)

    def watch_future(self, future):
        original = future.result
        def result(*args, **kwargs):
            with self.span('engram/future_wait'):
                return original(*args, **kwargs)
        future.result = result

    def __enter__(self):
        fd = self.engine.fast
        if fd is None:
            raise ValueError('timeline requires the fast decode path')
        for attr in ('step', 'draft', 'capture'):
            self.wrap(fd, attr, 'decode/' + attr)
        for layer, table in self.engine.tables.items():
            for attr in ('read_raw', 'to_device'):
                self.wrap(table, attr, f'engram/{layer}/{attr}')
        if getattr(self.engine, 'eg_pool', None) is not None:
            self.wrap(self.engine.eg_pool, 'submit', 'engram/submit', self.watch_future)
        if getattr(self.engine, 'ep', None) is not None:
            self.wrap(self.engine.ep, 'control', 'decode/control')
        # Graph replay timing is enqueue time, not GPU execution time. CUPTI
        # supplies actual kernel intervals, including NCCL, separately.
        return self

    def __exit__(self, *exc):
        self.active = False
        for obj, attr, had_local, local in reversed(self.originals):
            if had_local:
                setattr(obj, attr, local)
            else:
                delattr(obj, attr)

    def export(self, profiler, path, metadata):
        os.makedirs(os.path.dirname(os.path.abspath(path)), mode=0o700, exist_ok=True)
        # Reserve with private permissions before Kineto writes the trace.
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w'):
            pass
        profiler.export_chrome_trace(path)
        # Kineto may replace the reserved file rather than truncate it.
        os.chmod(path, 0o600)
        with open(path) as f:
            trace = json.load(f)
        base = trace.get('baseTimeNanoseconds')
        if base is None:
            raise ValueError('trace lacks clock base; cannot safely align host intervals')
        for event in self.events:
            item = dict(event)
            item['ts'] = (item.pop('ts_ns') - base) / 1000
            trace['traceEvents'].append(item)
        trace['decode_metadata'] = metadata
        with open(path, 'w') as f:
            json.dump(trace, f)
