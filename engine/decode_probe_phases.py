"""Bounded, opt-in timing/allocation census between decode graph replays."""
from collections import deque


class DecodeLoopPhases:
    # Allocation requests often reuse PyTorch's pool. Device allocations count
    # actual pool growth; neither counter measures DRAM traffic or kernel scratch.
    _counters = {"allocation_requests": "allocation.all.allocated",
                 "requested_bytes": "allocated_bytes.all.allocated",
                 "device_allocations": "num_device_alloc",
                 "device_frees": "num_device_free"}

    def __init__(self, base, device, captures, *, torch_module=None):
        if torch_module is None:
            import torch as torch_module
        self.base, self.torch, self.device = base, torch_module, device
        self.captures = captures
        self.rows = {}
        self.events = {}
        self._event = None
        self._counts = None
        self._captures = 0

    @property
    def steps(self):
        return self.base.steps

    @steps.setter
    def steps(self, value):
        self.base.steps = value

    def _sample(self):
        stats = self.torch.cuda.memory_stats(self.device)
        return {k: int(stats.get(v, 0)) for k, v in self._counters.items()}

    def start(self):
        self.base.start()
        self._counts = self._sample()
        self._captures = self.captures()
        self._event = self.torch.cuda.Event(enable_timing=True)
        self._event.record()

    def mark(self, name):
        end = self.torch.cuda.Event(enable_timing=True)
        end.record()
        counts = self._sample()
        captures = self.captures()
        row = self.rows.setdefault(name, dict(intervals=0, capture_intervals=0,
                                              **{k: 0 for k in self._counters}))
        row["intervals"] += 1
        row["capture_intervals"] += int(captures != self._captures)
        if self._counts is not None:
            for key in self._counters:
                row[key] += max(0, counts[key] - self._counts[key])
        if self._event is not None:
            self.events.setdefault(name, deque(maxlen=64)).append((self._event, end))
        self._event, self._counts, self._captures = end, counts, captures
        self.base.mark(name)

    def table(self, notes=()):
        return self.base.table(notes)

    def report(self):
        # Called only by the mirrored idle command, after its synchronization.
        phases = []
        for name, counters in self.rows.items():
            times = [a.elapsed_time(b) for a, b in self.events.get(name, ())]
            phases.append(dict(name=name, **counters,
                host_total_ms=self.base.acc.get(name, 0) * 1000,
                stream_span_mean_ms=sum(times) / len(times) if times else None,
                stream_samples=len(times)))
        return dict(steps=self.steps, phases=phases,
            measurement="host wall time; last 64 default-stream spans per phase include GPU idle gaps",
            allocations="PyTorch allocation requests may reuse pool; device allocations/frees measure pool growth",
            capture_intervals="exclude cold graph capture intervals when assessing steady decode reuse")
