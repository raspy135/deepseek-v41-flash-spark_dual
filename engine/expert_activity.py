"""Bounded CPU-only load telemetry; never touches tensors or collectives."""
from collections import deque
from contextlib import contextmanager
import threading
import time


class ExpertActivity:
    def __init__(self, capacity=256, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.active = {}
        self.events = deque(maxlen=capacity)
        self.sequence = 0

    @contextmanager
    def loading(self, key, slot):
        started = self.clock()
        entry = dict(layer=int(key[0]), expert=int(key[1]), slot=int(slot), started=started)
        with self.lock:
            self.active[slot] = entry
        ok = False
        try:
            yield
            ok = True
        finally:
            finished = self.clock()
            with self.lock:
                self.active.pop(slot, None)
                self.sequence += 1
                self.events.append(dict(layer=entry['layer'], expert=entry['expert'], slot=entry['slot'],
                                        sequence=self.sequence, finished=finished,
                                        duration_ms=round((finished-started)*1000, 2), ok=ok))

    def snapshot(self):
        now = self.clock()
        with self.lock:
            active, events, sequence = list(self.active.values()), list(self.events), self.sequence
        return dict(sequence=sequence,
                    active=[dict(layer=e['layer'], expert=e['expert'], slot=e['slot'],
                                 elapsed_ms=round((now-e['started'])*1000, 1)) for e in active],
                    recent=[{k:v for k,v in e.items() if k != 'finished'} |
                            {'age_ms':round((now-e['finished'])*1000, 1)} for e in reversed(events)])
