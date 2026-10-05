"""Optional pre-proposal decision to emit one ordinary target sample.

The policy only reads completed steps. It never sees a proposal from the step
whose mode it selects, so it is valid for greedy and rejection sampling alike.
The experiment uses the existing two-row graph (root + discarded dummy), not
an eager serial forward or an unsupported odd compressor width.
"""
from collections import deque
import math
import os
import statistics

VERSION = 1


def enabled():
    value = os.environ.get("DSV41_DRAFT_BYPASS", "0")
    if value not in ("0", "1"):
        raise ValueError("DSV41_DRAFT_BYPASS must be 0 or 1")
    return value == "1"


class DraftBypassPolicy:
    MIN_SAMPLES = 8
    PROBE_INTERVAL = 16
    MARGIN = .03

    def __init__(self):
        self.pinned = None  # benchmark only: True skips draft, False uses DSpark
        self.reset_request()

    def reset_request(self):
        self.spec_samples = deque(maxlen=32)
        self.bypass_samples = deque(maxlen=32)
        self.steps = {"draft": 0, "bypass": 0}
        self.probes = 0
        self._since_probe = 0

    def decide(self):
        if self.pinned is not None:
            return bool(self.pinned)
        if len(self.spec_samples) < self.MIN_SAMPLES:
            return False
        rate = sum(n for n, _ in self.spec_samples) / sum(t for _, t in self.spec_samples)
        if self.bypass_samples:
            bypass_rate = 1 / statistics.median(self.bypass_samples)
            # Periodically recheck drafting after sustained bypass; otherwise a
            # prose->code transition could leave the policy permanently stale.
            if self._since_probe >= self.PROBE_INTERVAL:
                self._since_probe = 0
                return False
            return bypass_rate > rate * (1 + self.MARGIN)
        # Price the fallback only where recent proposals rarely pay for themselves.
        poor = sum(n for n, _ in self.spec_samples) / len(self.spec_samples) < 1.5
        if poor and self._since_probe >= self.PROBE_INTERVAL:
            self.probes += 1
            self._since_probe = 0
            return True
        return False

    def observe(self, bypass, emitted, seconds):
        self.steps["bypass" if bypass else "draft"] += 1
        self._since_probe += 1
        if seconds is None or not math.isfinite(seconds) or seconds <= 0 or emitted <= 0:
            return
        if bypass:
            self.bypass_samples.append(seconds)
        else:
            self.spec_samples.append((emitted, seconds))

    def report(self):
        return dict(steps=dict(self.steps), probes=self.probes,
                    bypass_ms=(1000 * statistics.median(self.bypass_samples)
                               if self.bypass_samples else None))
