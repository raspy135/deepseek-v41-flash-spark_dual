"""Rolling expert-miss trigger over completed verification bursts, CPU bookkeeping only."""
from collections import deque


class UrgentAdaptWindow:
    def __init__(self, tokens, misses, window=30, threshold=.10, cooldown=150):
        self.window, self.threshold, self.cooldown = window, threshold, cooldown
        self.samples = deque([(tokens, misses)] if misses is not None else [])
        self.last_attempt = None

    def observe(self, tokens, misses):
        if misses is None:
            self.samples.clear()
            return None
        if self.samples and (tokens <= self.samples[-1][0]
                             or misses[1] < self.samples[-1][1][1]):
            self.samples.clear()
        self.samples.append((tokens, misses))
        cutoff = tokens - self.window
        # Keep the nearest completed-burst boundary at or before the target.
        # Including the entire burst avoids inventing per-token miss counts.
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()
        begin, before = self.samples[0]
        if before is None or tokens - begin < self.window or misses[1] <= before[1]:
            return None
        rate = (misses[0] - before[0]) / (misses[1] - before[1])
        ready = self.last_attempt is None or tokens - self.last_attempt >= self.cooldown
        return dict(reason='urgent', output_tokens=tokens, window_tokens=tokens-begin,
                    miss_rate=rate, threshold=self.threshold,
                    cooldown=not ready, triggered=ready and rate > self.threshold)

    def attempted(self, tokens, misses):
        # A no-op plan is still an attempt; do not repeatedly plan on the same evidence.
        self.last_attempt = tokens
        self.restart(tokens, misses)

    def restart(self, tokens, misses):
        """Discard stale/warmup evidence without changing the attempt cooldown."""
        self.samples.clear()
        if misses is not None:
            self.samples.append((tokens, misses))
