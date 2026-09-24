"""Per-request speculative depth policy for DSV41_BLOCK_DYNAMIC (two depths, e.g. 3 and 5).

Why: the best draft depth depends on how predictable the text is. Measured on TP2
(2026-09-23, docs/decode-dynamic-depth.md): depth 5 makes every verify step ~16% slower
(107 -> 124 ms) and pays that back only where the drafter is right -- code went 35.8 -> 44.6
tok/s, while prose and stories LOST 9-13%. So the engine re-decides as it goes.

The rule compares expected tokens per second, never acceptance alone:

  * At the deep depth the shallow alternative is known exactly. The drafter always drafts the
    deep block and the shallow verifier would have checked its first `lo` drafts, so a step that
    accepted `a` drafts would have yielded min(a, lo) + 1 tokens at the shallow depth. Switch
    down when that rate, at the shallow step time, beats the real one.
  * At the shallow depth nothing past `lo` drafts is visible. A step that accepted all `lo`
    drafts is "saturated"; each would have gained on average `extra` more tokens at the deep
    depth, learned from deep-depth steps that got at least `lo` accepted. Switch up when that
    estimate, at the deep step time, beats the real rate.
  * Step times are measured per depth on this process (EWMA of the loop's wall time, which is
    what the user waits for), seeded from the measured ratio until both are known.
  * A decision happens at most once per `interval` emitted tokens, and needs a `margin` win.

Rank 0 decides; the engine broadcasts the depth with the per-step control flag, so both ranks
always verify the same width (EPDistributed.control). Rank 1 never consults its own copy.
"""
from __future__ import annotations

import os


class DepthPolicy:
    def __init__(self, depths, start=None, interval=None, margin=None, step_ratio=None, extra=None):
        self.lo, self.hi = sorted(depths)
        env = os.environ.get
        self.start = int(start if start is not None else env("DSV41_BLOCK_DYNAMIC_START", self.lo))
        if self.start not in (self.lo, self.hi):
            raise ValueError(f"DSV41_BLOCK_DYNAMIC_START={self.start} is not one of {self.lo}, {self.hi}")
        self.interval = int(interval if interval is not None else env("DSV41_BLOCK_DYNAMIC_TOKENS", "60"))
        self.margin = float(margin if margin is not None else env("DSV41_BLOCK_DYNAMIC_MARGIN", "0.03"))
        # hi/lo step-time ratio until both have been measured: 124/107 ms on TP2 at 3 vs 5
        self.prior_ratio = float(step_ratio if step_ratio is not None else 1.16)
        # Mean extra drafts accepted past `lo` on a saturated deep step, learned per request as
        # a prior-weighted mean: EXTRA_WEIGHT pseudo-observations of the prior, so a handful of
        # real deep steps outweigh it. (An EWMA at 5%/step was tried first and oscillated: after
        # correctly dropping to `lo`, the stale estimate pulled the policy straight back up.)
        self.extra_prior = float(extra if extra is not None else 1.5)
        self.step_s = {self.lo: None, self.hi: None}
        self.pinned = None          # tests/benchmarks: force one depth
        self.reset_request()

    # ------------------------------------------------------------------ per request
    def reset_request(self):
        """New request: back to the start depth. Step times and `extra` are kept -- they
        describe this process, not the request -- while the counters are per request."""
        self.depth = self.pinned if self.pinned is not None else self.start
        self.switches = 0
        self.last_switch = None
        self.steps = {self.lo: 0, self.hi: 0}
        self._extra_sum, self._extra_n = 0.0, 0
        self._clear_window()

    EXTRA_WEIGHT = 5

    @property
    def extra(self) -> float:
        return ((self.extra_prior * self.EXTRA_WEIGHT + self._extra_sum)
                / (self.EXTRA_WEIGHT + self._extra_n))

    def _clear_window(self):
        self.n = 0              # steps in the window
        self.tok = 0.0          # tokens actually produced (accepted + 1 each)
        self.tok_lo_cf = 0.0    # deep window: what the shallow depth would have produced
        self.saturated = 0      # shallow window: steps that accepted every draft
        self.emitted = 0

    # ------------------------------------------------------------------ observations
    def observe(self, depth: int, accepted: int, emitted: int, step_s: float | None):
        """One verify step at `depth` accepted `accepted` drafts and emitted `emitted` tokens
        (fewer at a stop or max_tokens); `step_s` is its wall time, or None if unknown."""
        self.steps[depth] = self.steps.get(depth, 0) + 1
        if step_s is not None and step_s > 0:
            prev = self.step_s[depth]
            # A cold graph capture or a scheduling hiccup is not the step's cost.
            if prev is None or step_s < 4 * prev:
                self.step_s[depth] = step_s if prev is None else 0.9 * prev + 0.1 * step_s
        if depth != self.depth:
            return   # a step queued before the last switch; not evidence about the current depth
        self.n += 1
        self.tok += accepted + 1
        self.emitted += emitted
        if depth == self.hi:
            self.tok_lo_cf += min(accepted, self.lo) + 1
            if accepted >= self.lo:
                self._extra_sum += accepted - self.lo
                self._extra_n += 1
        elif accepted >= self.lo:
            self.saturated += 1

    def _times(self):
        lo, hi = self.step_s[self.lo], self.step_s[self.hi]
        if lo is None and hi is None:
            return 1.0, self.prior_ratio
        if lo is None:
            return hi / self.prior_ratio, hi
        if hi is None:
            return lo, lo * self.prior_ratio
        return lo, hi

    # ------------------------------------------------------------------ decision (rank 0)
    def decide(self) -> int:
        """Depth for the next verify step."""
        if self.pinned is not None:
            self.depth = self.pinned
            return self.depth
        if self.emitted < self.interval or self.n == 0:
            return self.depth
        t_lo, t_hi = self._times()
        if self.depth == self.hi:
            rate_hi = self.tok / (self.n * t_hi)
            rate_lo = self.tok_lo_cf / (self.n * t_lo)
            switch = rate_lo > rate_hi * (1 + self.margin)
            target = self.lo
        else:
            rate_lo = self.tok / (self.n * t_lo)
            rate_hi = (self.tok + self.saturated * self.extra) / (self.n * t_hi)
            switch = rate_hi > rate_lo * (1 + self.margin)
            target = self.hi
        if switch:
            # The evidence behind the switch, for the server log (engine pops it on rank 0).
            self.last_switch = {"from": self.depth, "to": target, "steps": self.n,
                                "tokens_per_step": round(self.tok / self.n, 2),
                                "rate": {self.lo: round(rate_lo, 1), self.hi: round(rate_hi, 1)},
                                "estimated": self.lo if self.depth == self.hi else self.hi}
            self.depth = target
            self.switches += 1
        self._clear_window()
        return self.depth

    def pop_switch(self):
        """The last switch's evidence, once; None if the depth did not change since."""
        sw, self.last_switch = self.last_switch, None
        return sw

    def report(self) -> dict:
        t_lo, t_hi = self.step_s[self.lo], self.step_s[self.hi]
        return {"depths": [self.lo, self.hi], "steps": {str(k): v for k, v in self.steps.items()},
                "switches": self.switches, "extra": round(self.extra, 3),
                "step_ms": {str(self.lo): None if t_lo is None else round(1000 * t_lo, 1),
                            str(self.hi): None if t_hi is None else round(1000 * t_hi, 1)}}
