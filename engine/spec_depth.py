"""Speculative depth policies: recent acceptance and opt-in per-draft confidence.

ConfidenceDepthPolicy implements DSV41_BLOCK_CONFIDENCE for all temperatures.
The original DSV41_BLOCK_DYNAMIC controller below supports greedy and sampling:

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
  * Step times are measured per depth on this process: the median of the last STEP_SAMPLES
    loop wall times (what the user waits for), the minimum until MIN_SAMPLES exist, and the
    measured ratio for a depth not yet seen. Steps that captured a CUDA graph are not timed
    (the engine passes None). An EWMA seeded by its first sample was used first and failed
    in serving: the first request after a restart timed its graph captures, depth 3 read as
    ~190 ms/step instead of ~105, and the policy went to 5 on prose and never came back.
  * A decision happens at most once per `interval` emitted tokens, and needs a `margin` win.

Rank 0 decides; the engine broadcasts the depth with the per-step control flag, so both ranks
always verify the same width (EPDistributed.control). Rank 1 never consults its own copy.
"""
from __future__ import annotations

import os
import math
import statistics
from collections import deque


def verify_odd_enabled():
    """Opt-in target widths 1/3/5, with the trained DSpark block unchanged."""
    value = os.environ.get("DSV41_VERIFY_ODD", "0")
    if value not in ("0", "1"):
        raise ValueError("DSV41_VERIFY_ODD must be 0 or 1")
    return value == "1"


def confidence_depths(dynamic_depths=None):
    """Resolve the opt-in policy without importing CUDA."""
    value = os.environ.get("DSV41_BLOCK_CONFIDENCE", "0")
    if value not in ("0", "1"):
        raise ValueError("DSV41_BLOCK_CONFIDENCE must be 0 or 1")
    if value == "0":
        return None
    odd = verify_odd_enabled()
    allowed = tuple(range(1, 6)) if odd else (1, 3, 5)
    if ((not odd and dynamic_depths not in (None, (3, 5)))
            or (odd and dynamic_depths is not None and any(d not in allowed for d in dynamic_depths))):
        raise ValueError("DSV41_BLOCK_CONFIDENCE requires depths in 1..5 with DSV41_VERIFY_ODD=1, "
                         "otherwise DSV41_BLOCK_DYNAMIC=3,5 or unset")
    fixed = os.environ.get("DSV41_BLOCK", "").strip()
    if fixed not in ("", "off", "default", "5"):
        raise ValueError("DSV41_BLOCK_CONFIDENCE requires DSV41_BLOCK=5 or unset")
    return allowed


class ConfidenceDepthPolicy:
    """Per-draft expected tokens / measured step cost.

    DSpark's logits estimate conditional acceptance, so survival to position k is
    the product of the first k probabilities. Include the guaranteed verifier token.
    Never consult the current step's actual acceptance to choose its width.
    Greedy uses whole-block lookahead. Sampling extends a guaranteed prefix at
    width boundaries, without looking at proposals it might exclude.

    The 88/106/124 ms priors are historical TP2 measurements. Unseen widths inherit
    the median measured/prior scale; observed widths use robust local timings.
    Captures are excluded by the caller. This does not save any drafter passes:
    DSpark computes all five backbone positions together.
    """
    depths = (1, 3, 5)
    priors = {1: .088, 3: .106, 5: .124}
    VERSION = 4  # Opt-in intermediate depths preserve the sampled proposal-prefix rule.

    def __init__(self):
        if verify_odd_enabled():
            self.depths = (1, 2, 3, 4, 5)
            # Interpolate the historical 1/3/5 priors only until this process
            # measures the new widths. These are not claimed measurements.
            self.priors = {1: .088, 2: .097, 3: .106, 4: .115, 5: .124}
        self._samples = {d: deque(maxlen=32) for d in self.depths}
        self.step_s = {d: None for d in self.depths}
        self.refresh_interval = int(os.environ.get("DSV41_CONF_COST_REFRESH", "0"))
        if self.refresh_interval < 0:
            raise ValueError("DSV41_CONF_COST_REFRESH must be >= 0")
        self._cost_age = dict.fromkeys(self.depths, 0)
        self._refresh_depth = None
        self._refresh_samples = 0
        self.cost_refreshes = 0
        self.pinned = None  # qualification only
        self.reset_request()

    def reset_request(self):
        self.depth = 3
        self.steps = dict.fromkeys(self.depths, 0)
        self.switches = self.invalid_confidence = 0
        self.selection = None

    def _times(self):
        ratios = [t / self.priors[d] for d, t in self.step_s.items() if t is not None]
        scale = statistics.median(ratios) if ratios else 1.0
        return {d: self.step_s[d] or self.priors[d] * scale for d in self.depths}

    def decide(self):
        # Provisional width in the loop-control message. Final choice follows draft().
        return self.depth

    def choose(self, logits):
        self.selection = "lookahead"
        if self.pinned is not None:
            if self.pinned not in self.depths:
                raise ValueError(f"confidence depth must be one of {self.depths}")
            target = self.pinned
        elif (refresh := self._refresh_target()) is not None:
            target = refresh
        elif len(logits) != 5 or any(not math.isfinite(x) for x in logits):
            self.invalid_confidence += 1
            target = 3
        else:
            times = self._times()
            survival, expected = 1.0, 1.0
            rates = {}
            for k, x in enumerate(logits, 1):
                # Stable sigmoid, including arbitrarily large finite logits.
                z = math.exp(-abs(x))
                survival *= 1 / (1 + z) if x >= 0 else z / (1 + z)
                expected += survival
                if k in times:
                    rates[k] = expected / times[k]
            target = max(self.depths, key=lambda d: rates[d])  # ties prefer shallower
        return self._select(target)

    def choose_sampled(self, logits):
        """Extend the configured depths using an already selected proposal prefix.

        Confidence i depends on proposals BEFORE i (the Markov embedding), never
        proposal i. At depth d, confidence[d] is therefore safe to use when
        deciding whether to include the next proposals. Estimate their added
        confidences from that boundary score instead of peeking at their tokens.

        This makes the block length a stopping time of the proposal prefix:
        inclusion of proposal i is decided without consulting proposal i or later
        proposals. Its q remains the original conditional distribution, so the
        existing min(1, p/q) acceptance and residual correction remain valid.
        A whole-block argmax or rounding a token-dependent cutoff DOWN to a graph
        width would violate that condition. Accuracy of the confidence estimate
        affects cost, not the target distribution.
        """
        self.selection = "prefix"
        if self.pinned is not None:
            if self.pinned not in self.depths:
                raise ValueError(f"confidence depth must be one of {self.depths}")
            return self._select(self.pinned)
        refresh = self._refresh_target()
        if refresh is not None:
            return self._select(refresh)
        if len(logits) != 5:
            self.invalid_confidence += 1
            return self._select(3)
        times = self._times()
        survival, expected, seen = 1.0, 1.0, 0
        for depth, deeper in zip(self.depths, self.depths[1:]):
            # All these positions are already included. Never validate future
            # logits as a group: even an invalid-value fallback can leak lookahead.
            for x in logits[seen:depth]:
                if not math.isfinite(x):
                    self.invalid_confidence += 1
                    return self._select(depth)
                survival *= self._sigmoid(x)
                expected += survival
            seen = depth
            x = logits[depth]
            if not math.isfinite(x):
                self.invalid_confidence += 1
                return self._select(depth)
            probability = self._sigmoid(x)
            extra = survival * sum(probability ** k for k in range(1, deeper - depth + 1))
            if (expected + extra) / times[deeper] <= expected / times[depth]:
                return self._select(depth)
        return self._select(self.depths[-1])

    @staticmethod
    def _sigmoid(x):
        z = math.exp(-abs(x))
        return 1 / (1 + z) if x >= 0 else z / (1 + z)

    def _select(self, target):
        self.switches += target != self.depth
        self.depth = target
        return target

    def _refresh_target(self):
        """Occasionally remeasure an unselected width; never inspect draft tokens.

        Three non-capture steps replace a stale estimate. Without this, an old
        expensive depth 3 can block the sampled 1->3->5 policy indefinitely.
        Opt-in: extra probes can cost throughput, so qualify before enabling.
        """
        if not self.refresh_interval:
            return None
        if self._refresh_depth is None:
            depth = max(self.depths, key=lambda d: self._cost_age[d])
            if self._cost_age[depth] < self.refresh_interval:
                return None
            self._refresh_depth = depth
            self._refresh_samples = 0
            self._samples[depth].clear()
            self.step_s[depth] = None
            self.cost_refreshes += 1
        return self._refresh_depth

    def observe(self, depth, accepted, emitted, step_s):
        self.steps[depth] += 1
        for d in self.depths:
            self._cost_age[d] += 1
        if step_s is not None and math.isfinite(step_s) and step_s > 0:
            self._cost_age[depth] = 0
            if self._refresh_depth == depth:
                self._refresh_samples += 1
                if self._refresh_samples >= 3:
                    self._refresh_depth = None
            samples = self._samples[depth]
            samples.append(step_s)
            self.step_s[depth] = statistics.median(samples) if len(samples) >= 3 else min(samples)

    def pop_switch(self):
        return None  # per-step switches belong in aggregate stats, not the server log

    def report(self):
        return {"policy": "confidence", "selection": self.selection,
                "cost_refresh_interval": self.refresh_interval, "cost_refreshes": self.cost_refreshes, "depths": list(self.depths),
                "steps": {str(d): n for d, n in self.steps.items()},
                "switches": self.switches, "invalid_confidence": self.invalid_confidence,
                "step_ms": {str(d): None if t is None else round(t * 1000, 1)
                            for d, t in self.step_s.items()},
                "estimated_step_ms": {str(d): round(t * 1000, 1) for d, t in self._times().items()}}


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
        self._samples = {self.lo: deque(maxlen=self.STEP_SAMPLES), self.hi: deque(maxlen=self.STEP_SAMPLES)}
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
    STEP_SAMPLES = 32
    MIN_SAMPLES = 3

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
            q = self._samples[depth]
            q.append(step_s)
            # A median ignores the occasional slow step; with too few samples for that, the
            # minimum is the safe side (a slow first step is an overhead, never a speedup).
            self.step_s[depth] = statistics.median(q) if len(q) >= self.MIN_SAMPLES else min(q)
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
