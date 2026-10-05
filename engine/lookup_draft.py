"""Request-local exact-continuation drafts.

This is prompt-lookup decoding, not a learned cache. If the last N settled token
ids occurred earlier in the same request and enough tokens followed that earlier
occurrence, those known continuation ids can be proposed to the target verifier.
The proposal is deterministic given settled history. Its sampling distribution
is a delta at each copied token, so the ordinary rejection verifier remains
lossless at every temperature when it receives that exact proposal distribution.
Proposal selection and depth must happen before the target logits are inspected.

The index stores only a bounded number of recent occurrences per N-token key.
Among them, lookup prefers the occurrence with the longest matching context
before the key. Keeping the state request-local avoids cross-user token leakage.
The engine keeps this cache on rank 0 and broadcasts its selected proposal.
"""
from __future__ import annotations

VERSION = 2  # Deterministic proposal probabilities support sampled verification.


def deterministic_draft_probs(drafts, vocab_size: int, *, dtype=None):
    """Return q[B,V] for a fixed copied continuation, on the draft IDs' device.

    Keep q independent of the target's logits, temperature, nucleus, grammar and
    penalties. Those define p after masking. For q=delta_d the ordinary verifier
    accepts d with probability p[d]; on rejection (p-q)+ is p with d removed.
    Both the sequential and batched verifier can therefore consume this q without
    changing their sampling or stop semantics. This first implementation builds
    dense q rows to reuse those already-qualified paths.
    """
    import torch

    if drafts.ndim != 1 or drafts.numel() < 1 or vocab_size < 1:
        raise ValueError("expected nonempty draft IDs [B] and positive vocabulary size")
    if drafts.dtype not in (torch.int32, torch.int64):
        raise ValueError("lookup draft IDs must be int32 or int64")
    if dtype is None:
        dtype = torch.float32
    q = torch.zeros((drafts.numel(), vocab_size), dtype=dtype, device=drafts.device)
    return q.scatter_(1, drafts.to(torch.int64).unsqueeze(1), 1.)


class ExactDraftCache:
    """Map exact suffixes to continuations already settled in this request."""

    def __init__(self, history, min_match: int, max_candidates: int = 8,
                 max_match: int | None = None):
        if min_match < 2:
            raise ValueError("lookup draft min_match must be at least 2")
        if max_candidates < 1:
            raise ValueError("lookup draft max_candidates must be positive")
        if max_match is None:
            max_match = max(64, min_match)
        if max_match < min_match:
            raise ValueError("lookup draft max_match must be >= min_match")
        self.min_match = int(min_match)
        self.max_candidates = int(max_candidates)
        self.max_match = int(max_match)
        self.history = [int(t) for t in history]
        self.index: dict[tuple[int, ...], list[int]] = {}
        self.stats = {
            "enabled": True,
            "version": VERSION,
            "min_match": self.min_match,
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "draft_tokens": 0,
            "accepted_tokens": 0,
            "full_accepts": 0,
            "match_tokens": 0,
            "dspark_skipped": 0,
        }
        # end is the first continuation position. The current suffix (end == len)
        # is deliberately absent because its continuation is not known yet.
        for end in range(self.min_match, len(self.history)):
            self._remember(end)
        self._last_depth = 0

    def _remember(self, end: int) -> None:
        key = tuple(self.history[end - self.min_match:end])
        positions = self.index.setdefault(key, [])
        positions.append(end)
        if len(positions) > self.max_candidates:
            del positions[:-self.max_candidates]

    def extend(self, tokens) -> None:
        """Commit settled tokens and make each preceding suffix available for future lookup."""
        for token in tokens:
            end = len(self.history)
            if end >= self.min_match:
                self._remember(end)
            self.history.append(int(token))

    def _match_length(self, end: int, current_end: int) -> int:
        matched = self.min_match
        left = end - self.min_match - 1
        right = current_end - self.min_match - 1
        while (matched < self.max_match and left >= 0 and right >= 0
               and self.history[left] == self.history[right]):
            matched += 1
            left -= 1
            right -= 1
        return matched

    def propose(self, depth: int) -> list[int] | None:
        """Return depth known continuation tokens for the current suffix, if any."""
        depth = int(depth)
        self.stats["lookups"] += 1
        self._last_depth = 0
        end_now = len(self.history)
        if depth <= 0 or end_now < self.min_match:
            self.stats["misses"] += 1
            return None
        key = tuple(self.history[-self.min_match:])
        positions = self.index.get(key)
        if not positions:
            self.stats["misses"] += 1
            return None

        best_end = best_match = -1
        # Newest wins an equal-length tie. Skip occurrences whose continuation
        # has not yet accumulated enough settled tokens for this verify width.
        for end in reversed(positions):
            if end + depth > end_now:
                continue
            matched = self._match_length(end, end_now)
            if matched > best_match:
                best_end, best_match = end, matched
        if best_end < 0:
            self.stats["misses"] += 1
            return None

        proposal = self.history[best_end:best_end + depth]
        self._last_depth = depth
        self.stats["hits"] += 1
        self.stats["draft_tokens"] += depth
        self.stats["match_tokens"] += best_match
        self.stats["dspark_skipped"] += 1
        return proposal

    def record_accept(self, accepted: int) -> None:
        if not self._last_depth:
            return
        accepted = min(int(accepted), self._last_depth)
        self.stats["accepted_tokens"] += accepted
        self.stats["full_accepts"] += int(accepted == self._last_depth)
        self._last_depth = 0

    def report(self) -> dict:
        out = dict(self.stats)
        hits = out["hits"]
        proposed = out["draft_tokens"]
        out["hit_rate"] = round(hits / max(out["lookups"], 1), 4)
        out["accept_rate"] = round(out["accepted_tokens"] / max(proposed, 1), 4)
        out["mean_match_tokens"] = round(out["match_tokens"] / max(hits, 1), 2)
        return out
