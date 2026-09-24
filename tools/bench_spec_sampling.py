"""Isolated sampled-verifier latency; no model load or serving configuration changes.

Includes CPU launch gaps and the final device result readback. These synthetic distributions
exercise a realistic vocabulary size but are not full-generation tokens/second measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from engine.spec_sampling import sample_probs_batch, verify_sampled
from engine.v41_engine import sample_probs


def sequential(logits, q, drafts, temperature, top_p):
    """Existing sampled branch, including its scalar synchronizations and RNG calls."""
    accepted, new, bonus = 0, [], None
    for i in range(len(drafts)):
        p = sample_probs(logits[i], temperature, top_p)
        d = int(drafts[i])
        r = torch.rand((), device=logits.device)
        ok = bool(r < (p[d] / q[i, d].clamp_min(1e-20)).clamp(max=1))
        if ok:
            accepted += 1
            new.append(d)
        else:
            residual = (p - q[i]).clamp_min(0)
            if float(residual.sum()) <= 0:
                residual = p
            bonus = int(torch.multinomial(residual / residual.sum(), 1))
            break
    if bonus is None:
        p = sample_probs(logits[accepted], temperature, top_p)
        bonus = int(torch.multinomial(p, 1))
    return accepted, new, bonus


def compare(arms, iters):
    for fn in arms.values():
        for _ in range(5):
            fn()
    torch.cuda.synchronize()
    samples = {name: [] for name in arms}
    accepts = {name: [] for name in arms}
    names = list(arms)
    for i in range(iters):
        for name in names if i % 2 == 0 else names[::-1]:
            start = time.perf_counter()
            result = arms[name]()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - start) * 1000)
            if isinstance(result, tuple):
                accepts[name].append(result[0])
    return dict(ms={k: statistics.median(v) for k, v in samples.items()},
                mean_accepted={k: statistics.mean(v) for k, v in accepts.items() if v})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--vocab", type=int, default=129280)
    parser.add_argument("--drafts", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--accept-pattern", choices=("mixed", "all", "first-reject"), default="mixed",
                        help="Also measure best/worst-case wasted batched probability work")
    args = parser.parse_args()
    if args.iters < 1 or args.vocab < 2 or any(b < 1 for b in args.drafts):
        parser.error("positive iterations/drafts and vocabulary >= 2 required")
    torch.manual_seed(20260922)
    for b in args.drafts:
        logits = torch.randn(b + 1, args.vocab, device="cuda")
        q = torch.softmax(logits[:-1] / .6 + .7 * torch.randn_like(logits[:-1]), -1)
        drafts = torch.multinomial(q, 1).squeeze(-1)
        for top_p in (.95, 1.):
            if args.accept_pattern == "all":
                q = sample_probs_batch(logits, .6, top_p)[:-1].clone()
                drafts = torch.multinomial(q, 1).squeeze(-1)
            elif args.accept_pattern == "first-reject":
                # Concentrate the proposal on a very unlikely target token. Report actual
                # accepted counts rather than asserting a stochastic event is impossible.
                drafts = logits[:-1].argmin(-1)
                q = torch.zeros_like(logits[:-1]).scatter_(1, drafts[:, None], 1.)
            torch.cuda.synchronize()
            result = compare({
                "sequential": lambda: sequential(logits, q, drafts, .6, top_p),
                "batched": lambda: verify_sampled(logits, q, drafts, .6, top_p),
            }, args.iters)
            print(json.dumps(dict(drafts=b, vocab=args.vocab, top_p=top_p,
                                  accept_pattern=args.accept_pattern, **result)), flush=True)
            result = compare({
                "rowwise_probs": lambda: torch.stack([sample_probs(row, .6, top_p) for row in logits]),
                "batched_probs": lambda: sample_probs_batch(logits, .6, top_p),
            }, args.iters)
            print(json.dumps(dict(drafts=b, vocab=args.vocab, top_p=top_p, stage="probabilities", **result)),
                  flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
