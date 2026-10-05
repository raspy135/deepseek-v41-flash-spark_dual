"""CPU screening of lookup indexing, hit/miss selection and delta-q construction.

This prices host overhead only, not model throughput. No GPU work is submitted.
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.lookup_draft import ExactDraftCache, deterministic_draft_probs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tokens', type=int, nargs='+', default=[16384, 98304])
    ap.add_argument('--steps', type=int, default=512)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--output')
    args = ap.parse_args()
    records = []
    for count in args.tokens:
        for kind in ('periodic-hit', 'random-miss'):
            rng = random.Random(27)
            history = ([i % 256 for i in range(count)] if kind == 'periodic-hit'
                       else [rng.randrange(129280) for _ in range(count)])
            build, rounds = [], []
            for _ in range(args.reps):
                start = time.perf_counter()
                cache = ExactDraftCache(history, min_match=16)
                build.append(time.perf_counter() - start)
                start = time.perf_counter()
                for step in range(args.steps):
                    depth = 3 if step % 2 else 5
                    proposal = cache.propose(depth)
                    # Commit actual, independently settled tokens after each call.
                    settled = ([(count + step * 3 + i) % 256 for i in range(3)]
                               if kind == 'periodic-hit' else [rng.randrange(129280) for _ in range(3)])
                    cache.record_accept(min(depth, len(settled)) if proposal else 0)
                    cache.extend(settled)
                rounds.append((time.perf_counter() - start) / args.steps)
            records.append({'prompt_tokens': count, 'kind': kind, 'ngram': 16,
                            'depths': [3, 5], 'steps': args.steps, 'reps': args.reps,
                            'index_build_ms_median': 1000 * statistics.median(build),
                            'append_propose_us_median': 1e6 * statistics.median(rounds),
                            'lookup': cache.report()})
    import torch
    torch.set_num_threads(1)
    q_records = []
    for depth in (1, 3, 5):
        ids = torch.arange(depth)
        for _ in range(20):
            deterministic_draft_probs(ids, 129280)
        timings = []
        for _ in range(args.reps):
            start = time.perf_counter()
            for _ in range(args.steps):
                deterministic_draft_probs(ids, 129280)
            timings.append((time.perf_counter() - start) / args.steps)
        q_records.append({'depth': depth, 'vocab': 129280, 'device': 'cpu', 'threads': 1,
                          'q_bytes': depth * 129280 * 4,
                          'build_q_us_median': 1e6 * statistics.median(timings)})
    result = {'host_lookup': records, 'delta_q': q_records,
              'limitation': 'CPU overhead only; no target weights, CUDA or end-to-end throughput measurement.'}
    payload = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()
