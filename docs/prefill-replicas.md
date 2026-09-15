# Experimental prefill replicas (default off)

This is whole-expert EP relocation for prefill, not TP. Original weights and decode lookup
tables remain resident and unchanged. It is not yet a demonstrated performance improvement.

## Lifecycle

1. `DSV41_PREFILL_REPLICA_GB` reserves dedicated tail slots in the main packed FP4 arena,
   additional to the original arena budget. Maximum 1 decimal GB per rank (~53 experts).
   These slots are excluded from both the LRU and transient ring. Memory safety checks include them.
2. With two or more chunks, the first chunk records actual post-pruning expert assignments.
   This is separate from the existing *wanted/unpruned* demand statistics; those are unchanged.
3. Rank 0 plans whole-expert moves that reduce predicted maximum assignment counts per layer.
   Destination capacity is global per rank, not per layer. The plan is broadcast to both ranks.
4. Each destination loads replicas from its local checkpoint using the existing I/O pools and
   staging buffers. Waves contain at most eight moves globally. Both ranks check completion/failure
   after every wave, including a rank that has no loads. A failure activates no replicas.
5. Private prefill LUTs route moved experts only to the destination; originals are masked to the
   null slot for that computation. The router's expert selection and weights are unchanged.
6. Tables are cleared before decoder replay, adaptation, and decode, and again on request cleanup.
   Returning to the original routing requires no weight reload. No mid-decode adaptation is added.

`DSV41_PREFILL_REPLICA_BUDGET_MS=500` is a **soft admission budget**, including planning/loading.
The next wave is not started if elapsed time plus the preceding wave duration exceeds it. Active
disk reads cannot be cancelled safely; a wave, status checks, or table creation may overrun the
budget. Reported `load_ms` and `budget_exceeded` expose that, rather than claiming a hard bound.
The planner predicts assignment counts, not elapsed-time savings. A positive count improvement
does not prove that loading replicas pays for itself.

## Scope and safety

- Supported initial scope: EP2, software FP4, pruned all-resident device-LUT serving.
- Flags/capacity/budget are in the boot-time cross-rank guard. Every new status collective is
  reached by both ranks; load errors are coordinated before any routing activation.
- Prefix reuse is disabled while this experiment is enabled. Regrouping routed FP32 sums can
  change rounding and downstream routing; do not reuse a baseline prefix under another grouping.
- Originals are never evicted for replicas. Existing swap cadence and decode graph LUT addresses
  remain unchanged. Replica slots are reused for fresh plans on subsequent requests.
- No token text or IDs are logged; statistics describe capacity, timing, and predicted work.

## Validation status

CPU unit tests cover planner limits, identical execution ownership, compact-route/LUT agreement,
original-state restoration, invalid plans, budget exhaustion, and load failure. A two-process
CPU/Gloo test exercises the actual collective sequence and coordinated fallback when rank 0's
load fails. Nine tests including chunk timing passed. A synthetic 40x384 Poisson routing histogram
was planned in ~4.9 ms; that is a planner microbenchmark, not an engine speedup.

Run CPU checks without interrupting serving:

```bash
.venv/bin/python -m unittest engine.test_prefill_replicas engine.test_prefill_replicas_dist
```

Before enabling serving, stop the service and run the real-weight GPU regrouping check:

```bash
MODEL_DIR=/home/ryan/models/DeepSeek-V4.1-Flash DSV41_FP4_DOT_SCALED=0 \
  .venv/bin/python tools/test_fp4_replicas.py
```

That GPU test is added but has **not yet been run**. Full-model quality, real I/O time, and net
prefill benefit still require validation. Short nesting prompts alone do not exercise replicas:
use a passing long-context control with at least two prefill chunks, and inspect `loaded` to
ensure the path actually activated. Keep the feature off until those checks pass. Do not apply
it to decode or split weights into TP shards as part of this experiment.
