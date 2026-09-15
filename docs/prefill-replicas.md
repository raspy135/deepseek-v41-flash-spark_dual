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

The GPU test was run and **failed the strict numerical gate at T=2048**. With real layer-0
weights, relocation preserved final BF16 output at T=1/6/63/512 (FP32 maximum differences
0 / 0 / 5.96e-8 / 2.38e-7), but a 2.38e-7 FP32 difference crossed a BF16 rounding boundary
at T=2048. Restoring original routing remained bit-exact in every case. This demonstrates
regrouping sensitivity, not a measured full-model quality failure. Serving activation and speed
benchmarking were initially withheld. The user subsequently authorized evaluating generated
quality and speed despite this tiny numerical difference; that measured trial is recorded below.

The initial GPU harness incorrectly placed real replica slots above the null sentinel, producing
a large false failure in small-batch routing. The harness now matches production: null is last,
above every real and replica slot. Do not attribute that initial failure to production routing.

Broader model quality still requires validation. Short nesting prompts alone do not exercise replicas:
use a passing long-context control with at least two prefill chunks, and inspect `loaded` to
ensure the path actually activated. Keep the feature off until those checks pass. Do not apply
it to decode or split weights into TP shards as part of this experiment.

## Serving trial: no net speedup on captured input

The first enabled startup caught an input-validation bug: the constructor's optional world-size
argument can be None while EPDistributed resolves WORLD_SIZE=2 from the environment. Validation
now uses `self.ep.world`. Both nodes then ran image
`cbfc721c7f174cbb17ab1bf04a9ec5b0095e5bca473a176ed866021b3a58e4bd` with replica GB=1,
budget=500 ms, chunk=2048, ring=4096, normal adaptive keep=.59/spec ON and profiling ON.

`bench/replica_quality.py` supplies 6,082-token synthetic context followed by depth-8 or depth-10
nesting. Both controls passed off and on. On: 66 then 64 replicas were successfully loaded globally,
in 285.104 / 236.548 ms (planning, I/O, coordination and table preparation). Both were within
the budget and activated, so these checks exercised the feature rather than merely testing short
prompts that bypass it. These two passing controls do not establish broad quality parity.

Saved private 14,396-token real input, output capped at 32 tokens, no prefix reuse:

| mode | prefill seconds | tok/s | replicas loaded, pair total | setup ms |
|---|---:|---:|---:|---:|
| off, quiet baseline after build/transfer | 16.849 | 854.39 | 0 | 0 |
| on, first replay | 18.250 | 788.83 | 47 | 198.217 |
| on, second replay | 17.488 | 823.19 | 50 | 210.435 |

47/50 replicas represent 0.884/0.940 GB of expert payload globally, not per node. Capacity reserved
was 53 slots (~0.996 GB) on each node, whether used or not. Each plan loaded all proposed replicas;
there was no need to consume the full 500 ms. Setup time is not pure disk-copy time.

On the second replay, rank 0/1 expert-routing GPU envelopes were 5.965/6.161 s, and combine/wait
envelopes 1.578/2.058 s. Assignment-count predictions improved (~139,203 -> 129,024 summed
per-layer maxima), but end-to-end prefill did not. Adaptive generations/settings changed between
requests and startup occurred between arms, so this is not an isolated attribution of slowdown.
Replicas were disabled again rather than retain extra memory without a demonstrated benefit.
Do not claim memory pressure caused the difference: no OOM occurred during this serving trial,
and concurrent memory-pressure telemetry was not collected.
