# Decode timeline instrumentation

This is a diagnostic runner, not a serving feature. Nothing is imported into the
normal server, and no new always-on timers or logging are added. The file names
use the existing benchmark/test exclusion so instrumentation-only edits do not
invalidate persistent prefix caches.

`tools/bench_decode_timeline_tp.py` runs an isolated TP2 engine with the saved
demand ranking, frozen swaps, and prefix persistence disabled. It performs two
unprofiled generations, then profiles at most 16 output bursts (eight by default)
after at least 128 output tokens by default (`--warmup-tokens`). Actual draft, verification, sampling,
Engram fetching, and TP communication remain in the execution path. This is not
the old fixed-step test with prefetched Engram rows.

Hooks label host calls for verification, draft, cold graph capture, loop control,
Engram worker reads, waiting for read futures, and host-to-device conversion.
PyTorch/CUPTI records actual kernel and copy intervals. The raw clock base aligns
worker-thread host spans with the CUDA timeline; do not align different nodes'
traces using their host timestamps.

The analyzer reports:

- GPU-busy interval union and gaps within the measured window, including its ends;
- NCCL, expert, FP8 and FP32 kernel families, without treating overlapping times
  as additive;
- host intervals and their overlap with GPU gaps;
- kernel counts, largest gaps, accepted-token statistics and unprofiled baselines;
- whether the profiled output matches each unprofiled generation.

An Engram host call includes queueing, copies and possible waits for earlier GPU
work. Its host duration is **not** pure disk latency. Overlap with a GPU gap is a
correlation, not proof that this call caused that gap. Hardware DRAM throughput
is not measured: Nsight GPU-metric access on these hosts currently requires
privileges unavailable to this session. No driver permissions were changed.

## Run only after coordinating a service pause

Use `tools/run_two_node_gate.sh` with an identical image on both nodes and an
isolated, synchronized source snapshot. Create a private output directory on
both hosts first. Optional `--capture` accepts only a trusted local capture from
the existing `DSV41_CAPTURE_NEXT` mechanism; torch pickle loading can execute
code. Vision and grammar-constrained captures are rejected. The exact prompt IDs
are retained, but output is capped by `--max-tokens` (512 by default) and a missing
seed becomes 42. Without a capture, the prompt asks for a simple coffee-shop HTML
page with a small stylesheet. It does not suppress EOS to force output length.
Input hashes and profiling-window settings must agree across ranks before generation.

```bash
GATE_IMAGE=<image-id> GATE_LOG_DIR=results/<new-private-folder> \
GATE_SOURCE_ROOT=<snapshot-on-both-hosts> \
bash tools/run_two_node_gate.sh bench_decode_timeline_tp.py \
  --capture /app/results/<new-private-folder>/capture.pt \
  --out /app/results/<new-private-folder>

python3 tools/bench_decode_timeline_analysis.py results/<new-private-folder>/rank0.json --trim-edges
python3 -m unittest tools.test_decode_timeline
```

The runner does not record prompt text, token IDs, generated text, tensor values,
or stack traces. Traces and JSON summaries use mode 0600. Treat them as private
nonetheless: shapes, timings, hashes and configuration can reveal workload
characteristics. Raw files stay under ignored `results/`, not in Git or the cloud.
Profiler overhead, disabled adaptation, and the output cap must be stated with
any result. Inspect `decode/capture` spans for cold graph work before interpreting
a sample as steady-state decode. Restart normal serving after the gate completes.

## First real-capture measurement

September 16: existing 14,435-token text-only capture, output capped at 64 tokens,
HC32, TP2, 90 GB arena, keep=0.61, 512K capacity. The traces contain no prompt or
completion text. Both profiled outputs exactly matched both unprofiled runs.
Warm unprofiled throughput was 22.42 / 22.40 tok/s on ranks 0 / 1; the first
unprofiled run was 16.27 / 16.12, illustrating why cold and warm runs must not
be mixed. Prefill was about 24 seconds in each run, with disk prefixes disabled.

Profiler startup caused a ~68 ms initial idle gap on rank 1. Excluding the first
and last iterations using host loop-control boundaries leaves six iterations:

| Mean milliseconds per iteration | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| Observed interval | 133.90 | 135.61 |
| Any GPU kernel/copy active | 116.21 | 123.13 |
| No recorded GPU activity | 17.70 | 12.48 |
| Expert projections | 41.35 | 36.32 |
| Dense/grouped FP8 projections | 28.70 | 27.84 |
| FP32 GEMMs | 8.65 | 8.52 |
| NCCL kernels (including waits) | 8.20 | 23.19 |
| Host waiting for Engram futures | 12.00 | 3.16 |
| That wait overlapping GPU inactivity | 7.56 | 1.08 |

The Engram layer-1 and layer-14 host reads each averaged ~14.7–14.9 ms on rank 0
versus ~5.9–6.0 ms on rank 1. Those two readers overlap; their durations must not
be added as critical-path costs. Conversely, layer-14 `to_device` occupied
~178/221 ms of host time across six steps, but overlapped GPU inactivity for only
~8.4/4.5 ms: most of that apparent host wait was concurrent with GPU work, not an
idle pipeline. This distinction was invisible in the previous aggregate timers.

The evidence points to asymmetric progress: rank 0 has slower expert work and
Engram preparation while rank 1 spends more time in collectives. NCCL duration
cannot be read as pure wire-transfer cost. The next bounded investigation is
rank-0 Engram preparation/read latency and CPU contention, not assuming bandwidth
alone limits decode. This is one short profiled workload, not proof of a permanent
node imbalance or a guaranteed gain from removing all observed idle time.

Raw captures/traces are private, ignored artifacts in
`results/decode-timeline-20260916-a/`. The analyzer's `--trim-edges` view is the
basis for this table; original exported summaries include window boundaries.

## Longer HTML response

September 16 follow-up: a simple coffee-shop HTML page (87 prompt tokens), no
JavaScript or external assets. Both unprofiled runs emitted the full 512-token
budget at 25.13 and 25.03 tok/s on rank 0 (25.11 / 25.02 on rank 1). Mean accepted
length was 3.45 tokens per speculative iteration, versus 3.0 on the earlier real
capture. This is a different workload and context length, not an engine speedup.
The output cap can truncate the page; this is a timing test, not HTML validation.

The third generation profiled 12 bursts, from output token count 130 to 170.
Both ranks' profiled output matched both unprofiled outputs exactly. No cold
graph capture was recorded in this window. Trimming its first and last iterations
leaves ten iterations:

| Mean milliseconds per iteration | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| Observed interval | 129.06 | 129.07 |
| Any GPU kernel/copy active | 113.95 | 118.91 |
| No recorded GPU activity | 15.12 | 10.16 |
| Expert projections | 38.23 | 36.30 |
| Dense/grouped FP8 projections | 30.71 | 27.99 |
| FP32 GEMMs | 9.28 | 8.53 |
| NCCL kernels (including waits) | 7.66 | 20.35 |
| Host waiting for Engram futures | 0.72 | 1.59 |
| That wait overlapping GPU inactivity | 0.52 | 0.14 |

Rank-1 collective time remains higher, but the long rank-0 future wait from the
first workload does not reproduce here. Its Engram layer-1/layer-14 reads average
8.83/8.42 ms, versus 6.22/6.04 ms on rank 1, and overlap GPU inactivity. Those two
reads overlap each other too: do not add them as independent savings. This narrows
the claim to workload-dependent CPU/read scheduling and asymmetric GPU progress,
not a fixed 12 ms Engram-future bottleneck. The trace still does not measure DRAM
bandwidth or establish a specific optimization's benefit.

Identical settings to the first diagnostic: saved ranking, swaps frozen, prefix
reuse/persistence disabled, no production precision changes. Normal serving was
restored afterward with instrumentation off. Private raw files remain under
`results/decode-timeline-html-20260916/` on the local machines.
