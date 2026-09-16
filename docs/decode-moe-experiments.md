# Decode MoE microbenchmark — September 16, 2026

Goal: identify a bounded optimization for native MXFP4 without changing the
quality-sensitive FP32 accumulation and BF16 rounding boundaries. No experimental
kernel is enabled in serving. Both model services were stopped during measurement.

## Method

`tools/bench_fp4_decode_tp.py` loads 64 real layer-0 experts from the native
checkpoint, split over two Sparks using the serving output-layout TP path. Inputs
and routes are synthetic; these are real decode shapes, not captured user activations.
Rows 1, 4, and 8 cover one token, k=3 verification, and two-request verification.
Routes cover six shared experts, a mixed 16-expert pool, and disjoint experts.

The measured path includes both TP gathers and fixed-order expert reduction.
CUDA graph replay avoids confusing eager Python launch overhead with serving cost.
Each timing is the median of five sets of 20 replays; the reported result is the
slower rank's median. Baseline stage events are measured separately, so their
instrumented timings do not necessarily sum to the uninstrumented replay time.
Gather stages include packing and waiting for the other rank, not just wire transfer.

Every candidate must match the baseline's FP32 output bitwise on both ranks before
timing. This does not prove full-model quality or benchmark arbitrary routing mutations.
No full generation quality run was needed because no candidate was promoted.

## Results

An initial tile/warp/pipeline sweep tested 49 case/configuration combinations.
All were bit-exact, but small wins were inconsistent and baseline drift was large
for highly shared routes. No scheduling defaults changed.

A separate gate/up projection prototype reduces per-block register pressure by
computing each projection in separate blocks, followed by the same activation and
BF16 conversion. It passed all 35 initial and 56 confirmation comparisons exactly.
Initial apparent gains of 30–49% in selected cases did **not** reproduce in the
alternating confirmation and must not be advertised as improvements.

Four alternating baseline/candidate pairs, median milliseconds per expert call:

| Rows / routing | Baseline | Separate gate/up | Interpretation |
| --- | ---: | ---: | --- |
| 1 / shared | 0.916 | 0.943 | slower |
| 4 / shared | 1.069 | 1.042 | small/inconsistent |
| 4 / mixed | 1.223 | 1.192 | small/inconsistent |
| 4 / disjoint | 1.548 | 1.631 | slower |
| 8 / shared | 1.077 | 1.024 | small/inconsistent |
| 8 / mixed | 1.261 | 1.235 | small/inconsistent |
| 8 / disjoint | 2.617 | 2.700 | slower |

**Rejected for serving.** The extra projection/activation layout did not establish
a robust gain. The candidate's gate/up stage itself was flat or slower; much of
the earlier apparent improvement was in collective/wait timing.

For the four-row mixed baseline, one instrumented pass measured about 0.523 ms
gate/up, 0.224 ms down, 0.040 ms routing, 0.237 ms intermediate gather/packing/wait,
and 0.020 ms final reduction/gather. The intermediate stage ranged widely across
cases/repetitions (roughly 0.03–0.53 ms in the recorded passes). These are not a
full-engine breakdown and cannot be extrapolated directly to output tok/s.

## Full-engine trace

`tools/bench_decode_trace_tp.py` measures TP2 with a 90 GB arena, prune_keep=0.61,
512K capacity, frozen adaptation, and a synthetic 2,737-token Python review prompt.
After two 64-token warm generations, a fixed four-token verification at position
2,803 takes 122.85 ms on both ranks (ten unprofiled steps). Engram rows are
prefetched for this repeated-step test; draft generation and host fetching are not
included. This is not output-token latency.

Five profiled steps give these rank-0 CUDA totals per step:

| Kernel family | ms / step |
| --- | ---: |
| MXFP4 expert gate/up | 28.03 |
| Dense FP8 linear | 21.73 |
| cuBLAS FP32 small-N GEMM | 20.28 |
| MXFP4 expert down | 12.83 |
| NCCL AllGather | 10.45 |
| Grouped FP8 linear | 4.15 |

Expert projections dominate more than communication on this workload. Do not
infer a communication bottleneck from the earlier isolated microbench drift.
FP32 projection precision and fixed decode padding are correctness-sensitive;
their cost is not justification to lower precision.

## Fused decode routing prototype

A benchmark-only Triton router replaces the small Torch sort/scatter sequence.
Each expert's first pair owns its block, and pairs within the expert retain their
original order. It changes block scheduling, not expert selection or arithmetic.
Four alternating baseline/candidate pairs passed all 56 bitwise comparisons:

| Rows / routing | Baseline ms | Fused router ms |
| --- | ---: | ---: |
| 1 / shared | 0.671 | 0.651 |
| 4 / shared | 0.795 | 0.721 |
| 4 / mixed | 1.058 | 1.003 |
| 4 / disjoint | 1.504 | 1.470 |
| 8 / shared | 0.825 | 0.840 |
| 8 / mixed | 1.131 | 1.084 |
| 8 / disjoint | 2.684 | 2.621 |

These are complete expert-call timings, not routing-only or full-engine speedups.
The eight-row shared case regressed. Full-engine alternating graph replay and
generation comparisons are required before considering serving adoption.

The full-engine comparison subsequently passed: identical logits for a fixed
four-token step and identical 64-token greedy output. Four alternating pairs gave
median step times of 120.34 ms baseline and 118.49 ms fused (1.54% lower). Both
series drifted down during the run, so this small difference needs order-balanced
confirmation. The one candidate generation also captured a missing graph parity;
its end-to-end throughput is not a fair baseline comparison. No serving promotion.

The first full-logit attempt used an invalid comparison setup: the repeated-step
driver retained pending compressor state as scratch-buffer views and compared a
cold candidate capture to a warm baseline. The driver now clones that history,
restores it before each replay, excludes cold capture, and asserts baseline
self-reproducibility. These are test-driver fixes, not serving changes. Separately,
`tools/test_decode_routing_cuda.py` passed 227 route mutations during graph replay,
checking exact pair coverage/order against a CPU oracle.

## Dense FP8 scheduling sweep

`tools/bench_fp8_decode_tiles.py` loads six real layer-0 dense weights. It tests
seven scheduling configurations at four input rows on both ranks, keeping the
K tile (and thus reduction grouping), weight format, and precision unchanged.
All 42 comparisons were bit-exact. The narrow K=5120 projections benefit from
BLOCK_N=64, four warps, two stages instead of 128/four/three:

| Projection (N × K) | Baseline ms | Candidate ms |
| --- | ---: | ---: |
| Query A (1280 × 5120) | 0.0432 | 0.0271 |
| KV (512 × 5120) | 0.0436 | 0.0248 |
| Shared expert up (2304 × 5120) | 0.0432 | 0.0248 |

The large query B/output projections and shared expert down did not establish a
benefit from that tile. Do not apply it globally. A combined routing/narrow-FP8
full-engine experiment uses reversed order on alternate timing pairs and warms
candidate graph parities before measuring generation; it also compares depth-8
nesting tokens and scores against baseline.

The combined test preserved full logits and every generated token. However,
order-balanced step medians were 111.06 ms baseline versus 110.52 ms candidate
(0.49%), and warm end-to-end results did not establish a meaningful gain:

| Generation order | Path | Decode seconds | Output tok/s |
| --- | --- | ---: | ---: |
| 1 | baseline | 2.964 | 21.25 |
| 2 | candidate | 3.403 | 18.51 |
| 3 | candidate | 2.949 | 21.36 |
| 4 | baseline | 3.400 | 18.53 |

Both used 23 verification steps, with identical accepted output. The median
decode-time difference is only about 0.2%; observed variation is much larger.
**Not promoted to serving.** The microkernel savings are not evidence of a
user-visible throughput improvement.

Depth-8 nesting emitted identical token sequences on both paths, but neither
128-token response contained a parseable JSON object. This frozen-pruning run is
therefore an equivalence check, not a quality pass. No conclusion about the
user's adapted serving quality should be drawn from it.

## Reproduction

Use the existing two-node gate launcher with a synchronized source snapshot, with
all model services stopped. It requires identical image IDs and both checkpoints.

```bash
GATE_IMAGE=<identical-image-id> \
GATE_LOG_DIR=results/decode-moe-sweep \
GATE_SOURCE_ROOT=<same-absolute-source-snapshot-on-both-nodes> \
bash tools/run_two_node_gate.sh bench_fp4_decode_tp.py
```

Set `DSV41_BENCH_DECODE_EXPERIMENT=separate-up` for the prototype sweep, or
`DSV41_BENCH_DECODE_EXPERIMENT=confirm-separate` for alternating confirmation.
Use `fused-routing` with the MoE microbench for router comparisons. With
`bench_decode_trace_tp.py`, `fused-routing` tests routing alone and `routing-fp8`
tests the combined candidate. `bench_fp8_decode_tiles.py` runs the dense sweep.
This variable is read by the benchmark only. Raw local output is under
`results/decode-moe-{sweep,separate,confirm}/rank{0,1}.log`; it is not committed.
The only serving-source edits are two optional `stage_mark` callbacks, inactive
without an explicitly supplied profiler. No quantization, reduction order, TP
communication sequence, or default kernel configuration changed.
