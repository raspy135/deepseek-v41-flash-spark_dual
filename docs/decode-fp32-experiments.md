# FP32 decode projections — September 16, 2026

HC decode padding now defaults to 32 rows; `DSV41_HC_MM_TILE=16` restores the
previous path. Weights, inputs and accumulation remain FP32; TF32 is disabled
in the layout microbenchmark. The new choice changes rounding order, not precision.

## Where the time goes

Grouping the earlier full-engine CUDA trace by launch geometry separates the
126 small FP32 GEMMs per verification step:

| Family | Calls / step | Grid | CUDA ms / step |
| --- | ---: | --- | ---: |
| Hyper-connections | 80 | 3 × 2 | 17.17 |
| Expert router | 40 | 48 × 2 | 2.67 |
| Compressor | 6 | 64 × 2 | 0.45 |

The HC weight is only 24 output rows but has 20,480 input columns. The cuBLAS
kernel selected for the fixed 16-row decode shape launches six blocks on a
48-SM Spark. The long reduction is poorly parallelized. This explains why such
small weights can consume more time than their byte count suggests.

## Real-weight layout/padding microbenchmark

`tools/bench_fp32_decode_layouts.py` loads layer-0 HC attention/FFN and expert
router weights, plus the first available compressor gate. Four random BF16
activation rows are promoted to FP32 and padded to the baseline's 16 rows.
CUDA graph replay times the GEMM alone, excluding one-time padding/layout
preparation. Results use the slower rank's median of five sets of 50 replays.

| HC projection | Baseline ms | Pad input to 32 rows, ms | Max absolute difference |
| --- | ---: | ---: | ---: |
| Attention | 0.1106 | 0.0308 | 6.20e-6 |
| FFN | 0.1165 | 0.0315 | 9.06e-6 |

**Not bit-exact.** The changed cuBLAS algorithm uses a different reduction order,
despite retaining FP32. Padding output columns was bit-exact but did not establish
a meaningful gain. Transposing HC weight storage was slower and non-exact.
The expert router and compressor benefited from transposed storage, but those
results were also non-exact; they are not part of the full-engine candidate.

The isolated `hc32` candidate changes only HC decode projections, always padding
1–16 input rows to the same 32-row shape. Both single-token and speculative
verification use that same shape. It does not change prefill, quantization,
expert selection policy, or TP communication. Its full-model effects must be
measured before considering any serving option. A small projection error must
not be equated with a small downstream logit or quality change.

## Full-engine and quality results

TP2, native MXFP4, 512K capacity, 90 GB arena, 0.61 keep fraction, k=3. The
2,737-token synthetic coding prompt uses frozen trace-only selection for the
performance comparison. Four alternating pairs (reversing order on alternate
pairs) measured median fixed verification time **122.47 → 107.52 ms**, 12.2%
less time. This excludes draft/Engram host fetching; it is not output-token latency.

Warm 64-token generation times were 3.779/3.262 seconds baseline and 2.749/2.754
seconds candidate. Candidate throughput was 22.87–22.92 tok/s versus 16.67–19.31
baseline. Both baseline runs were slower, but their spread makes the 12% step
measurement a more conservative description than claiming a universal throughput
gain. All four measured generations emitted the same 64 tokens. Fixed-block
logits were not exact (maximum difference 0.5), but all four argmax tokens agreed.
Subsequent user testing did not reveal a noticeable real-world difference. The
12% result describes this fixed verification workload, not a confirmed improvement
in real-request throughput or latency.

A separate low-volume quality run loaded the saved demand ranking (78.6% observed
weight), disabled further swaps and disk-prefix reuse, and did not save demand.
It used the existing `llm_benchmark/quality_quant2.py` prompts and graders:

| Subset | HC16 | HC32 |
| --- | ---: | ---: |
| Nesting depths 4, 6, 8, 10 | 0.96875 | 1.0 |
| First three word-constraint cases | 0.0 | 0.0 |
| Three character-count cases | 0.0 | 0.0 |

Nine of ten graded outputs were token-identical. Depth 8 changed from seven
levels to the correct eight. Neither configuration passed the selected counting
or word-constraint cases; these are unchanged weaknesses, not quality passes.
A short Japanese explanation changed wording and remained imperfect. This small
sample found no graded regression, but cannot rule out regressions elsewhere.
The user accepted small quality variation in exchange for measured speed.

`tools/test_hc_mm_cuda.py` passed 16 checks: exact single-token/verify row
invariance at both HC settings, exact agreement with explicit padding, unchanged
larger prefill, and unchanged non-HC projection behavior. The setting is checked
across ranks at boot and exposed as `hc_mm_tile` in health. Prefix namespaces
already include source and DSV41 environment settings, preventing cross-setting
reuse of disk snapshots. The other routing/FP8 prototypes remain disabled.

## Reproduction

Use the existing two-node gate with a synchronized source snapshot and identical
container image IDs, with model services stopped:

```bash
GATE_IMAGE=<image-id> GATE_LOG_DIR=results/decode-fp32-layouts \
GATE_SOURCE_ROOT=<snapshot> \
bash tools/run_two_node_gate.sh bench_fp32_decode_layouts.py

DSV41_BENCH_DECODE_EXPERIMENT=hc32 GATE_IMAGE=<image-id> \
GATE_LOG_DIR=results/decode-hc32-full GATE_SOURCE_ROOT=<snapshot> \
bash tools/run_two_node_gate.sh bench_decode_trace_tp.py
```

For the small quality comparison, copy the existing
`../llm_benchmark/quality_quant2.py` into the isolated snapshot's tools directory
as `bench_quality_quant2_fixture.py`, synchronize that snapshot to the worker,
and run `bench_hc_quality_tp.py` through the same gate. The fixture is not a
serving dependency and is not copied into the repository or container image.

The initial microbench attempt omitted cropping padded output columns in its
comparison and raised a shape error; `decode-fp32-layouts-fixed` is the completed
run. No serving process was affected.
