# Decode kernel launches

`DSV41_DECODE_LEAN=1` (default) cuts the kernels a TP2 decode step launches from **8,457 to
4,115**, bit-identically: same logits path, same token hashes on every measured workload. It is in
the EP2 boot guard with its two sub-switches, `DSV41_DECODE_LEAN_ROPE` and
`DSV41_DECODE_LEAN_SOFTMAX`. `0` on both ranks restores the previous spelling.

## Why launches

A graphed decode step was 8,457 kernels (2026-09-30, current build, python prompt), and
7,600 of them were torch elementwise/copy/reduce kernels of 1-8 us. TensorFold's GLM-5.3 round on
the same pair of Sparks is 1,853 kernels (its `docs/THEORY-2.md`), about 41 per layer against our
180. Inside a CUDA graph a tiny kernel still costs its own duration (~1-2 us minimum on GB10) plus
a sub-microsecond gap, so the cost is per kernel, not per byte. Measured here: the first round
removed 2,072 kernels and 4.6 ms of kernel time, **~2.2 us per kernel removed**.

## How to see a layer's program

`tools/bench_decode_kernels_tp.py` (run through `run_two_node_gate.sh` with a
`GATE_SOURCE_ROOT` snapshot) generates three greedy workloads twice, hashes the output token ids,
then profiles 8 bursts and writes the ordered kernel sequence. `tools/decode_layer_program.py
<dir>/kernels-seq-rank0.tsv` prints one layer: everything between two consecutive routed-expert
`moe_down` kernels. An ordinary layer went 180 -> 134 -> 80 kernels; compressor/indexer layers
are 30-120 more.

## What changed, and why each is exact

The rule: every op whose result depends on reduction order or on the row count stays the same
torch op on the same shape. Only what surrounds it moves.

| Site | Before | After | How |
| --- | ---: | ---: | --- |
| rmsnorm (104 / step) | 12 | 4 | static zero-padded [16, D] fp32 buffer; torch `square().mean()` on it; one kernel for `+eps`, `rsqrt`, the two multiplies and the cast; fp32 weight made once |
| HC coefficients (86 / step) | 15 (+3 copies) | 6 | same padded buffer feeds the padded cuBLAS GEMM (`torch.mm(..., out=)`) and the padded mean; `rsqrt` and `mixes * rsqrt` move into the Sinkhorn kernel, which writes the layer's static buffers |
| hc_post / hc_pre+norm | + copies | in place | `hc_post(out=self.h)` (a program reads its tile's residual streams before writing that tile), `hc_pre_rmsnorm(out=self.y)` direct bf16 store |
| RoPE (~4 / layer) | 4-5 | 1 | one kernel; torch's complex multiply is `re = fma(a, c, -(b*d))`, `im = fma(b, c, a*d)` in this build (below) |
| router | ~23 | ~10 | padded gate GEMM from a static buffer; `softplus`/`sqrt`/`+bias`/prune mask as one kernel; `topk(out=route_idx)`; torch gather + sum; `+1e-20`/divide/scale as one kernel; `index_select(out=slots)` |
| attention softmax | ~12 | 3 | `*scale`, mask, `amax` (exact in any order), `clamp`, `exp` in one kernel; torch `sum`; sink term and division in one kernel |
| attention keys | cat + 2x `.float()` | 1-2 copies | written once into an fp32 buffer |
| shared expert | 7 elementwise | 1 | SwiGLU tail: torch's silu is `x / (1 + exp(-x))` with libdevice `exp` and an IEEE division |
| routed + shared | `.float()`, `+=`, `.to(bf16)` | 0 | `hc_post(x2=)` forms `(routed + shared.float()).to(bf16)` in-kernel |
| per-step index math | per layer | per step | window slots/mask, `freqs[pos]`, compress lengths and the indexer's clamp/mask memoized per captured graph key |

`tools/test_decode_lean.py` (14 tests, real layer-3 weights from `MODEL_DIR`) checks each against
the torch spelling with `torch.equal`, eager and under graph replay.

## Two traps for anyone writing the next one

- **Triton flushes denormals in libdevice by default** (`enable_reflect_ftz=True`); torch's nvcc
  build does not. The first router and softmax kernels differed by 1.2e-38: `exp` of a very
  negative logit is a denormal in torch and 0 in Triton, and `sqrt(softplus(x))` turns that into
  1e-19 vs 0. Every lean kernel that calls libdevice launches with `enable_reflect_ftz=False`
  (and `enable_fp_fusion=False`). Triton's plain `/`, `tl.exp` and `tl.sqrt` are approximate; use
  `libdevice.div_rn`, `libdevice.exp`, `libdevice.sqrt_rn`.
- **FMA contraction is part of torch's numerics.** Checking RoPE at bf16 output accepted three
  contractions of the imaginary part, because bf16 rounding hides fp32 differences. At fp32, only
  one of nine matches: 0 of 4.2M products differ, every other pairing 24-33 %
  (`test_rope_fp32_contraction`). Compare before the final rounding.

## Measured (TP2 pair, frozen ranking, greedy, 256 tokens)

Arms alternate; run-to-run spread of one build is 8-10 %, so single runs decide nothing.

Round 2 (everything above), order lean, off, lean, off:

| Arm | html tok/s | python tok/s | explain tok/s | kernels / step | kernel ms / step |
| --- | ---: | ---: | ---: | ---: | ---: |
| lean | 43.37 | 44.99 | 22.50 | 4,115 | 119.6 |
| off | 39.61 | 43.29 | 21.45 | 8,457 | 123.8 |
| lean | 45.14 | 46.66 | 23.20 | 4,115 | 114.1 |
| off | 41.83 | 43.32 | 21.63 | 8,457 | 123.2 |
| **mean change** | **+8.7 %** | **+5.8 %** | **+6.1 %** | **-51 %** | |

Per step that is -9.1 ms (html), -6.2 ms (python), -5.8 ms (explain), from the arms' mean
ms/step (104.8 vs 113.9, 107.1 vs 113.2, 95.4 vs 101.2). Round 1 alone (rmsnorm, HC, buffers;
6,385 kernels) measured 42.23 / 42.08 html against 40.30 off. The profiled `kernel ms / step`
includes profiler overhead and is only comparable between arms.

Token hashes (`df46ab5e`, `73a30f4e`, `285bf77a`) are identical in every arm. Results:
`results/kernels-20260930/`.

## What is left in a layer (80 kernels)

- the kept cores: 2 x (padded GEMM + split-K reduce + square + mean) for HC, 2 x (square + mean)
  per norm;
- five NCCL all-gathers, each followed by a reorder copy (routed intermediate and output, shared
  intermediate and output, `wo_b` input and output). The routed/shared pairs share a column split,
  so they can be one collective each with the same bits;
- ~5 small kernels around the native MoE routing call (null-slot keys);
- the torch attention's fp32 staging: `q.float()`, ring and compressed-row gathers, their copies.
