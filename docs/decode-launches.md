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

## Round 3: shared collectives (`DSV41_DECODE_LEAN_MOE`)

Output-layout TP gathered the routed intermediate [P, 1152] and the shared-expert intermediate
[T, 1152] separately, then the routed and shared outputs separately: four all-gathers and four
reorder copies per layer. Both pairs have the same column split, so `_moe_merged` stacks the two
intermediates into one gather, sums the two outputs locally in fp32 before one gather
(`routed + shared.float()` is elementwise, so the order of sum and gather does not matter), and
`hc_post(split_w=2560)` reads the gathered [2, T, 2560] as is. The native routing prologue and
epilogue (null-slot keys, block remap) became two kernels. `tools/test_decode_lean.py` covers the
pieces; the two-node run covers the collectives:

| Arm | html tok/s | python tok/s | explain tok/s | kernels / step | NCCL / step |
| --- | ---: | ---: | ---: | ---: | ---: |
| merged | 43.99 | 45.42 | 22.69 | 3,715 | 170 (7.9 ms) |
| separate | 41.70 | 44.72 | 22.18 | 4,115 | 250 (8.9 ms) |
| merged | 43.90 | 45.22 | 22.77 | 3,715 | |
| separate | 43.62 | 45.20 | 22.55 | 4,115 | |

Token hashes identical. The gain is ~1 ms per step and within the noise of the second pair:
80 fewer collectives saved only ~1 ms of NCCL time, because the remaining ones wait longer. The
collective cost here is latency and rank skew, not bytes -- merging more of them will not buy much.

## Attention staging (`DSV41_DECODE_LEAN_ATTN`): exact, fewer launches, no measurable gain

`_attention_lean` has RoPE emit q already widened to fp32, gathers the window and compressed keys
straight into the fp32 key block (`keys_f32`: the packed-KV decode reproduced bit for bit,
negative zero included), and has the output RoPE round the einsum result itself. 158 fewer
kernels per step, same token hashes -- and no speed change:

| Arm | html tok/s | python tok/s | explain tok/s | kernels / step |
| --- | ---: | ---: | ---: | ---: |
| on | 44.42 | 46.01 | 22.81 | 3,557 |
| off | 45.18 | 46.83 | 23.16 | 3,715 |
| on | 45.92 | 47.44 | 23.67 | 3,557 |
| off | 44.66 | 46.92 | 23.26 | 3,715 |

The removed kernels were copies, and writing the gathered keys as fp32 moves the same bytes the
copies did. Kept on because it is exact and costs nothing; do not count it as a speedup.

## Routed-expert kernels: v2 (`DSV41_FP4_CUDA_V2`, default on)

Distinct routed experts per layer per verify step (`DSV41_ROUTE_STATS=1`, 256 greedy tokens):
python 21.7, html 20.3, explain 16.5 (the last mostly at depth 3). At TP2 a slot is 6.27 MB for
`up` (w1, w3, scales) and 3.13 MB for `down`, so python reads ~136 + 68 MB per layer.

`tools/bench_fp4_moe_decode.py` (one GPU, L2-cold calls, graph replay) put v1 at 150-185 GB/s at
these U against a 205-225 GB/s plain read of the same bytes; below U = 14 it fell to 80-160. v1
gives each lane 2 bytes of a 64-byte warp tile per K step and walks the row again for each routed
pair. v2 gives each lane whole 32-value scale groups (one 16-byte load per matrix), keeps the row
in registers across the pairs, and replays v1's relaxed reduction exactly: each group's 8 virtual
lane partials with v1's fma pattern and shuffle tree, then each subgroup's running sum as a
sequential chain through shared memory. `tools/test_fp4_moe_v2.py`: bit-identical on TP and
single-box shapes, T = 4 and 6, U = 6..36, with and without null-slot pairs. (Whether v1's
`p *= scale; acc += p` was contracted into an fma does not matter: UE8M0 scales are powers of two.)

| U | up v1 -> v2 (us) | GB/s | down v1 -> v2 (us) | GB/s |
| ---: | ---: | ---: | ---: | ---: |
| 14 | 554 -> 409 | 159 -> 214 | 312 -> 299 | 140 -> 147 |
| 20 | 680 -> 599 | 184 -> 209 | 364 -> 303 | 172 -> 207 |
| 26 | 820 -> 760 | 199 -> 214 | 417 -> 391 | 195 -> 209 |

In the engine (alternating, token hashes identical):

| Arm | html tok/s | python tok/s | explain tok/s | up / down per layer |
| --- | ---: | ---: | ---: | --- |
| v2 | 45.87 | 48.22 | 23.99 | 709 / 355 us |
| v1 | 44.55 | 45.98 | 22.98 | 842 / 380 us |
| v2 | 46.75 | 48.18 | 24.00 | 730 / 355 us |
| v1 | 45.88 | 47.68 | 23.73 | 770 / 360 us |

+2.4 to +2.9 % (-2.5 to -3 ms per step), less than the microbenchmark promised, and the reason is
the ceiling: in the engine the shared expert's w1||w3 (~12 MB) streams on the side stream during
`moe_up`, so `moe_up`'s window moves ~148 MB in ~710 us, ~208 GB/s. **The routed experts are now
at the DRAM ceiling in situ.** What is left of their ~46 ms per step is bytes -- ~22 distinct
experts x 9.4 MB x 40 layers -- and only fewer or smaller expert reads will move it, not a kernel.
v2's own weak spot, many pairs on few experts (U < 14), is not where serving runs; batching the
chain replay across pairs would fix it and was not done.

## What is left in a layer (~66 kernels)

- the kept cores: 2 x (padded GEMM + split-K reduce + square + mean) for HC, square + mean per
  norm, topk and the k-way sum in the router, the softmax sum;
- three all-gathers (MoE intermediates, MoE outputs, and the `wo_b` input/output pair), two
  reorder copies;
- the matmuls themselves.

Further exact trims exist -- the ring write folded into the kv RoPE, the HC and router padding
copies written by the kernel before them, `hc_post` reading the `wo_b` gather in place -- and add
up to well under 1 ms per step. The time is elsewhere now (per step, python prompt, profiled):
routed experts ~50 ms (the native `moe_up`/`moe_down` at an estimated 165-180 GB/s against a
~235 GB/s streaming ceiling), dense fp8 projections ~30 ms, fp32 GEMMs ~10 ms (HC, router gate,
torch attention), the bf16 LM head ~7.7 ms (read twice per step), NCCL ~7.9 ms.

## Dense FP4 (`DSV41_DENSE_FP4`): no decode gain on this build (null result)

Measured 2026-10-01 on the round-3/v2 build, TP2, native FP4 experts, `DSV41_TP_ATTN=1`,
`DSV41_DRAFT_HEAD_FMT=off` (to isolate the variable), greedy, 256 tokens, arms alternating. Six
`bench_decode_kernels_tp.py` gate runs:

| dense_fp4 | run | python tok/s | python ms/step | html tok/s | explain tok/s | profiled dense ms/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 1 | 49.2 | 101.6 | 44.8 | 22.1 | 29.5 |
| attn,wo_a | 1 | 48.8 | 102.4 | 45.4 | 22.1 | 30.2 |
| attn,wo_a | 2 | 48.6 | 102.8 | 45.5 | 22.1 | 29.5 |
| off | 2 | 48.4 | 103.4 | 43.3 | 21.5 | 30.9 |
| attn | 1 | 49.8 | 102.4 | 46.0 | 22.3 | 31.1 |
| wo_a | 1 | 49.3 | 103.5 | 43.3 | 22.1 | 30.6 |

Every arm is inside the build's 8-10 % run-to-run spread, and the profiled dense family stays near
30 ms/step even though the fp4 weights are ~half the bytes -- but the aggregate hides a bimodal
kernel. Achieved bandwidth at M=6 (L2-cold, real weights, `tools/test_fp4_linear.py` method):

| weight | N x K | fp8 GB/s | fp4 GB/s | fp4 time / fp8 |
| --- | --- | ---: | ---: | ---: |
| wq_b | 32768x1280 | 224-231 | 199-217 | 0.62 |
| shared w2 | 5120x2304 | 214-218 | 162-174 | 0.67-0.70 |
| wo_b | 5120x8192 | 197 | 105-111 | 0.94-1.00 |
| shared w1 | 2304x5120 | 160-204 | 113 | 0.96 |
| wq_a | 1280x5120 | 185-190 | 67 | 1.47 |
| wkv | 512x5120 | 120-131 | 33 | 1.81-2.11 |

The format is fine: on wide N the fp4 kernel reaches 199-217 GB/s against the fp8 kernel's 224-231
and is 1.6x faster, essentially the 0.53 byte ratio. On narrow N it collapses to 33-67 GB/s against
fp8's 120-190 on the same shape and bytes. That is a parallelism bug in the fp4 kernel, not a
property of 4 bits: `BLOCK_N=32` gives `wkv` 16 CTAs for 48 SMs with a 40-iteration serial K loop of
8 dependent dots each, and a sweep of `BLOCK_N` x `num_warps` x `num_stages` does not lift it (best
still 35 GB/s for `wkv`, 68 for `wq_a`). `attn` is mostly these narrow shapes, so its wide-N win
cancels their loss; `wo_a`'s grouped kernel has a G=8 group axis for parallelism, which is why it is
the one clear win (~1 ms). The fix is a split-K variant with a fixed-order reduction -- named as the
gap in the 2026-09-11 notes and still not built -- not a different launch config.

The earlier pre-round-3 gain (134.4 -> 125.6 ms step) is therefore not "gone"; it was measured on a
build whose narrow shapes were equally starved, and the intervening fp8 decode scheduling moved the
baseline under it. Dense fp4 also changes tokens (it is a re-quantization), so enabling `attn` today
is a wash that costs quality; `wo_a` alone is the honest subset.

Two notes from the exercise:

- `DSV41_DENSE_FP4` could not load under TP attention before this date: `shard_attention` called
  `.shard()` on `wq_b`/`wo_b`, which `FP4Weight` did not implement, and it had no
  `FP4GroupedWeight` branch for `wo_a`. Added (`tools/fp4_linear.py`,
  `engine/tensor_parallel.py`), with a dequant-equals-slice check in `tools/test_fp4_linear.py`.
  The gate arms above load and pass with it.
- The narrow-N shapes are the ones that lose, and it is a missing split-K, not the format: the
  same kernel reaches 199-217 GB/s on `wq_b`/`w2`. A split-K with a fixed-order reduction is the
  change that would move this, and it is not built. `wo_a` (grouped, G=8) already works and is ~1 ms.
