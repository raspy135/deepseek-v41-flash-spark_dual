# Decode projection scheduling, merging and prune-miss fusion

Four decode changes, all in the EP2 boot guard. **On by default since 2026-09-23**, after the
full-engine A/B below. To roll one back, set it to `0` (`128` for the tile) on both ranks. Each is meant
to leave logits, hidden states and tokens bit-identical. Each is checked for that in unit tests
and again in the full-engine gate.

| Switch | What changes |
| --- | --- |
| `DSV41_FP8_DECODE_BLOCK_N=auto` | Decode fp8 (M ≤ 16) output-column tile; `auto` = widest of 128/64/32 that fills all 48 SMs, else 16 |
| `DSV41_DECODE_MERGED_PROJ=1` | One launch and one activation quantization for `wq_a‖wkv` and shared `w1‖w3` |
| `DSV41_PRUNE_MISS_FUSED=1` | `DSV41_PRUNE_MISS` accounting as one Triton launch per decode layer |
| `DSV41_FP8_ACT_QDQ_FUSED=1` | Activation fp8 quantize-dequantize inside the fp8 GEMM (step 1 below) |

## Why they are exact

- **BLOCK_N** only partitions output columns. Each output element is still a single fp32
  accumulation over K in the same `BLOCK_K` steps.
- **Merging** stacks weights along N. Column-wise it is the same GEMM, and the activation is
  quantized once instead of twice, which gives the same values. The attention slices are made
  contiguous so the norms' reductions see the same tensors as before. The stacked weight replaces
  the originals' storage (they become row views), so no memory is added. TP-wrapped or FP4
  weights are skipped.
- **Prune-miss fusion** does not touch routing. Its only output is the demand database. Rank 0
  plans swaps from that database and broadcasts the plan, so the two ranks cannot diverge
  through it.

## Unit tests (GB10, serving image, synthetic weights at TP2-local shapes)

`python3 -m unittest tools.test_fp8_decode_tiles engine.test_prune_miss_fused`: 11 tests.

- BLOCK_N 64, 32, 16 and auto, at 2 and 4 warps, give `torch.equal` results against 128 for
  every served shape. Rows tested: 1, 4, 6, 10, 16. Output dtypes tested: bf16 and fp32.
- The same holds for the grouped `wo_a` kernel.
- Merged `qlinear` slices equal the separate calls, including a downstream `rmsnorm`.
  Merged `expert_ffn` equals the unmerged one.
- Graph capture works for both.
- Fused prune-miss counts and miss totals equal the torch spelling exactly. This holds for fresh
  and for integer-seeded databases. Score mass agrees to 1e-12, and the difference is only
  summation order. Graph replay accumulates correctly, and blocks larger than 16 rows fall back
  to torch.

## Microbenchmark, 2026-09-23

Single GB10, serving stopped, M=4 (`DSV41_BLOCK=3`). `tools/bench_decode_projection_micro.py`
uses CUDA-graph replay over **40 distinct weight copies**, arms alternating in order.

**Pitfall recorded:** the first version replayed one weight 40 times. It stayed in L2 and
reported up to 758 GB/s on a ~273 GB/s part. That run is kept as
`micro-l2-hot-invalid.log` and must not be used.

| Projection (N×K) | 128 (µs) | auto (µs) | auto tile |
| --- | ---: | ---: | ---: |
| wq_a 1280×5120 | 39.6 | 35.8 | 16 |
| wkv 512×5120 | 37.4 | 21.2 | 16 |
| sh_w1/w3 TP-local 1152×5120 | 38.5 | 32.3 | 16 |
| sh_w2 TP-local 5120×1152 | 28.8 | 28.5 | 64 |
| wq_b TP-local 16384×1280 | 94.8 | 94.6 | 128 |
| wo_b TP-local 2560×8192 | 100.9 | 99.6 | 32 |

Findings from the table:

- `wq_b` and `wo_b` already run at 208–222 GB/s. A narrower tile buys nothing there, which
  matches the earlier sweep.
- BLOCK_N 16 at 4 warps loses on `sh_w2` (1.67×). `auto` avoids that tile for `sh_w2` by
  picking 64.

**Merged vs separate, per call, including `qlinear`'s activation quantization:**

| Pair | Separate, 128 | Merged, 128 | Separate, auto | Merged, auto |
| --- | ---: | ---: | ---: | ---: |
| wq_a+wkv | 110.2 µs | 66.3 µs | 88.4 µs | 67.6 µs |
| sh_w1+w3 (local) | 115.6 µs | 77.2 µs | 98.6 µs | 74.6 µs |

- Most of the merge's saving is the second `act_qdq_fp8` (~16 µs per call, about 13 small
  kernels), not the GEMM itself.
- Once merged, the tile policy adds nothing: the merged kernels already reach 200–209 GB/s.

**Prune-miss accounting, one layer:**

- torch spelling: 25.0 µs.
- Fused 2D kernel: 6.7 µs at 4 warps (the default); 1, 2 and 8 warps are slower.
- An earlier row-serial version of the kernel took 13.7 µs.

**Sum:**

| Candidate | Saving per 40-layer verify |
| --- | ---: |
| Merged projections | ~3.3 ms |
| Tile policy on the remaining projections | ~0.1 ms |
| Prune-miss fusion | ~0.7 ms |
| **Total** | **~4 ms** |

That is roughly 3% of the ~129 ms iteration in the 2026-09-16 trace. These are kernel-level
numbers, **not a decode-rate claim**. The full-engine gate decides.

## Step 1: activation quantization inside the fp8 GEMM

`DSV41_FP8_ACT_QDQ_FUSED=1`, default off, in the boot guard. `R.act_qdq_fp8` costs ~16 µs per
`qlinear` call in graph replay (about 13 small kernels). It runs about 5–7 times per layer.

With the flag on, decode-sized bf16 activations going into an fp8 weight are quantized inside
`_fp8_linear_kernel` instead. Each 32-wide quantization group lies inside one K tile, so each
block quantizes its own tile. This covers:

- plain weights: `wq_a`, `wkv`, `wq_b`, the indexer's `wq_b`, and shared `w1`/`w3`;
- the TP wrappers: `wo_b` is output-parallel and quantizes after the gather; shared `w2` is
  row-parallel and quantizes its own K shard. Both give the same groups as before.

Three cases keep the torch spelling: prefill-sized input, fp32 input, and a swapped-out
`act_qdq_fp8` (engine act-quant off).

Why it is exact:

- `amax` does not depend on reduction order.
- Division, `log2` and `exp2` go through libdevice's IEEE versions. Triton's default `/` and
  `tl.log2` are approximate, and either could move `ceil(log2(·))` across an integer.
- The power-of-two scale multiply is exact.
- The e4m3 cast rounds to nearest even on both sides.

Compiled for sm_121 without a GPU, the PTX has no `lg2.approx` and no `div.full`. It does have
flush-to-zero variants, which cannot change a result here; the reasoning is at
`_act_qdq_tile`.

`tools/test_fp8_act_qdq.py` checks the quantizer element by element through an identity weight.
It targets these edge cases:

- amax exactly on a power-of-two boundary, and one ulp above it;
- groups under the 1e-4 floor;
- e4m3 rounding ties;
- negative zero.

It also checks real shapes, merged weights, the shared expert, both TP wrappers (with a stub
collective) and graph capture. All pass on GB10 (2026-09-23).

**Negative result for the served config.** Serving runs with activation quantization off
(`act_quant=False`, `DSV41_ACT_QUANT` unset). In that mode `Model.__init__` replaces
`R.act_qdq_fp8` with a bf16 cast, so the ~13-kernel quantization this step removes never runs.
The fused path's guard correctly declines. The full-engine A/B was token- and logit-exact,
and every timing difference was noise (+0.8, +1.0, −1.1, 0.0 ms). The microbenchmark measured
the `act_quant=True` spelling, which serving does not use. **Check `act_quant` before counting
`qlinear` quantization as serving cost.** Keep the switch only for `DSV41_ACT_QUANT=1`.

The first in-kernel version was 2.3× slower than torch (`wq_a` 56 → 129 µs), because it used an
IEEE `div_rn` per element inside the K loop. Dividing by the power-of-two scale was replaced by
multiplying by `exp2(-e)`, which gives the same rounded value. That version is 15–20 µs faster
per call than torch; this applies to `act_quant=True` only.


## Full-engine results, 2026-09-23

`tools/run_decode_window.sh`, results in `results/decode-window-20260923-2201/`. Setup: the
served config (TP2, `DSV41_BLOCK=3`, shared overlap and native Engram on, CUDA FP4), HTML prompt,
512 tokens, frozen ranking. Rank-0 fixed-verify batches were timed in alternating order.

**Exactness:** in every experiment, logits and hidden states were exact on both ranks, all six
generations were token-identical, and depth-8 nesting was identical.

**Timing** (per-batch pairs, candidate minus baseline):

| Experiment | Pairs | Per step |
| --- | --- | ---: |
| prune-miss | −0.5 ms (median) | ~−0.5 ms |
| merged | −2.4, −1.6 (2 clean pairs; both arms had slow outliers) | ~−2 ms |
| act-qdq | +0.8, +1.0, −1.1, 0.0 | none (see above) |
| block-n auto | −1.5, −2.4, −1.2, −2.4 | ~−1.9 ms |
| **all four** | **−4.0, −3.7, −4.0, −3.7** | **~−3.8 ms (−4.1%)** |

The `all` generations ran at 30.1 and 31.8 tok/s baseline versus 32.9 and 33.0 candidate.
That is noisy (the series drifts), but it points the same way.

**All five A/Bs reported `FAIL`, and every one was an artifact.** Rank 0's demand database holds
a non-finite score mass. The mass check subtracted before from after, which gives NaN in both
arms, and the check treated that as a mismatch. Counts and miss totals were exact. The check
now compares finite entries and reports how many non-finite values the database holds.

**Gate forwarding bug (fixed).** The first attempt that day ran every stage on code defaults
(EP2, `DSV41_BLOCK=5`, no overlap). `run_two_node_gate.sh` piped `env` through `rg`, which
existed only as a shell function in the interactive shell, so no `DSV41_*` setting was
forwarded. Both ranks agreed with each other, so the boot guard stayed silent. The gate now
uses `grep`, refuses to start if `.env` sets `DSV41_*` and nothing is forwarded, and accepts
`GATE_ENV="DSV41_X=v ..."` overrides. The 2026-09-16 runs were not affected: their configs
show `tp_experts: true`.

## Maintenance window

Stop serving on both nodes and wait for the memory pool to be released. Then run:

```bash
GATE_IMAGE=$(docker image inspect -f '{{.Id}}' deepseek-v41-flash-spark:local) \
bash tools/run_decode_window.sh            # or name experiments: merged act-qdq ...
```

The script does the following:

1. Refuses to start while serving runs or with less than 90 GiB available.
2. Snapshots `engine/`, `tools/` and `server/` to the same path on both nodes (the image holds
   older code) and verifies that the file hashes match.
3. On this node: runs the unit tests and the microbenchmark.
4. **Step 0:** profiles decode with the current `.env` config (`bench_decode_timeline_tp.py`).
5. Runs `bench_decode_projection_tp.py` once per candidate, then with all of them together.

Output goes to `results/decode-window-<stamp>/`, with a `summary.txt`.

The A/B procedure:

1. Separate graph pools per arm.
2. Order-balanced 512-token HTML generations (A B A B B A), with every token compared.
3. Fixed-step logits and hidden states compared, plus the decode demand each arm records.
4. Four alternating 10-step verify batches, timed.
5. Depth-8 nesting output compared.

A run that differs anywhere still reports its timings but ends `FAIL`.
