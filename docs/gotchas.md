# Gotchas

Every one of these was hit for real, or is written down because the code had to be shaped
around it. Sources: [`NOTES.md`](../NOTES.md) and [`LIMITATIONS.md`](../LIMITATIONS.md).

## The box hard-resets when host memory goes negative

There is no OOM killer moment to catch and no log line to find afterwards. On unified
memory the GPU allocator and the page cache draw on the same pool; push `MemAvailable`
towards zero and the machine goes down. Everything below follows from that:

* The engine keeps a hard `keep_free_gb` floor (20 GB) under `MemAvailable` when sizing the
  arena, and the auto factor was lowered from 0.88 to 0.82 after a warm start peaked at
  111 GiB of 121 with 10 GiB left.
* `start.sh` and the container entrypoint **refuse to start** below `MIN_FREE_GIB` (90),
  naming the processes that hold the pool.
* `stop.sh` and `./run.sh stop` block until the pool is actually back. Pinned and
  page-cache-backed memory is reclaimed lazily; starting the next server before that lands
  is the classic way to wedge the box.
* `compose.yaml` uses `restart: on-failure:1`, not `unless-stopped`. A load that fails
  usually fails because the pool is spoken for, and restarting into the same condition
  turns "the server did not start" into "the power button is the only way back".

## Do not set a memory limit on the container

`--memory` / `mem_limit` on this hardware is a cap on **GPU** allocations too, and the
arena auto-sizer will silently shrink to fit it — giving a small hot set and a permanently
NVMe-bound server with no error anywhere. There is no cgroup knob that means "host RAM but
not GPU" here. Leave it unset.

## `torch.cuda.mem_get_info()` lies on GB10

It counts the page cache as used. Right after the checkpoint download it reported **32.0 GB
free on a box with 99.9 GiB `MemAvailable`** — which would have handed the engine a 23 GB
arena, about 7% of the routed experts, and a server that streams every token. The engine
now takes the larger of `mem_get_info()` and `/proc/meminfo`'s `MemAvailable` and logs
both. If you write anything that sizes a buffer from free memory on this box, do the same.

## The transient ring must hold a full layer

Prefill touches nearly every expert of a layer (370–381 of 384 measured at layer 0), so
prefill misses go to a small ring of transient slots instead of through the LRU — otherwise
every prompt evicts the hot set. The ring's size is not a tuning knob with a soft floor:
below 384 slots it **wraps inside a single layer** and the engine computes that layer with
the wrong experts. No exception, no warning — just an 0.88 relative error in the output,
which is exactly what the first smoke test produced. Default 400. Do not lower it.

## An expert is two reads, not six and not one

The six tensors of an expert (`w1/w2/w3` × weight and scale) are not adjacent in the shard:
all the scales sit near the front, all the weights far behind, but within each group an
expert's three tensors are contiguous. Read it as six `preadv`s and you pay six `O_DIRECT`
round trips — and at a large arena a decode step misses only about one expert per layer, so
nothing else is in flight to hide the latency. The rate collapses from ~4.7 GB/s to
~0.6 GB/s. `ShardFile.expert_runs` groups them into exactly two runs: 1.1 MB of scales,
17.7 MB of weights.

## `O_DIRECT` alignment, and the last tensor of a shard

`O_DIRECT` reads must start and end on 4 KiB boundaries and land in an aligned buffer, so
every read is widened outward to `ALIGN` and the wanted bytes are sliced out afterwards. Two
consequences that bite:

* The staging buffers are pinned and over-allocated by `8 × ALIGN` for exactly that
  widening.
* The **aligned tail can run past EOF** on the last tensor of a shard, and the short read
  that comes back is not an error — the loop tops up until it has what it asked for and
  accepts a short final read. Code that treats a short `preadv` as a failure will break here
  only on some shards, which is the worst way to find out.

And the whole path needs a real filesystem: `O_DIRECT` does not work on overlayfs, so the
checkpoint must be a **bind mount**, never a `COPY` into the image or a network filesystem.

## Engram rows are too small for `O_DIRECT`

48 random 264-byte reads per token. `O_DIRECT` on those is all overhead: the engram reader
uses **buffered** `preadv` in a thread pool with `POSIX_FADV_RANDOM` and a small
process-local row cache, and it is not the bottleneck (3,792 rows in 0.65 s on a 64-token
generation). Do not "optimise" it into the `O_DIRECT` path.

## The CDN rate-limits single-range requests

`tools/engram_rows.py` fetches only the n-gram rows a corpus needs, straight out of the two
101 GB shards on the Hub, without downloading them. Single-range requests get HTTP 429 at
around 400/s. **Multipart** range requests are the way: 500 rows per request, ~314 requests
per layer, 10–11 s for a 10,760-token corpus. If you rewrite that fetcher, keep the
multipart batching or you will be rate-limited into hours.

## cuBLAS makes the same GEMM give different bits

Two separate mechanisms, both of which had to be disarmed for the engine to be chunk-invariant:

* **split-K reduction in bf16.** `F.linear(x[:6], w) != F.linear(x, w)[:6]` by ~2.4e-3 on
  the N=512 `wkv` projection, and the attention softmax amplifies that ~2× per layer.
  `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` cuts it to
  ~9e-5.
* **tiling chosen from M.** cuBLAS picks its tiling and split-K from the number of rows, and
  for several shapes even a row's *offset inside* the tile changes its last bits — 8 and 16
  are offset-invariant, 32/64/128 are not. So every activation GEMM and last-dim reduction
  runs in fixed 16-row tiles. It costs about +10%: a 512-token prefill chunk issues 32 GEMM
  launches per projection instead of one.

## Bit-exactness stops at 512 tokens

Beyond ~1024 tokens of context the indexer's top-512 stops keeping every visible compressed
position, and which 512 it keeps is decided by scores computed against a key cache whose
*length* differs between a chunked and a single-chunk run. Ties then break differently and
the two runs can select different positions. The engine is close but not identical past that
point — and so is the reference implementation, for exactly the same reason. Do not write a
long-context regression test that asserts equality.

Related: `Caches.rollback(n)` only restores the compressor's pending token if `n` lies inside
the last forwarded chunk (or exactly at its start). Further back it raises rather than
silently producing a wrong latent. That covers the speculative-decoding use and nothing else.

## 16-byte tiles cap at ~130 GB/s on GB10

The FP4 MoE kernel is bandwidth-bound on the 18.8 MB every active expert weighs, so tile
width is not a micro-optimisation. Narrow (16-byte) tiles top out around 130 GB/s on this
part; the 64-byte-wide tiles the kernel uses reach 193–197 GB/s effective at decode sizes,
against a ~210 GB/s practical copy ceiling on a nominally 273 GB/s box. If you retune
`BM`/`BN`, watch the effective bandwidth, not the occupancy.

## A native CUDA spelling is not automatically faster than Triton

`tools/fp4_moe_cuda.cu` is a CUDA C++ decode implementation of the same packed-FP4
operation (`DSV41_FP4_CUDA=1`). It uses the hardware E2M1 conversion, coalesced 64-byte row loads,
the same per-32 UE8M0 boundary, BF16 linear-output rounding, a one-kernel stable router, and is
CUDA-graph capturable. After the tuning below, native CUDA with relaxed reduction was selected as
the default on 2026-09-22. The initial negative measurements are retained here as history.

Real layer-0 weights, 32 experts, FP32 routed output, 20 order-balanced A/B/B/A measurements on
GB10 (another resident process made the absolute bandwidth lower than an isolated run):

| tokens / distinct experts | Triton | native CUDA | Triton/native |
|---|---:|---:|---:|
| 1 / 6 | 0.774 ms | 0.847 ms | 1.09x |
| 6 / 30 | 5.453 ms | 9.365 ms | 1.72x |
| 8 / 32 | 5.613 ms | 9.594 ms | 1.71x |

The T=6 relative output error against Triton was `7.73e-7` (`2.29e-6` at T=8), and graph replay
passed. A rows-per-warp sweep over 1, 2, 4, 8 and 16 made one row the least-slow arm; none won.
Triton's tensor-core tile reuses each activation across many output rows, while the CUDA-core
row kernel saves padded-pair math but pays scalar reductions and repeated activation traffic.
These eager timings also include host launch
gaps and different router implementations, so they do not isolate the source of the difference.

A later source-level pass made power-of-two index arithmetic explicit, hoisted weight/scale row
bases, and moved the up kernel's `pair / topk` out of the dot-product loop into packed router
metadata. With the competing worker stopped, two fresh runs of 100 order-balanced measurements
showed a real crossover rather than a general win:

| tokens / distinct experts | Triton range | native CUDA range | result |
|---|---:|---:|---:|
| 1 / 6 | 0.801-0.803 ms | 0.745-0.746 ms | CUDA 1.074-1.077x faster |
| 6 / 30 | 3.028-3.034 ms | 3.775-3.960 ms | CUDA 1.245-1.305x slower |
| 8 / 32 | 3.245-3.300 ms | 4.418-4.661 ms | CUDA 1.339-1.436x slower |

The output errors remained zero at T=1, `7.73e-7` at T=6, and `2.29e-6` at T=8. This is enough
to retain the CUDA arm for single-token investigation, but not to enable it globally: decode call
size varies, and the larger calls still lose. At that stage `DSV41_FP4_CUDA=0` remained the default;
the subsequent CUDA-core tuning below supersedes that decision.

### CUDA-core tuning with graph replay (2026-09-22)

The next review moved the routed-pair loop outside K, keeping only one pair's accumulators and
activation base pointer live. The previous version kept sixteen pairs' state and checked sixteen
validity branches per K tile, despite the usual 1-2 pairs per expert. The new version rereads weights
for each pair, but the smaller register footprint wins on these sparse decode routes. Activation
loads are now one aligned 64-bit load per lane instead of four scalar BF16 loads. The ordered
four-subgroup reduction drops its redundant lane-0 broadcast. These changes preserve the old CUDA
arithmetic order. Up register usage initially fell from 64 to 40 with the pair-loop change.

Use graph replay to exclude host launch gaps. The earlier 7.5% T=1 eager win was **not** proof
that source-level shifts/hoisting accelerated the kernel: the pre-tuning CUDA version was slower
than Triton at T=1 under graph replay (0.725 vs 0.697 ms in the first four-arm comparison).

Two 100-repetition, alternating-order graph runs on GB10 with the serving worker stopped, real
layer-0 weights in a 32-expert arena, top-k=6, FP32 routed output:

| tokens / distinct experts | Triton (ms) | previous CUDA (ms) | tuned CUDA (ms) | relaxed CUDA (ms) |
|---|---:|---:|---:|---:|
| 1 / 6 | 0.697-0.702 | 0.725-0.726 | 0.556-0.559 | 0.534-0.537 |
| 6 / 30 | 2.875-2.890 | 3.679-3.763 | 2.799-2.803 | 2.685-2.727 |
| 8 / 32 | 3.108-3.150 | 4.659-4.866 | 3.099-3.126 | 2.896-2.967 |

The tuned original-order mode was bit-identical to the previous CUDA output in all three cases. The
`DSV41_FP4_CUDA_RELAXED=1` combines even/odd partials before subgroup reduction and accumulates
each subgroup across K before the final warp reduction. It retains FP32 accumulators and the BF16
linear-output boundaries, but changes summation order: relative errors against Triton were
`4.59e-8`, `2.54e-5`, and `2.17e-4` respectively. These are kernel errors, not language-quality
measurements. Following these results, `DSV41_FP4_CUDA=1` and `DSV41_FP4_CUDA_RELAXED=1` were
selected as defaults at the user's request. Explicit `DSV41_FP4_CUDA=0` restores Triton;
`DSV41_FP4_CUDA_RELAXED=0` retains native CUDA with the original summation order. Prefill and
unsupported shapes still use Triton. Full-engine timing and generation-quality gates remain
outstanding; changing the defaults does not expand the measured validation scope.
The relaxed flag and CUDA source digest are included in the EP2 boot-time agreement guard.

Rejected candidates, compared against the vector-load, eight-warp version in graph replay:

* Whole activation-row staging into 10 KiB shared memory per CTA: 0.606/3.034/3.541 ms versus
  0.560/2.783/3.110 ms at T=1/6/8 (8-14% slower), with identical results.
* Four, sixteen, and thirty-two warps per CTA: no consistent improvement over eight; T=8 was
  3.225/3.265/3.494 ms versus about 3.10-3.14 ms with eight.

Six GPU regression tests cover top-k=1/2/3/4/6, up to sixteen pairs on one expert, null experts,
partial output and scatter layouts, output-row tails, graph replay, and long router runs. Native
routing now splits runs longer than sixteen into additional blocks instead of dropping pairs.
Both reduction modes pass. Compute Sanitizer memcheck on the default mode reports zero errors.
No two-rank serving run or full-model generation quality evaluation was performed for this tuning.

Reproduce current arms with `MODEL_DIR=/path/to/checkpoint python tools/bench_fp4_moe_cuda.py
--graph --compare-relaxed --iters 100`. `--baseline-source PATH` adds a saved ABI-compatible CUDA
source as the fourth arm; the previous source SHA256 was
`1f33165b6759be39368eb77605a3ff9013445c22daf730037fc3a54e55d7b0c0`.

### Casting consolidation and predecoded weights: negative results (2026-09-22)

The follow-up tested whether fewer conversions, two-pair reuse, or extra weight memory help the
**already tuned relaxed CUDA default**. Each row below is a separate 100-repetition alternating
CUDA-graph A/B run on GB10, with the serving worker stopped, real layer-0 weights in a 32-expert
arena, top-k=6, and FP32 routed output. Entries are baseline -> candidate milliseconds; compare
within each entry, not across rows (clocks drift). No candidate was promoted.

| candidate | T=1 / 6 experts | T=6 / 30 experts | T=8 / 32 experts |
|---|---:|---:|---:|
| Pair FP32->FP16 activation conversions with half2 | 0.535 -> 0.536 | 2.683 -> 2.685 | 2.913 -> 2.886 |
| Convert each activation tensor once before up/down | 0.534 -> 0.542 | 2.720 -> 2.729 | 2.917 -> 2.917 |
| Share weight decode across two routed pairs | 0.536 -> 0.567 | 2.712 -> 2.852 | 2.894 -> 3.108 |
| Combine four FP4 values' unpacking in one asm block | 0.532 -> 0.532 | 2.722 -> 2.731 | 2.899 -> 2.934 |
| Predecode FP4 codes to FP16; retain group scales | 0.536 -> 1.878 | 2.762 -> 11.383 | 2.879 -> 13.969 |
| Predecode and fold group scales into FP16 weights | 0.535 -> 1.822 | 2.819 -> 11.018 | 3.051 -> 14.082 |

Every candidate was bit-identical to its CUDA baseline on these three real-weight inputs. Relative
errors against Triton remained 4.59e-8 / 2.54e-5 / 2.17e-4; this is not a general guarantee for
prescaling other weights or a language-quality evaluation. All candidates keep FP32 accumulation.

The packed conversion changes show no consistent useful win. A second 100-repetition packed
activation run gave 0.537 -> 0.534 / 2.719 -> 2.691 / 2.894 -> 2.904 ms: the roughly 1% advantage
moved between workloads, and T=8 changed from winning to losing. A second precasting run gave
0.534 -> 0.543 / 2.695 -> 2.701 / 2.882 -> 2.884 ms, confirming no win. Precasting removes repeated
BF16->FP16 conversion from the row loops but adds two conversion launches and scratch buffers;
those costs are included in its times. Two-pair reuse raises up-kernel registers from 40 to 54
(no spills) and loses 5-7% on these sparse routes. It might behave differently on denser routing;
these results do not prove every possible two-pair implementation loses.

FP16 weight expansion grows just the weight payload from 566,231,040 to 2,264,924,160 bytes for
32 experts, excluding scales and the original FP4 copy retained for A/B. Expansion is outside the
timed region, making these optimistic steady-state cache numbers. Even eliminating scale loads
and multiplies does not offset the extra weight traffic: these variants lose 3.4-4.9x. Freeing memory
by reducing context length therefore does not make this particular FP16 CUDA-core cache faster.
This is not a measurement of a different tensor-core GEMM implementation.

The serving CUDA source was restored byte-for-byte (SHA256
`f28eb41f84b0e7b607335ca7837055a0850d9972116da1074e0410089875eede`), keeping both CUDA defaults
enabled. All six GPU regression tests pass in both reduction modes after restoration. No engine settings, serving
allocations, or quality assumptions changed.

Reproduce with the worker stopped and `MODEL_DIR` pointing at the checkpoint:

```sh
python tools/bench_fp4_casting.py packed --iters 100
python tools/bench_fp4_casting.py precast --iters 100
python tools/bench_fp4_casting.py pair2 --iters 100
python tools/bench_fp4_casting.py decode4 --iters 100
python tools/bench_fp4_predecoded.py --iters 100
python tools/bench_fp4_predecoded.py --iters 100 --scaled
```

The benchmark-only patches under `tools/experiments/` are applied to temporary sources with
zero patch fuzz. They are not production switches; patch failures after kernel changes should be
reviewed rather than silently adapting a stale experiment.

### Aggressive half2 arithmetic: no useful speed/accuracy tradeoff (2026-09-22)

The next experiment changes the arithmetic, not just the casts. `half2_group` keeps decoded
FP4 values and rounded activations in packed FP16, uses `__hmul2`/`__hfma2` for the local
products, and performs the three width-8 shuffle/add stages in packed FP16. Only then are
the even/odd partials widened to FP32, combined, scaled, and accumulated across K. No
unscaled partial crosses a 32-element scale boundary. The less aggressive `half2_local`
widens immediately after the local two-term packed sums and retains FP32 subgroup reduction.
Both are benchmark-only patches; neither changes the serving source or boot configuration.

Two 100-repetition alternating graph runs of `half2_group`, real layer-0 weights, top-k=6,
32-expert arena, stopped serving worker, relaxed baseline, FP32 output:

| tokens / distinct experts | baseline -> candidate, run 1 (ms) | run 2 (ms) | candidate relative L2 error vs Triton |
|---|---:|---:|---:|
| 1 / 6 | 0.5361 -> 0.5329 | 0.5362 -> 0.5342 | 0.002738 |
| 6 / 30 | 2.7086 -> 2.7160 | 2.7028 -> 2.7134 | 0.002511 |
| 8 / 32 | 2.9155 -> 2.9156 | 2.8880 -> 2.8836 | 0.002660 |

That is within about +/-0.6% of baseline, with substantially greater numerical error. The
baseline errors remain 4.59e-8 / 2.54e-5 / 2.17e-4. The CUDA 13.0 sm_121a disassembly confirms
actual HMUL2/HFMA2 and packed reduction instructions; up registers fall from 40 to 38 with
no spills. Fewer instructions alone do not establish a speedup on this workload.

`half2_local`'s first 100-repetition run gave 0.5353 -> 0.5340 / 2.7427 -> 2.7101 /
2.9165 -> 2.8721 ms, with errors 0.002147 / 0.001805 / 0.001867. These 0.3-1.5% improvements
are small and all three errors still exceed the existing 1e-3 kernel gate. This is not a
language-quality measurement, and the gate was not loosened to promote either candidate.

The aggressive candidate also fails the synthetic parity checks (seven routing/shape subcases,
relative error 0.0030-0.0033) and the mixed-null parity assertion (0.00318). Graph replay,
down scatter/tail layout, and both routing tests pass. These accuracy failures are distinct
from the following deliberate range stress, not evidence of a routing bug.

The range stress uses a down projection with K=128, all FP4 weights +6, scales 1/128, and
constant BF16 activations. At activation 1024 the exact and baseline result is 6144, while
`half2_group` produces infinity before the scale can bring it back into range. `half2_local`
remains finite there, but overflows at activation 8192 (baseline 49152). These deliberately
large inputs are not a measured serving activation distribution; they demonstrate lost
range, not observed full-model collapse. Neither candidate has a full-model quality test.

Reproduce without changing the serving kernel:

```sh
MODEL_DIR=/path/to/checkpoint python tools/bench_fp4_casting.py half2_group --iters 100 --stress
MODEL_DIR=/path/to/checkpoint python tools/bench_fp4_casting.py half2_local --iters 100 --stress
```

For these approximate candidates the benchmark reports `kernel_error_gate_passed=False` but
continues through all workloads. Its process exit status is **not** an accuracy gate. The
production CUDA source retains the SHA256 recorded above, and its six GPU tests still pass.

### Batching the sampled verifier can waste work after early rejection

The opt-in `DSV41_BATCHED_VERIFY=1` prototype removes per-proposal scalar readbacks and
batches probability calculation. At three drafts/top_p=0.95, synthetic mixed-acceptance
sampler latency improved 0.855 -> 0.719 ms, but first-rejection latency regressed
0.344 -> 0.720 ms because unused target rows were sorted. The absolute saving is small,
seeded outputs change, and no end-to-end speedup is established. It stays default-off;
see [sampling tests and measurements](spec-sampling.md) before enabling or extending it.

### Mapped host weights avoid a copy but can slow repeated reads

GB10's shared physical DRAM does not make the current pinned staging buffer and CUDA arena the
same allocation. `ExpertStore._read_leased` reads SSD data into pinned host memory, then
`_load_into_slot` copies it into the arena. NVIDIA also documents that ordinary `cudaMalloc`
allocations cannot be coherently accessed by the CPU/I/O complex on Spark; see the
[Spark CUDA porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html).
A zero-copy design needs an appropriate mapped allocation and ownership/synchronization protocol.

`tools/bench_fp4_mapped.py` compares the same native kernel with device weights and pinned CPU
weights obtained through `cudaHostGetDevicePointer`. A 60-repetition alternating graph run,
32 real experts, original-order tuned reduction (`DSV41_FP4_CUDA_RELAXED=0`), gave:

| tokens / distinct experts | device weights | mapped weights |
|---|---:|---:|
| 1 / 6 | 0.565 ms | 0.588 ms |
| 6 / 30 | 2.871 ms | 3.493 ms |
| 8 / 32 | 3.198 ms | 4.434 ms |

Outputs were bit-identical, but mapped reads were 4%, 22%, and 39% slower. This excludes the
initial copy and SSD I/O: it rejects assuming mapped memory is free for a hot resident arena,
not the possibility of saving latency on one-use cold experts. Measure load-plus-compute and
buffer lifetime before adopting it. The serving expert allocator/loader was not changed.

## `persistent_topk` does not fit in GB10's shared memory

This one is inherited, and it is the reason no upstream engine runs this model on one
Spark. The sparse-attention indexer's `persistent_topk` kernel wants 128 KB of shared memory
per block; GB10 has 99 KB. The four-Spark vLLM build works around it with
`top_k_per_row_decode` instead, plus block size 64/128 for the sparse SWA and indexer caches.
Anyone porting a datacenter kernel to this box should expect to meet the same wall — and
`DeepSelect`, DeepSeek's own top-k kernel, is `sm_100a`/`sm_103a` only.

## Triton has to be able to compile for `sm_121a`

The MoE kernel is JIT-compiled on the first call and emits inline PTX
(`cvt.rn.f16x2.e2m1x2`), so `ptxas` inside the container must know the target. Triton wheels
bundle their own `ptxas`, and that copy has shipped behind the driver before — the CUDA-12.8
one in the triton 3.5 wheels could not name `sm_121` at all. The image is therefore built on
`nvidia/cuda:13.0.2-devel-ubuntu24.04` with `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.
If a kernel launch dies with something about `gpu-name` or an unknown target, that variable
is the first thing to check.

If Triton cannot compile at all the engine falls back to a dequantise-then-GEMM path — it
is correct and very slow, and it will not announce itself except in `engine_config.kernel`
on `/health`. Check that field before believing a slow row.

## The warm start is not optional (but it is silent)

Without a `coverage.json` the arena is filled in `(layer, expert)` index order, which is a
measurably worse hot set than the traced one, and the only sign is a lower `expert_hit_rate`
in the benchmark. The container's entrypoint auto-discovers the newest
`results/trace-*/stats/coverage.json` and logs which one it used, or says it is falling back
to index order. Read that line.

## First token can be minutes, and startup can be tens of minutes

The socket is bound only after the warm start has read tens of GB off NVMe (measured:
~63 s for the 19 GB of non-expert weights, then 73.2 GB of experts in 16 s), and a cold
prompt misses almost every expert. Health waits are 20 minutes native, 45 in the container,
and any client timeout has to be set accordingly. Nothing is wrong at minute 6.

## Chunk counts lie under speculative decoding

Several accepted tokens arrive per SSE chunk. Use `usage.completion_tokens`; counting chunks
under-reports decode speed by 3–5×. See [benchmarking](benchmarking.md).

## "The GPU is idle" is usually the accounting, and then it is usually the engram

Two separate traps, hit in that order while chasing a decode that showed 70–90% GPU utilization.

**First, `decode_accounting` measured the wrong path.** Its `attn`/`moe` timers live in the eager
forward; with CUDA graphs on, decode replays a captured graph and those lines never execute, so
every number in that dict is the *prefill* figure and the whole decode step lands in
`unaccounted` — which read 46–75% and looks exactly like an idle GPU. Fixed, but the lesson
generalizes: a host wall-clock timer around a graph replay measures queueing, not work. The
numbers that can be trusted are `gpu_timing` (`DSV41_GPU_TIMING=1`, CUDA events on the GPU
timeline) and the `[step timing]` table (`DSV41_STEP_TIMING=1`).

**Then the engram read really was the gap.** `_gather_rows` sized its thread tasks
`max(1024, len(ids) // nt + 1)`. A decode verify block asks for `T_VERIFY × 24 ≈ 144` unique
rows, and 144 ≤ 1024, so every decode gather ran as ONE task — serially, one synchronous page
fault at a time, 83 ms of a 196 ms step. The rows were always correct, so nothing ever pointed at
it. Sizing tasks from the row count took the same gather from 93–105 ms to 4.5–6.0 ms.

What did NOT help, both measured rather than reasoned about:

* **Dropping the EP row split at decode.** Halving 144 rows is worth about a millisecond and
  costs a collective; removing it moved the step time not at all. (Kept anyway — it is size-gated
  now, and at prefill volumes the split is worth ~100 ms.)
* **Pinned staging for the H2D.** It works and it cuts the engram wait 58 → 11 ms/step, and the
  decode rate does not move: the host stops blocking in `step` and starts blocking in `verify`
  instead. Its original note said exactly this, and it is still true for a new reason — once the
  gather is fixed, the GPU is genuinely the bottleneck (`layers_ms_per_step` ≈ 150 of a ~167 ms
  step). Leave `DSV41_ENGRAM_PINNED` off unless something changes upstream of it.

The residual utilization dip is therefore real but small. Anything further has to come from the
GPU side: fewer bytes per step, or more accepted tokens per step (`accept_len` is ~2.5 of 6).

## Prefill halved when the fused attention kernel was defaulted off

### Live-profile regression after the nesting investigation (2026-09-14)

The local serving `.env` still had `DSV41_PREFILL_FUSED_ATTN=0`, despite the engine and
`env.example` defaulting to 1. Restoring it to 1 retains the short (<=128-token) reference-shaped
prefill branch and the corrected expert rounding; it is not enabling fused **decode** attention.

Replayed an existing local 14,393-token model-ready capture with EP2, native FP4,
keep=0.59, spec ON, demand logging and adaptation ON. Prompt/completion content was not printed
or committed. GPU phase timings were enabled. Every full-prefill row reused **zero** prefix tokens:

| configuration | prefill seconds | prefill tok/s | decode tok/s |
|---|---:|---:|---:|
| fused prefill OFF | 46.721 | 308.06 | not meaningful: saved request capped output at 2 tokens |
| fused prefill ON, first replay | 23.380 | 615.61 | 14.92 (239 tokens, acceptance 3.16) |
| fused prefill ON, subsequent replay after prefix eviction | 24.909 | 577.83 | 17.15 (221 tokens, acceptance 3.17) |

The unfused softmax phase alone cost 14.587 s, versus 1.210 s with fusion. EP-combine event
time fell from 14.697 to 5.159 s, then varied to 8.238 s; this includes waiting for the peer,
not just network transfer. Adaptive expert generations changed, and completion lengths differed:
these are live-profile observations, not an isolated throughput guarantee. The previously cited
warm 4.5k synthetic-code test with demand logging OFF understated the user's experienced loss.
An unfused cached-prefix decode-only replay produced 181 tokens at 14.82 tok/s, acceptance 2.97;
its cached prefill rate is deliberately not reported as prefill throughput.

Normal-profile nesting after restoration passed 4/4. `tools/test_prefix_invariance.py --tries 1`
was **inconclusive**: adaptation moved expert generation 8 -> 12 during the comparison. This is
not a new prefix-parity pass or a demonstrated parity failure. The historical 800–1000 tok/s
prefill range had not yet been reproduced at this point. Later warmed runs below reached it,
but do not establish consistent speed recovery.

Follow-up: the original 14,393-token capture reached 907.64 tok/s after restoring unrestricted
CPU affinity. A P-core-only trial was not retained: adaptation changed between runs, so the
668.29 -> 689.48 -> 907.64 sequence does not isolate affinity. A new real 14,396-token capture
replayed twice at 588.57 and 862.06 tok/s (24.459 and 16.700 s), both with zero prefix reuse and
2048-token chunks. EP-combine GPU envelopes were 6.993 and 0.516 s; MoE envelopes were 7.352
and 6.184 s. Adaptation remained enabled (expert generations 3 and 5). These are eight equal
chunk boundaries in both runs, not evidence that chunk dispatch costs are equal.

`DSV41_PREFILL_TIMING=1` records per-chunk host-call duration, host gaps, GPU stream envelopes
and GPU gaps on **both** ranks; JSON logs contain positions/timings only, no token IDs or text.
Events are resolved after generation, never synchronized at chunk boundaries. Host duration
includes existing blocking operations; GPU envelopes include idle/collective waits, not just
kernel execution. Do not subtract host and GPU durations as "launch overhead", or subtract
timestamps across ranks. With `DSV41_ATTN_TIMING=1`, both ranks also log their aggregate phases.
These diagnostics locate stalls; a CPU/GPU timeline may still be needed to prove dispatch starvation.

Instrumented EP2 replay of the 14,396-token capture (32 output tokens, zero prefix reuse) measured
667.61 then 762.84 prefill tok/s (21.564 / 18.872 s). Both ranks recorded eight chunks. Host
gaps between calls were 8–22 microseconds; CUDA-stream gaps were 1–4 microseconds. Thus an
expensive pause *between* chunks is not supported by these runs. Dispatch stalls *inside* a
chunk are not ruled out. First-chunk GPU envelopes were ~5.3 then ~3.7 s; most later full chunks
were 2.2–3.0 s. Rank-0/rank-1 aggregate combine envelopes were 2.316/3.633 s in run one and
3.014/2.123 s in run two: the greater waiter switched ranks. The first replay also includes
post-restart warm-up effects; do not treat the difference as an isolated code speedup.

Doubling chunks to 4096 (with RING=8192 to preserve history) was not retained. Same real input,
zero prefix reuse, normal adaptive EP2, 32-token output cap: first new-shape run 61.153 s
(235.41 tok/s), then warmed runs 21.473 and 20.969 s (670.42 / 686.55 tok/s). Four chunks were
confirmed. New-shape first/last chunks dominated the cold run; do not use it as steady-state
throughput. Warm performance did not beat the preceding 2048-token profiles (667.61 / 762.84),
and host MemAvailable fell to ~2.8 GB during the trial. Nesting passed 4/4, but those short
probes do not prove long-context chunk-size parity. Adaptive generations changed, so this is
not an isolated causal comparison. Restore chunk=2048, ring=4096 rather than retain a larger
working set without a demonstrated benefit. The public default remains unchanged.

HC final-output-store cleanup: keep the FP32 normalization scratch and reduction order, but
store the final result directly to BF16 for T>=512 instead of an FP32 store plus torch cast.
`python -m engine.test_hc_output_store` compares against the old store/cast path: bit-identical
at T=1/6/60/64/128/512/2048/2108. Two alternating timing sweeps measured T=2048 at
0.850/0.844 ms old versus 0.674/0.675 ms direct, T=512 at 0.179/0.192 versus 0.157/0.181 ms.
Smaller batches retain the old default (T=128 showed no consistent gain). This is only an
estimated ~0.095 s across 560 full-chunk calls on a 14k-token prompt, not a measured end-to-end
speedup. It is a minor cleanup, not an explanation or solution for the multi-second regression;
do not prioritize more restarts/tuning around this sub-millisecond site.

Rejected global schedule change: `tools/tune_fp4_prefill.py --confirm` alternated software-FP4
down-projection tuples (128,8,3) and (128,4,2) using fixed serving-style routing. FP32 outputs
were bit-identical. Candidate improved T=512 (~22.1–22.5 vs 23.4–23.8 ms), but not consistently
T=2048 (~33.3–33.9 vs 32.8–34.2 ms); retain the current default. Wider up tiles (BN=128)
were substantially slower in the initial sweep. Do not infer whole-engine gains from that sweep.

Capture remains the existing `DSV41_CAPTURE_NEXT` mechanism in `server/app.py`, now taking a
count: it captures the next N requests and disarms itself (`captured_request.pt` for N=1, else
`captured_request_0.pt`, `_1.pt`, ...), so two requests that should share a prefix can be diffed
token-for-token without a second capture mechanism. `bench/replay_capture.py` still replays
the N=1 filename, overrides the old diagnostic output cap, and writes only timings, usage, and
content hashes to a mode-0600 result file. It rejects vision/grammar captures rather than
silently replaying a different workload.

Upstream inspection at `45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f` found that the headline
24–36 decode tok/s rows used CB3 experts, `DENSE_FP4=attn,wo_a`, an FP8 head, and mostly high-
acceptance code. Its newer core changes focus on contribution-based expert ranking and routing
modes, not a replacement fast-decode kernel. Those settings are not a like-for-like native-FP4,
BF16-dense EP2 comparison. No merge or quantization rollback was performed.
See [upstream measurements](https://github.com/0xBakeer/deepseek-v41-flash-spark/blob/45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f/RESULTS.md#43-shipped-configuration).

### Earlier investigation

`DSV41_PREFILL_FUSED_ATTN` went from `"1"` to `"0"` in `692d5c1`, three minutes after `8ccf9a3`
landed the prefix cache. Neither commit has a message. Prefill went 1045–1135 → 544 tok/s on the
same 4,513-token prompt, and `softmax_attn` went back to dominating attention, which is ~73% of
prefill.

The apparent reasoning: the fused kernel rounds differently from the torch path (~1e-2 relative,
bf16 level), and a cache that resumes a prefill has to agree with itself. But that compares the
wrong two things. Nothing requires the fused path to match the torch path. What must match is a
**resumed prefill against a cold one under the same kernel** — and the fused kernel is internally
chunk-invariant: each query row is its own program and the key axis is reduced in fixed `BLOCK_N`
tiles, neither of which depends on the chunk length. (`split` must stay 1 at prefill; it is
derived from available parallelism, which *does* depend on T.)

`tools/test_prefix_invariance.py` is the test that was missing. It resumes at 2,259 tokens —
deliberately not a multiple of the 2,048 chunk size, so the two runs tile the same tokens
differently — and asserts both halves: that the output is identical, **and** that the cache was
actually used (`prefix_cached_tokens == 2259`). Equality alone proves nothing, since a cache that
silently stopped working leaves both runs cold and agreeing.

Back on by default: prefill 544 → 1047 tok/s, `generation_gate.py` 6/6, cache used and
byte-identical. Note that `DSV41_PREFILL_ATTN_BLOCK_H`/`_BLOCK_N` in `.env` were tuned in
`df74991` while this path was switched off, so those values measured nothing — they are read per
call at the `_prefill_attn` call site and are worth re-tuning now that the path is live.

## A wider verify block makes utilization better and decode slower

`DSV41_BLOCK=9` (T_VERIFY=10 instead of 6) was tried because the dense weights are read once per
step however many token rows ride along — measured, at the shapes decode uses:

```
wq_b  M=6: 0.193 ms   M=16: 0.200 ms      MoE  T=6, 30 experts: 2.824 ms
w1    M=6: 0.044      M=16: 0.044              T=64, 32 experts: 3.389 ms
```

So rows look free, and GPU utilization does visibly improve with the wider block. Decode got **27%
slower**:

```
              step      accept   decode
T=6           162.7 ms   2.95    18.10 tok/s
T=10          217.4 ms   2.87    13.24 tok/s
```

Two reasons. **Rows are free for dense weights but not for routed experts** — more tokens route to
more *distinct* experts, and that read grows nearly linearly with T until it saturates:

```
T=6    21.13 distinct experts/layer    7.94 GB/step
T=10   29.85 distinct experts/layer   11.23 GB/step
```

Total bytes 14.86 → 18.1 GB. And **acceptance did not move** (2.95 → 2.87): the DSpark head is
trained at `dspark_block_size=5`, so drafts 6–9 are outside its horizon and are essentially never
accepted. Four more candidate positions, paid for, none collected.

This is the clearest case in the repo of utilization and throughput pointing opposite ways. Fuller
tiles are not the goal; tokens per byte read is.

Note also that the documented `DSV41_BLOCK` range (odd, 1–15) is wrong. The graph-capturable
routing path (`build_routing_small`) handles P ≤ 64 pairs and P = T_VERIFY × 6, so T_VERIFY ≤ 10
caps it at 9; above that the MoE falls into a `torch.unique` path that cannot be captured at all.

## Low decode utilization is mostly skinny work, but EP null routing made it worse

The GPU timeline on the live FP4/EP2 configuration is nearly full: 140.9 ms of backbone plus
14.7 ms of DSpark drafting in a 162.6 ms decode loop. A 70–90% utilization reading therefore does
not imply 10–30% host-idle time. The work consists mostly of small-row dense projections and FP4
routed-expert kernels, interleaved with one collective per layer; occupancy and arithmetic-unit
utilization are low even while the device is continuously executing.

There was still one exact EP2 loss. Every non-owned route mapped to a shared zero expert, so about
half of a 36-pair verify block collided on one arena slot. The safe implementation raised the MoE
tile from the single-device-tuned BM=16 to BM=64 for *all* experts. Passing the known null slot and
giving each null pair a temporary unique routing key lets the kernels skip those pairs and restores
BM=16 for the real experts. On the focused GB10 kernel test the call moved 1.58 → 1.03 ms; at full
model scale, with essentially matched demand (~21.8 distinct experts/layer), backbone time moved
from roughly 150 to 140.9 ms/step. The mathematical result is unchanged; the EP2 oracle test keeps
the same 0.16% FP4-kernel error against dequantized GEMM.

After that change only about 7 ms/step is outside the GPU timeline. The next single-request gains
must reduce executed bytes/work or improve DSpark acceptance. Dense TP and a wider verify block
have both been measured in the wrong direction; top-k 5 is faster but exceeds the prose loss gate,
and the BF16 LM head remains intentionally unquantized.

## The demand half-life has to be read against your traffic, not as a constant

`DSV41_PRUNE_HALFLIFE` is the EWMA half-life of the expert-demand history, in *routing slots*, and
the decay applied is `0.5 ** (grown / half)`. The number only means something next to how many
slots a request actually produces: a 4.5k-token prompt here generates **~588,000**.

`.env` carried `2e6` against a code default of `2e7`, i.e. a factor of ten more aggressive. At
`2e6` each request decays the history to `0.5 ** (588000/2e6) = 0.815` — the whole history
half-lives every ~3.4 requests. The ranking genuinely moved that much, the planner dutifully
swapped it, and the result was ~100 experts moved per request (near the `DSV41_PRUNE_SWAP_MAX=128`
cap) at a **0.9% miss rate**, costing ~1 GB of NVMe and ~0.2 s every request to fix nothing.

Restoring `2e7` (0.98 per request, stable over ~34) on the same workload:

```
                swaps/request                        routed-miss
2e6    89, 116, 96, 92, 72, 71                       2.3 → 0.9%
2e7    38, 50, 44, 43, 41, 41, 34 … 21, 18, 20, 16   0.9 → 0.4%
```

Churn fell ~5x **and** the miss rate halved. That direction is the useful part: a steadier history
is not a staler one here, it is a more accurate one — the twitchy version was chasing each prompt
in turn and never settling on the set that serves all of them.

The symptom to watch for is the swap count sitting near `DSV41_PRUNE_SWAP_MAX` while the miss rate
is already low. That combination always means the ranking is churning, never that there is real
work to do; reach for the half-life before `DSV41_PRUNE_SWAP_MIN_GAIN`, because the gain threshold
only suppresses the symptom.

## Structural corruption: test expert arithmetic, not just engine switches

**Correction to 01ed3e2 (2026-09-14):** the earlier conclusion below that the checkpoint's
MXFP4 scale grid caused nesting corruption was not established by those tests. They all kept
the custom expert arithmetic. Dequantizing the *same* packed expert weights to BF16 and using
BF16-output matmuls passed all four nesting probes (1.000 versus 0.750 for the custom kernel
in the single-box handoff). That fallback is diagnostic only: host-side `unique().tolist()`
makes it uncapturable in CUDA graphs, and dequantizing each expert per call is much slower.

The fused kernel omitted the BF16 output boundary of the gate/up linear before FP32
clamp/SwiGLU. It also omitted each down projection's BF16 output boundary; conversely the
EP2 path rounded the routed aggregate *before* adding the shared expert. These are engine
differences, not missing expert weights or a different checkpoint. Preserve FP32 accumulation,
round each linear where the reference rounds, then sum routed and shared results in FP32
before the final BF16 cast. "More precision" at a different boundary is not reference parity.

The handoff's gate/up correction made software decode (`DOT_SCALED=0`) pass 1.000, while
`DOT_SCALED=1` remained at 0.750. Do not interpret a pass at one greedy rounding boundary as a
quality guarantee. The corrected standalone test disables reduced-precision BF16 reduction,
as the engine already does: at T=512, relative error is 2.88e-4 (software) / 1.40e-4 (scaled),
not the previously reported ~4.5e-3. Both pass a 1e-3 tolerance; the scaled path having *lower*
random-tensor error did not predict its nesting score.

Fresh-process margin check, full experts (`PRUNE_KEEP=1`, no adaptive pruning, speculation
off), same canonical spaced depth-8 answer, one-token decode after prefill:

| expert arithmetic | final `}}}}` logit | competing `}}` logit | target-minus-competitor |
|---|---:|---:|---:|
| software decode + BF16 boundaries | 32.00 | 31.75 | +0.25 |
| dot_scaled + BF16 boundaries | 32.00 | 32.25 | -0.25 |
| BF16-dequant control | 33.00 | 32.75 | +0.25 |

The control itself is borderline. This verifies a rounding-sensitive decision, **not** a
comfortable-margin or general-quality fix. `MARGIN=1 SPEC=0 PRUNE_KEEP=1` with
`tools/nesting_arm.py` reproduces the check; run each arithmetic arm in a fresh process because
activation-quantization monkeypatching is process-global. The canonical target is not the only
valid tokenization; use actual generation alongside margins when evaluating a candidate.

Rejected speed-preserving attempt: resetting dot_scaled's accumulator every 128 K and adding
the partials outside MMA. Depth-8 generation passed via an earlier switch to *compact* JSON,
but the same spaced-prefix closing margin worsened to -1.0; T=512 MoE time rose from 8.8 to
10.6 ms. This is exactly why a pass alone is insufficient. Keep the original scaled kernel as
an opt-in arm; the serving profile selects software decode, without claiming that its +0.25
margin is robust. Initial layer-0/32-expert microbenchmarks measured T=512 at 15.1 ms software
versus 8.8 ms scaled; these are kernel timings, not whole-engine prefill throughput.

Second rejected speed-preserving attempt: explicit scaled **prefill** plus software **decode**
(including prefill dispatch during decoder replay). Full-expert single-node depth-8 generation
failed with an extra closing brace; the forced closing margin worsened to **-0.75**
(target 31.50, competitor 32.25). The experimental dispatch was removed, not enabled in serving.

Matched EP2 timing, both with corrected rounding and cycle breaker OFF: arena 88 GB,
keep=0.59, spec ON, MAX_SEQ=262144, adaptation/demand logging OFF. Three deterministic
cache-busted 4514-token prompts, 64-token completions per arm; all prompt hashes and generated
texts matched and every run reported zero prefix-cached tokens. Warm runs (1 and 2):

| arithmetic | prefill tok/s, run 1 / 2 | decode tok/s, run 1 / 2 |
|---|---:|---:|
| software (passing configuration) | 348.52 / 491.77 | 30.99 / 30.87 |
| scaled (comparison arm) | 431.07 / 520.37 | 31.54 / 31.77 |

Software was 19.2% / 5.5% slower in warm prefill and 1.7% / 2.8% slower in decode.
Run 0 includes cold/JIT/cache effects (software prefill 164.64, scaled 289.15 tok/s) and is
not a steady-state comparison. Warm results also vary substantially: do not claim a universal
percentage or "no speed regression." This compares two arithmetic settings of the corrected
engine, not a matched historical pre-fix build. Command: `bench/prefill_ab.py OUT --runs 3
--cache-bust --max-tokens 64`. No full quality benchmark was run at keep=1.

Final normal-profile deployment (EP2, keep=0.59, spec ON, arena 88 GB, adaptive settings
restored) passed the arithmetic floor smoke test: 10/10, 503 tokens, 26.0 decode tok/s.
`generation_gate.py --only arithmetic --max-tokens 600` checks basic operation, not general
quality; the full-expert nesting result above is the separate quality evidence.

Single-node regression checks with software decode and these rounding boundaries:
`tools/test_fp4_moe.py` passes at 1e-3 for both BF16 and FP32 routed output, with bit-identical
repeated runs and prefixes at call sizes 1, 3, 6, 7, 17, 64, 148, 256, and 300.
`engine/test_fastdecode.py` passes exact equality for logits, hidden state, and drafts at both
tested parities (keep=0.25, transient slots=16). `engine/test_spec_lossless.py --max-tokens 64`
passes both prompts: all 64 tokens identical with speculation off versus on, using
`results/trace-union/stats/coverage.json`, keep=0.25, and 16 transient slots. These are bounded
regression checks, not long-output quality certification.

Other retained fixes: HC pre-mix must round to BF16 before RMSNorm, HC post-mix adds the branch
after the residual mix, and short reference-shaped attention applies only to **prefill**.
Applying the compact attention path to decode broke FastDecoder/eager agreement (relative
logit error 0.104, top-1 agreement 0.83); the prefill gate restores the same decode shape.
The removed window-KV FP8 QDQ call had been an identity with `act_quant=False`: it was not
evidence of reference FP8-cache parity. Compressed-KV QDQ is separate and remains graph-safe.

### Historical negative tests (before the expert rounding correction)

**HTTP comparison confound found 2026-09-14:** these served nesting failures also had a
default-on `Penalties` cycle breaker. Direct `eng.generate()` probes did not construct it.
The breaker bans the correct `n` token at answer position 13 in canonical depth-8 JSON.
Thus the historical HTTP failures below do not independently establish kernel/checkpoint
corruption. With corrected software-decode kernels, EP2, full experts, and speculation ON,
changing **only** `DSV41_CYCLE_BREAK=1` to `0` raised the four-probe nesting score from
0.458 (1.000 / 0.833 / 0 / 0) to **1.000 (4/4)**. Both nodes used identical image
`6c5a316bafb4187d86a27471f5c1ca4e69a53ccd0cecae510056f601431e865d`.
This is a bounded quality check, not a guarantee for arbitrary prompts. The breaker is now
opt-in, covered by `engine/test_penalties.py`, and exposed in health and the EP boot guard.

Additional eliminated suspects: the single-node depth-8 run still passed with serving's
MAX_SEQ=262144 and PREFILL_FUSED_ATTN=0 (same +0.25 closing margin); prompt IDs and deployed
engine source hashes matched. `tools/test_fp4_ep_sum.py` with 30 real layer-0 experts measured
full versus parity-split FP32 sums at T=1/6/63/512: maximum differences 0 / 2.98e-8 / 2.38e-7 /
4.77e-7, and identical BF16 results after adding the shared term. This does not prove exact
full-model EP parity, but no extra arithmetic change was needed for the four served probes.

Symptom, found comparing this engine against MiaAI's EXL3 2.9 bpw build of the same checkpoint on a
nesting probe (`llm_benchmark/quality_quant2.py --only nesting`): asked to emit `{"n": ...}` nested
N deep, it instead produces, deterministically,

```
depth=4   {"n": {"n": {"n": {"n": 44}}}}                          ok
depth=8   {"n": {"n": {"n": {"n": {"n": {{"n": {"n": 48}}}}}}}}}   doubled brace
depth=10  {"n": {"n": {... {"n": "n": {...  "nn": 50}}...        dropped quote, merged key
```

Depth 4 is always right, 6 marginal, 8+ broken, and the corruption is character duplication and a
lost quote -- not a semantic substitution. It reproduced on every engine configuration then tried:

| hypothesis | test | result |
|---|---|---|
| checkpoint corrupt | sha256 of all 48 shards vs the Hub's LFS metadata | all match |
| wrong base revision | ours `dba1be0a` vs the quant's source `fb2764a5` | one encoding commit apart; weights unchanged |
| pruning | `PRUNE_KEEP=1.0` (full model, streaming) | still fails |
| speculative decoding | `SPEC=0` | still fails |
| `tl.dot_scaled` MoE | `DSV41_FP4_DOT_SCALED=0` | still fails |
| Decoder SWA Bounded Replay | documented bit-exact for ≤128-token prompts; this prompt is ~50 | not reachable |
| chat template | official `encode_messages(thinking_mode="chat")`; `reasoning_effort` is inert outside thinking mode | standard |
| detokenizer | canonical nesting strings through `IncrementalDetokenizer`, 1- and 6-token bursts | byte-exact |
| prefix cache | cold (evict), warm, cold again | identical corruption |

The same weights under a per-row codebook requant (EXL3 2.9 bpw, vLLM) answer all four depths
exactly. That comparison changes both quantization and runtime; it cannot isolate the scale grid
as the cause. The same-weight BF16-dequant experiment above is the more relevant control.

**Do not generalize it.** The battery's macro is decided by `nesting` + `constraint` +
`char_count`; `counter`, `json_strict`, `verbatim`, `copy_transform` and `escape_json` saturate at
1.00 for both models. Neither a four-probe nesting pass nor teacher-forced loss establishes
whole-model quality parity.

Requantization alternatives investigated at the time (not established fixes for this bug):

* **NVFP4** (E2M1 + E4M3 per 16) is the small change -- only the routed experts differ, and e4m3
  scales carry 3 mantissa bits instead of a power of two. It is **not reachable natively here yet**:
  `tl.dot_scaled`'s shape validator accepts the layout (`is_fp8e4nv()` selects block 16) but the
  sm_121a backend asserts in `TritonGPUAccelerateMatmul` (`type.getElementType().isIntOrIndex()`).
  MXFP4 (e8m0, block 32) lowers fine. So NVFP4 needs a software MoE path (decode ~free, it is
  byte-bound; prefill pays) or a newer Triton.
* **Size runs the other way.** MXFP4 is 18.80 MB/expert (4.25 bits/weight), NVFP4 19.91 MB
  (4.5 b/w, +5.9%), CB3 14.45 MB. At one arena NVFP4 costs ~4% of residency, itself a quality cost
  on a model where residency is the dominant lever. The bet is whether E4M3-per-16 beats
  UE8M0-per-32 by more than 4% fewer experts costs.
* Public "NVFP4" checkpoints are MXFP4→NVFP4 casts of this same checkpoint (`base_model:quantized`),
  so they redistribute the same error; there is no higher-precision source to download.

Keep these rejected alternatives as history, not as a reason to rule out engine arithmetic.

## The dense fp4 and fp8 head buy ~5% decode and cost structure

Prompted by the byte budget: per decode step a rank reads ~16 GB of replicated dense against ~6 GB
of split routed experts, so the dense is the *bigger per-step read*, and the fork keeps it at
checkpoint precision on purpose. Measured anyway, on the served pair (`generation_gate.py` 6/6 in
every arm; ms/step is `1000/tok_s * accept_len`, so it does not move with acceptance):

| arm | dense_fp4 | head | ms/step | nesting (3 reps) | copy_transform (3 reps) |
|---|---|---|---|---|---|
| B | off | bf16 | 180.4 | 0.458 | 1.00 |
| C | off | fp8 | **171.0** | 0.483 | **0.50** |
| A | attn,wo_a | fp8 | 170.9 | **0.278** | — |

Two results, both negative:

* **All of the ~5% is the head, and it costs exact-string fidelity.** C is 5.2% faster and
  nesting-neutral, but `copy_transform` "level radar civic" comes back "lavev radar civic" in 3/3
  reps where B is correct 3/3. The single-box measurement had the fp8 head at +0.0014 nats and
  byte-identical greedy output; under EP2 that margin is gone on a palindrome reversal, which is a
  near-tie on the last characters. The head stays bf16.
* **Attention/`wo_a` fp4 is free of speed and full of cost.** A is not faster than C
  (170.9 vs 171.0) despite removing another 2.4 GB/step, because the attention projections at M=6
  are launch/latency-bound, not byte-bound -- the same reason the dense-TP experiment moved
  nothing. And it drops nesting depth=6 from 0.83 to 0.11. It stays off.

The lesson matches the entry above: on this model the levers that move are residency and the expert
bytes, not the dense ones, and "it looks right" is not the quality these near-tie probes measure.

`dense_fp4` and `head_fmt` are now in the EP2 boot config guard: they are load-time numerics
choices read from the environment, so a pair split across them would diverge silently.

## A per-request timestamp in the prompt defeats the exact-match prefix cache

Six consecutive requests from a local app on 2026-09-15, captured with the (now count-taking)
`DSV41_CAPTURE_NEXT`. The two "identical" prompts were token-for-token identical
(`captured_request_0.pt == captured_request_1.pt`, LCP 14,336), and the server reported
`prefix=14336/14336 (100%)`, prefill 20.43 s -> 0.35 s. The cache itself works.

The misses were caused by the prompt template embedding `[User current time: 2026-09-15 07:5X]`
at token index 14,331 -- five tokens from the end of the prompt. The next request on the next
minute changes that digit, so the prompts share only 14,331 of 14,336 tokens, and
`_restore_prefix` -- which requires the previous prompt to be an exact prefix of the new one --
drops all 14,331. Measured `prefix=0/14336` and 16-23 s prefills on the re-sent prompts. The
continuations within a minute (`_0 -> _1`, `_3 -> _4 -> _5`) still hit, because there the cached
prompt really is a prefix.

The cheap fix is app-side: a timestamp belongs in the history once, not re-stamped on every
request, and a growing conversation then extends the cached prompt normally. The engine-side
one is `DSV41_PREFIX_SNAPSHOTS=N`: keep the N most recent chunk-boundary prefix snapshots and,
on a full miss, resume at the longest one whose tokens are still a real prefix. With N=8 on the
pair, the timestamped re-send that used to be `prefix=0/14336` and 20.94 s (684.54 tok/s) came
back `prefix=12288/14337 (85.7%)` and 4.58 s (3131.84 tok/s): the 14,331-token LCP rounded down
to the 2,048-token snapshot granularity. It is off by default; each snapshot clones the 128-row
window and replay tail (~6 MB at window 128 / head_dim 512 / 21 window layers), and it is in the
EP2 boot config guard because the restored length decides how many prefill collectives a request
issues. Filled under `DSV41_PREFIX_SNAPSHOTS=8` in the local `.env`.

## A gate that forwards nothing still passes, and serving does not quantize activations

Two traps from 2026-09-23 ([decode projection fusion](decode-projection-fusion.md)):

* **Check the gate's reported config, not `.env`.** `run_two_node_gate.sh` builds its
  `-e DSV41_*` flags from `env`. When the filter command (`rg`) existed only as an interactive
  shell function, nothing was forwarded. Both ranks booted on code defaults (EP2,
  `DSV41_BLOCK=5`), agreed with each other, and passed. The boot guard compares ranks with each
  other, not with `.env`. The gate now uses `grep` and refuses to start when nothing is
  forwarded. Still read `tp_experts` / `draft_tokens` from the result's `config` before
  believing a number.
* **`qlinear` costs no quantization in serving.** With `act_quant=False` (the default),
  `R.act_qdq_fp8` is replaced by a bf16 cast. Removing its ~13 kernels saved 15–20 µs per call
  in a microbenchmark and exactly nothing in the engine. Count a cost only on the path serving
  actually takes.
