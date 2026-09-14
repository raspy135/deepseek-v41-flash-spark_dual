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

## Structural corruption in greedy decode is the expert quant, not the engine

Symptom, found comparing this engine against MiaAI's EXL3 2.9 bpw build of the same checkpoint on a
nesting probe (`llm_benchmark/quality_quant2.py --only nesting`): asked to emit `{"n": ...}` nested
N deep, it instead produces, deterministically,

```
depth=4   {"n": {"n": {"n": {"n": 44}}}}                          ok
depth=8   {"n": {"n": {"n": {"n": {"n": {{"n": {"n": 48}}}}}}}}}   doubled brace
depth=10  {"n": {"n": {... {"n": "n": {...  "nn": 50}}...        dropped quote, merged key
```

Depth 4 is always right, 6 marginal, 8+ broken, and the corruption is character duplication and a
lost quote -- not a semantic substitution. It reproduced on every engine configuration tried, so no
switch the engine exposes is the cause:

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
exactly. So this is a property of the checkpoint's **MXFP4 scale model** -- UE8M0, one power of two
per 32 K -- on repeated low-margin tokens, not of this engine. The effective precision at 4
bits/weight is gated by that scale grid, not by the code width, which is why a well-fitted 3-bit
codebook can win this particular family.

**Do not generalize it.** The battery's macro is decided by `nesting` + `constraint` +
`char_count`; `counter`, `json_strict`, `verbatim`, `copy_transform` and `escape_json` saturate at
1.00 for both models. Accurate JSON does not mean 4-bit loses everywhere against 3-bit: the native
MXFP4 is competitive on the other families and on held-out teacher-forced loss. The finding is
"repeated structural output is where the MXFP4 scale model shows", nothing broader.

**If you want to fix it**, the fix is a requant and the lever is the scale/error model, not the bit
width:

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

If you serve the native checkpoint, expect this family to fail and do not spend engine time on it.

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
