# DeepSeek-V4.1-Flash on TWO DGX Sparks (EP2 fork)

> This is a fork of [0xBakeer/deepseek-v41-flash-spark](https://github.com/0xBakeer/deepseek-v41-flash-spark),
> which serves this model on **one** GB10 box. This fork adds a **second box**: the 15,360 routed
> experts are split by parity across the pair (expert parallel, world size 2) and everything else
> — attention, Engram, router, the DSpark drafter, KV, sampling — stays bit-identically replicated.
> [`README.md`](README.md) is upstream's, kept byte-for-byte, and still describes the engine
> this is built on.

## Quick start

Head spark should be able to SSH to worker spark.

```bash
cp env.example .env          # set PEER, MASTER_ADDR, MODEL_DIR, NCCL_SOCKET_IFNAME
scripts/download-model.sh    # 510 GB, on BOTH boxes -- each needs its own local copy,
                             # because the engine reads experts with O_DIRECT
scripts/dual-build.sh        # build once here, ship the identical image to the peer
scripts/dual-up.sh           # start both ranks; --check runs preflight and exits
scripts/dual-down.sh         # stop both
```

There is no single-box path in this fork: `start.sh`/`stop.sh` are removed, and the engine is
started only through the container scripts above. Upstream still serves one box if that is what
you want.

**What the second box buys.** Not raw FLOPs, but **residency**: 82 GB of expert arena 
per box instead of one box's 73.8 GB, so 28.4 % of the routed experts stay resident with the rest 
streamed from NVMe, and the working set is re-fitted to the traffic as it arrives.

Measured on the development pair (two GB10 / DGX Spark, 200 GbE direct-attach RoCE):

| | |
|---|---|
| prefill | **~1047 tok/s** on a 4,513-token prompt, when cache hits it can go 10k+ token/sec |
| decode | **13–24 tok/s**, set by draft acceptance (2.1 on prose, 4.5 on code) at a ~163 ms step |
| context | 256k (`MAX_SEQ=262144`) |
| residency | 4,361 of 15,360 experts (28.4 %), 82 GB arena per box, total about 60% of expert is loaded. Loaded expert will be swapped live. See adaptive expert loading for details. |


## About this recipe

- It uses custom engine, made by 0xBakeer. In this fork, it's heavily modified.
- Concurrency is 1.
- Expert weight quant is fp4. 
- It can't load all expert weight to two machines, so some of weights are not loaded.
  However, adaptive expert loading measures missed expert weight and the engine will load missed 
  expert eventually. It works well with continuous conversation with harness.

## What this fork adds on top of the upstream engine

- **Docker image** - Docker image created for portability
- **EP2 expert parallelism** — routed experts split `expert % 2 == rank`, one fp32 all-reduce per
  MoE layer, captured inside the decode CUDA graphs. A boot-time guard refuses to start when the
  two ranks disagree about how to compute, because a skewed pair does not fail — it quietly
  computes different things.
- **Adaptive expert loading** — the engine records what the router *wanted* (not what it got),
  blends it with the shipped trace, and swaps the arena toward observed demand. It adapts twice
  per request: once at the prefill→decode boundary, so a long prompt's demand is applied before
  the answer is written, and once after. The database survives restarts.
- **Prefix cache** — a continuing conversation re-prefills only its new tokens
  (95–100 % hits in practice). `tools/test_prefix_invariance.py` asserts the property the cache
  rests on: resuming from a cached prefix reproduces a cold prefill **byte for byte**, and that
  the cache was actually used.
- **Vision** — the ViT + aligner path, with image spans aligned to prefill chunk boundaries.
- **Speed work, all measured** — the engram row gather was running single-threaded at decode
  (83 ms of a 196 ms step); the fused prefill attention had been switched off and is worth
  544 → 1047 tok/s.

--------------------------

**The upstream engine's own documentation is [`README.md`](README.md), preserved unmodified.**
Everything it describes about the engine still applies; this file covers only what the fork adds
on top, and the parts of it that differ for a two-box deployment.


## Knobs this fork adds

All are environment variables read at process start, and `scripts/dual-up.sh` forwards every
`DSV41_*` it sees to **both** ranks. The table gives the *code* default; the shipped pair profile
in `env.example` differs where noted, and `/health` reports what is actually live.

**Expert parallelism (the pair)**

| knob | default | what it does |
|---|---|---|
| `WORLD_SIZE` | `1` | `2` splits routed experts `expert % 2 == rank`; everything else is replicated |
| `DSV41_ENGRAM_ROW_SPLIT` | `0` (profile `1`) | each rank reads half the Engram rows; one all-reduce rebuilds the union |
| `DSV41_ENGRAM_SPLIT_MIN` | `4096` | rows below which a gather stays local — a 144-row decode gather is not worth a collective |
| `DSV41_DIST_TIMEOUT_S` | `600` | process-group timeout; `DSV41_EP_PING_S` keeps an idle pair inside it |

> **These must match on both ranks.** A rank that computes differently from its peer does not
> fail — it emits different tokens, silently. The engine broadcasts 11 such fields at boot and
> refuses to start if they disagree.

**Adaptive expert residency** — record what the router *wanted*, then re-fit the arena to it.

| knob | default | what it does |
|---|---|---|
| `DSV41_PRUNE_MISS` | `0` (profile `1`) | record demand (the unmasked top-k) — the input to everything below |
| `DSV41_PRUNE_SWAP` | `0` (profile `1`) | actually move experts in the arena |
| `DSV41_PRUNE_PRIOR` | `2e7` (profile `2e4`) | per layer; the blend is `w = observed/(observed + PRIOR)` |
| `DSV41_PRUNE_HALFLIFE` | `2e7` | EWMA half-life **in routing slots**, and a 4.5k prompt makes ~588,000 of them. `2e6` half-lives the history every ~3.4 requests and churns ~100 experts per request at a 0.9 % miss rate |
| `DSV41_PRUNE_SWAP_MAX` | `64` (profile `512`) | swaps per pass. Binds on a real deficit: a 26k out-of-distribution prompt can start near 31 % miss, where 128 (2.9 % of the arena) cannot catch up |
| `DSV41_PRUNE_SWAP_MIN_GAIN` | `0.05` (profile `0.02`) | per-swap bar. Reach for the half-life first — this only suppresses the symptom |
| `DSV41_PRUNE_SWAP_PREFILL` | `0` (profile `1`) | adapt at the prefill→decode boundary too, so a long prompt's demand is applied **before** the answer is written |
| `DSV41_PRUNE_SWAP_PREFILL_MIN` | `1024` | newly prefilled tokens below which that pass declines |
| `DSV41_PRUNE_SWAP_PREFILL_MIN_MISS` | `0.10` | …and the prefill's own miss rate below which it declines, so a well-served prompt is left alone |
| `DSV41_PRUNE_DB` | `results/prune_demand.npz` | survives restarts; gitignored, rebuilt from the shipped trace if absent |

**Prefill**

| knob | default | what it does |
|---|---|---|
| `DSV41_PREFILL_FUSED_ATTN` | `1` | fused sinked-softmax attention. **544 → 1047 tok/s**, and chunk-invariant, so the prefix cache still reproduces a cold prefill byte for byte |
| `DSV41_PREFIX_CACHE` | `1` | exact prompt-prefix reuse; a continuing conversation re-prefills only its new tokens |
| `DSV41_INDEX_SCORE_BF16` | `1` | the indexer score buffer ranks, it does not compute — bf16 halves prefill's largest transient |
| `DSV41_PREFILL_CHUNK` | `2048` | prefill chunk size; image spans are aligned so none straddles a boundary |
| `DSV41_ENGRAM_GATHER_THREADS` | `64` | Engram row gather width. It ran single-threaded at decode and cost 83 ms of a 196 ms step |

**Decode**

| knob | default | what it does |
|---|---|---|
| `DSV41_BLOCK` | `5` | drafted tokens per verify block (odd). The DSpark head is trained at 5; raising it reaches more *distinct* experts per step and measured slower here |
| `DSV41_VISION` | `1` | load the ViT + aligner |

The engine carries other switches that are experimental, half-finished or measured harmful, and they are deliberately not listed here — they are default-off, and the ones worth knowing about are written up with their numbers in [`docs/gotchas.md`](docs/gotchas.md).

**Instrumentation**, all default `0` and cheap: `DSV41_STEP_TIMING` (per-phase decode table),
`DSV41_GPU_TIMING` (GPU-timeline ms per step), `DSV41_ATTN_TIMING` (prefill attention phases),
`DSV41_ROUTE_STATS` (distinct experts per layer per verify block — adds two ops *inside* the
decode graph, so measurement only).
