# DeepSeek-V4.1-Flash on two DGX Sparks

A fork of [0xBakeer's engine](https://github.com/0xBakeer/deepseek-v41-flash-spark)
with **two-node tensor parallelism (TP2)** and adaptive expert loading. Routed
experts default to an **EXL3 2.9 bpw** pack that fits 1.4x as many experts in the same
memory; the checkpoint's **native MXFP4** experts remain a supported profile on the same
image and scripts. The OpenAI-compatible API supports streaming responses, thinking,
tool calls and images.

| Default EXL3 profile, 2026-10-08 | |
| --- | --- |
| Decode, essay / HTML game / Python | **31.0 / 64.4 / 46.7 tok/s** (FP4: 25.6 / 49.3 / 39.4) |
| Cold prefill, 8K / 32K | **1,445 / 1,394 tok/s** (FP4: 1,332 / 1,441) |
| Resident routed experts at 92.3 GB/node | **13,813 of 15,360 (90%)** (FP4: 9,793, 64%) |

The routed experts do not all fit in memory; residency follows demand and is shared
across layers. Adaptation improves coverage; it does not guarantee full-model quality.

## Why use this engine?

- **Vision support.** Send images and text through the same chat API. 
- **Flexible memory configuration, including room for other small services.** Adjust the resident
  expert count, context allocation and each node's Engram cache to fit your setup.
- **EXL3 experts by default, native MXFP4 when you want it.** EXL3 trades a
  requantization (2.9 bpw) for 90% residency and 18-31% faster decode; the FP4 profile
  keeps the checkpoint's expert weights. Switch by restarting with the other example
  profile; nothing is rebuilt.
- **Let your workload shape memory allocation.** Router-score history learns which
  experts to keep, and persists across restarts. This changes residency, not model
  weights. A bundled learned seed gives fresh installs a starting distribution.
- **Move capacity between layers.** One shared arena replaces fixed per-layer
  quotas: a less-used layer can give its slots to a busier one. The total memory
  budget stays fixed, with only the routing minimum reserved in each layer.
- **Adapt while answering.** Normal adaptation runs after prefill and during long
  answers. An urgent trigger can replace experts when recent decode misses rise;
  it does not require streaming every missing expert from disk.
- **See what the engine is doing.** The live expert map shows loading, residency
  and arena ownership. Request statistics expose both routing-count and
  router-score-weighted misses, making selection changes inspectable.

TP2 output sharding, EXL3 and native FP4 kernels, RoCE communication, speculative decoding
and RAM prefix reuse support this design. The distinctive feature is their
combination with a persistent, observable, globally adaptive expert working set.

## EXL3 (default) or FP4 experts

Both profiles serve the same model: dense layers, attention, the vocabulary head, Engram,
shared experts and the DSpark drafter always come from the native checkpoint. Only the
384 x 40 routed experts differ.

| At `ARENA_GB=92.3` per node | **EXL3** (`.env.example.exl3`, default) | FP4 (`.env.example.fp4`) |
| --- | --- | --- |
| Routed expert weights | [Mia-AiLab's](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) EXL3, 2.9 bpw average (2-bit in layers 18-22), 6.67 MB/node each | Checkpoint MXFP4, 9.4 MB/node each |
| Resident experts | **13,813 (90%)** | 9,793 (64%) |
| Decode, essay / HTML / Python | **31.0 / 64.4 / 46.7 tok/s** | 25.6 / 49.3 / 39.4 tok/s |
| Cold prefill, 8K / 32K | 1,445 / 1,394 tok/s | 1,332 / 1,441 tok/s (parity within noise) |
| Extra disk | 98 GB pack per node, built once from the 198 GB EXL3 checkpoint | none |

EXL3 reads fewer bytes per expert, which is most of a decode step, and leaves fewer
experts missing; that is why it is the default. The cost is requantization error in the
routed experts: choose FP4 to serve the checkpoint's own expert weights. Our tests
do **not** establish that either profile answers better; they are throughput checks
plus the engine's floor tests. See [the 2026-10-08 measurements](RESULTS.md) and
[the EXL3 design notes](docs/exl3-plan.md).

## Performance

Measured on two DGX Sparks on **2026-10-08**, both profiles on the same image at 92.3 GB/node:
512K context allocation, native dense precision, abliteration enabled, dynamic speculative
depth and adaptation active. TTS was not loaded.

| Workload | EXL3 decode | EXL3 time to first token | FP4 decode |
| --- | ---: | ---: | ---: |
| Python LRU cache with TTL, thread safety and tests | **46.7 tok/s** (45.8–47.7) | 0.47 s | 39.4 tok/s (38.6–40.2) |
| Coastal-ecosystem essay | **31.0 tok/s** (30.3–31.6) | 0.55 s | 25.6 tok/s (25.5–25.7) |
| Angry Birds single-file HTML game | **64.4 tok/s** (63.9–65.0) | 0.24 s* | 49.3 tok/s (49.1–49.5) |

Medians of two 512-token runs after one warmup each; thinking off, temperature zero,
fixed output length. Decode speed excludes time to first token. *The HTML prompt reused
its 62 cached prompt tokens; the others reused none. FP4's first-token times were 0.7–0.9 s
except two requests that queued 16–24 s behind idle-time expert re-planning right after the
profile switch. These are short throughput checks, not answer-quality tests. Prompt content,
draft acceptance, expert history and cache reuse affect speed. See [full results](RESULTS.md).

Cold prefill (uncached random-word prompts; medians of 5 at 8K and 2 at 32K):

| Prompt | EXL3 | FP4 | FP4 before 2026-10-08 |
| --- | ---: | ---: | ---: |
| 8K | **1,445 tok/s** (5.7 s) | 1,332 tok/s (6.2 s) | 390–470 tok/s (17–21 s) |
| 32K | **1,394 tok/s** (23.5 s) | 1,441 tok/s (22.7 s) | 470 tok/s (70 s) |

The first requests after a boot or profile switch run 1.5–3x slower while adaptation moves
experts and kernels warm up. Prefill switches: `DSV41_ENGRAM_DIRECT=prefill`,
`DSV41_PREFILL_CHUNK=2048`, `DSV41_PREFILL_ATTN_INDEXED=1`, `DSV41_HC_PREFILL_FUSED=1`,
`DSV41_INDEX_FUSED=1`, `DSV41_FP4_PREFILL_DOT_SCALED=1`, `DSV41_FP4_PREFILL_SCALED_TILES=1`;
EXL3 adds its prefill tile kernel (`DSV41_EXL3_PREFILL_KERNEL=tile`, the default).
A larger `ARENA_GB` leaves less memory for prefill scratch: at 95.3 GB the engine halved its
prefill chunk and 8K took 7.4 s.

Decode history (FP4, same benchmark on 2026-10-08):

| Workload | Before | Arithmetic switches | + Draft | + Kernels |
| --- | ---: | ---: | ---: | ---: |
| Coastal-ecosystem essay | 21.9 tok/s | 23.1 tok/s | 25.4 tok/s | **26.4 tok/s** |
| Angry Birds HTML | 43.3 tok/s | 50.3 tok/s | 52.7 tok/s | **53.4 tok/s** |
| Python LRU cache with TTL | 32.3 tok/s | 37.4 tok/s | 38.4 tok/s | **40.4 tok/s** |

Switches: `DSV41_ATTN_STAGED=2`, `DSV41_ROUTER_BF16=1`, `DSV41_HC_KERNEL=1`. Draft:
`DSV41_TP_DRAFT_HEAD=1`, `DSV41_DRAFT_HEAD_FMT=fp8`, `DSV41_DRAFT_MARKOV_TP=1`,
`DSV41_TP_DRAFT_ATTN=1`. Kernels: `DSV41_FP8_DECODE_BLOCK_N=32`, `DSV41_FP8_DECODE_BLOCK_K=256`,
`DSV41_LEAN_RMS_FUSED=1`, `DSV41_HC_FRONT_FUSED=1`. Paired 22-prompt averages: **+9%**,
**+2.3%** and **+2.1%**, **+1.8%** and **+4.0%**. Both profiles also queue the next draft
before the host reads each verify result (`DSV41_EARLY_DRAFT`/`DSV41_EARLY_VERIFY`, on by
default): -2.0 ± 0.5 ms a step, outputs bit-identical.

Optional `DSV41_HEAD_KERNEL=packed` losslessly packs the native BF16 vocabulary
head. The short TP2 code/prose comparison saved **146 MiB/node** and improved
decode round speed about **2%**, with identical tested tokens and acceptance.
The draft head must be `off`/`bf16` unless `DSV41_TP_DRAFT_HEAD=1` builds a separate
FP8 draft shard; the examples enable both.

Reproduce an individual workload against a running server:

```bash
python3 bench/bench.py --model deepseek --workload code --osl 512 \
  --warmup 1 --runs 2 --temperature 0 --ignore-eos --label readme-20261008-code
```

## Setup

You need two DGX Sparks, Docker with NVIDIA GPU support, a working RoCE link,
and roughly 600 GB of local NVMe space per node for the checkpoint. Clone this
repo to the same absolute path on both nodes and enable head-to-worker SSH.

```bash
cp .env.example.exl3 .env      # default; .env.example.fp4 for the native MXFP4 experts
```

Set `MODEL_DIR`, `PEER`, `MASTER_ADDR`, `NCCL_SOCKET_IFNAME` and
`GLOO_SOCKET_IFNAME`. Keep credentials in `.env`. Both profiles need the native
checkpoint on both nodes; the default EXL3 profile also needs its expert packs, built
once below before the first `dual-up.sh`. Use your existing checkpoint;
if it is missing, run the download script on each node (requires Python 3 and
`huggingface_hub`):

```bash
(set -a; source .env; set +a; bash scripts/download-model.sh)
```

Then run on the head:

```bash
bash scripts/dual-build.sh       # builds once and copies the image to the worker
bash scripts/dual-up.sh --check
bash scripts/dual-up.sh
```

Startup takes several minutes. The template binds to localhost; set `HOST` to
a trusted interface to allow remote clients. The API has no authentication.
Optional dual-rail networking needs both interfaces/subnets configured as described
in the example profiles.

`dual-up.sh` refuses an image built from different engine code than the checkout (rebuild
with `dual-build.sh`, or boot the checkout directly with `scripts/dual-dev.sh` while
developing), and refuses to report success if an EXL3 profile came up without its CUDA
kernel (`/health` must show `"kernel": "exl3-cuda"`).

### EXL3 expert packs (once, for the default profile)

Only the head needs the EXL3 checkpoint (about 198 GB). After `dual-build.sh`:

```bash
hf download Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw \
  --local-dir /path/to/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw   # next to the native checkpoint
bash scripts/exl3-packs.sh --check
bash scripts/exl3-packs.sh       # ~8 min a rank, then copies rank 1's pack to the worker
```

This writes one 98 GB pack per node to `<models>/exl3-packs/`, where the engine looks for it.
The EXL3 checkpoint and the head's copy of rank 1's pack can then be deleted. Switch
profiles by copying the other example over `.env` (keeping your edits), then
`dual-down.sh` and `dual-up.sh`; nothing is rebuilt.

## Use

- API: `http://<host>:8000/v1`, model name **`deepseek`**.
- Health: `/health` shows the actual serving configuration and allocated/reserved CUDA memory.
  `DSV41_GRAPHS_MAX=8` bounds cached graph variants by rotating their shared pool.
  `DSV41_PREFILL_ADAPT=1` shrinks prefill chunks under memory pressure on either node;
  `DSV41_PREFILL_CHUNK` remains the maximum. The scratch estimate is not an OOM guarantee.
- Expert map: `/expert-map` shows resident, loading and missing experts by layer
  or arena sector. `/v1/expert-map` returns JSON; it does not expose prompt text.
- Logs: `docker logs -f deepseek-v41-ep2-rank0`. Container names retain `ep2`,
  but the template runs TP2.
- Stop: `bash scripts/dual-down.sh`. Restart both nodes after changing settings.

See the [API reference](server/README.md) for requests, thinking and tool calls.

## Example profiles

[.env.example.exl3](.env.example.exl3) (default) and [.env.example.fp4](.env.example.fp4) target
running the engine on its own. They differ only in `EXPERT_FORMAT` and the predictor database:

| Setting | Profile |
| --- | --- |
| Context allocation | 524,288 tokens, including the answer |
| Expert memory | 92.3 GB/node; residents follow the arena (EXL3 13,813, FP4 9,793); 8 transient slots |
| Expert selection | Global dynamic allocation, router-score ranking, high sensitivity, prior 4 |
| Adaptation | After prefill, during long answers, on urgent misses, and between requests |
| Vision | Tower on the second node; embeddings shared with both ranks |
| Engram row cache | 256 MiB on the head, 1,024 MiB on the worker |
| Prompt cache | RAM reuse enabled; disk and post-response caching disabled |
| Generation | Speculative depth 3/5; thinking off by default, effort 75 when enabled |

**All weights resident:** [.env.example.exl3.allweight](.env.example.exl3.allweight) keeps every
routed expert in memory (15,360 of 15,360, no expert misses) by giving the 1,920 2-bit EXL3 experts
slots of their own size (`DSV41_EXL3_EXACT_SLOTS=1`: 98.37 GB/node instead of 102.6), with a 200K
context and a 256 MiB worker Engram cache. Measured 2026-10-08: decode 30.2 / 64.9 / 43.7 tok/s
(essay / HTML / Python), cold prefill 8K 7.3 s and 32K 35.4 s. The ~7 GB left per node makes the
prefill budget choose 512-row chunks, so prefill is 22-34% slower than the 92.3 GB profile; the
lowest free memory under a 32K-prompt stress run was 4.6 GB. Use it when nothing else shares the boxes.

For **TTS alongside the engine**, use `ARENA_GB=90.2`, as on the development pair; the
resident expert count follows the arena (every slot but 16). The engine-only
example adds about 2.1 GB per node. Its slot capacity is checked,
but sustained peak-memory headroom has not yet been validated. Both TP nodes
need room for the increase, even when TTS runs on only one.

There are no fixed per-layer quotas beyond the routing minimum. Fresh installs
use the bundled learned expert seed; existing local demand history takes priority.
The learned history survives restarts. **RAM prompt caches do not.** The retained
300-second disk-save interval has no effect while disk caching is disabled, as
required by the current dynamic allocation mode.

Dense FP4 re-quantization and expert streaming are off. Predictive prefill is an
optional experiment: `shadow` learns and evaluates without moving experts;
`apply` makes predicted swaps before prefill. Actual post-prefill adaptation stays
active. The template leaves prediction off until you choose to collect examples.

Memory headroom depends on other services. Lower the arena (the resident count follows)
or context allocation, if needed. Full-length 512K quality is not established.
Keep demand databases, predictor banks and prompt caches private and out of Git.

## Basic knobs

Edit `.env` on the head, then restart the pair with `scripts/dual-down.sh` and
`scripts/dual-up.sh`. The launcher sends the settings to both ranks. Values below
refer to the example profile, not every engine fallback default.

| Setting | What it controls |
| --- | --- |
| `EXPERT_FORMAT=exl3` | `exl3` (default profile; needs the packs, see Setup) or `fp4` (the checkpoint's native MXFP4 experts). Same image; restart to switch. |
| `ARENA_GB=92.3` | Expert arena GB **per node**; the resident count follows it (every slot but 16, at most all 15,360 experts, logged at boot) for either `EXPERT_FORMAT`. Use `90.2` for the TTS profile. `DSV41_RESIDENT_EXPERTS=<n>` pins a count for experiments. |
| `MAX_SEQ=524288` | Context allocation including generated tokens. Lower it to reduce cache memory needs. |
| `DSV41_ADAPT_SENSITIVITY=high` | How quickly new traffic changes expert ranking: `low`, `medium`, `high`, `max`; `off` freezes adaptation. Higher follows changes faster but can displace useful experts sooner. |
| `DSV41_ADAPT_PRIOR=4` | Weight of the shipped routing trace. Lower values let your observed traffic dominate sooner. |
| `DSV41_ADAPT_DECODE_TOKENS=600` / `DSV41_ADAPT_URGENT=1` | Periodic and urgent adaptation during an answer. Setting the interval to `0` disables both decode triggers; post-prefill adaptation remains. |
| `DSV41_PREFILL_CHUNK=2048` | Tokens processed per prefill chunk. Smaller chunks reduce temporary memory and give finer prefix-cache boundaries; `DSV41_PREFILL_ADAPT=1` shrinks them under memory pressure. |
| `DSV41_ENGRAM_CACHE_MB=256` / `DSV41_ENGRAM_CACHE_MB_PEER=1024` | Engram row-cache budgets in MiB on the head and worker. Separate from expert residency and prompt caching. |
| `DSV41_VISION=1` / `DSV41_VISION_MODE=peer` | Enable images and put the vision tower on rank 1. Set vision to `0` for text-only serving. |
| `DSV41_PREFIX_CACHE=1` | Reuse matching prompt state in RAM. Keep `DSV41_PREFIX_DISK=0` and `DSV41_PREFIX_RESPONSE=0` with dynamic allocation. |
| `SPEC=1` | Enable DSpark speculative decoding; `0` disables drafting. Speed depends on draft acceptance. |
| `DSV41_HEAD_KERNEL=off` | Optional `packed` stores the native BF16 head losslessly and uses a decode kernel. Requires `DSV41_DRAFT_HEAD_FMT=off` or `bf16`. Saves about 146 MiB/node; measured code/prose decode gain was about 2%. |
| `DEFAULT_THINKING=off` / `DEFAULT_EFFORT=75` | Defaults when the client does not specify thinking. Effort is 1–100 and applies when thinking is enabled; clients can override it per request. |

The two `ADAPT_*` ranking knobs derive the normal swap thresholds; copying old
`DSV41_PRUNE_*` threshold overrides is unnecessary. `DSV41_USER_PROMPT_MAX_LOADS`
only caps experimental streaming loads, **not normal adaptive expert swaps**.
See [adaptation details](docs/adaptive-experts.md) for advanced settings.

## Optional abliterated weights

Set `DSV41_ABLIT_WOB` to a compatible native
`wo_b_l10_35.safetensors` overlay (about 1.1 GB), as supported by
[the overlay loader](engine/ablit.py). It replaces the attention output projections
in layers 10–35 at load time. Experts and other checkpoint tensors retain their
original weights; no second full checkpoint or modification of the original files
is needed. This intentionally changes model behavior.

Place the **same overlay file on both nodes** inside the host models directory.
For example, if the checkpoint is `/srv/models/DeepSeek-V4.1-Flash`, place it at
`/srv/models/dsv41-wo-b-ablit/wo_b_l10_35.safetensors`. For the Docker launcher,
use the corresponding **container path** in `.env`:

```dotenv
DSV41_ABLIT_WOB=/models/dsv41-wo-b-ablit/wo_b_l10_35.safetensors
```

Restart both ranks and check `engine_config.ablate_wob` in `/health`. The boot
guard checks that the overlay contents agree across nodes. For a native launch,
use the host's absolute path instead. To return to stock weights, remove or empty
`DSV41_ABLIT_WOB` and restart. The overlay substitutes weights rather than adding
another model, so disabling it does not free an extra expert arena.

## Details and measurements

- [Adaptive expert loading and learned seed](docs/adaptive-experts.md)
- [Predictive prefill experiment](docs/predictive-prefill.md)
- [Results](RESULTS.md), [historical measurements](docs/serving-measurements.md),
  and [known issues and rejected experiments](docs/gotchas.md)
- [TP memory](docs/tp-memory.md), [packed KV](docs/packed-kv.md),
  and [speculative decoding](docs/decode-dynamic-depth.md)
- Decode dashboard at `/decode-probe`: [timing, buffer reuse and local replay controls](docs/decode-probe.md)

Original engine by 0xBakeer; see [credits](CREDITS.md). Repository code is
[MIT-licensed](LICENSE); model use is governed by the checkpoint's own license.
