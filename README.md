# DeepSeek-V4.1-Flash on two DGX Sparks

A fork of [0xBakeer's engine](https://github.com/0xBakeer/deepseek-v41-flash-spark)
with **two-node tensor parallelism (TP2)** and adaptive expert loading. Routed
experts keep the checkpoint's native MXFP4 weights. The OpenAI-compatible API
supports streaming responses, thinking, tool calls and images.

The model does not fit fully in memory. The engine-only example selects **9,800 of
15,360 routed experts**, sharing their memory across layers as demand changes.
Adaptation improves coverage; it does not guarantee full-model quality.

## Why use this engine?

- **Vision support.** Send images and text through the same chat API. The vision
  tower can run on the second node and share image embeddings with both ranks;
  this freed **0.97 GB** on the head in our tests while preserving image support.
- **Flexible memory configuration, including room for TTS.** Adjust the resident
  expert count, context allocation and each node's Engram cache to fit your setup.
  The development pair runs this engine alongside TTS using the smaller expert
  budget documented below. You can trade some expert capacity for another service
  without downloading or converting a different weight pack. See the
  [coexistence checks and memory limitations](RESULTS.md).
- **Keep the original expert weights.** Routed experts use native MXFP4 from the
  checkpoint, with no additional EXL3 conversion or separate quantized weight pack.
  The default also keeps dense layers at their checkpoint precision.
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

TP2 output sharding, native FP4 kernels, RoCE communication, speculative decoding
and RAM prefix reuse support this design. The distinctive feature is their
combination with a persistent, observable, globally adaptive expert working set.

## Compared with an EXL3 recipe

The main difference is **where the memory saving comes from**:

| | This engine's example profile | A fully resident EXL3 configuration |
| --- | --- | --- |
| Expert weights | Original MXFP4; only a selected subset stays resident | Additional compression, such as the tested 2.9 bpw pack |
| Routing coverage | Missing experts can affect answers, even after adaptation | All routed experts available when the compressed set fits |
| Workload adaptation | Changes which experts occupy the fixed memory budget | Weight quantization stays fixed; memory policy depends on the runtime |
| Main tradeoff | Preserve stored expert precision while accepting residency misses | Accept requantization error to fit more expert weights |

Choose this engine when you want image input, flexible memory allocation for
services such as TTS, and checkpoint expert precision with residency tailored
to your own traffic. It is especially useful for experimenting with expert selection and
seeing the effect directly in the map and routing statistics.

EXL3 remains a strong alternative: fitting all experts avoids this engine's
residency misses, and lower-bit compression can be less damaging than missing an
important expert. Our tests do **not** establish general quality superiority over
Mia's recipe or the official API. Speed, context capacity, vision and concurrent
serving depend on the particular EXL3 runtime; this profile serves one request at
a time. See [recorded experiments](RESULTS.md) rather than treating native weight
precision or a lower miss rate as an answer-quality guarantee.

## Setup

You need two DGX Sparks, Docker with NVIDIA GPU support, a working RoCE link,
and roughly 600 GB of local NVMe space per node for the checkpoint. Clone this
repo to the same absolute path on both nodes and enable head-to-worker SSH.

```bash
cp .env.example .env
```

Set `MODEL_DIR`, `PEER`, `MASTER_ADDR`, `NCCL_SOCKET_IFNAME` and
`GLOO_SOCKET_IFNAME`. Keep credentials in `.env`. Use your existing checkpoint;
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
in `.env.example`.

## Use

- API: `http://<host>:8000/v1`, model name **`deepseek`**.
- Health: `/health` shows the actual serving configuration.
- Expert map: `/expert-map` shows resident, loading and missing experts by layer
  or arena sector. `/v1/expert-map` returns JSON; it does not expose prompt text.
- Logs: `docker logs -f deepseek-v41-ep2-rank0`. Container names retain `ep2`,
  but the template runs TP2.
- Stop: `bash scripts/dual-down.sh`. Restart both nodes after changing settings.

See the [API reference](server/README.md) for requests, thinking and tool calls.

## Example profiles

[.env.example](.env.example) targets running the engine on its own:

| Setting | Profile |
| --- | --- |
| Context allocation | 524,288 tokens, including the answer |
| Expert memory | 92.3 GB/node; 9,800 residents; 8 transient slots |
| Expert selection | Global dynamic allocation, router-score ranking, high sensitivity, prior 4 |
| Adaptation | After prefill, during long answers, on urgent misses, and between requests |
| Vision | Tower on the second node; embeddings shared with both ranks |
| Engram row cache | 256 MiB on the head, 1,024 MiB on the worker |
| Prompt cache | RAM reuse enabled; disk and post-response caching disabled |
| Generation | Speculative depth 3/5; thinking off by default, effort 75 when enabled |

For **TTS alongside the engine**, use `ARENA_GB=90.2` and
`DSV41_RESIDENT_EXPERTS=9574`, as on the development pair. The engine-only
example adds 226 residents and about 2.1 GB per node. Its slot capacity is checked,
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

Memory headroom depends on other services. Lower the resident budget and arena,
or context allocation, if needed. Full-length 512K quality is not established.
Keep demand databases, predictor banks and prompt caches private and out of Git.

## Basic knobs

Edit `.env` on the head, then restart the pair with `scripts/dual-down.sh` and
`scripts/dual-up.sh`. The launcher sends the settings to both ranks. Values below
refer to the example profile, not every engine fallback default.

| Setting | What it controls |
| --- | --- |
| `DSV41_RESIDENT_EXPERTS=9800` + `ARENA_GB=92.3` | Expert count and allocated GB **per node**. Change together; a lower count alone does not shrink the arena. The exact count overrides `PRUNE_KEEP`. Use `9574` / `90.2` for the TTS profile. |
| `MAX_SEQ=524288` | Context allocation including generated tokens. Lower it to reduce cache memory needs. |
| `DSV41_ADAPT_SENSITIVITY=high` | How quickly new traffic changes expert ranking: `low`, `medium`, `high`, `max`; `off` freezes adaptation. Higher follows changes faster but can displace useful experts sooner. |
| `DSV41_ADAPT_PRIOR=4` | Weight of the shipped routing trace. Lower values let your observed traffic dominate sooner. |
| `DSV41_ADAPT_DECODE_TOKENS=600` / `DSV41_ADAPT_URGENT=1` | Periodic and urgent adaptation during an answer. Setting the interval to `0` disables both decode triggers; post-prefill adaptation remains. |
| `DSV41_PREFILL_CHUNK=1024` | Tokens processed per prefill chunk. Smaller chunks reduce temporary memory and give finer prefix-cache boundaries. |
| `DSV41_ENGRAM_CACHE_MB=256` / `DSV41_ENGRAM_CACHE_MB_PEER=1024` | Engram row-cache budgets in MiB on the head and worker. Separate from expert residency and prompt caching. |
| `DSV41_VISION=1` / `DSV41_VISION_MODE=peer` | Enable images and put the vision tower on rank 1. Set vision to `0` for text-only serving. |
| `DSV41_PREFIX_CACHE=1` | Reuse matching prompt state in RAM. Keep `DSV41_PREFIX_DISK=0` and `DSV41_PREFIX_RESPONSE=0` with dynamic allocation. |
| `SPEC=1` | Enable DSpark speculative decoding; `0` disables drafting. Speed depends on draft acceptance. |
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

Original engine by 0xBakeer; see [credits](CREDITS.md). Repository code is
[MIT-licensed](LICENSE); model use is governed by the checkpoint's own license.
