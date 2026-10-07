# DeepSeek-V4.1-Flash on two DGX Sparks

A fork of [0xBakeer's engine](https://github.com/0xBakeer/deepseek-v41-flash-spark)
with **two-node tensor parallelism (TP2)** and adaptive expert loading. Routed
experts keep the checkpoint's native MXFP4 weights. The OpenAI-compatible API
supports streaming responses, thinking, tool calls and images.

The model does not fit fully in memory. The engine-only example selects **9,800 of
15,360 routed experts**, sharing their memory across layers as demand changes.
Adaptation improves coverage; it does not guarantee full-model quality.

## Why use this engine?

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
- **Budget memory around other services.** Tune the exact resident count, context
  allocation and each node's Engram cache. Peer-only vision freed **0.97 GB** on
  the head in our tests, leaving more room for experts or a companion service such
  as TTS. See [measurements and limitations](RESULTS.md).

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

Choose this engine when you want to retain checkpoint expert precision, tailor
residency to your own traffic, and control how memory is shared with other
services. It is especially useful for experimenting with expert selection and
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

## Details and measurements

- [Adaptive expert loading and learned seed](docs/adaptive-experts.md)
- [Predictive prefill experiment](docs/predictive-prefill.md)
- [Results](RESULTS.md), [historical measurements](docs/serving-measurements.md),
  and [known issues and rejected experiments](docs/gotchas.md)
- [TP memory](docs/tp-memory.md), [packed KV](docs/packed-kv.md),
  and [speculative decoding](docs/decode-dynamic-depth.md)

Original engine by 0xBakeer; see [credits](CREDITS.md). Repository code is
[MIT-licensed](LICENSE); model use is governed by the checkpoint's own license.
