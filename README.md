# DeepSeek-V4.1-Flash on two DGX Sparks

A fork of [0xBakeer's engine](https://github.com/0xBakeer/deepseek-v41-flash-spark)
with **two-node tensor parallelism (TP2)** and adaptive expert loading. Routed
experts keep the checkpoint's native MXFP4 weights. The OpenAI-compatible API
supports streaming responses, thinking, tool calls and images.

The model does not fit fully in memory. The current profile keeps **9,574 of
15,360 routed experts**, sharing their memory across layers as demand changes.
Adaptation improves coverage; it does not guarantee full-model quality.

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

## Current profile

[.env.example](.env.example) contains the portable serving settings:

| Setting | Profile |
| --- | --- |
| Context allocation | 524,288 tokens, including the answer |
| Expert memory | 90.2 GB/node; 9,574 residents; 8 transient slots |
| Expert selection | Global dynamic allocation, router-score ranking, high sensitivity, prior 4 |
| Adaptation | After prefill, during long answers, on urgent misses, and between requests |
| Vision | Tower on the second node; embeddings shared with both ranks |
| Engram row cache | 256 MiB on the head, 1,024 MiB on the worker |
| Prompt cache | RAM reuse enabled; disk and post-response caching disabled |
| Generation | Speculative depth 3/5; thinking off by default, effort 75 when enabled |

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
