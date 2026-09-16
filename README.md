# DeepSeek-V4.1-Flash on two DGX Sparks

A fork of [0xBakeer/deepseek-v41-flash-spark](https://github.com/0xBakeer/deepseek-v41-flash-spark),
using a custom PyTorch/Triton engine, two-node tensor parallelism, adaptive expert
loading, and persistent prefix caches. Routed experts use the checkpoint's native
MXFP4 weights, without another quantization step.

The model does not fit entirely in two Sparks. The usual configuration keeps about
63% of routed experts resident and changes that selection as demand changes.
Pruning is a quality/speed tradeoff: adaptation improves coverage, but does not make
a pruned request equivalent to running every expert.

The server provides an OpenAI-compatible API, streaming, tool calls, thinking mode,
vision, and DSpark speculative decoding. It processes one request at a time.

## Technical highlights

- **Native MXFP4 checkpoint.** Uses the original routed-expert weights without requantizing them to another format.
- **Vision enabled.** Supports text and image inputs, not just text-only inference.
- **Adaptive expert loading.** Resident experts change with your workload, using observed routing demand to decide which weights to keep in memory.
- **One model, two DGX Sparks.** Tensor parallelism splits the same selected experts across both nodes, alongside attention and other model weights.
- **Persistent prefix cache.** Saves prompt prefixes to local disk for reuse across requests and server restarts.

## Setup

You need two DGX Sparks, a working RoCE link, Docker with NVIDIA GPU support, and
at least 600 GB of local NVMe space per node for the checkpoint. Each node reads
its own copy of the weights; do not put them on NFS. The development pair uses a
200 GbE direct link.

Clone this repository to the same absolute path on both nodes. The head must be
able to SSH to the worker without a password prompt.

```bash
cp .env.example .env
```

Set `MODEL_DIR`, `PEER` (`user@worker-ip`), `MASTER_ADDR` (the head's link IP), and
`NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` (the link interface). The template uses
the current 768K allocation and TP profile below. Full-length 768K quality has not
been validated. Keep credentials in `.env`, not the template.

```bash
# Run on each node; downloads the full checkpoint to local storage.
(
  set -a
  source .env
  set +a
  bash scripts/download-model.sh
)

# Run on the head.
bash scripts/dual-build.sh       # build once and copy the identical image to the worker
bash scripts/dual-up.sh --check  # preflight only
bash scripts/dual-up.sh
```

The download script needs Python 3 with `huggingface_hub` installed. Skip it if
the full checkpoint is already present on both nodes.

Startup loads weights before opening the API port and takes several minutes.
Both nodes need enough free unified memory; stop other GPU servers first.

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
docker logs -f deepseek-v41-ep2-rank0
```

The container names still contain `ep2`, even in TP mode. `/health` reports the
active layout, memory settings, speculation, and cache configuration.
Stop the pair with `bash scripts/dual-down.sh`.

## TP profile

These are the current serving settings, also in `.env.example`, not the engine's
fallback defaults. Arena sizes are decimal GB **per node**. Older quality and speed
checks below retain their original configurations.

```dotenv
MAX_SEQ=786432
ARENA_GB=92
PRUNE_KEEP=0.63
EXPERT_FORMAT=fp4
TRANSIENT_SLOTS=16
KEEP_FREE_GB=6
SPEC=1
DSV41_BLOCK=3
DSV41_PACKED_KV=1
DSV41_VISION=1

DSV41_TP_EXPERTS=1
DSV41_TP_DENSE=1
DSV41_TP_ATTN=1
DSV41_TP_HEAD=1
DSV41_TP_DRAFT_EXPERTS=1
DSV41_TP_EMBED=1
DSV41_TP_EXPERT_LAYOUT=output
DSV41_TP_LINEAR_LAYOUT=output

DSV41_DENSE_FP4=off
DSV41_FP4_DOT_SCALED=0
DSV41_HEAD_FMT=bf16
DSV41_ACT_QUANT=0
```

TP splits resident expert weights, draft expert weights, the input embedding,
main-model shared experts, attention projections, and the vocabulary head across
the pair. KV caches and some other weights remain replicated.
The `output` layouts preserve complete down-projection dot products; the older
`intermediate` layouts regressed nesting quality. Leave the precision settings above
alone unless you are testing a numerical change.

The main capacity and speed controls:

| Setting | What to change it for |
| --- | --- |
| `MAX_SEQ` | Total context allocation, including the answer. More context needs more cache and scratch memory. |
| `DSV41_PACKED_KV=1` | Pack FP4 history in 64-bit words. Saves about 0.67 GiB per request lane at 384K context; window caches and index keys stay BF16. Restart both nodes when changing it. |
| `ARENA_GB` / `PRUNE_KEEP` | More resident experts, at the cost of memory. Raise them together only when there is room. |
| `DSV41_TP_DRAFT_EXPERTS=1` | Split draft expert weights across TP2; saves 3.36 GiB per node. Requires native FP4 and the `output` expert layout. Adds draft collectives. |
| `DSV41_TP_EMBED=1` | Split input embedding columns across two ranks; saves 0.62 GiB per node. Adds one gather per lookup, without changing stored precision. |
| `SPEC=1` | Enable speculative decoding. Speed depends on how many draft tokens are accepted. |
| `DSV41_MAX_CONCURRENCY=1` | Experimental: `2` serves two requests together on TP. Needs extra cache memory; prefill still runs one prompt at a time. See [concurrency notes](docs/concurrency.md). |
| `DSV41_HC_MM_TILE=32` | Faster FP32 hyper-connection decode projections. `16` restores the previous summation order. See [measurements](docs/decode-fp32-experiments.md). |
| `DSV41_PREFILL_CHUNK=1024` | Prefill chunk size. Smaller chunks give finer prefix-cache boundaries; larger chunks reduce dispatch overhead. |
| `DSV41_PREFILL_FUSED_ATTN=1` | Keep fused prefill attention enabled. |
| `DSV41_ENGRAM_ROW_SPLIT=1` | Split large Engram row reads across the two nodes. |
| `DSV41_VISION=1` | Load image support. Set to `0` for text-only serving. |

At keep `0.63`, the engine selects 9,680 of 15,360 routed experts: 242 per layer.
Each TP shard is 9.40032 MB, so their weights occupy about 91.00 GB per node,
within the 92 GB arena. That is 440 more resident experts than the previous
88 GB / keep `0.60` profile. Sharding the draft experts and embedding frees
3.98 GiB per node; 4 decimal GB of that saving now goes to the larger arena.
The remaining memory must cover dense weights, the drafter, caches, and scratch.
The engine refuses a pruned configuration whose selected experts cannot all fit.

For an unpruned control, use `PRUNE_KEEP=1.0` and `TRANSIENT_SLOTS=384`. Experts outside
the arena are then streamed from NVMe. This is much slower; use short quality probes
before attempting a long benchmark.

### Context length

The profile allocates 768K tokens (`786432`), with packed history caches. A
14,435-token request completed with this allocation and the 92 GB arena; that
does not validate full-length quality or peak memory use at 768K. The checkpoint
declares a 1M-token maximum, but allocation alone does not establish correctness
or speed at that length. Lower `MAX_SEQ` if you do not need the extra context.

Command-line overrides leave `.env` unchanged. Stop the existing pair before restarting:

```bash
bash scripts/dual-down.sh
MAX_SEQ=524288 bash scripts/dual-up.sh
```

## Adaptive expert loading

The engine records the router's choices before pruning, blends that demand with a
routing trace, and replaces less-used resident experts. Swaps can happen after
prefill and at the end of a request, subject to the thresholds below. They do not
retroactively recompute tokens that were already processed.

Use these values for request-weighted adaptation:

| Setting | Value | Purpose |
| --- | --- | --- |
| `DSV41_PRUNE_MISS` | `1` | Record demand and report missed expert selections. |
| `DSV41_PRUNE_SWAP` | `1` | Enable request-boundary swaps. |
| `DSV41_PRUNE_SWAP_PREFILL` | `1` | Also allow swaps before decoding. |
| `DSV41_PRUNE_UNIT` | `request` | Give each request one vote instead of weighting by token count. |
| `DSV41_PRUNE_PRIOR` | `8` | Weight of the initial trace relative to observed requests. |
| `DSV41_PRUNE_HALFLIFE` | `20` | Age demand over 20 requests. |
| `DSV41_PRUNE_SWAP_MAX` | `512` | Maximum expert replacements per pass. |
| `DSV41_PRUNE_SWAP_MIN_GAIN` | `0.005` | Avoid swaps with little expected benefit. |
| `DSV41_PRUNE_SWAP_PREFILL_MIN` | `32` | Minimum newly prefilled tokens for a prefill swap. |
| `DSV41_PRUNE_SWAP_PREFILL_MIN_MISS` | `0.02` | Skip that pass if prefill misses less than 2% of selections. |
| `DSV41_PRUNE_DB` | `results/prune_demand_req.npz` | Save demand across restarts. Use a separate file when changing units. |

`TRACE_STATS` chooses the initial `coverage.json`; when unset, the launcher looks
under `results/trace-*/stats/`. Keep the demand database private and out of Git.
To freeze expert placement for an A/B test, set both swap flags to `0`.

## Persistent prefix cache

Matching prompts can resume from a saved prefix instead of processing it again.
With the template's 1K prefill chunks, snapshots retain chunk boundaries as well as
the complete prompt. Disk bundles survive requests and restarts.

| Setting | Value | Purpose |
| --- | --- | --- |
| `DSV41_PREFIX_CACHE` | `1` | Enable prefix reuse. |
| `DSV41_PREFIX_SNAPSHOTS` | `16` | Retain up to sixteen chunk-boundary snapshots in addition to the prompt boundary. |
| `DSV41_PREFIX_DISK` | `1` | Save and restore prefixes on disk. |
| `DSV41_PREFIX_DISK_GB` | `20` | Disk budget per node, with least-recently-used eviction. |
| `DSV41_PREFIX_DISK_STRICT` | `0` | Reuse historical prefixes even after expert selection changes. Set `1` to require the same selection. |
| `DSV41_PREFIX_RESPONSE` | `1` | Cache the generated answer after response delivery. Set `0` to disable; see limitations below. |

The default directory is `results/prefix-cache/rank-N`; override its parent with
`DSV41_PREFIX_DISK_DIR`. Both ranks must have a matching bundle. Code, model, and
numerical configuration changes can invalidate old entries.

Strict mode gives fewer hits but avoids reusing a prefix computed under a different
expert selection. Neither mode is a promise that every cached run is bitwise identical
to a fresh run. Token IDs and KV data can reveal prompt contents: keep these files local
and private. Vision prefixes match both token IDs and fingerprints of the processed
image pixels, layout, and position. Unchanged image history can be reused from RAM
or disk; a changed image invalidates snapshots after its start, while earlier
matching boundaries remain eligible. Image preprocessing still runs on each request,
but cached image spans do not run through the vision tower or prefill again.

With `DSV41_PREFIX_RESPONSE=1`, successful plain-text responses are flushed to the
client first, then the engine restores the input snapshot and prefills only the
answer tokens. The extended prefix stays in memory and is saved to disk when disk
caching is enabled. The next matching turn can skip the previous answer as well
as the previous input. This shifts work into the gap between requests; it does not
eliminate that work. An immediately arriving request can wait for preparation.

Thinking, tool-call, vision, and stop-string-truncated responses are skipped.
The next request must preserve the exact token prefix; edited history or different
chat serialization can prevent a match. The original input boundary is retained
as a fallback. Strict mode skips extension if adaptation changed the routing mask.
At concurrency=2, optional preparation is queued until both lanes are idle, and
already-queued inference requests take priority. Request timing excludes this work;
the server logs it separately as `prefix_response`. Restart both nodes to enable it.

## Measurements and limitations

The corrected TP path passed nesting depths 4/6/8/10 twice through the live API with
speculation and adaptation enabled. The second pass restored prefixes from disk.
That is a regression check, not a general quality guarantee.

Current bench suite (`bench/bench.py`, fixed output length, warm-up plus three
measured runs, medians). Configuration: TP2, native FP4, 90 GB arena per node,
keep 0.61, packed KV, `DSV41_BLOCK=3`, shared-expert overlap and native Engram
gather enabled:

| workload | prompt tokens | output tokens | prefill tok/s | decode tok/s | accept |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 62 | 512 | — | 26.3 | 3.10 |
| random 8K | 8,180 | 512 | 502 | 21.7 | 2.63 |
| prose | 45 | 512 | — | 16.0 | 1.94 |

The 8K prompt is fresh per run, so that prefill is uncached; the short prompts are
mostly fixed overhead. Acceptance varies on random text, so these are medians of a
wide spread.

On the development pair, an earlier 7,709-token README prompt with no prefix reuse
(before the shared-expert overlap and native Engram gather) measured:

| Run | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| First | 489 | 16.0 |
| Repeat 1 | 576 | 17.8 |
| Repeat 2 | 954 | 18.6 |

Configuration: TP2, native FP4, 90 GB arena per node, keep 0.61, speculation on,
frozen expert placement, 128 output tokens. These are three runs, not a matched
EP comparison. Other workloads and longer contexts can behave differently.
The measured prompt used the previous README, not this shortened version.

The newer draft/embedding sharding check used a frozen keep `0.60` mask, 88 GB
arena, packed KV, K=3, and a 768K allocation. On a 14,435-token prompt with 128
output tokens, all six runs (three per mode) produced identical output and mean
draft acceptance of 2.78. Allocation fell from 98.04 to 94.06 GiB per node.
Final warmed runs measured 19.55 versus 19.45 decode tok/s and 17.124 versus
17.321 seconds of prefill. This short test does not establish a speedup or noise
range. The subsequent 92 GB / keep `0.63` profile passed a single cold-request
smoke test, not a matched throughput comparison. See [the measurements](docs/tp-memory.md).

Packed KV is also a memory tradeoff: the controlled 384K-allocation test showed
about 3.5% longer prefill and 3.6% longer decode than the BF16-storage recheck.
See [packed KV measurements](docs/packed-kv.md) for the workload and full results.

The single-node engine path still exists (`WORLD_SIZE=1`, all TP flags off), but
the maintained launcher is for two nodes. Recent single-node serving has not been
revalidated; its adaptive-swap path currently needs a null-slot fix. EP2 also remains
available by disabling all six `DSV41_TP_*` enable flags, with memory and pruning
sized separately.

## Further reading

- [TP and persistent prefix implementation](docs/tp-and-persistent-prefix.md)
- [Draft and embedding TP memory savings](docs/tp-memory.md)
- [Packed KV storage and measurements](docs/packed-kv.md)
- [Post-response prefix preparation](docs/response-prefix.md)
- [Nesting regression: diagnosis, fix, and test results](docs/nesting-regression-tp.md)
- [Performance experiments and known issues](docs/gotchas.md)
- [API reference](server/README.md)
- [Benchmark harness](bench/README.md)
- [Historical results](RESULTS.md) and [design notes](NOTES.md) — include upstream single-node measurements, not just this fork

## Credits and license

The original engine is by 0xBakeer; see [CREDITS.md](CREDITS.md). This fork changes
the distributed execution, expert loading, and prefix caching. Repository code is
[MIT-licensed](LICENSE). Check the checkpoint's own license for model use.
