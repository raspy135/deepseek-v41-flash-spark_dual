# DeepSeek-V4.1-Flash on two DGX Sparks

A fork of [0xBakeer/deepseek-v41-flash-spark](https://github.com/0xBakeer/deepseek-v41-flash-spark),
using a custom PyTorch/Triton engine, two-node tensor parallelism, adaptive expert
loading, and persistent prefix caches. Routed experts use the checkpoint's native
MXFP4 weights, without another quantization step.

The model does not fit entirely in two Sparks. The usual configuration keeps about
61% of routed experts resident and changes that selection as demand changes.
Pruning is a quality/speed tradeoff: adaptation improves coverage, but does not make
a pruned request equivalent to running every expert.

The server provides an OpenAI-compatible API, streaming, tool calls, thinking mode,
vision, and DSpark speculative decoding. It processes one request at a time.

## Setup

You need two DGX Sparks, a working RoCE link, Docker with NVIDIA GPU support, and
at least 600 GB of local NVMe space per node for the checkpoint. Each node reads
its own copy of the weights; do not put them on NFS. The development pair uses a
200 GbE direct link.

Clone this repository to the same absolute path on both nodes. The head must be
able to SSH to the worker without a password prompt.

```bash
cp env.example .env
```

Set `MODEL_DIR`, `PEER` (`user@worker-ip`), `MASTER_ADDR` (the head's link IP), and
`NCCL_SOCKET_IFNAME` (the link interface). Then apply the TP profile below.
`env.example` still includes an older EP profile; the last assignment to a variable
wins, so replace its settings or put your overrides at the end of `.env`.

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

These are the settings used for the recent TP quality checks, not the engine's
fallback defaults. Arena sizes are decimal GB **per node**.

```dotenv
MAX_SEQ=262144
ARENA_GB=90
PRUNE_KEEP=0.61
EXPERT_FORMAT=fp4
TRANSIENT_SLOTS=16
KEEP_FREE_GB=6
SPEC=1

DSV41_TP_EXPERTS=1
DSV41_TP_DENSE=1
DSV41_TP_ATTN=1
DSV41_TP_HEAD=1
DSV41_TP_EXPERT_LAYOUT=output
DSV41_TP_LINEAR_LAYOUT=output

DSV41_DENSE_FP4=off
DSV41_FP4_DOT_SCALED=0
DSV41_HEAD_FMT=bf16
DSV41_ACT_QUANT=0
```

TP splits resident expert weights, shared experts, attention projections, and the
vocabulary head across the pair. KV caches and some other weights remain replicated.
The `output` layouts preserve complete down-projection dot products; the older
`intermediate` layouts regressed nesting quality. Leave the precision settings above
alone unless you are testing a numerical change.

The main capacity and speed controls:

| Setting | What to change it for |
| --- | --- |
| `MAX_SEQ` | Total context allocation, including the answer. More context needs more cache and scratch memory. |
| `ARENA_GB` / `PRUNE_KEEP` | More resident experts, at the cost of memory. Raise them together only when there is room. |
| `SPEC=1` | Enable speculative decoding. Speed depends on how many draft tokens are accepted. |
| `DSV41_PREFILL_CHUNK=2048` | Prefill chunk size. Larger chunks can reduce dispatch overhead but use more scratch memory. |
| `DSV41_PREFILL_FUSED_ATTN=1` | Keep fused prefill attention enabled. |
| `DSV41_ENGRAM_ROW_SPLIT=1` | Split large Engram row reads across the two nodes. |
| `DSV41_VISION=1` | Load image support. Set to `0` for text-only serving. |

At keep `0.61`, the engine selects 9,400 of 15,360 routed experts. Each TP shard is
9.40032 MB, so their weights occupy about 88.36 GB per node, within the 90 GB arena.
The remaining memory must cover dense weights, the drafter, caches, and scratch.
The engine refuses a pruned configuration whose selected experts cannot all fit.

For an unpruned control, use `PRUNE_KEEP=1.0` and `TRANSIENT_SLOTS=384`. Experts outside
the arena are then streamed from NVMe. This is much slower; use short quality probes
before attempting a long benchmark.

### Context length

The profile allocates 256K tokens; that is not a claim that full-length quality has
been validated. `MAX_SEQ=524288` is a candidate for 512K. The checkpoint declares a
1M-token maximum, but allocating the cache is not enough to establish correctness or
speed at that length. A starting 512K budget is `ARENA_GB=88` with `PRUNE_KEEP=0.60`;
it still needs a full-length test.

Command-line overrides leave `.env` unchanged. Stop the existing pair before restarting:

```bash
bash scripts/dual-down.sh
MAX_SEQ=524288 ARENA_GB=88 PRUNE_KEEP=0.60 bash scripts/dual-up.sh
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
With the default 2K prefill chunks, snapshots retain chunk boundaries as well as
the complete prompt. Disk bundles survive requests and restarts.

| Setting | Value | Purpose |
| --- | --- | --- |
| `DSV41_PREFIX_CACHE` | `1` | Enable prefix reuse. |
| `DSV41_PREFIX_SNAPSHOTS` | `8` | Retain up to eight chunk-boundary snapshots in addition to the prompt boundary. |
| `DSV41_PREFIX_DISK` | `1` | Save and restore prefixes on disk. |
| `DSV41_PREFIX_DISK_GB` | `20` | Disk budget per node, with least-recently-used eviction. |
| `DSV41_PREFIX_DISK_STRICT` | `0` | Reuse historical prefixes even after expert selection changes. Set `1` to require the same selection. |

The default directory is `results/prefix-cache/rank-N`; override its parent with
`DSV41_PREFIX_DISK_DIR`. Both ranks must have a matching bundle. Code, model, and
numerical configuration changes can invalidate old entries.

Strict mode gives fewer hits but avoids reusing a prefix computed under a different
expert selection. Neither mode is a promise that every cached run is bitwise identical
to a fresh run. Token IDs and KV data can reveal prompt contents: keep these files local
and private. Vision requests currently bypass disk persistence.

## API and diagnostics

Use `http://<head>:8000/v1` as the OpenAI base URL and the name in
`SERVED_MODEL_NAME` as the model. Thinking defaults to off; send
`"enable_thinking": true` to enable it for a request. `DEFAULT_THINKING` and
`DEFAULT_EFFORT` set server defaults. See [the API guide](server/README.md) for tools,
streaming, and effort mappings.

There is no API authentication. Leave `HOST=127.0.0.1` for local access. For remote
clients, bind a trusted interface or use an authenticated reverse proxy.

Completions include `x_engine_stats`: prefill/decode rates, prefix hits, draft acceptance,
and pruning misses. Compare uncached prefill with uncached prefill; a cache-hit rate
is not the speed of processing new tokens.

For a reproducible input capture, start with `DSV41_CAPTURE_NEXT=1`. The next request
is saved to `results/captured_request.pt`, then capture disarms. It overwrites that
filename and includes recoverable prompt data, so use it deliberately and do not commit it.
Leave detailed timing and route-stat instrumentation off for normal serving.

## Measurements and limitations

The corrected TP path passed nesting depths 4/6/8/10 twice through the live API with
speculation and adaptation enabled. The second pass restored prefixes from disk.
That is a regression check, not a general quality guarantee.

On the development pair, a 7,709-token README prompt with no prefix reuse measured:

| Run | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| First | 489 | 16.0 |
| Repeat 1 | 576 | 17.8 |
| Repeat 2 | 954 | 18.6 |

Configuration: TP2, native FP4, 90 GB arena per node, keep 0.61, speculation on,
frozen expert placement, 128 output tokens. These are three runs, not a matched
EP comparison. Other workloads and longer contexts can behave differently.
The measured prompt used the previous README, not this shortened version.

The single-node engine path still exists (`WORLD_SIZE=1`, all TP flags off), but
the maintained launcher is for two nodes. Recent single-node serving has not been
revalidated; its adaptive-swap path currently needs a null-slot fix. EP2 also remains
available by disabling the four TP flags, with memory and pruning sized separately.

## Further reading

- [TP and persistent prefix implementation](docs/tp-and-persistent-prefix.md)
- [Nesting regression: diagnosis, fix, and test results](docs/nesting-regression-tp.md)
- [Performance experiments and known issues](docs/gotchas.md)
- [API reference](server/README.md)
- [Benchmark harness](bench/README.md)
- [Historical results](RESULTS.md) and [design notes](NOTES.md) — include upstream single-node measurements, not just this fork

## Credits and license

The original engine is by 0xBakeer; see [CREDITS.md](CREDITS.md). This fork changes
the distributed execution, expert loading, and prefix caching. Repository code is
[MIT-licensed](LICENSE). Check the checkpoint's own license for model use.
