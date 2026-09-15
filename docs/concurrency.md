# Two-request serving

Implemented on `codex/concurrent-tp-serving`. Default serving remains one request.

The implementation uses two independent request lanes over the same weights and
expert arena. A single scheduler thread owns CUDA and the TP command stream.
Requests keep separate KV caches, Engram history, speculative rollback state,
sampling RNG, and prefix snapshots. The expensive expert operations combine the
two verify blocks; attention remains lane-local.

Prefill is serialized initially. A new prompt can therefore pause an existing
decode while it prefills. Expert adaptation also pauses both lanes at a scheduler
boundary. There is one demand database and one disk-prefix writer/budget per rank.
This is not paged attention or arbitrary-sized continuous batching.

`DSV41_MAX_CONCURRENCY=2` is the experimental switch. It requires TP, speculative
decoding, and a fully resident pruned expert set. Each lane allocates its own
context cache at `MAX_SEQ`; enabling it needs additional memory. The default `1`
does not construct the scheduler or allocate a second lane. Prefill replicas are
not supported with concurrent serving.

## Measurements

TP2, native MXFP4, keep 0.60, arena 88 GB, context allocation 8192, identical
six-token verify blocks, five timed iterations, September 15:

| Run | Two serial steps | Batched pair | Throughput ratio |
| --- | ---: | ---: | ---: |
| First corrected gate | 359.55 ms | 205.42 ms | 1.75× |
| Repeat load | 271.67 ms | 206.78 ms | 1.31× |
| Cold-state fix | 268.00 ms | 212.13 ms | 1.26× |
| Canonical prompt gate | 261.13 ms | 209.25 ms | 1.25× |
| Final isolation gate (image transfer in background) | 292.76 ms | 271.21 ms | 1.08× |

Both lanes matched the serial logits exactly in both runs. The serial timings
varied substantially; these are feasibility measurements, not an HTTP throughput
claim. Identical prompts maximize expert reuse, so different prompts can gain less.

The pre-existing batch prototype failed its first TP gate: relative logit error
1.03, only one of six argmax tokens matching. It performed an EP reduction after
TP had already reconstructed the expert result, doubling it, and rounded before
adding the shared expert. The corrected path follows `FastDecoder._layer_b`.

Independent-prompt testing then caught cold-capture state hazards. The greedy
verifier held a view of the drafter's output, which capture warm-up overwrote;
acceptance now compares against the actual verified block. Compressor rollback
state can also be a view into scratch overwritten during warm-up, so it is saved
across capture. Both lanes subsequently reproduced serial greedy outputs exactly.

At this frozen 0.60 expert set, the canonical depth-8 nesting probe produced one
extra closing brace in **both** serial and paired runs (score 0). Token equivalence
is not a claim that the baseline quality is perfect. The gate records that result
and checks that batching does not change it; production adaptation is tested separately.

Validation driver: `engine/profile_batch2.py --requests`. In addition to the
kernel check it compares independent greedy prompts, seeded sampling, cancellation,
and lane reuse against serial generation. Synthetic prompts only; the driver
disables adaptation and disk caches and does not write the production demand DB.

The complete isolation gate passed: serial vs paired greedy token equality for
two different prompts, request-local seeded sampling, cancellation of lane zero
without changing lane one, and reuse of both lanes. Cold and warmed lanes were
checked. This does not establish universal bitwise equivalence for every prompt.

## HTTP serving check

Both nodes ran the same image with `DSV41_MAX_CONCURRENCY=2`, context allocation
393216, arena 88 GB, keep 0.60, adaptation enabled, and the shared 20 GB/rank disk
cache. Two different short prompts (English B-tree explanation and Japanese binary
search), 96 generated tokens each:

| Mode | Total wall time | Aggregate output tok/s |
| --- | ---: | ---: |
| Sequential | 13.45 s | 14.27 |
| First paired run, cold paired graphs | 14.93 s | 12.86 |
| Second paired run | 11.27 s | 17.04 |

The warmed pair gained 19.4% aggregate throughput, while each individual request
was slower than running alone. The sequential requests reported 15.06 and 16.14
decode tok/s. Adaptation was active, so these are serving observations, not a
fixed-weight kernel comparison. Disk-prefix restores and expert swaps were
observed on both ranks. A streamed request was disconnected after its first content;
the other request completed and the server remained healthy.

The switch batches **two separate HTTP requests**; it does not implement `n=2`
completions within a request. Additional requests queue (up to 32 waiting).

Two concurrent long requests (15,168 and 15,172 tokens, 16 output tokens each)
also completed with adaptation enabled and 15-boundary prefix bundles saved.
Cold prefill took 37.32 and 35.82 seconds (406 and 424 tok/s); combined request
latency was about 79 seconds because prefill is serialized. This is a functional
long-context check, not a prefill improvement. Host memory available was about
6.7 GiB on the head and 9.4 GiB on the worker near the end of the run.
Repeating the pair restored all 15,168/15,172 prefix tokens: reported prefill time
was 0.394/0.728 seconds and both requests finished within 5.73 seconds.

The live test used a launch-time override, leaving the personal `.env` unchanged:

```bash
IMAGE=deepseek-v41-flash-spark:concurrency-test DSV41_MAX_CONCURRENCY=2 bash scripts/dual-up.sh
```

For a normal build, set `DSV41_MAX_CONCURRENCY=2` in `.env`, build/sync the image,
and restart both ranks. Set it back to `1` to retain the single-request scheduler
and avoid allocating the second lane. Performance and adaptation checks above
used `DSV41_PRUNE_UNIT=request`.
