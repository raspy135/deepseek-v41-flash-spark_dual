# CUDA graphs for prefill feed-forward blocks

`DSV41_PREFILL_GRAPHS=1` captures the position-independent feed-forward section
of each transformer layer: hyper-connection mixing and normalization, routing,
routed/shared experts and their TP collectives, and the residual update. The
existing arithmetic is shared with the eager path through `Model._ffn`.
Attention, KV writes, compressed-context selection and Engram reads remain eager.
This is partial prefill graph coverage, not a whole-prompt graph.

The switch defaults to 0. It requires resident FP4 tensor parallelism, fixed
prefill routing and concurrency 1. Replica experiments and prefill phase-timing
instrumentation are incompatible. The feature and routing mode are compared by
the two-rank boot guard. No rank makes a local memory-based admission decision.
A capture failure is fatal; silently falling back on one rank could desynchronize
the pair's collectives.

Graph keys are `(rows, layer)`, independent of context position. The only captured
row counts are the configured prefill chunk size and model window size (2048 and
128 in this recipe). Other chunk sizes, image-bearing chunks, prefix-cache
reconstruction and ordinary decode use the eager path. The decoder-tail prefill
also benefits when it has the full 128 rows. Tensor diagnostic hooks are not
supported while graph replay is enabled.

A shared graph memory pool reuses temporary allocations. Each captured row count
has one set of staging buffers shared across layers, overwritten with each FFN
output after its input reads complete. Intermediate outputs are borrowed until
the next FFN; encoder and decoder final outputs are cloned so replay tails and
prefix snapshots survive later use. Staging still adds memory traffic and is
included in full-engine measurements. Warmup inputs are restored before the first
real replay, since warmup also overwrites them.
Graph capture warms the original FFN once, then restores every GPU demand counter
and host statistic changed by warmup/capture. Actual replay records demand once.
Masks, compact routing maps and expert arena slots are updated in place by the
existing adaptation code; captured graphs read those same addresses.

`engine.test_prefill_graphs` tests admission rules plus actual CUDA capture with
changed inputs, shared pools, output lifetime, demand counter restoration and
in-place weight changes. The distributed gate `tools/bench_prefill_graphs_tp.py`
compares final prefill logits, output tokens and prefill demand counts on both
ranks, across different prompts and after a real expert swap. It measures eager /
graph / graph / eager after warming both paths. Capture cost and graph replay
counts are reported separately in `prefill_graphs` statistics.

## First prototype: correct but slower

The initial implementation used separate input/output staging buffers and cloned
every layer's output. Both ranks passed exact final-logit, output-token and demand
counter comparisons, including an expert swap after capture. It captured 40 graphs
in 6.86 seconds on rank 0. This capture cost is separate from warmed timings.

For the 2191-token prompt, eager/graph/graph/eager prefill took
2.186 / 5.847 / 2.289 / 2.236 seconds. The first warmed graph run had a large
outlier despite no new captures; do not silently discard it or attribute it to
staging copies without a device trace. Median throughput was 991.00 vs
665.94 tok/s (-32.8%). For the 6031-token prompt, the same sequence took
5.410 / 5.635 / 5.516 / 5.384 seconds: median throughput 1117.49 vs
1081.84 tok/s (-3.2%). The first prototype was not enabled for serving.

Evidence: `results/prefill-graphs-20261003/`. The second prototype reuses input
buffers for outputs and clones only the final encoder/decoder layer outputs,
to test whether reducing the extra copies changes the result.

## Reduced-copy prototype: no consistent speedup

The second implementation also passed exact logits, output tokens and demand
counters on both ranks, including replay after an expert swap. Capturing 40 graphs
took 4.69 seconds on rank 0. Peak allocated memory reached 103.21 GB during the
run; this is a cumulative peak, not a measurement of steady graph overhead.

Warmed eager/graph/graph/eager prefill times were:

| Prompt | Eager | Graph | Graph | Eager |
|---|---:|---:|---:|---:|
| 2191 tokens | 2.321 s | 2.253 s | 2.249 s | 2.234 s |
| 6031 tokens | 5.966 s | 7.301 s | 5.473 s | 5.409 s |

Median per-run throughput was 962.40 vs 973.50 tok/s (+1.2%) for the
shorter prompt and 1062.95 vs 963.98 tok/s (-9.3%) for the longer prompt.
The slower graph run again had no new captures. These two warmed samples per
condition do not establish a reliable gain, and the capture cost worsens initial
latency. Keep `DSV41_PREFILL_GRAPHS=0` for serving. The implementation remains
experimental for future profiling; fewer launches alone did not deliver the
expected improvement. Evidence: `results/prefill-graphs-v2-20261003/`.

The live service was restored to the previously qualified `confidence-depth`
image, which does not contain this experiment. Enabling it requires building the
updated source into the same image on both nodes as well as setting the flag.

The later `engram-cache` image includes this code on both nodes, still disabled.
With that image selected, set `DSV41_PREFILL_GRAPHS=1` in the head `.env` and
restart both nodes with `scripts/dual-down.sh` and `scripts/dual-up.sh` to opt in.
Set the flag back to `0` and restart to disable it. The measurements above remain
the reason it is off in the normal service.
