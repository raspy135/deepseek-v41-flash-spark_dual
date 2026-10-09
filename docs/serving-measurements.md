# Historical serving measurements

These measurements retain their original configurations. They are not claims about
the current `.env.example.fp4` / `.env.example.exl3` profiles; newer experiments are recorded in
[RESULTS.md](../RESULTS.md).


### Decode, 2026-10-01

Served configuration after this round (TP2, native FP4, dense FP4 `attn,wo_a`, dynamic depth 3/5,
RoCE all-gathers, L2 prefetch, 90.1 GB arena, keep 0.61); each change A/B'd in alternating arms.

| change | effect | bits |
| --- | --- | :---: |
| decode-lean rounds 1-2 | 8,457 → 6,385 kernels, +5.8-8.7% | identical |
| round 3 + v2 expert kernel | 3,557 kernels, +2.4-2.9% | identical |
| dense FP4 `attn,wo_a` + split-K | dense 30.6 → 24.6 ms/step, step −7% | output-changing |
| RoCE all-gather | −5.5 ms/step, +6.2% tok/s | identical |
| L2 prefetch 2 MB | −0.5 to −1.1 ms/step | identical |
| fp32 HC kernel (off) | −1.9 to −2.9 ms/step, acceptance −0.07..−0.10, tok/s a wash | output-changing |

| live (`bench.py`, warm-up + 3 runs) | prefill tok/s | decode tok/s | accept |
| --- | ---: | ---: | ---: |
| code | — | 37.34 | 3.72 |
| prose | — | 23.12 | 1.98 |
| random 8K | 508 | 25.57 | 2.70 |

Limits: experts are ~48 ms of the ~98 ms step at ~208 of ~235 GB/s; random-8K prefill (508) is below
2026-09-22's 611; thinking, `DSV41_MAX_CONCURRENCY > 1` and long context are unmeasured.

### Decode, 2026-09-23

Direct-engine runs through the two-node test gate (not `bench/bench.py`, no HTTP).
Setup: TP2, native FP4 with CUDA decode, shared-expert overlap, native Engram gather,
90 GB arena, keep 0.61, frozen expert ranking, prefix caches off, greedy, up to 512
output tokens. The arena and keep differ from the profile above, so compare within
this section, not with older rows. All columns come from one process: the "fixed"
columns are the dynamic build pinned to one depth (it still drafts 5). Every greedy
output was token-identical across depths on both nodes.

| workload | fixed depth 3 | fixed depth 5 | **dynamic 3,5** | tokens per step at 3 → 5 |
| --- | ---: | ---: | ---: | --- |
| Python module | 35.5 tok/s | 44.5 | **43.1** | 3.87 → 5.65 |
| HTML page | 31.0 | 35.0 | **34.3** | 3.36 → 4.29 |
| Prose explanation | 20.4 | 18.2 | **20.6** | 2.18 → 2.31 |
| Short story | 19.1 | 16.8 | **19.1** | 1.99 → 2.07 |
| Story, temperature 0.7 | 18.1 | — | 19.4 | both ran at depth 3; the gap is noise |

- **Why dynamic wins:** depth 5 makes each verify step about 16% slower (~106 → ~124
  ms), and pays that back only where the drafter is right. Dynamic depth switched to 5
  after the first ~60 tokens on code and HTML, and stayed at 3 on prose, costing
  nothing there. Separate fixed-depth runs in 3, 5, 5, 3 order, made before the
  projection defaults below were switched on, agreed within 4%.
- **Depth 1:** the step fell to 88 ms, but it lost on every workload (story 17.6,
  Python 22.4 tok/s).
- **Memory:** peak allocation 103.15 GB with both depths, versus 103.14 GB with one.
- **Not yet measured:** thinking-mode traces, long contexts, and requests that
  alternate between code and prose.

The decode projection defaults (merged `wq_a`‖`wkv` and shared `w1`‖`w3`, occupancy-sized
FP8 tiles, fused prune-miss accounting) were A/B-tested in the same process. Logits,
hidden states and all generated tokens were bit-identical. Four alternating batches of
fixed verify steps took 91.3–91.7 ms without them and 87.3–87.9 ms with them, about
3.8 ms (4.1%) less per step, on a 512-token HTML generation. Details:
[decode projection fusion](decode-projection-fusion.md) and
[dynamic depth](decode-dynamic-depth.md).

### Earlier measurements

The corrected TP path passed nesting depths 4/6/8/10 twice through the live API with
speculation and adaptation enabled. The second pass restored prefixes from disk.
That is a regression check, not a general quality guarantee.

Latest native-CUDA decode recheck (`bench/bench.py`, fixed 512-token output, one
warm-up plus three measured runs) on 2026-09-22:

| workload | prompt tokens | output tokens | decode tok/s | accept | expert hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 62 | 512 | 24.23 | 2.97 | 100% |

Configuration: TP2, native FP4, native CUDA BM=16 decode with relaxed reduction,
90.1 GB arena per node, keep 0.61, packed KV, speculation enabled. The three
measured runs were 24.23, 25.12, and 23.37 tok/s; acceptance varied from 2.81 to
3.04. The immediately preceding run on an older Triton-only container measured
24.41 tok/s, but rebuilding changed more than the MoE kernel, so this is not a
controlled CUDA-versus-Triton comparison. Raw rows are in
[`results/readme-cuda-rebuilt.json`](../results/readme-cuda-rebuilt.json) and
[`results/readme-cuda.json`](../results/readme-cuda.json).

The earlier broader bench suite used the same harness (fixed output length,
warm-up plus three measured runs, medians). Configuration: TP2, native FP4, 90.1 GB arena per node,
keep 0.62, packed KV, `DSV41_BLOCK=3`, shared-expert overlap and native Engram
gather enabled:

| workload | prompt tokens | output tokens | prefill tok/s | decode tok/s | accept |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 62 | 512 | — | 26.3 | 3.10 |
| random 8K | 8,180 | 512 | 611 | 23.9 | 2.97 |
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
smoke test, not a matched throughput comparison. See [the measurements](tp-memory.md).

Packed KV is also a memory tradeoff: the controlled 384K-allocation test showed
about 3.5% longer prefill and 3.6% longer decode than the BF16-storage recheck.
See [packed KV measurements](packed-kv.md) for the workload and full results.

The single-node engine path still exists (`WORLD_SIZE=1`, all TP flags off), but
the maintained launcher is for two nodes. Recent single-node serving has not been
revalidated; its adaptive-swap path currently needs a null-slot fix. EP2 also remains
available by disabling all six `DSV41_TP_*` enable flags, with memory and pruning
sized separately.
