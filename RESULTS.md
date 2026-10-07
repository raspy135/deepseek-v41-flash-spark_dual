# RESULTS — DeepSeek-V4.1-Flash on one DGX Spark (GB10, 121 GiB)

> **This file is append-only history.** Each tag has its own section with date, time and the exact
> configuration. Superseded numbers stay in place and are annotated; nothing is deleted.
> Sections: [v0.1.0-wip (2026-09-10)](#v010-wip--2026-09-10) · [v0.2.0-wip (2026-09-11)](#v020-wip--2026-09-11)

---

## v0.1.0-wip — 2026-09-10

> **Superseded (annotated 2026-09-11 09:20):** every number in this section was measured with a
> bug in the ported model math (`v41_ref.hc_post` mixed the Hyper-Connection residual with the
> transposed matrix). The engine ran, the numbers are what it did that day, but the model quality
> behind them was wrong (teacher-forced coding loss 2.16 nats instead of 1.37) and the DSpark
> acceptance was depressed (~2.4-3.0 instead of 3.0-3.75). See v0.2.0-wip below for the corrected
> state; NOTES.md ("2026-09-11 00:50") has the bug hunt.

**Measured 2026-09-10 on the box described below. Every number here was produced by a run on this
machine; nothing is extrapolated, scaled or quoted from elsewhere.** Where a planned measurement
was not taken it says so instead of guessing. The running log with the bug hunt behind these
numbers is NOTES.md ("Bring-up"); what still does not work is LIMITATIONS.md.

> **Status: work in progress.** Steps 1-5 of the bring-up (smoke, correctness, DSpark, engine/server
> API, serve) are complete and measured. Step 6 (the benchmark sweep) was stopped after
> the `code` row because the box was needed for interactive use, so **`prose`, the `angry-birds`/`mario`
> one-shots and the thinking-on run are not measured** and `results/oneshots/` is empty.

## Box and build

| | |
|---|---|
| machine | ASUS Ascent GX10, NVIDIA GB10 (sm_121a), 128 GB unified / 121 GiB visible, 20 cores, 1 NVMe (916 GB) |
| OS / driver | Ubuntu 24.04 (DGX OS base), driver 580.173.02, CUDA 13 |
| python | a venv with torch 2.13.0+cu130, triton 3.7.1, transformers 5.12.1 (docs/install.md) |
| model | `deepseek-ai/DeepSeek-V4.1-Flash`, full 48-shard checkpoint (510 GB) on local NVMe, FP4 routed experts read straight out of the shards |
| engine | this repo: `server/app.py --engine v41` -> `engine/v41_engine.py`, MoE on the Triton FP4 kernel `tools/fp4_moe.py` (`kernel: triton-fp4`) |
| other load | none — the box's other inference container was stopped for the whole of these runs, so the unified pool was ours alone |

The engine streams routed experts: only a resident hot set lives in the GPU arena and every miss is
an O_DIRECT read from the checkpoint. **A speed number from this recipe is meaningless without the
expert hit rate and the GB read that produced it**, so every table below carries them.

## 1. Load

Auto-sized arena, warm start ranked by `results/trace-full-20260910/stats/coverage.json`.

| stage | `max_seq` 8192 | `max_seq` 32768 (the served config) |
|---|---|---|
| non-expert weights (~19 GB) to GPU | 63 s | 63 s |
| DSpark experts (3 x 128 = 7.2 GB) resident | 5 s | 5 s |
| warm start | 3,891 experts / 73.2 GB in **16 s** (4.6 GB/s) | 3,526 experts / 66.3 GB in **14 s** (4.7 GB/s) |
| **total, process start to `ready` / `/health`** | ~85 s | **~90 s** |

Arena at `max_seq 32768`: **73.8 GB = 3,926 slots = 25.6 % of the 15,360 routed experts**
(3,526 LRU + 400 transient). Peak host use 99-101 GiB of 121; MemAvailable never below 19 GiB.

## 2. Correctness — teacher-forced NLL / top-1 vs the pure-torch reference port

`engine/v41_engine.py --teacher-forced corpus/trace_corpus.jsonl --act-quant`: 50 sequences,
each pushed through `Model.forward` in one chunk, next-token NLL and top-1 from the real head.
`--act-quant` matches the tracer's fp8 activation fake-quant (`results/trace-full-20260910/meta.json`);
serving runs with it off, which is strictly more precision.

Config: arena 80.7 GB / 4,291 slots (27.9 % resident), max_seq 8192, spec off, kernel triton-fp4,
act_quant on. 965 s of forward time.
Raw: `results/engine-tf-20260910/teacher_forced_engine_actquant.json`.

| corpus | tokens | reference (`tools/v41_ref.py`) NLL | **engine NLL** | delta | reference top-1 | **engine top-1** |
|---|---|---|---|---|---|---|
| coding | 5,459 | 2.1527 | **2.1586** | **+0.0059** | 0.6384 | **0.6410** |
| general | 5,251 | 3.4124 | **3.4380** | **+0.0256** | 0.4738 | **0.4769** |

Both well inside the ±0.05 nats bar. The serving engine's math — expert arena, Triton FP4 grouped
MoE, engram rows read off NVMe, fixed-tile GEMMs — agrees with the pure-torch port end to end.

## 3. DSpark speculative decoding

`--spec-ab`: one load, `spec` toggled between runs, so all three runs share the arena and the LRU
state. Config: **arena 74.5 GB / 3,960 slots (25.8 % resident), max_seq 8192, kernel triton-fp4,
act_quant off, thinking off**, prompt 16 tokens, 64 output tokens.
Raw: `results/engine-tf-20260910/spec_ab2.json`.

| run | temperature | decode tok/s | steps | accept_len_mean | expert hit rate | NVMe GB | attn s | moe s |
|---|---|---|---|---|---|---|---|---|
| greedy, spec **off** | 0 | 1.75 | 63 | — | 0.826 | 70.4 | 2.86 | 31.42 |
| greedy, spec **on** | 0 | **2.64** | 17 | **3.71** | 0.771 | 86.6 | 1.02 | 24.69 |
| sampled, spec **on** | 1.0 / top_p 0.95 | **2.64** | 20 | **3.40** | 0.794 | 92.0 | 1.21 | 25.55 |

**Greedy speculative output is token-for-token identical to greedy autoregressive output: 64 of 64,
first divergence `None`.** The verify loop is lossless as implemented. The sampled run is coherent.

DSpark is worth **1.5x** here: a 6-token verify block reads more expert bytes than a single token
does (86.6 vs 70.4 GB for the same 64 tokens) but amortises them over 3.7 accepted tokens.

## 4. Served throughput — `code` workload

Server: `./start.sh` with `.env` = `MAX_SEQ=32768`, `DEFAULT_THINKING=off`, `SPEC=1`,
`TRACE_STATS=results/trace-full-20260910/stats/coverage.json`, `ARENA_GB` auto -> **73.8 GB /
3,926 slots / 25.6 % resident, kernel triton-fp4, act_quant off**.
Bench: `bench/bench.py --workload code --runs 2 --osl 512 --ignore-eos`, thinking **off**,
temperature 0.6, top_p 0.95, 1 warm-up + 2 measured runs, **every run exactly 512 completion
tokens** (`finish_reason: length`). Raw: `results/bench-20260910/code.json`.

| run | TTFT | TPOT | decode tok/s | accept_len | expert hit rate | NVMe GB | engram rows |
|---|---|---|---|---|---|---|---|
| warm-up | 12.48 s | 338 ms | 2.96 | 3.47 | 0.820 | 496.7 | 45,440 |
| run 1 | 10.98 s | 369 ms | 2.71 | 3.02 | 0.834 | 517.6 | 51,792 |
| run 2 | 11.12 s | 379 ms | 2.64 | 3.04 | 0.827 | 543.1 | 51,408 |
| **median of the 2 measured runs** | **11.05 s** | **374 ms** | **2.68** | **3.03** | **0.830** | **530.3** | **51,600** |

Where the time goes (run 1): prefill 62 tokens in 10.82 s — 2,473 prefill expert misses = 46 GB at
4.3 GB/s; decode 514 tokens in 188.3 s over 170 DSpark steps — 25,044 decode expert misses = 471 GB
at **2.5 GB/s effective**, `moe_s` 160.5 s, `attn_s` 9.1 s, `engram_s` 3.6 s.

**The headline of this recipe: 0.92 GB of expert weights are streamed from NVMe per generated
token** at a 25.6 % resident set. Attention, the engram lookups and the Triton MoE kernel together
are under 8 % of the decode time; everything else is the SSD.

### Not measured

| planned row | status |
|---|---|
| `prose`, `--runs 2 --osl 512 --ignore-eos` | **not measured** — not run in this tag (box needed for interactive use) |
| `angry-birds` one-shot, thinking off, 8192 max output | **not measured** — not run in this tag (box needed for interactive use) |
| `mario` one-shot, thinking off, 8192 max output | **not measured** — not run in this tag (box needed for interactive use) |
| `angry-birds` one-shot, thinking on, effort 75 | **not measured** — not run in this tag (box needed for interactive use) |
| `results/oneshots/*.html` | **empty** — no one-shot completed |
| long-context (`random --isl 8192`) | never attempted in this tag |

## 5. Performance work done during bring-up (A/B, same box, same work)

Two bugs outside the model math dominated the first runs. Both A/Bs are honest in the way that
matters here: greedy decoding at a fixed arena size reads the *same* expert bytes before and after,
so only the time changed.

| measurement | before | after |
|---|---|---|
| expert read, **1** in flight (the large-arena decode regime) | 8.11 ms/expert, 2.32 GB/s | **4.78 ms/expert, 3.93 GB/s** |
| expert read, 12 in flight | 4.02 ms/expert, 4.68 GB/s | 3.95 ms/expert, 4.76 GB/s |
| decode, 20 GB arena, 64 greedy tokens, no spec (152.05 GB read both times) | 0.93 tok/s | **1.21 tok/s** |
| decode, ~75 GB arena, 64 greedy tokens, no spec (~70 GB read both times) | 0.76 tok/s | **1.75 tok/s** |
| decode, ~75 GB arena, 64 greedy tokens, DSpark on | 1.74 tok/s | **2.64 tok/s** |

* **The LM head was converted bf16 -> fp32 on every token** — a 2.65 GB allocation per token
  (`head` is [129280, 5120]), plus 132 MB per drafted token for the Markov head. Next to a 74 GB
  arena that pushes the caching allocator into `cudaFree`/`cudaMalloc`. Both are stored fp32 once
  at load now (+1.33 GB and +66 MB resident), which is also what the reference does.
* **The expert reader issued six O_DIRECT reads per expert and synchronised the compute stream six
  times per miss.** An expert is now read as its **two** maximal contiguous file runs (a 1.1 MB
  scale run and a 17.7 MB weight run — the shards group all scales at the front and all weights
  behind them), verified byte-exact against `safetensors.safe_open`; and the pinned staging buffer
  goes straight into the arena with `non_blocking=True` on a **per-io-thread CUDA stream** that
  first waits on the compute stream.

## 6. Reference points (not our measurements)

For scale only — different hardware, all experts resident, no streaming: a public **4x** DGX Spark
TP4 vLLM build reports 39-77 tok/s single stream, TTFT 0.27-0.58 s, DSpark acceptance 3.56
(NOTES.md 0.5). That build needs four boxes and states "TP2 does not fit either way". This repo runs the same model, at FP4 expert quality, on **one** box, at 2.6-2.7 tok/s.


---

## v0.2.0-wip — 2026-09-11

Measured 2026-09-11 00:50-09:15 on the same box (Qwen container stopped, pool ours alone), same
checkpoint. Commits `bd24743` (hc_post fix) .. `22bd9a8`+ (FP8 dense, pruning, CB3). Python venv as
in v0.1.0-wip. Every row below is one run of the stated command; no benchmark sweeps were run
(this recipe records a single decode number per configuration).

### 2.1 The bug and what it changed (2026-09-11 00:50, commit bd24743)

`tools/v41_ref.py::hc_post` summed the 4x4 Hyper-Connection `comb` matrix over the wrong index
(comb @ residual instead of the reference's combᵀ @ residual). Found by proving decode == single-chunk
prefill bit-for-bit at every layer (so caches were innocent) and re-reading the reference line by line.

| teacher-forced, trace corpus (engine, one chunk per sequence) | before fix | after fix |
|---|---|---|
| coding NLL / top-1 (5,459 tokens) | 2.1599 / 63.9 % | **1.3708 / 74.4 %** |
| general NLL / top-1 (5,251 tokens) | 3.4263 / 47.1 % | **2.8635 / 55.1 %** |

Same code prompt, greedy: before the fix every path stuttered ("LRLR", "time-to-llive"); after it,
clean production-quality code. DSpark acceptance length on that prompt 2.4 -> 3.75.

### 2.2 Decode paths (2026-09-11 00:10-07:05)

`engine/fastdecode.py`: CUDA graphs per layer (attention+HC+router graph, host slot resolve, MoE+residual
graph), fused Sinkhorn Triton kernel, bf16 head, fixed-length masked indexer scoring.
`tools/fp8_linear.py`: dense projections read in their stored FP8 form (Triton, 223 GB/s of FP8 at
M=6, 1.9x the bf16 GEMM); the bf16 copies are gone, which grew the auto arena from 74 to 79 GB.

| verify step (6 tokens), everything resident | wall |
|---|---|
| reference path (`Model.forward`), 2026-09-10 23:5x | 436 ms |
| fast path, bf16 dense (00:10) | 183 ms + 16 ms draft |
| fast path, FP8 dense (07:00) | **173 ms + 15 ms draft** |

Greedy argmax agreement fast vs reference path: 100 % on the tested positions; hidden states differ
2-5 % from bf16 GEMM noise amplified by near-tie router flips (documented in fastdecode.py).

### 2.3 Speed ladder (greedy, temperature 0, same 40-token code prompt, 160-200 output tokens, DSpark on, fast path, FP8 dense)

| configuration (all 2026-09-11) | resident experts | decode tok/s | accept len | hit rate | NVMe GB / request |
|---|---|---|---|---|---|
| unpruned, streaming, arena 79 GB (07:01) | 27 % | 3.5 | 3.24 | 0.826 | 208 |
| keep 40 % (07:06) | 68 % of kept | 6.4 | 3.02 | 0.943 | 78 |
| keep 30 % (07:04) | 91 % of kept | 9.5 | 2.76 | 0.986 | 23 |
| **keep 31 %, arena 90.5 GB = 4,813 slots, transient ring 16 (08:12)** | **100 %** | **12.9** | 2.99 | 1.000 | 0.08 |
| keep 25 %, arena 79 GB (07:00) | 100 % | 13.6 | 3.09 | 0.999 | 7 |

Prefill (from the 2026-09-10 23:xx prefill work, still valid): 1,860-token prompt TTFT
118.5 s -> 33.7 s with 2048-token chunks + Decoder SWA Bounded Replay; short prompts 5-11 s.

### 2.4 Quality ladder of pruning (teacher-forced, held-out corpus `corpus/heldout_corpus.jsonl`: code and prose the trace never saw; 5,444 + 5,270 tokens)

Router restricted per layer to the top-N experts by trace frequency (mixed profile); loss in nats.

| kept / layer | coding NLL (Δ) | general NLL (Δ) | time |
|---|---|---|---|
| 384 (100 %) | 1.5067 | 3.1884 | 02:45 |
| 192 (50 %) | 1.5232 (+0.017) | 3.2528 (+0.064) | 02:45 |
| 154 (40 %) | 1.5285 (+0.022) | 3.3017 (+0.113) | 02:45 |
| 154 (40 %) + all kept experts at simulated 3-bit codebook | 1.5392 (+0.033) | 3.2122 (+0.024) | 07:50 |
| 120 (31 %) — the resident configuration above | 1.5729 (+0.066) | 3.3788 (+0.190) | 09:13 |
| 116 (30 %) | 1.5962 (+0.090) | 3.4241 (+0.236) | 02:45 |
| 116 (30 %) + coldest 40 % at simulated 3-bit | 1.5817 (+0.075) | 3.4187 (+0.230) | 08:38 |
| 96 (25 %) | 1.6687 (+0.162) | 3.5079 (+0.320) | 02:45 |

In-sample (trace corpus) deltas are in NOTES.md and are slightly smaller. The simulated 3-bit rows
use `engine/codebook_sim.py` (per-row 8-of-16 subset of the FP4 grid, 21 % relative weight error);
the packed format `tools/cb3.py` is bit-exact with it, its kernel `tools/cb3_moe.py` is correct but
not yet fast (54 GB/s vs 190 for FP4), so no CB3 speed row exists yet.

### 2.5 NVMe (2026-09-10 16:5x, O_DIRECT, 18.8 MB objects; unchanged)
1 in flight 4.1 GB/s · 8 in flight 5.4 GB/s · 32 in flight 5.6 GB/s.

### What is not measured in this tag
Thinking-on decode, long-context (>2k) serving, sampled (temperature 1.0) quality A/B, any bench
sweep, the CB3 format at speed, the container image end to end.

### 2.6 Addendum 2026-09-11 09:30-09:50 — decode step after the routing fix (same config as the keep-31 % row)

| change (commit) | verify step, everything resident | e2e decode tok/s (greedy, code prompt, 200 tokens) |
|---|---|---|
| baseline of 2.3 (22bd9a8) | 195 ms + 15 ms draft | 12.9-13.1 |
| torch routing for decode-sized calls instead of the per-arena-slot Triton router, bf16 gate GEMM, device slot LUT (9f172fb) | **168 ms + 15 ms draft** | **15.2-15.7** (acceptance 2.97-3.12) |
| Engram rows read in background threads, overlapped with the graphs (next commit) | unchanged | 15.4 (within run-to-run noise; the reads were 16 ms/step, now hidden) |

Profile of the 168 ms: expert kernels ~86 ms (at the 273 GB/s floor for 30 experts x 18.8 MB x 40
layers), FP8 dense ~29 ms (at floor), remaining bf16 GEMMs (wo_a, head, draft) ~20 ms, fp32 mixing
GEMMs ~8 ms, ~3,000 small elementwise/reduction kernels ~25 ms. Run-to-run spread of the e2e number
is ±5 % (greedy acceptance varies with bf16 nondeterminism: 2.97-3.12 on the same prompt).

### 2.7 Addendum 2026-09-11 09:50 — thinking on (served, keep 31 % resident, arena 90.5 GB, transient 8, LUT)

One request through the gateway, `chat_template_kwargs.thinking=true`, `reasoning_effort=high`,
temperature 0, 400 tokens (all reasoning): **TTFT 3.9 s, decode 21.4 tok/s, DSpark acceptance 4.11**,
hit rate 1.0. Same prompt with thinking off (2.6 addendum): 15.2-15.7 tok/s at acceptance ~3.

### 2.8 Addendum 2026-09-11 10:20 — long prompt through the served resident config (keep 31 %, arena 90.5 GB, transient 8, LUT)

One request through the gateway with an 8,192-token prompt (the server's context clamp), greedy,
thinking off, 200 output tokens: **TTFT 39.6 s (207 prompt tok/s), decode 16.9 tok/s, acceptance
3.28**, output a coherent summary of the prompt. The 8k prefill runs through the chunked encoder +
decoder-replay path (2048-token chunks); decode at an 8k KV is not slower than at 100 tokens
because the CSA2 index keeps the attended set at 512 tokens.

### 2.9 Addendum 2026-09-11 10:55 — FP8 grouped `wo_a` kernel and fused decode attention (same served config)

Step A/B on `engine/profile_fast.py` (keep 31 %, arena 90.5 GB): **165.7 → 152.7 ms** verify step,
draft 14.3 → 13.7 ms. Two hundred greedy tokens, same prompt and flags as 2.6: **16.86 tok/s**
(acceptance 3.06) with the new kernels vs 15.28 (acceptance 2.97) with `DSV41_WOA_FP8=0
DSV41_FUSED_ATTN=0` back to back. The `wo_a` projection now runs from its stored FP8 (7.95 ms/step,
was 13.89 as a bf16 einsum) and attention scores/softmax/PV run in one Triton kernel with bf16
keys and fp32 math (1.0 ms/step, was 3.1 fp32 SIMT). Unit tests in `engine/test_kernels.py`;
details and caveats in NOTES.md (2026-09-11 10:25-10:55).

### 2.10 Addendum 2026-09-11 11:20 — split-K fp32 kernel for the HC mixing projections, no `kv_all` copy

The two Hyper-Connection mixing GEMMs per layer (M=6, N=24, K=20480, fp32) ran on a cuBLAS kernel
at 29 GB/s (84 µs each); a split-K Triton kernel (`tools/fp32_skinny.py`, fp32 math, 4e-7 relative
to cuBLAS, both at the fp32 floor against an fp64 check) runs them at 108 GB/s (22.7 µs). The
compressor projections (N=512) stay on cuBLAS, which is faster there. The attention kernel now
reads the window ring and the CSA2 rows through two base pointers instead of a concatenated copy
(bit-identical output). Verify step on `engine/profile_fast.py`: **152.7 → 147.2 ms**, draft
13.7 → 13.0 ms; GPU time per step −9.1 ms.

The single 200-token greedy decode line moved the other way: 16.71 tok/s (acceptance 2.83, 71
steps) vs 17.11 (acceptance 3.06, 65 steps) with `DSV41_HC_KERNEL=0`. The 4e-7 change in the
mixing values flips borderline routing decisions, the greedy text diverges after a few tokens (both
outputs are coherent), and this prompt landed on a lower-acceptance trajectory; one sample cannot
separate that from run-to-run acceptance spread (±5 %, see 2.6). Every prompt-independent number
(step time, GPU time, the un-graphed comparison in `engine/test_fastdecode.py`) improved, so the
kernel stays on by default. Details in NOTES.md (2026-09-11 11:00-11:20).

### 2.11 Addendum 2026-09-11 12:10 — CB3 (3-bit) expert kernel at speed; graph merging measured as no gain

**CB3 kernel (`tools/cb3_moe.py` v3, unit test `tools/test_cb3_moe.py`):** one real expert at decode
shapes, expert bytes per second:

| kernel | ms | GB/s of expert bytes |
|---|---|---|
| FP4 (18.80 MB/expert) | 2.14 | 184.1 |
| CB3 v1 (the parked gather variant) | 17.41 | 19.9 |
| CB3 v2 (new plane layout, Triton byte ops) | 1.97 | 154.0 |
| **CB3 v3 (512-weight blocks, inline PTX decode)** | **1.67** | **181.5** (best run 182.0) |

A 3-bit expert now costs 0.787× the time of an FP4 one; dequant is bit-identical to
`engine/codebook_sim.py`, kernel output within 8.6e-5 of the FP4 kernel on the same re-quantized
weights. The decisive factor was tile width, not instruction count: on this box a row-strided read
runs at 101 GB/s for 16-32 B tiles and 185-218 GB/s from 64 B up, so the format's blocks were
widened to 512 weights (w2's K=2304 is packed as 4×512 + 256). Arena arithmetic at 90.5 GB: 4,813
experts all-FP4 (31.3 %), 6,260 all-CB3 (40.8 %). Keep 40 % at simulated 3-bit was measured in 2.4
at coding 1.539 / general 3.212 nats held-out, better than the served keep-31 % FP4 on both. The
kernel is not yet wired into the serving path (a second arena tier); that is the next step.

**Graph merging:** 41 graph replays per step → 3 (at the Engram boundaries) and pinned staging for
the Engram rows: step 147.2 → 146.6 ms, decode 16.56 vs 16.63 tok/s (bit-identical output). GPU
busy time is 144.9 of the 146.6 ms; the rest is per-kernel latency inside the graphs (about 5,300
kernels per step), not launch count. Segmentation stays on (`DSV41_GRAPH_SEGMENTS=0` restores);
pinned staging is off by default (`DSV41_ENGRAM_PINNED=1`).


## v0.3.0-wip — 2026-09-11

Measured 2026-09-11 10:00-14:10 on the same box and checkpoint, the pool ours alone. Commits
`94a96a6` .. this tag. Every row is one run of the stated command; no benchmark sweeps were run.
The kernel work that led here is in the dated addenda 2.8-2.11 above; this section is the shipped
configuration that changed.

### 3.1 Shipped default: keep 40 %, every resident expert in the 3-bit CB3 format

`PRUNE_KEEP=0.40 EXPERT_FORMAT=cb3 ARENA_GB=90.5 TRANSIENT_SLOTS=8 KEEP_FREE_GB=10`, everything
resident, CUDA-graph decode path, device slot LUT. Same prompt and flags as 2.6 for the decode
line; teacher-forced on `corpus/heldout_corpus.jsonl`; TTFT on the 1,806-token prompt of 2.x.

| | keep 31 %, FP4 (v0.2.0-wip default) | **keep 40 %, CB3 (this tag)** |
|---|---|---|
| resident experts | 4,800 = 90.2 GB (31.3 %) | **6,160 = 89.0 GB (40.8 %)** |
| warm start (packing on the GPU) | 19 s | 183 s |
| decode, 200 greedy tokens | 16.61 tok/s (acceptance 2.83, 71 steps) | **18.98 tok/s** (acceptance 3.03, 66 steps) |
| TTFT, 1,806-token prompt | 11.11 s | **9.81 s** |
| held-out coding NLL | 1.5705 | **1.5384** (−0.032) |
| held-out general NLL | 3.3790 | **3.2087** (−0.170) |

The CB3 row reproduces the simulated 3-bit keep-40 % row of 2.4 (1.5392 / 3.2122) to 0.0008 /
0.0035 nats: the packed format, the kernel and the simulation are the same arithmetic. Against the
full unpruned model on the same corpus (2.4: 1.5067 / 3.1884) this configuration costs +0.032
(code) / +0.020 (prose) nats.

Prefill does not run the CB3 decode kernel: above 64 token-expert pairs the experts are unpacked
to FP4 codes on the fly (bit-exact) and the FP4 kernel runs; on one layer at 2,048 tokens that is
1.35x the MoE time of an FP4 arena of the same size. The two-tier arena (hot experts back at FP4,
cold in CB3) is not built; at 40.8 % all-CB3 there is no headroom in 90.5 GB for it.

### What is not measured in this tag
Thinking-on decode in this configuration, sampled quality A/B, long-context (8k+) serving in this
configuration, the container image end to end.

### 3.2 Addendum 2026-09-11 16:00 — attention projections in FP4 (served config, `DSV41_DENSE_FP4=attn`)

Dense bytes per verify step (from the safetensors headers): attention projections 3,734 MB,
shared experts 1,417 MB, `wo_a` 1,344 MB, other 436 MB. A dense FP4 kernel (`tools/fp4_linear.py`,
E2M1 codes + one UE8M0 scale per 32 weights along K, quantized at load from the stored FP8; unit
test `tools/test_fp4_linear.py`) reads 0.53x the bytes and wins on the wide matrices (`wq_b`,
`wo_b`: 199 GB/s of FP4 vs 221 of FP8) but not on the narrow ones. Held-out teacher-forced, one run
per setting, against the keep-40 % CB3 baseline 1.5384 / 3.2087:

| group in FP4 | coding | general | decision |
|---|---|---|---|
| shared experts | 1.5531 (+0.015) | 3.2740 (+0.065) | kept in FP8 |
| attention projections | **1.5403 (+0.002)** | **3.1738 (−0.035)** | **default from this addendum** |
| both | 1.5527 (+0.014) | 3.2323 (+0.024) | kept in FP8 |

The −0.035 on general is within what a 53-sequence corpus can resolve, not a gain. Decode line
back to back: 19.11 tok/s (acceptance 3.03) → **20.85 tok/s** (acceptance 3.23); prefill 4.12 →
3.28 s on the same prompt; verify step on `engine/profile_fast.py` 134.4 → 125.6 ms; 1.75 GiB of
resident weights freed. The shared experts are the one dense FFN every token passes through and
the FP4 weight error (12 % relative) shows there; `wo_a` stays FP8 through the grouped kernel.

### 3.3 Addendum 2026-09-11 17:40 — `wo_a` in FP4 and a leaner decode loop

The output projection's first factor (`attn.wo_a`, 1,343 MB read per verify step) now runs through a
grouped FP4 kernel (`tools/fp4_linear.py::fp4_grouped_linear`, one group per third grid axis),
quantized at load from the stored FP8. Held-out teacher-forced against the 3.1 baseline
(1.5384 / 3.2087): **coding 1.5346 (−0.004), general 3.1498 (−0.059)** — inside what this corpus can
resolve, and on the good side of zero. Resident weights 8.34 → 7.70 GiB. Kernel time in the step
7.95 → 4.56 ms; at T=6 in isolation the FP4 grouped kernel is 150-158 GB/s against the FP8 one's
207-215, and reads 0.53x the bytes, so 1.31-1.44x in wall time.

`DSV41_LEAN_STEP=1` (default; `=0` restores the previous code) computes the greedy accept/reject on
the GPU and reads back one 7-element pinned tensor instead of up to eleven separate syncs, builds
the verify block into a preallocated buffer, and drops redundant clones and a duplicate buffer
preparation. Sampling semantics are unchanged: at temperature 0 both paths produce byte-identical
text, and the temperature path is the original code.

| | verify step (`engine/profile_fast.py`) | wall per step, 200-token run |
|---|---|---|
| 3.2 configuration (`attn`) | 125.6 ms | 156.5 ms |
| this addendum (`attn,wo_a`, lean step) | **118.9 ms** | **149.1 ms** |

The decode line on the standard prompt moved 20.62 → 20.02 tok/s because the DSpark acceptance on
that one prompt fell from 3.23 to 2.99; at equal acceptance the new configuration is 21.7 tok/s.
Per-step time is the number this addendum claims. Prefill on the same prompt 3.27 → 2.61 s.

Instrumenting the loop (`DSV41_STEP_TIMING=1`) also corrected an earlier reading: the gap between
the graph harness and a fresh 200-token run is not removable Python. It is one un-graphed drafter
call on the first step, the Engram host-to-device copy absorbing queued graph work by design, and a
cold Engram row cache — a second decode in the same process costs ~140 ms/step and a third ~133 ms,
tracking the Engram read time and nothing else.

### 3.4 Addendum 2026-09-11 19:00 — the LM head in FP8, and a 2-bit expert tier that was not built

**The head.** `head.weight` is [129280, 5120] bf16 = 1.324 GB and is read in full twice per decode
step (the verify step and the DSpark draft); in 3.3's profile it is 5.79 ms of cutlass at 232 GB/s.
`DSV41_HEAD_FMT` = `bf16` (default) | `fp8` | `fp4` stores it in the dense projections' format
(e4m3 + one UE8M0 scale per 32x32 block, 0.663 GB, `tools/fp8_linear.py::quantize_to_fp8`) or the
routed experts' (E2M1 + one scale per 32 K weights of a row, 0.352 GB), quantized on the GPU at
load; above decode-sized M a quantized head dequantizes 16,384 vocabulary rows at a time into
cuBLAS. Held-out teacher-forced against the 3.3 baseline (1.5346 / 3.1498), one run each:

| head | coding | general | weight rel err | head GEMM at M=6 | decision |
|---|---|---|---|---|---|
| bf16 | 1.5346 | 3.1498 | — | 5.68 ms, 233 GB/s | — |
| **fp8** | **1.5351 (+0.0004)** | **3.1512 (+0.0014)** | 0.027 | **2.96 ms, 224 GB/s** | **default from this addendum** |
| fp4 | 1.5502 (+0.0156) | 3.1572 (+0.0074) | 0.118 | 2.38 ms, 148 GB/s | rejected on coding |

With the fp8 head the 200-token greedy output is byte-identical to the bf16 one at matched positions
(the second decode of each process; in every arm, bf16 included, a process's first decode differs
from its own second at token 9, because the first step drafts eagerly before any graph exists).
`engine/test_fastdecode.py` argmax agreement 1.00 / 1.00 on both parities, step 116.5 / 117.8 →
113.2 / 113.9 ms and draft 14.3 / 13.5 → 11.6 / 10.6. Decode line: 143.0 → 137.2 ms per step,
21.62 → 22.88 tok/s. In one process with the head swapped under a fixed arena, the verify step is
118.9 → 114.6 ms and the draft 13.4 → 10.7, with the two CB3 expert kernels unchanged — the
separate-process form of that A/B measures the arena's page placement instead and reports the
opposite sign (NOTES 2026-09-11).

**The 2-bit tier: measured, not built.** A CB3 slot filled with a four-entry row codebook repeated
to its eight entries carries exactly a 2-bit format's arithmetic at unchanged size, so the quality
question was answered inside the shipped configuration (`--sim-cb2-frac`). Held-out teacher-forced,
keep 0.40, against the same 1.5346 / 3.1498:

| coldest fraction of the kept set at 2 bits | coding | general |
|---|---|---|
| none (shipped) | 1.5346 | 3.1498 |
| 0.30 (1,840 of 6,160 experts) | 1.5471 (+0.0125) | 3.1966 (**+0.0468**) |
| 0.50 (3,080 of 6,160 experts) | 1.5563 (+0.0217) | 3.1940 (**+0.0442**) |

The whole prose penalty is paid by the coldest 30 % and does not grow after it, at 1.6x the budget a
tier would have to fit in. The packed CB2 format and its kernel were built and measured anyway
(`tools/cb3_moe.py::CB2ArenaV2`, 9.99 MB per expert = 0.691x CB3, bit-exact against the FP4 kernel
on the same weights): 143.8 GB/s of its own bytes against CB3's 168.7 in the same run, i.e. 0.81x in
wall time for the cold experts, which carry 22 % of the routed pairs at a 0.50 cold fraction — about
3 ms of a 119 ms step even before the loss is counted. `EXPERT_FORMAT` keeps its two values; nothing
in the serving path changed.


### 3.5 Addendum 2026-09-11 19:30 — thinking-on decode, the router's top-k, and the verify block size

**Thinking on, in this configuration.** One request each through the gateway, temperature 0, 400
output tokens, the same algorithmic prompt (2.7's number came from the much slower 09:50 engine and
a different prompt):

| | thinking off | thinking on (`reasoning_effort=high`) |
|---|---|---|
| TTFT | 1.67 s | 2.19 s |
| decode | **25.86 tok/s** | **21.44 tok/s** |
| DSpark acceptance | 3.51 | 2.90 |
| wall per step | 135.7 ms | 135.2 ms |

The step costs the same to 0.4 %; the whole difference is acceptance, so a single prompt's tok/s is
a property of the prompt as much as of the engine.

**How many experts a step activates.** `DSV41_ROUTE_STATS=1` (`engine/diag_topk.py`) counts the
DISTINCT routed experts a verify block asks for per layer — the quantity that sets the bytes, since
an expert is read once however many of the block's six tokens route to it. Measured on the real
decode path, keep 0.40 pruned: **20.96 per layer at the checkpoint's top-6**, not the ~30 that 36
routed pairs would suggest, i.e. **12.12 GB of expert reads per step** at 186 GB/s over the two CB3
kernels' 65.0 ms.

**Reducing the router's top-k** (`DSV41_TOPK`, default the checkpoint's 6; the gate weights are
renormalized over the survivors by the line that already normalizes the six; the DSpark drafter's
own top-3 of 128 is untouched). Held-out teacher-forced, one run each, the k=6 row re-measured here:

| `DSV41_TOPK` | distinct experts/layer | expert bytes/step | coding | general | decision |
|---|---|---|---|---|---|
| 6 | 20.96 | 12.12 GB | 1.5351 | 3.1512 | **shipped** |
| 5 | 17.58 (0.839x) | 10.17 GB | 1.5396 (+0.0045) | 3.1762 (**+0.0250**) | rejected on prose |
| 4 | 15.30 (0.730x) | 8.85 GB | 1.5890 (+0.0539) | 3.2330 (+0.0818) | rejected on both |

k=5 would have been worth 8.3 ms of a 114.9 ms verify step (106.6 ms; expert kernels 65.02 → 57.44,
less than proportional because 17.6 experts per layer gives the kernel fewer concurrent programs and
it gives back 5 % of its per-byte rate) and 23.45 → 24.87 tok/s on the standard 200-token prompt.
`engine/test_fastdecode.py` at k=5 is argmax agreement 1.00/1.00 on both parities, so the switch is
the same computation graphed and un-graphed. It fails the +0.015-nat gate on prose by 1.7x and is
not shipped. Prose leans on the tail of the router's distribution and code does not — the same split
the 2-bit expert tier showed in 3.4.

**The verify block size** (`DSV41_BLOCK`, drafted positions, odd, default the checkpoint's 5; the
verify width must stay even for the ratio-2 compressor's parity). One run per setting:

| verify block | step + draft | acceptance | **ms per accepted token** | 200 greedy tokens |
|---|---|---|---|---|
| 4 | 110.7 ms | 2.70 | 41.0 | 23.72 tok/s |
| **6 (shipped)** | 124.8 ms | 3.14 | **39.7** | 23.45 tok/s |
| 8 | 137.0 ms | 3.45 | **39.7** | 23.00 tok/s |

The step grows nearly linearly in the block and acceptance sublinearly, exactly as expected; their
ratio is flat between 6 and 8 (a tie within this box's ±5 % acceptance spread) and worse at 4. Block
8 also drafts beyond the horizon the DSpark head was trained for and lengthens the per-burst
latency, so the block stays at 6. Nothing in the serving configuration changed in this addendum.

### 3.6 Addendum 2026-09-11 23:40 — the 3-bit expert format degenerates in free generation, and is withdrawn

Asked for a single-file HTML game, the configuration shipped in 3.1 (keep 40 %, every resident
expert in the 3-bit CB3 format) writes one token over and over until the output cap:

```
```html
<!<!DOCTYPE><!DOCTYPE><!DOCTYPE><!DOCTYPE> ...
```

Everything else was eliminated one variable at a time, same prompt, greedy, 200-300 tokens each
(`results/htmlbug/`): the graphed decode path, CUDA graphs entirely, the fused attention kernel,
speculative decoding (off: identical loop), the FP4 dense projections and the fp8 head (both
reverted: identical loop), and sampling (temperature 0.0 / 0.3 / 0.6 / 0.8: identical loop, so the
distribution itself is degenerate, not the choice rule). The last variable left was the expert
configuration, and it is decisive:

| experts | output |
|---|---|
| CB3 3-bit, keep 40 % (3.1) | the loop, distinct-token ratio 0.03 |
| FP4, keep 31 % (v0.2.0-wip) | `<!DOCTYPE html / <html> / </html>`, closes and stops, ratio 0.71 |

**The shipped default returns to FP4 experts at keep 31 %** (box `.env`: `PRUNE_KEEP=0.31
EXPERT_FORMAT=fp4`), keeping the changes that are independently gated: the fp32 router (3.7 below),
the FP4 attention and `wo_a` projections, and the fp8 head. A 600-token Python class on that
configuration comes back complete and well-formed at 30.1 tok/s (acceptance 4.72).

**What this says about the gate.** 3.1 measured *better* than the configuration that works —
held-out teacher-forced 1.5384 / 3.2087 against 1.5705 / 3.3790 — and 2.4 predicted it from the
simulator to three decimal places. Teacher-forced loss scores the next token of text the model is
shown; it never lets an error compound, so it cannot see a model that cannot stay on its own
trajectory. Every quantization decision in this repo was taken on that number alone. A format that
improves it can still be unusable, and nothing here measured free-running generation until a user
asked for an HTML file.

`EXPERT_FORMAT=cb3` and its kernel remain in the tree, measured and documented, and must not be a
default again without a generation gate (`engine/test_spec_lossless.py` plus a long-generation
check) in front of it.

### 3.7 Addendum 2026-09-11 21:25 — the graphed decode path routed tokens to the wrong experts

`Model.moe` computes the router gate as `mm(y.float(), gate_w)`; the graphed path computed it in
bf16, added 2026-09-11 morning as a micro-optimization. The gate picks 6 of 384 experts and its
scores are dense with near-ties, so bf16 changed which experts ran: at layer 0, where the inputs are
bit-identical, 11 % of the picks differed, rising to 31 % in the middle layers, and each layer then
ran a different FFN than the reference.

| layer | activation error before / after | routed experts equal before / after |
|---|---|---|
| 0 | 0.0000 / 0.0000 | 0.89 / **1.00** |
| 6 | 0.1070 / **0.0073** | 0.83 / **1.00** |
| 12 | 0.1361 / **0.0087** | 0.69 / **1.00** |
| 36 | 0.0803 / **0.0131** | 0.69 / 0.94 |

Logit error against the reference 0.049 → 0.012, argmax agreement 1.00 on both parities. The fused
attention kernel (`DSV41_FUSED_ATTN`, default 0 since this addendum) is a second, smaller source of
the same divergence: with it on, deep-layer error returns to 0.048-0.093 and routed experts equal
falls to 0.72.


## v0.4.0-wip — 2026-09-12

Measured 2026-09-12 03:00-05:40 on the same box and checkpoint. Every row is one request through the
server; the quality column is the generation gate described in 4.1, not a loss number.

### 4.1 The gate this tag is built on

Teacher-forced loss cannot see a model that has stopped being able to stay on its own trajectory: it
scores the next token of text the model is shown and never lets an error compound. The configuration
shipped in v0.3.0-wip measured **better** on it (1.5384 / 3.2087 against 1.5705 / 3.3790) and wrote
`<!DOCTYPE><!DOCTYPE><!DOCTYPE>` for as long as it was allowed. Every configuration in this tag is
gated on free generation instead: five prompts (a story and an essay at temperature 0.7, a Python
module, a single-file HTML game, a JavaScript module at temperature 0), **900 to 2,000 tokens each**,
and the output must keep a distinct-token ratio above 0.25, repeat no line more than 30 % of the
time, and — where the generation finished on its own — be structurally intact.

Both thresholds were learned the hard way. A 300-token gate passed configurations that collapse at
900. Repetition ratios alone pass output whose CSS has decayed into `inset - 00 1 pix - 00 1 pix`,
so the gate checks balanced tags, closed fences and unit spelling too.

### 4.2 The keep-set is a cache policy, and it must be sampled from every workload

`corpus/trace_corpus.jsonl`, which ranked the experts for every earlier tag, is 50 documents whose
only content marker is Python: no HTML, no JavaScript, no CSS, no SQL, no configuration files. The
experts that write markup never fired while the trace was taken, ranked cold, and were dropped by
every pruned configuration. That single fact explains the degeneration chased through v0.3.0-wip.

| keep-set (all at keep 44 %, CB3, otherwise identical) | story | Python | HTML |
|---|---|---|---|
| original corpus | 0.27 | 0.51 | **0.03** |
| + a 12-gram repeat ban | 0.32 | 0.51 | **0.07** |
| `trace_corpus_v2` (web, code, config, technical prose) | 0.43 | 0.50 | 0.40 |
| `trace_corpus_v3` (narrative fiction and dialogue) | 0.54 | 0.58 | **0.04** |
| **union of both traces** | **0.56** | **0.47** | **0.59** |

(distinct-token ratio; <= 0.15 is degenerate.) A corpus of web and code fixes markup and leaves long
prose repeating; a corpus of fiction fixes prose and loses markup. At 44 % of the experts the two
rankings compete for the same slots, and the answer is to rank on both: an expert trace is a
per-token histogram, so `results/trace-union` is the concatenation of the two traces' per-layer
arrays (190 sequences, 36,250 tokens) with the statistics rebuilt from it. No third trace run.

### 4.3 Shipped configuration

`PRUNE_KEEP=0.44 EXPERT_FORMAT=cb3 ARENA_GB=98 TRANSIENT_SLOTS=8 KEEP_FREE_GB=6`,
`TRACE_STATS=results/trace-union/stats/coverage.json`, `DSV41_DENSE_FP4=attn,wo_a`,
`DSV41_HEAD_FMT=fp8`, `DSV41_FUSED_ATTN=0`, penalties 0. 6,779 experts resident = 44.1 % of all
routed experts, expert hit rate 1.0, no NVMe traffic during decode.

| workload | tok/s | DSpark acceptance |
|---|---|---|
| single-file HTML game | 36.6 | 5.07 |
| SQL schema and query | 31.8 | 4.69 |
| JavaScript module | 28.2 | 4.12 |
| German technical writing | 25.2 | 3.68 |
| arithmetic with working | 25.1 | 3.56 |
| Python module | 24.3 | 3.56 |
| thinking on, effort high | 18.6 | 2.66 |
| explanation (temperature 0.7) | 18.1 | 2.64 |
| long story (temperature 0.7) | 17.1 | 2.46 |

Prefill on a 5,014-token prompt: **14.9 s = 337 tok/s** (the all-resident arena reads nothing from
the SSD; the same prompt through the streaming configuration is 87 tok/s). A tool call returns
`finish_reason: tool_calls` with well-formed arguments. The generated HTML game passes every
structural check — doctype, balanced `<style>` and `<script>`, a 3x3 grid, a win check, a reset
button, click handlers, no corrupted CSS units — and stops on its own at 982 tokens.

The step is ~145 ms in every case; the spread is entirely how well the DSpark drafter predicts each
kind of text, about 5 accepted tokens per step on markup against 2.5 on prose.

### 4.4 What this tag fixes in the engine

* The graphed decode path computed the router gate in bf16 while `Model.moe` computes it in fp32.
  The gate picks 6 of 384 experts and its scores are dense with near-ties, so 11 % of the picks
  differed at layer 0 — where the inputs are bit-identical — and up to 31 % deeper in. Every layer
  after that ran a different FFN than the reference. Now fp32: routed experts agree 1.00 at layer 0,
  logit error against the reference 0.049 -> 0.012.
* `DSV41_FUSED_ATTN` defaults to 0: the fused decode-attention kernel returns deep-layer agreement
  to 0.072-0.093 and routed agreement to 0.72.
* A keep-set can now be built from `coverage.json` alone (the per-category histograms are written
  into it), so a checkout reproduces one without the per-layer trace arrays.

### What is not measured in this tag
Sampled quality A/B at scale, long-context (8k+) generation quality, the container image end to end,
and the tool grammar of `server/tool_grammar.py` on real weights (it is off by default).

### Router-score placement smoke (2026-10-05)

Added opt-in `DSV41_PRUNE_METRIC=score`: observed placement uses raw positive
router scores for the original top-k selections, with separately normalized
request score history. The existing frequency trace remains the cold-start prior.
Native FP4 weights and numerical settings stayed unchanged; this is not EXL3.

The two-Spark live profile uses keep 0.61 (235 experts/layer), a 90.1 GB arena,
16 transient slots, 524288 context, speculation and the existing attention overlay.
Both ranks agreed on 119 guarded fields. After five calibration prompts disjoint
from evaluation, the same five fixed MMLU-Pro questions used in the preceding
full-streaming smoke scored **4/5**, all valid, in **11.39 seconds total**
(median 1.58 seconds). Answers A/G/A/B/D exactly matched that full-streaming
run, which scored 4/5 in 52.38 seconds. Earlier frequency runs scored 3/5 and 4/5.

On these five score-mode requests, aggregate selection misses were **17.44%**;
score-weighted misses were **15.68%**. Historical frequency selection misses on
the same questions were 23.14% and 22.21%, but their history, compiler warmup
accounting, and cache state differed. This is a diagnostic smoke, not an isolated
metric A/B, broad accuracy estimate, or sustained-throughput measurement. No
large benchmark was run. Forty-seven selected tests passed in the serving image,
including fused GPU recorder parity, graph replay, history units, and swap checks.

Raw results, manifests, source hashes, launch settings and rollback settings:
`../llm_benchmark/results/custom-router-score-20261005/`.

### 2026-10-05: fixed-history expert ranking and bounded contribution rescue

Small MMLU-Pro diagnostic: 14 new questions (seed 20261006, one per category),
greedy direct answers, all swaps frozen, 9,400 resident experts at 61% kept.
Each launch started from the same saved request-unit history, prior 4. Frequency,
raw router score, score plus confidence depth, and score plus calibrated prefill
rescue all answered 9/14, with identical letters and no invalid requests.
Frequency and score differed by 257 placements; selection misses were 13.36% and
13.99%, respectively. This gives no evidence that the lower missing rate or raw
score ranking improves answers. It is a small diagnostic, not an accuracy estimate.

The rescue profile used two disjoint public texts (132 tokens) and layers 0–3
only. Nine layer/expert rescue events caused five actual expert reads across the
14 questions, without correcting an answer. Median question time rose from
0.897 to 0.973 s. Keep the prototype disabled. Details and limitations are in
[critical-prefill.md](docs/critical-prefill.md).

Two timed repetitions after one warmup, 128 output tokens each, unique prefixes
(no prefix reuse): confidence depth improved code from 45.50 to 52.325 decode
tok/s (+15.0%), with identical outputs; prose went 20.19 to 19.71 (-2.4%), also
with identical outputs. End-to-end median times were code 3.193/2.835 s and
prose 6.671/6.828 s. This is an opt-in code speed result, not a general default
win; confidence remains off in production.

Offline global layer allocation retained 95.765% versus uniform 95.678% of the
blended calibration mass at the same budget. It was not run live after the goal
was clarified as preserving critical experts rather than maximizing routing mass.

Raw artifacts and paired comparisons: `/home/ryan/git/llm_benchmark/results/
expert-tuning-20261005/`; launch/config/profile artifacts: `results/expert-tuning-20261005/`.

### Full-routing control and layer allocation (2026-10-06)

On the identical 14-question manifest above, unrestricted routing (keep 1,
384 transient slots, same native FP4 weights and numerical settings) scored
**11/14**, versus **9/14** for frozen score/uniform placement. The only changed
answers were math 7867 (A -> J) and philosophy 11054 (I -> B), both recoveries.
The math answer independently matches the analytic derivative, -153.5947587.
Every request was valid and had zero prefix reuse. Full-routing request time
totaled 101.87 s versus 35.20 s for uniform, including first-request warmup;
medians were 6.121 versus 0.897 s. This control isolates two pruning-sensitive
answers without changing the model format; it is not a broad quality estimate.

An explicit fixed layer budget then kept 330 experts in layers 0–4 and 35–39,
204 in layers 5–14, and 203 in layers 15–34: the same 9,400 total slots, no
streaming, same ranking/history, all swaps frozen. It recovered math but lost
biology 2868 (E -> F), leaving **9/14**. Philosophy remained wrong. The other
category changed I -> A and remained wrong. This rejects this particular edge
allocation as a quality improvement; it does not establish a general layer
importance ranking. The allocation option is experimental and unset by default.

Isolating the late budget (330 experts in layers 35–39, 222 in 0–14, 221 in
15–34; again 9,400 total) scored **10/14**: math recovered, biology stayed correct,
and all other letters matched uniform. All requests were valid with zero NVMe
expert reads. This is a placement-only recovery on the diagnostic set; a separate
validation set is needed before treating it as a general policy improvement.

### Immediate layer streaming qualification and group probes

`DSV41_STREAM_LAYERS` now supports prefill and decode, loading current router
picks before MoE through the transient ring. The keep set stays at 9,400 experts;
160 transient slots fit in the same arena without evictions of residents. Both
ranks guard the selected layers and graph-split policy at boot. Fifty focused
tests passed, including current-token load ordering and transient-only decode.

All-layer streaming matched unrestricted routing on biology 2868, math 7867,
and philosophy 11054: E/J/B, all correct. Separately streaming 0–4, 18–22, or
35–39 produced E/A/I in every arm, matching the pruned baseline. None of these
five-layer groups alone explains the recoveries. The late-only retention result
above is consequently not proof of late-layer causal dominance.

A short 39-token generation with layers 35–39 streamed correctly counted 1–20
in nine completed verification steps, 2.53 s total, 1.64 GB expert reads and
34.96 decode tok/s. This qualifies that mixed graph path on a short prompt;
it is one functional check, not a sustained-throughput comparison. See
[layer-stream.md](docs/layer-stream.md) for constraints. Answer probes and the
multi-token check are saved under `../llm_benchmark/results/expert-tuning-20261005/`.

The follow-up block-necessity sweep used only the two previously identified
pruning-sensitive cases. It fully routed all layers except one five-layer block,
which kept the original frozen 235-expert mask. These are post-hoc diagnostic
cases, not a held-out accuracy estimate. All 16 requests were valid, with identical
question prefixes and no prefix reuse. Each arm began from the same demand seed.

| Pruned block; all others fully routed | Math 7867 | Philosophy 11054 |
| --- | --- | --- |
| 0–4 | A, wrong | B, correct |
| 5–9 | J, correct | I, wrong |
| 10–14 | J, correct | B, correct |
| 15–19 | J, correct | B, correct |
| 20–24 | J, correct | B, correct |
| 25–29 | J, correct | B, correct |
| 30–34 | J, correct | B, correct |
| 35–39 | J, correct | B, correct |

Thus early blocks are necessary under this particular full-routing control;
restoring 0–4 alone was insufficient. Lack of damage from individually pruning
later blocks does not prove they can all be pruned together, or that later layers
are generally unimportant. Protocol: `results/expert-tuning-20261005/layer-ablation-protocol.json`;
raw responses and summary: `../llm_benchmark/results/expert-tuning-20261005/layer-ablation-summary.json`.

Streaming 0–9 with all other layers pruned at the original budget recovered
philosophy but not math. On the full 14-question diagnostic it scored **10/14**;
every other letter matched the 9/14 baseline. This demonstrates that the two
early block-necessity results cannot simply be combined into a sufficient policy
for both examples. It is a targeted diagnostic gain, not an independent accuracy
estimate. Artifact: `stream-first10-quality.json` in the benchmark results directory.

Fine-grained diagnostics used persistent per-layer mask tensors and an
authoritative per-request broadcast, avoiding ten full model reloads. Each of
layers 0–4 was separately pruned on math, and 5–9 on philosophy, in the otherwise
fully routed model. **No individual layer broke its tested answer**. Pruning
0–2 or 3–4 separately also preserved math; pruning 5–7 broke philosophy, while
pruning 8–9 preserved it. These are joint effects, not proof of a single dominant
layer or expert. Restoring the first **20** layers recovered both answers and
preserved the biology control. Full-routing controls matched before and after.

Fifty-one focused tests passed for the revised mask-control path, including
root-authoritative masks with no peer file, in-place address preservation, and
errors broadcast to both ranks. The first harness pass stopped on a transient
busy state after its three correct reference answers; partial results were kept,
and the harness now waits for idle before changing masks. Complete measurements:
`single-layer-ablation-retry.json`; the partial pass is `single-layer-ablation.json`.

Reducing the sufficient 20-layer policy to **0–9 plus 15–19** (15 streamed layers)
preserved all three probe answers. Streaming **0–14** (also 15 layers) still lost
math. Thus the middle block 15–19 matters under partial routing, despite its
individual block ablation causing no damage under full routing. Layer importance
depends on the other routing decisions, not just position. These two arms and an
unchanged full-routing control are in `fifteen-layer-sufficiency.json`.

Further preset subsets all preserved E/J/B on the three probes: 0–7 plus 15–19
(13 layers), 0–2 plus 5–7 plus 15–19 (11), and **3–7 plus 15–19 (10)**.
All had an unchanged full-routing control afterward. The ten-layer subset was
then frozen for normal serving-path qualification and a separate 14-question
validation seed 20261007. It is the smallest passing policy among these tested
subsets, not a globally minimal or generally optimal layer set.

### Ten-layer policy: matched validation and cost

The fixed **3–7 plus 15–19** policy was then tested through the normal mixed
serving path, with diagnostic mask changes disabled. All arms used native FP4,
the same initial score-history seed, 235 residents per layer (9,400 total), the
same 90.1 GB arena, 160 transient slots, and all swaps/prefix reuse disabled.

| Manifest | Frozen resident routing | Ten layers fully routed | All layers fully routed |
| --- | --- | --- | --- |
| Diagnostic seed 20261006, 14 questions | 9/14 | 11/14 | 11/14 (earlier keep-1 control) |
| Separate seed 20261007, 14 questions | 9/14 | 9/14 | 12/14 (all-layer streaming control) |

The matched resident diagnostic reproduced every original baseline letter.
Ten-layer streaming recovered math 7867 and philosophy 11054, and changed law
1789 from G to A while still wrong. All other diagnostic letters matched the
baseline. On the separate manifest, **every ten-layer answer letter matched
the resident baseline**. Full routing recovered business 834 (G -> I),
engineering 12071 (A -> J), and history 5003 (F -> C), with no other changes.
Thus the ten-layer policy misses pruning-sensitive cases outside its selection
set. It is not a demonstrated general quality improvement and remains off.

A short identical-output decode check counted 1–20 (39 output tokens), using
unique prompt identifiers, one warmup and two measured repetitions per arm.
Resident / ten-layer medians were **1.219 / 2.184 s** per request and
**45.35 / 39.58 decode tok/s**. Ten-layer median reported expert reads were
0.95 GB versus zero for resident routing. No resident promotions occurred.
This is a functional qualification and an exploratory cost comparison on a short
fixed answer, not a sustained throughput estimate. On the separate 14-question
manifest, median request times were 0.733 / 1.553 s for resident / ten-layer.

The ten-layer policy was frozen before the separate manifest's responses;
subsequent block analysis of its three full-routing recoveries is post-hoc.
Raw responses, health configs and comparison: `candidate-10-quality.json`,
`candidate-10-validation.json`, `validation-uniform-quality.json`,
`validation-uniform-diagnostic.json`, `validation-full-stream-quality.json`,
and `layer-policy-summary.json` under the benchmark results directory.
The policy freeze and source hashes are in `results/expert-tuning-20261005/`.

### New-case block sensitivity: early, middle, and later layers

After validation, the three newly identified pruning-sensitive cases were used
for a separate **post-hoc** block sweep. Graph topology stayed fixed with all
layers split; only the authoritative routing masks changed. All 39 short requests
were valid, no prefixes were reused, and expert generation stayed zero. Full
routing controls before and after gave the same correct I/J/C answers.

Combining the two earlier diagnostic cases with these three new cases gives this
limited sensitivity map. Each row prunes only that block; all other layers retain
full routing. A blank finding means no answer loss on these five cases, not proof
that the block is unimportant in general.

| Pruned block | Correct reference answers lost |
| --- | --- |
| 0–4 | Math 7867 |
| 5–9 | Philosophy 11054; history 5003 |
| 10–14 | Engineering 12071 |
| 15–19 | Engineering 12071 |
| 20–24 | None on these cases |
| 25–29 | None on these cases |
| 30–34 | Business 834 |
| 35–39 | None on these cases |

Three broad sufficiency policies were specified before this sweep's responses:

| Full-routing layers; other layers pruned | Business 834 | Engineering 12071 | History 5003 |
| --- | --- | --- | --- |
| 0–19 | G, wrong | J, correct | C, correct |
| 20–39 | I, correct | A, wrong | F, wrong |
| 0–9 plus 30–39 | I, correct | A, wrong | C, correct |

This provides direct counterexamples to a universal early/late preference:
engineering depends on middle blocks, business on a later block, and history on
an early block. Engineering also needs more than restoring 15–19 in the
ten-layer policy, showing necessity and sufficiency differ. These findings guide
future expert-level protection; they do not justify streaming whole regions by
default or selecting a new policy on the same validation cases and calling it
held out. Protocol: `results/expert-tuning-20261005/fresh-block-ablation-protocol.json`;
responses and condensed table: `fresh-block-ablation.json` and
`fresh-block-ablation-summary.json` in the benchmark results directory.

Final state: the normal `deepseek` EP2 pair is restored on the latest local code
image, with streaming, diagnostic mask control, explicit layer allocations,
calibrated rescue and confidence depth disabled. The original 0.61 keep ratio,
16 transient slots, 90.1 GB arena, production score database, prefix cache and
adaptive swap settings are restored. Both running ranks and the workspace match
on all six modified runtime modules; hashes are in `runtime-source-verified.json`.
The final health/config check passed on both ranks, and a floor request returned
A. Fifty-one focused tests passed; `git diff --check` passed. TensorFold remains
stopped. No model downloads or conversions were performed for these trials.
Final checks are saved in `results/expert-tuning-20261005/final-production-*`.

## 2026-10-06: author's topic database and maxmin selection (TP2)

Ported the author's fixed topic-selection path at upstream commit
`45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f`: 39 topics, frequency and saliency
histograms for every 40 x 384 layer/expert position, plus the greedy maxmin
ranker. The 13,048,828-byte statistics file is pinned by SHA256
`eb5214a78791a1e8cc0db51353f6ea7931f4f2d18142776f90784e01e67f16d3`.
Only statistics were obtained; no model weights were downloaded or converted.
The tiny image overlays use the already-built runtime on both nodes.

Saliency is the trace's accumulated gate-weight times expert-output norm.
Maxmin normalizes each topic separately and repeatedly admits the best unused
expert for the least-covered topic. Its priorities are selection order, not
router-score mass. The port requires fixed placement: no history blending,
prefill/end/decode/urgent swaps, or unequal layer allocation. Default counts/sum
with no explicit topics retains the existing adaptive coding/general trace path.
The existing unconditional score broadcast remains authoritative; no collective
was added. The TP2 configuration guard now includes source, ranker, topics,
statistics SHA256 and actual initial expert-ID SHA256, rather than only mask size.
Profile coverage is reported as a trace proxy, not answer accuracy.

Exact CPU parity with the pinned upstream maxmin function passed for **both
families across all 39 topics, 40 layers, 384 experts**. Sixty focused tests
passed, including nine profile tests. Both image copies have matching engine
and profile modules and identical statistics checksums. Setup and parity records
are in `results/expert-profile-20261006/`.

### Fixed comparison and separate validation

All arms use current output-sharded TP2, native FP4, 0.61 keep, **235 residents
per layer / 9,400 total**, a **90.1 GB arena**, 16 transient slots and the same
existing numerical settings/weight overlay. Streaming, calibrated rescue,
explicit layer budgets, confidence scheduling, prefix reuse, prefill graphs and
all swaps are disabled. No inference kernels changed for the profile port.

The baseline reconstructs the normal two-topic trace plus an immutable snapshot
of the current request-unit score database, with prior 4 and mean observed
history weight **75.16%**. A CPU reconstruction verifies the exact baseline
expert-ID fingerprint. New profiles use all 39 topics and no observed-history
blend. Thus baseline-to-topic comparison changes both prior and selection
policy; frequency-to-saliency comparison isolates histogram family at the same
maxmin policy and topic list. They replace 3,085 / 3,036 of the baseline's
resident IDs; frequency and saliency differ by 1,277 IDs.

The frozen protocol uses MMLU-Pro direct-letter, nonthinking, greedy, max 16
output tokens, one question per category. Seeds 20261008 and 20261009 and their
manifests were recorded before responses. The first-set winner alone receives
separate validation; a candidate must beat baseline there before promotion.
No topic subset, keep budget or ranking parameter was tuned on these answers.

| Expert selection | First set, 14 questions | Separate set, 14 questions | First-set raw miss rate |
| --- | --- | --- | --- |
| Current trace + router-score history, frozen | 8/14 | 12/14 | 13.89% |
| 39-topic frequency + maxmin, fixed | 10/14 | 9/14, including one format failure | 27.19% |
| 39-topic saliency + maxmin, fixed | 8/14 | Not selected for validation | 27.84% |

Frequency recovered business 857 (G -> I) and law 1621 (I -> G) in the first
set with no newly wrong correct answers. Chemistry 4225 changed between wrong
answers. Saliency recovered business but lost biology 3425 (C -> D); chemistry
and psychology changed between wrong answers. Its net score tied baseline.

On the separate set, frequency lost engineering 11975 (E -> B) and health
6764 (B -> D). Chemistry 3555 started explaining the Joule-Thomson coefficient
instead of returning a letter and reached the unchanged 16-token limit.
The original harness stopped there; its partial result is retained, and an
explicit continuation ran only the remaining 11 questions. **No response was
retried or given extra tokens.** All 14 observations are combined in
`counts-validation-complete.json`; the invalid response is a failure in the
all-14 denominator, not silently excluded as in valid-only accuracy.
Physics changed between wrong answers. No baseline error was recovered.

This is a small comparison, not an aggregate model-quality estimate. It rejects
these fixed all-topic profiles as a serving default under the predeclared rule.
The first-set result also demonstrates that raw miss rate alone cannot rank
quality: frequency had almost twice the misses but two more correct answers.
All quality requests reported zero expert NVMe reads and expert generation zero.

### Short generation and timing checks

Both topic profiles generated run-length encoding Python that parsed and passed
seven cases, including empty and Unicode strings. Frequency gave exact 1–20
counting outputs on all three short requests and converted `nihongomo ikeru?`
to `日本語もいける？`. Saliency passed two exact counting checks; one echoed
`run_1:` before otherwise correct numbers. It wrote `日本語も行ける？`, which
fails the selected exact kana-form check but is a lexical variation, not an
encoding error. Both outputs were valid UTF-8. These checks catch practical
format/code failures; they do not establish broad quality.

The same LRU-code and coastal-climate prompts were also timed at 128 output
tokens, with one warmup and two measured repetitions per arm, prefix reuse off.

| Workload | Baseline median request / decode rate | Frequency/maxmin median request / decode rate |
| --- | --- | --- |
| LRU Python code, 128 output tokens | 3.234 s / 44.85 tok/s | 3.996 s / 39.62 tok/s |
| Coastal climate prose, 128 output tokens | 6.895 s / 19.44 tok/s | 5.894 s / 22.87 tok/s |

Outputs differ between policies and are truncated by the token cap, so this is
an exploratory throughput comparison, not a passing code-quality test or a
universal speedup. The functional Python qualification above uses a complete,
separate task. Saliency did not qualify for this additional timing comparison.

Protocol, manifests, original/continued raw responses, statistics provenance,
selection IDs, qualification code and condensed comparison are under
`results/expert-profile-20261006/`. See `protocol.json`, `summary.json`,
`selection-comparison.json`, `upstream-parity.json`, `*-qualification.json` and
the `*-discovery.json` / `*-validation*.json` measurements. The production
demand file and original environment stayed unchanged throughout the trials.
The optional profile implementation remains available; the new image is used
with the original adaptive selection settings.

Final production health passed for `deepseek` on TP2. All original numerical,
budget, adaptive swap and prefix settings match the pretrial service; only the
image changes to include the optional profile code. The exact initial expert
IDs reproduce the frozen baseline, and both running ranks match the workspace
on seven runtime modules. No experimental profile, streaming, mask control,
layer allocation, calibrated rescue or confidence policy is enabled.
Normal-service counting, Japanese conversion and Python checks passed, including
seven functional code cases. One preliminary functional check stopped while
post-request adaptive work was still busy; the retained retry waits for idle and
is not included in the frozen timing comparison. Production history resumes
its normal updates after restoration. TensorFold remains stopped.
Final evidence: `runtime-source-verified.json`, `production-runtime-env.json`,
`production-restored-health.json`, `production-qualification-retry.json` and
`production-final-health.json`. `git diff --check` passed.

## 2026-10-06: lower priority for Mia's 2-bit layers

Parsed the already-present EXL3 quantization metadata, SHA256
`2949806ec66ff269a539caebe8afc55ba7d7c7220e011edc478fa67812a95ae1`.
Every routed w1/w2/w3 matrix in layers **18–22** uses 2 bits (1,152 matrices per
layer); the other 35 backbone layers use 3 bits. This is quantization tolerance
evidence, not a measurement of expert-omission damage.

One fixed **0.8 layer-priority factor** was chosen before responses. Normalizing
to the same 9,400 residents yields **193 per layer in 18–22, 241 elsewhere**,
instead of uniform 235. It removes 210 slots from those five layers and spreads
six additional slots to each other layer. Lowering a layer's scalar scores with
an unchanged uniform quota would not change its selected experts, so the test
uses the existing explicit layer budgets and their TP2 boot guard. No runtime
engine code, weights, numerical kernels or format changed for this test.

Both arms use native FP4 TP2, 90.1 GB arena, 16 transient slots, the same trace,
sum prior, router-score ranking and immutable current-history seed
`beef789badad39c7c29b2eb5b1576ad1ea4ef9cb1d2e189345cc2012d06858e9`.
Every swap/rescue/streaming trigger, prefix cache and prefill graph is disabled.
Expert generation stays zero. History is copied to isolated arm databases, and
the production environment and history file remain untouched during testing.

The protocol predefines fresh MMLU-Pro seed **20261010**, one question in each
of 14 categories, plus the five previously identified pruning-sensitive cases:
math 7867, philosophy 11054, business 834, engineering 12071, history 5003.
All requests are nonthinking, greedy, direct-letter, max 16 output tokens.
The candidate must improve fresh accuracy with valid responses, avoid regression
on the known-case total, and pass short functional checks to be promoted.
No discount or topic was retuned after responses.

| Allocation | Fresh questions | Prior sensitive probes | Total |
| --- | --- | --- | --- |
| Uniform 235 per layer | 8/14 | 2/5 | 10/19 |
| Mia-informed 193 / 241 | 7/14 | 3/5 | 10/19 |

All 38 responses were valid. The candidate lost law 1501 (E -> H) on the fresh
set, recovered history 5003 (F -> C) on the probes, and changed engineering
12071 between wrong answers (A -> G). Every other answer matched. The new
current-history baseline already answers math and business correctly; older
history's baseline results are therefore not appropriate controls here.

The policy fails the predeclared fresh-set criterion and stays disabled. This
is a measured tradeoff for one discount, not evidence that all bit-informed
priorities fail. Compression keeps each selected expert's contribution while
changing its weights; pruning changes which expert contributes. The latter
needs its own quality measurement. No full streaming, TensorFold launch,
model downloads, conversion or new service was used.

Metadata, fixed quotas, manifest, seed hash, raw arm responses, source provenance
and answer changes: `results/mia-layer-priority-20261006/protocol.json`,
`layer-counts.txt`, `*-fresh.json`, `*-known*.json`, and `summary.json`.

During this test, a separate user-message extraction helper was added and tested
without changing serving or expert ranking. It selects the last actual user
message from the original API input before the model encoder merges tool results
or treats mid-conversation system messages as user-like. Nontext attachments,
tool results and other messages are excluded. It preserves original messages
and exact text spacing/Unicode. Six focused tests passed. Unmarked documents
inside the same plain-text field require a client boundary. The current router
recorder still counts the full request; token-scope/priority integration is a
separate change. See `server/latest_user.py` and `docs/gotchas.md`.

Final health passed on the original `deepseek` TP2 configuration: uniform layer
budget restored, all experimental policies off, original adaptive swaps and
prefix settings on. Its exact initial expert IDs match this trial's frozen
baseline. The production environment and saved history still match the pretrial
copies. Both containers are running; TensorFold remains stopped. Final record:
`results/mia-layer-priority-20261006/production-final-health.json`.
`git diff --check` passed.

## 2026-10-06 — latest-user discovery, protected admission and 5% layer discount

Implemented `DSV41_USER_PROMPT_STREAM`: verified tokenizer ranges locate only
the last original user's explicit text before tool merging. During a streaming
discovery prefill, those rows receive full expert routing. Their normalized
gate-score mass orders admissions within the fixed layer quotas. Admitted used
experts are protected from prefill, decode and idle adaptive eviction until the
next request replaces the protection. Older messages, system/developer input,
tool results and structured attachments are excluded from this temporary score;
the conversation is still available for attention. Text-only and concurrency 1
are currently required. Plain-text documents inside that same user text cannot
be separated without a client boundary.

Mia's 2-bit layers 18–22 receive relative layer weight 0.95, others 1.0.
Largest-remainder rounding preserves 9,400 residents: 225 in those five layers,
237 in layers 0–14 and 236 elsewhere. This moves 50 slots compared with uniform
235/layer. The 90.1 GB arena remains fixed; 160 transient slots accommodate all
cold experts in the smallest layer while retaining all residents. Native FP4
checkpoint weights were unchanged. No model downloads or TensorFold launch.

The first prototype loaded/promoted after first-answer logits were computed.
It scored 10/14, losing engineering and math while recovering business. That
timing is too late for a direct-letter answer. Version 2 therefore rebuilds the
prompt once using the new resident set after admission, with streaming and
demand collection paused. Both the encoder state and the first answer now use
the chosen residents. The normal 128-token bounded decoder replay stays on;
upper-layer discovery rows are the user-text intersection with that window.
Protection/plan broadcast is unconditional on both ranks, including empty
selections; the feature and version are in the TP2 boot guard.

| Policy | Correct / 14 | Valid / 14 | Mean question wall time |
|---|---:|---:|---:|
| Frozen uniform control | 11 | 14 | 1.530 s |
| Frozen 5% discount only | 11 | 14 | 1.551 s |
| Prompt admission v1, rejected first-token timing | 10 | 14 | 8.944 s |
| Prompt admission v2 + 5% discount | 12 | 14 | 8.659 s |

All arms started from immutable history SHA256
`beef789badad39c7c29b2eb5b1576ad1ea4ef9cb1d2e189345cc2012d06858e9`, with
other adaptive swaps and prefix/response caching disabled. Questions were fixed
before answers (seed 20261011; manifest
`3e60f71315a4d09288a50a36cab1ae986942951aae2cc3af0aa7feaf2b777443`).
Admission arms retain their promoted masks across this ordered small suite,
as the real policy does. Version 2 reused the same questions to assess the
timing repair; this is exploratory, not an independent holdout or broad quality
claim. The discount alone changed no answers. Final admission recovered
business q671 (H→D) and psychology q2367 (F→I), but lost engineering q11754
(I→A). Law q1581 remained wrong. No response was retried or discarded.

Three short counting requests per arm (one warmup, two measured) averaged
1.465 s total baseline vs 2.886 s final policy, with prefill 0.408→1.869 s and
decode 0.985→0.952 s. Outputs matched the 1–20 instruction. Japanese conversion
passed and the generated Python RLE function passed seven cases, including
Unicode and empty input. With no user selection/admission, full resident prompt
rebuilds at 6 and 426 tokens reproduced the original logits exactly. The final
question arm promoted 689–1,490 experts per request, protected 4,814–6,645,
and reported at most 74 layer/expert overflow entries. All decode expert cache
miss counters were zero: decode stays resident, rather than streaming.

62 focused unit/integration checks and 17 mock-server end-to-end checks passed.
The initial benchmark mistakenly required all 40 layers to process all user
tokens despite bounded replay. Its assertion and first completed response were
kept; the corrected check verifies exact window intersections and the suite
resumed without repeating that question. Version 1 responses remain saved.

The user-requested policy and 5% discount are retained as an explicit pilot,
with the original production demand DB and adaptive settings restored. Required
prefix/disk/response caches and prefill graphs stay off; transient slots are
160. The policy offers a small measured net gain and substantially higher prompt
latency. It does not establish that prompt router mass reliably predicts all
later answer experts. Detailed protocol, raw responses, functional checks,
source provenance, timings, rejected version and pilot settings are in
`results/user-prompt-priority-20261006/`.

Final live pilot health passed after restoring the original adaptive settings.
A multi-turn counting check had 567 prompt tokens but only 15 selected latest
user tokens: all 40 scheduled layers streamed those 15 rows, admitted 105
experts and protected 1,604. Prefill and end-of-request adaptation also applied
261 and 114 swaps, respectively, without evicting protected experts; the pilot
remained healthy with those 1,604 still protected after idle maintenance. The
five implementation modules match byte-for-byte on both running ranks and the
workspace. Native `deepseek` serves on port 8000; TensorFold remains stopped.
Final state: `pilot-smoke.json`, `pilot-final-health.json`, `rank*-source.json`,
`pilot-changes.json`. `git diff --check` passed. Benchmark votes used separate
demand DBs; the live pilot resumed the original DB and its qualification request
contributed normally to that history.

## 2026-10-06 — 100 resident loads and fully dynamic sector allocation

The expert arena already maps `(layer, expert)` to fixed-size slots. TP2 stores
9,400,320 bytes per slot per GPU; the peer stores the matching shard. Added
cross-layer transfers, a global demand-based initial allocation, and
`DSV41_DYNAMIC_EXPERTS`. Layer counts can grow/shrink without changing the total
9,400 residents or 90.1 GB arena. No fixed quotas or Mia discount remain in the
dynamic policy; six residents per layer is the routing minimum. Decode mask/LUT
addresses remain stable. Changed layers' eager compact prefill maps are rebuilt.

Full latest-user discovery now uses rank-0-authoritative token batches whose
unique cold experts fit the transient ring, allowing layers below the previous
224-resident capacity bound. Each row still executes all its chosen experts.
No decode streaming, new quantization, checkpoint download or TensorFold launch.
Both the policy and global-allocation version/settings are in the TP2 boot guard.

`DSV41_USER_PROMPT_MAX_LOADS=100` bounds logical resident expert replacements
across the entire request, including ordinary prefill/decode/end/idle adaptation.
Both TP ranks apply the same 100-expert plan to their own half weights. It is
neither a per-layer limit nor a total I/O limit: transient discovery stays
uncapped. Deferred absentees are not protected as if they were resident.

Same ordered 14-question MMLU-Pro set (seed 20261011, greedy max16), history
SHA256 `beef789badad39c7c29b2eb5b1576ad1ea4ef9cb1d2e189345cc2012d06858e9`,
normal adaptive swaps disabled, cache off. One new fixed-cap arm and one new
dynamic arm; compared with saved uncapped v2, without retries or answer-driven
tuning. This is exploratory reuse, not a holdout. Dynamic also removes the
discount and changes startup selection, so the comparison does not isolate
the effect of cross-layer transfers.

| Whole policy | Correct / 14 | Valid / 14 | Mean question wall time |
|---|---:|---:|---:|
| Saved uncapped prompt v2, fixed quotas + 5% discount | 12 | 14 | 8.659 s |
| Fixed quotas + 5% discount, cap 100 | 11 | 14 | 6.338 s |
| Dynamic allocation, no discount, cap 100 | 10 | 14 | 6.998 s |

The fixed cap lost psychology q2367 (I→F). Dynamic additionally lost math q8468
(E→I); all other direct answers matched the fixed-cap arm. Every request used
exactly 100 admissions. Dynamic made 95–100 cross-layer transfers per question
(mean 97.57), with resident counts 206–271 at startup and 203–260 after the
suite, total 9,400 throughout. Promotion mean was 0.327 s fixed-cap versus
0.451 s dynamic. Resident rebuild mean was 0.775 versus 2.201 s. Discovery
recorded 878–1,893 cold load events fixed-cap and 869–1,861 dynamic: the 100 cap
does not remove streaming discovery cost. Neither quality nor speed improved
over the fixed-cap arm on this small workload.

Three counting requests (one warmup, two timed) averaged 2.682 s fixed-cap and
3.382 s dynamic. Counting, Japanese conversion and the generated RLE function
passed, including seven empty/Unicode cases. No-user resident rebuilds at 6
and 426 tokens had exact logit delta zero. 57 focused checks and 17 mock-server
tests passed. A separate six-token real-weight TP2 gate forced three discovery
batches with six residents and eight transient slots: both ranks exactly
matched an all-resident MoE, max absolute delta 0, resident directory unchanged.
The first attempt to run that gate beside the live arena failed CUDA context
initialization with OOM on both ranks; the successful gate ran during restart.

At the user's explicit request, dynamic allocation stays enabled despite the
measured regression. Original production history and ordinary adaptive settings
are restored for live evaluation; benchmark demand files are separate. No
parameter search or quality-based rollback. Protocol, all responses, timings,
source hashes, batch gate and live qualification are in
`results/dynamic-experts-20261006/`; the fixed-cap comparison is in
`results/user-prompt-cap-100-20261006/`.

Final live qualification passed with the original production DB and ordinary
adaptation enabled. The 567-token multi-turn request selected only 15 latest
user tokens, admitted 89 experts and protected 1,599. Ordinary adaptation used
the remaining 11 loads: final total 100, remaining budget zero, resident total
still 9,400. Both ranks passed the 145-field boot guard; seven implementation
modules match the workspace byte-for-byte. `deepseek` is healthy on port 8000
with context 524,288 and dynamic mode retained. TensorFold's stop marker remains.

The launcher twice returned curl exit 23 after reporting healthy startup:
`curl | head` under `pipefail` stopped reading the enlarged health response.
`dual-up.sh` now reads the complete response before truncating its display,
using the configured bind address; `bash -n` and `git diff --check` passed.
The failed display exits and successful independent health checks are retained.

## 2026-10-06 — Resident prefill with dynamic allocation retained

The custom-harness incident had 15,434 total prompt tokens and 9,847 selected
latest-user tokens. Discovery was still running after more than 113 seconds,
before the first answer token. Role selection had not included every token,
but mixed chunks entered the host streaming path and full discovery cold reads
were unbounded. The 100-load limit covered resident replacements only. The
incident record contains aggregate counters, not the user's prompt.

At the user's request, `DSV41_USER_PROMPT_STREAM=0` is now persisted on both
nodes. Dynamic allocation no longer requires prompt streaming. The resident
replacement budget remains active when streaming is off; previously its
disabled feature flag also disabled that budget. Global startup/adaptation,
the six-resident layer floor, no fixed quotas/discounts, and the total 9,400
resident experts remain. Prompt discovery and the extra first-answer rebuild
are off. Policy versions: user prompt 5, dynamic allocation 3.

Two predeclared synthetic counting requests used normal production adaptation,
the original history DB, greedy max32 and thinking off. Both returned the exact
sequence 1 through 10. The cold 38-token request took 8.553 s including initial
graph warmup (prefill 6.699 s); the 10,014-token request took 41.326 s (prefill
40.465 s). Both had zero discovery loads, zero streamed rows/batches and zero
store resolves, and remained within 100 resident replacements after normal
adaptation completed. Neither rebuilt prefill for prompt priority. These are
operational checks, not a quality benchmark or a matched speed comparison with
the incident. Resident compute still scales with prompt length; routed pruning
misses remain even when the streaming I/O miss counters are zero.

59 focused tests and 17 mock-server tests passed. Both TP2 ranks passed the
145-field boot guard and seven implementation module hashes matched the
workspace. The normal launcher exited successfully; TensorFold stayed stopped.
Artifacts: `results/dynamic-resident-20261006/`. At this stage RAM/disk/response
prefix caches were still off; the RAM restriction is addressed below.

## 2026-10-06 — RAM prefix reuse with dynamic expert allocation

The constant 0% prefix reuse was caused by the persisted `PREFIX_CACHE=0` and
an overly broad dynamic-mode boot guard inherited from prompt streaming.
RAM snapshots store encoder state, not expert-sector pointers. Resident
prefill can therefore reuse historical KV after global sector transfers, as
the original within-layer adaptive policy already did. Removed the RAM-cache
restriction for dynamic allocation; prompt streaming still requires caches off.
Disk/response caching and prefill graphs remain outside this prototype.
Dynamic policy version 4 and `PREFIX_CACHE=1` are in the TP2 config guard.

Three predeclared synthetic counting requests used native TP2, normal
production adaptation/history, no prompt streaming, cap100, greedy max32,
thinking off and seed 20261013. Both ranks applied the same global plans.

| Request | Prompt tokens | Cached tokens | Prefill | Wall time |
|---|---:|---:|---:|---:|
| Cold base, includes graph warmup | 2,289 | 0 | 13.548 s | 15.319 s |
| Exact repeat | 2,289 | 2,289 (100%) | 0.381 s | 1.346 s |
| Extended conversation | 2,329 | 2,289 (98.3%) | 2.364 s | 2.927 s |

All three returned the exact sequence 1 through 10. Expert generation advanced
0→1→2→3, with 100 resident replacements per request and 9,400 residents
throughout. Discovery loads, streamed rows and store resolves remained zero;
there was no priority rebuild. These verify cache use across changing resident
sets, not equality with fresh prefill under a new selection or general quality.
The cold timing includes compilation; it is not a steady-state throughput arm.

40 focused cache/media/disk/global/prompt tests and four prefix-state function
checks passed. The new CPU integration check applies a cross-layer transfer
on each TP rank, restores the historical encoder snapshot and verifies that
the new expert directory remains active. Both live ranks passed the 145-field
boot guard; seven module hashes match both ranks and the workspace. The pair
was restarted successfully with RAM prefix caching enabled on both nodes;
TensorFold remains stopped. `git diff --check` passed. Artifacts and all three
synthetic responses: `results/dynamic-prefix-20261006/`.

## 2026-10-06 — Correct the cap to streaming cold loads only

The requested cap was on temporary streaming loads. Applying it to generic
resident adaptation was a scope error. Prompt policy 6 removes the request-wide
resident clamp from `plan_swaps`, fixed-layer swaps and global transfers.
Prefill, periodic/urgent decode, request-end and idle adaptation now use their
existing rules. The previous cap could consume all 100 loads during prefill
and silently leave urgent adaptation with an empty plan.

Bounded latest-user streaming now selects cold experts before routing and
shares the allowed set and cold-load budget across TP2 ranks. Common transient
hits are free; each selected cold set fits the ring. Later calls use resident
routing once the cap is spent. This online policy prioritizes current rows,
not future layers. Streaming remains OFF in production; RAM prefix reuse and
fully dynamic allocation stay ON. The configured 100 cap is consequently
inactive during ordinary production adaptation.

70 focused tests passed, including the real store resolver with cap1 on both
simulated ranks, a 101-transfer resident plan after exhausting the stream budget,
and urgent decode planning with the exhausted budget. Two short live counting
checks (19 prompt tokens, greedy max32) returned the exact count: resident loads
were 160 then 110, with zero streaming loads. The repeat reused all 19 tokens.
Wall times were 6.664 s (cold graph warmup) and 1.636 s; no quality or speed gain
is claimed. Urgent adaptation is enabled at its existing 10%/~30-token threshold
and 150-token cooldown; the short live responses did not exercise its trigger.
Both ranks passed the 145-field boot guard and seven source hashes matched.
Artifacts: `results/stream-load-cap-20261006/`.


## 2026-10-06 — Live expert memory map

`/expert-map` now displays all 15,360 routed experts and all 9,584 arena sectors.
`/v1/expert-map` exposes the same snapshot as JSON. Resident, transient, missing
and actively loading experts have distinct states; a bounded completion history
keeps fast loads visible for three seconds. The page polls once per second,
with layer/expert inspection, sector view, zoom, pause and state highlighting.
It reads rank 0 CPU metadata without the generation lock, CUDA reads or new
collectives. This is an approximate live snapshot, not an independent TP peer
audit. Missing from the arena is not the same as a routed miss.

45 focused tests, five HTTP/map tests and 17 mock server tests passed. A short
19-token counting request returned the correct sequence while 30 map snapshots
were sampled: 98 loads recorded, active transfers observed, 9,400 final residents,
5,960 missing and zero streaming loads. Median snapshot fetch was 15.292 ms,
maximum 90.117 ms in this one check; no inference speed or quality improvement
is claimed. Browser checks covered both views, selection, zoom and pause/resume;
no console errors were reported. Inspection also caught live polling overwriting
number inputs; focused input text is now preserved and selection updates on input.

Image `expert-map-v8` is persisted on both nodes. The final HTML was hot-updated
in both running containers and included in the rebuilt image for future starts.
Artifacts, snapshots, test logs and screenshot: `results/expert-map-20261006/`.


## 2026-10-06 — Native TP2 and TTS coexistence

Reduced the native expert arena from 90.1 to 88.7 GB per node and transient
slots from 160 to 32. Capacity is now 9,435 sectors; all 9,400 resident experts
still fit. The arena allocation decreases by 149 sectors × 9,400,320 bytes =
1.401 GB per node. Dynamic allocation, prompt streaming OFF, 524,288 context
and RAM prefix caching remain unchanged. TTS uses a 3,072 token buffer,
reduced from 4,096 at the user's request; the 2,048 proposal was superseded.

Starting TTS after DeepSeek had loaded and generated failed at CUDA memory
initialization (`cudaMemGetInfo` reported out of memory), despite Linux reporting
reclaimable memory. This failed order is not evidence that the two services
cannot coexist. After stopping the pair, starting and warming TTS first, then
starting DeepSeek, both came up with the requested settings. No model downloads
or expert-budget reduction were needed. Both TP ranks passed the 145-field
config guard and loaded 9,400 residents.

One short text request and one short speech request were submitted concurrently.
The 19-token count prompt (max32, greedy) returned exactly 1 through 10 in
7.944 s including cold graph work. TTS returned a 153,644-byte mono 24 kHz PCM
WAV in 7.532 s. This checks coexistence, not sustained-load capacity, audio
quality or throughput improvement. Host MemAvailable afterward was about
7.8 GiB; swap was already in use. TTS buffer savings were not separately
measured. Artifacts: `results/tts-headroom-20261006/`.


## 2026-10-06 — One previously failed math question on the live engine

At the user's request, repeated only MMLU-Pro math question 7867, selected
before its new response from the earlier `score-uniform.json` run. Identical
prompt/options, greedy temperature 0, top_p 1, thinking off, max16; one request,
no retries, calibration or extra diagnostic prompts. The user explicitly
accepted this request contributing to production adaptation.

Earlier frozen uniform allocation answered A (-75.98), incorrect. Current live
dynamic allocation answered J (-153.59), matching the dataset key, in 2.723 s.
The response was a valid single letter with stop termination. This is one
historical failure recovered, not an overall benchmark score or a causal
quality estimate: history, placement and other serving settings differ, and
the question was selected because it previously failed. Exact protocol, raw
response and before/after health: `results/single-math-20261006/`.


## 2026-10-06 — Package the learned distribution for fresh starts

Captured the idle generation-44 working set: 9,400 exact resident expert IDs,
179–285 residents per layer, plus the current aggregate request-unit counts and
router-score mass. The 231 KiB `profiles/learned-experts-v1.npz` includes no
prompts, responses, KV caches or model weights. Its companion JSON records the
layer counts, capture generation and checksum. The source history includes the
single math retest the user explicitly authorized earlier; this export does not
claim to be an uncontaminated benchmark arm or a generally optimal distribution.

Fresh dynamic score/request-mode starts use the bundle only if the local demand
DB path does not exist. At the 9,400 budget the exact captured map is restored;
other budgets use normal global selection from the seeded demand blend. The
same history initializes the model accumulators, so ordinary future write-back
creates the new user's own DB. Existing DBs always bypass the bundle, even if
incompatible or unreadable; explicit seed opt-out also bypasses it. The original
trace remains the adaptation prior. No local production DB was replaced.

Rank 0 alone reads the seed and sends it in the existing ranking broadcast;
seed version, enabled flag and checksum join the boot guard. The example config
now selects 0.61 keep, dynamic score placement and compatible cache switches.
32 CPU tests passed, including the real boot-ranking segment with conflicting
peer history, exact expert IDs, budget resizing, opt-out, invalid seed rejection
and local-history precedence. Both nodes' new `learned-seed-v9` images load the
identical seed checksum and 9,400 IDs in an offline CPU check. Both launch files
point to that image for the next restart. The live engine and TTS were not
restarted, and no inference/quality benchmark was run for this change.
Artifacts: `results/default-distribution-20261006/`.


## 2026-10-06 — 1,024-token prefill chunks beside TTS

After the live 20,855-token session exhausted host headroom with 2,048-token
chunks, reduced only `DSV41_PREFILL_CHUNK` to 1024 on both nodes. The 524,288
context capacity, 88.7 GB arena, 32 transient slots, all 9,400 residents, dynamic
allocation, prefix caching and TTS max-seq-len 3072 remain unchanged. No watchdog
relaxation or CUDA-cache-reclamation code was added.

A bounded text-only memory check used a copied demand DB: 20,775 prompt tokens
with no cached prefix, then a 20,801-token extension producing 319 tokens while
TTS synthesized one short sentence pair. Full prefill took 38.475 s; extension
reused 20,480 tokens (98.5%) and prefilled in 2.646 s. Both text responses met
their simple format checks, TTS returned a 399,404-byte WAV, and both ranks
remained healthy. The main host was sampled every 0.25 s: minimum MemAvailable
6.948 GB, final 7.404 GB. This is one memory coexistence check, not a matched
speed comparison, quality benchmark or a guarantee for 524k/vision workloads.
An urgent decode pass also occurred (64 swaps at output token 187), without
exhausting the reserve.

The production demand file remained byte-identical through the test. The test
instance was stopped and production relaunched from that original history.
Artifacts and memory samples: `results/memory-chunk-20261006/`.


## 2026-10-06 — Peer-only vision and 100 more dynamic residents

Added optional `DSV41_VISION_MODE=peer` for TP2. Rank 1 keeps the vision tower,
aligner and delimiter embeddings; rank 0 keeps only metadata for input preparation.
Both ranks exchange encoding status before broadcasting the complete image span.
The boot guard includes ownership mode, enabled state and protocol version; an
owner load/encode failure fails the pair rather than leaving the head waiting for
an absent tensor. Text-only requests have no added vision collective.

A standalone paired-GPU probe compared replicated and peer-only splice using
synthetic patch tensors at 546×546 (184 embedding rows) and 1176×1344 (926 rows).
Seven measured iterations per mode followed two warmups, alternating mode order,
with synchronized timings covering the slower rank. Every output was bit-exact.
Median replicated→peer times were 57.588→57.615 ms (+0.047%) and
496.262→462.887 ms (−6.73%). These are tower/splice timings, not full-request
throughput measurements. Dropping rank 0's tower released exactly 970,536,960
allocated bytes. No inference history was read or modified by that probe.

Added `DSV41_RESIDENT_EXPERTS` for an exact global dynamic budget, checked against
the routing floor/model capacity and included in the boot guard. This avoids
rounding the intended 9,500 to a multiple of 40 through `PRUNE_KEEP`. Static
allocation retains its existing budget semantics. The existing resident-capacity
check still requires the complete selected set to fit on each TP rank.

Local configuration uses 9,500 residents, arena 89.7 GB, 32 transient slots,
Engram caches 256 MiB / 1,024 MiB and peer-only vision. The extra resident weights
cost 940,032,000 bytes per rank. Cache budgets free 805,306,368 bytes on rank 0
and 3,221,225,472 on rank 1; combined with removed vision weights, this exceeds
the 1 GB arena increase by about 0.776 GB / 2.221 GB respectively. These are
allocation-budget differences, not guarantees about peak available memory.
26 focused CPU tests passed (vision ownership/error ordering, exact budget,
learned-seed selection and global residency). Artifacts: `results/peer-vision-20261006/`.

An API smoke check with a synthetic red square returned `Red` while TTS returned
a valid 126,764-byte WAV. It used a copied demand database; production history
remained byte-identical. At 0.25 s sampling, rank 0 MemAvailable stayed at least
7.701 GB during this short concurrent check. Both ranks loaded all 9,500 residents
and agreed on all 152 guarded fields; logs confirmed 256/1,024 MiB Engram caches.
The isolated test was stopped before restarting against the original production
history. This does not measure long-context multimodal headroom or quality gains
from the additional experts, nor decode-speed effects of the smaller row caches.


## 2026-10-06 — Increase the live dynamic budget to 9,550

Raised the exact resident budget from 9,500 to 9,550 and each arena from
89.7 to 90.2 GB. The extra 50 TP expert shards use 470,016,000 bytes per node.
Kept 32 transient slots, peer-only vision, 256/1,024 MiB Engram caches, 1,024-token
prefill chunks and the 524,288 context limit. Both ranks restarted, loaded all
9,550 residents and agreed on 152 guarded fields; the API and expert map confirmed
9,550 residents (180–292 per layer at startup). Production demand history remained
byte-identical through the restart, and TTS remained running. No inference or
quality benchmark was run for this configuration-only increase. Artifacts:
`results/residents-9550-20261006/`.


## 2026-10-06 — Reassign 24 transient sectors to residents

Reduced `TRANSIENT_SLOTS` from 32 to the supported minimum of 8 and increased
`DSV41_RESIDENT_EXPERTS` from 9,550 to 9,574. The 90.2 GB arena stays unchanged:
9,574 residents + 12 spare resident slots + 8 transient slots + 1 null slot =
9,595 sectors. Streaming, critical rescue and prefill replicas are off; normal
global adaptation overwrites donor resident slots directly and does not consume
the transient reserve. This configuration does not establish capacity for a
future unbounded streaming workload.

Both ranks loaded all 9,574 residents, passed the existing 152-field config guard
and became ready; the peer launch environment was also checked for the matching
8-slot reserve. API and expert map confirmed the new allocation. Learned demand
history remained byte-identical and TTS remained running. No inference benchmark
was run for this configuration-only change. Artifacts:
`results/residents-9574-20261006/`.


## 2026-10-06 — One live retest of computer-science question 10632

Retested the exact saved MMLU-Pro prompt and option order once with the live
9,574-resident dynamic engine, temperature 0, top_p 1, thinking disabled and
max_tokens 16. It returned `H` (incorrect; frozen key `F`) in 4.874 s, with 168
prompt tokens and one output token. Historical custom passes both answered `I`
(incorrect); Mia's two saved passes answered `H` (incorrect); the official API's
two saved passes answered `F` (correct). This item did not improve to a correct
answer. No repetition or alternative prompt was tried, and the reference
endpoints were not rerun. The user accepted the single request's contribution
to live adaptation. Prompt identity, protocol, response and health snapshots:
`results/single-cs10632-20261006/`.

At the user's request, two additional identical live attempts returned `H` /
`H`, both incorrect (0.963 s / 0.798 s). Each reused all 168 prompt tokens from
the RAM prefix cache, with adaptation active (reported generations 108 / 109).
They are cached repeats, not independent full-prefill evaluations of the updated
resident sets. Across the three live attempts, correctness was 0/3. Raw responses,
health and protocol are in the `attempt-2/` and `attempt-3/` subdirectories of
`results/single-cs10632-20261006/`. No further requests were sent.


## 2026-10-06 — One live retest of law question 1911

One request using the exact historical MMLU-Pro prompt and option order,
temperature 0, top_p 1, thinking disabled and max_tokens 16 returned `F`
(incorrect; frozen key `A`) in 3.071 s. Historical custom passes both returned
`A` (correct), Mia's two passes returned `C` (incorrect), and the official API's
two passes returned `A` (correct). This is a worse result on this individual
question than the earlier custom runs; it does not isolate the effects of
resident count, learned history or other intervening settings. No repeated
request or reference-endpoint rerun was made. Protocol, paired historical
answers, response and health snapshots: `results/single-law1911-20261006/`.

One user-requested repeat of 1911 returned `F` again (incorrect) in 0.878 s.
Reported request routing misses fell from 31.31% to 9.56%, and score-weighted
misses from 34.09% to 5.52%. However, the repeat reused all 405 prompt tokens;
recorded routing selections were 15,552 versus 66,582 on the first request.
These rates cover different executed work and do not establish a reduction on
fresh prefill of the full question. Reported expert generation advanced from
111 to 112. Artifacts: `results/single-law1911-20261006/attempt-2/`.


## 2026-10-06 — Thinking-on then thinking-off on law question 1911

Ran one user-requested best-case adaptation probe on the original question:
thinking on at effort 75, temperature 0/top_p 1, with a bounded 4,096-token
reasoning-plus-answer budget; then the exact historical thinking-off request
with max_tokens 16. Neither request included any answer or reasoning from the
preceding response. Prompt-template rendering differed near the beginning;
both runs reported zero cached prompt tokens without changing cache settings.

The thinking run used all 4,096 tokens for reasoning and returned no final answer
(finish=length), taking 148.695 s. It is incomplete, not a scored wrong answer.
During decode, six urgent passes loaded 150 experts in total, and three periodic
passes loaded 41. Prefill and post-response adaptation also remained active.

The subsequent thinking-off run answered `A` correctly in 4.703 s. Against the
original fresh thinking-off attempt (`F`, incorrect), request routing misses
fell from 31.31% to 5.74%, and score-weighted misses from 34.09% to 3.12%.
Both off runs executed 66,582 recorded routing selections and reused zero prompt
tokens. This is a positive single-item warmup observation, not a general quality
result or isolation of urgent adaptation: previous repeats, all other adaptation
paths and run-to-run variation remain confounders. The truncated thinking pass
does not establish whether thinking would eventually finish correctly. The
engine remained healthy. Artifacts: `results/law1911-thinking-pair-20261006/`.


## 2026-10-06 — Repeat thinking after the successful direct answer

One further thinking-on attempt of law question 1911 used exactly the prior
request body and rendered token IDs: effort 75, temperature 0, top_p 1, shared
reasoning/answer cap 4,096. The server default effort was confirmed as 75
(high); 60 maps to medium. The request did not contain prior reasoning or an
answer, and reported zero cached prompt tokens.

It again exhausted all 4,096 tokens in reasoning without a final answer,
finish=length, in 149.134 s (previous thinking attempt: 148.695 s). Treat this
as incomplete, not an incorrect selected option. Routing misses were 5.54%
and score-weighted misses 3.60%, versus 7.38% / 5.00% on the previous thinking
attempt. Better routing coverage did not make this bounded effort-75 attempt
finish. The earlier fresh thinking-off attempt had answered correctly, so the
observed additional thinking was unproductive within this budget; this does
not establish how a larger budget or a lower effort would behave. No further
inference requests were sent. Engine health remained OK.
Artifacts: `results/law1911-thinking-repeat-20261006/`.


## 2026-10-06 — Lower thinking effort to 50 for law question 1911

One follow-up used the same question, temperature 0, top_p 1 and 4,096-token
cap, with effort reduced from 75 to 50 and thinking explicitly enabled.
The prompt-debug endpoint confirmed thinking=true and effort=50. This was a
per-request setting; the server default remains 75.

The run finished normally in 17.397 s, using 411 reasoning tokens (413 completion
tokens total), and answered `C`, incorrect against the frozen key `A`. It reused
zero cached prompt tokens. Routing misses were 4.67%, score-weighted misses 2.79%.
Unlike both effort-75 attempts, it did not exhaust the budget; however, the lower
effort did not reproduce the earlier correct thinking-off answer. This is one
sequential live-adaptation comparison, not an isolated effort-only experiment.
No additional inference requests were made. Engine health remained OK.
Artifacts: `results/law1911-thinking-effort50-20261006/`.


## 2026-10-06 — Thinking effort 60 on law question 1911

One user-requested run used the same question, temperature 0, top_p 1 and
4,096-token cap with thinking explicitly enabled and effort 60. Prompt-debug
confirmed both settings. It finished normally in 60.963 s, using 1,549 reasoning
tokens (1,551 completion tokens total), and answered `C`, incorrect against the
frozen key `A`. Zero prompt tokens were cached. Request routing misses were
4.01%, score-weighted misses 2.42%. This completed within the cap, unlike the two
effort-75 runs, but used more reasoning than effort 50 (411 tokens) and selected
the same incorrect option. Sequential live adaptation makes this an exploratory
comparison rather than an isolated effect of effort. The server default remains
75 and the engine remained healthy. No additional inference requests were made.
Artifacts: `results/law1911-thinking-effort60-20261006/`.

### 2026-10-06 — Predictive prefill prototype: CPU cost and shadow selection

Added an optional bounded nearest-neighbor predictor that feeds provisional
prefill demand into the existing global ranking policy before model execution.
Actual post-prefill adaptation remains in place. Predictions never enter the
persisted demand history; measured observations are folded by the normal path.
This is an approximation, not an exact calculation of deep-layer routing from
input tokens. Apply remains opt-in pending evidence of selection quality.

Validation used copied demand history and separate predictor banks. Production
history remained byte-identical to the pre-test backup. The live test overrides
minimum examples to 1 to exercise the path; the normal minimum is 8. Every request
used synthetic inventory notes, thinking off, greedy decoding, max 16 output
tokens, and returned `READY`. These are integration checks, not answer-quality
benchmarks.

| Shadow request | Prompt / cached tokens | Prediction + planning | Result |
| --- | --- | --- | --- |
| Cold example | 10,260 / 0 | 12.250 ms | Collected first example; no prediction |
| Similar fresh example | 10,260 / 0 | 36.859 ms | Predicted 189 promotions vs 202 from actual demand |
| Cached extension | 10,349 / 10,240 (98.95%) | 1.355 ms | No similar suffix; safely skipped prediction |

On the similar fresh example, feature cosine was 0.999876, mean demand total
variation was 0.1113, promotion precision 76.72%, and recall 71.78%. Only two exact
incoming/outgoing pairs matched. The 145 shared incoming experts do not establish
quality equivalence: near-identical text still produced different routing after
normal adaptation changed the residents. Keep this negative result when considering
lower similarity thresholds or promoting the prototype to an apply default.
The cached-extension example had cosine 0.5213 to the two full-prefill examples;
reusing a prefix does not make new-suffix routing interchangeable with full-prefill
routing. The bank needs relevant continuation examples as well.

Observation/evaluation/persistence took 31.038, 81.101 and 78.927 ms respectively.
Request wall times were 24.657, 14.471 and 2.467 s. These are sequential requests
with different warmup/cache/residency states, not an A/B speed or quality claim.

The separate CPU feature+neighbor microbenchmark (128 entries, five timed runs
after warmup) measured medians 1.715 ms at 10k new tokens, 16.151 ms at 200k context
with 2k new tokens, and 22.424 ms at 524,288 context with 5,243 new tokens. It excludes
planning, expert weight I/O and model execution. Bank array storage was 17,301,504
bytes. No GPU kernel was introduced. Focused unit coverage: 51 passing tests.

Artifacts: `results/predictive-prefill-20261006/` (`feature-bench.json`,
`live-summary.json`, per-request health snapshots, isolated histories and logs).

A separate apply-mode integration probe restored the original pre-test demand
history and used the isolated bank above (three entries, minimum 1). A fresh
10,260-token near-variant loaded 362 predicted experts before prefill in
1,167.950 ms, following 41.177 ms prediction/planning. It returned `READY`, with
7.79% request routing misses and 6.88% score-mass misses (cold shadow probe:
21.81% / 20.26%). Prefill took 18.421 s, request wall time 19.702 s. Observed demand
TV against prediction was 0.08198; recording took 24.756 ms. Post-prefill correction
remained active. This validates the distributed early-load path but uses nearby
training examples and a trivial answer; it is not proof of preserved answer quality
or a controlled speed comparison. Weight I/O is much larger than feature cost.

Normal service uses the new image with `DSV41_PREDICTIVE_PREFILL=shadow`, minimum
8, and an empty production bank. It retains the original production history,
9,574 residents, eight transient sectors, prefix caching, peer vision and Engram
cache sizes. Synthetic predictor banks are not copied into the production bank.

### 2026-10-06 — Repository checkpoint and portable profile

Compared the local serving environment with `.env.example` using resolved
adaptation settings, not just raw variable names: adaptation now agrees exactly.
The example selects high sensitivity/prior 4, 9,574 dynamic residents in a 90.2 GB
arena with eight transient slots, a 524,288-token allocation, peer vision,
256/1,024 MiB Engram caches, native dense precision, and speculative depth 3/5.
RAM prefix reuse is explicit. Disk/post-response caches remain disabled; the
300-second disk-save interval is documented as inactive in this mode. Experimental
prediction stays opt-in in the template; the local server collects in shadow mode.
Machine addresses, credentials, optional ablation weights and local image tags
are not copied into the template. Older raw pruning overrides are omitted because
the adaptation knobs derive their effective values.

The README was reduced to setup, operation and the current profile. Detailed
adaptation notes and historical measurements remain under `docs/`, including
negative results. Pre-commit checks: 168 passed, three CUDA-only tests skipped
with GPUs hidden to avoid interfering with live serving. This comprises 117
focused unit tests, 30 persistence/response tests, four prefix-cache function
tests, and 17 mock-API tests. Shell syntax, relative documentation links and diff
whitespace checks passed. No new live quality benchmark was run for this commit.
