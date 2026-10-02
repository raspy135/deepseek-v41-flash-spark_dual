# Dynamic speculative depth

`DSV41_BLOCK_DYNAMIC=3,5` lets the engine choose the draft depth per request, as acceptance
changes. Code off by default; `.env.example` enables it. It is in the EP2 boot guard, and it
cannot be combined with `DSV41_BLOCK` or with `DSV41_MAX_CONCURRENCY` > 1.

## Why

The best fixed depth depends on how predictable the text is. The measurements below come from
TP2 (served config), greedy, 512 tokens, two samples per depth in the order 3, 5, 5, 3
(`tools/bench_decode_block_tp.py`, `results/decode-block-20260923-2223/`). The last column is
tokens per step at depth 3 → depth 5.

| Workload | Depth 1 | Depth 3 | Depth 5 | Tokens per step (3 → 5) |
| --- | ---: | ---: | ---: | --- |
| python | 22.4 tok/s | 35.8–36.1 | 44.3–44.6 | 3.87 → 5.65 |
| html | 21.6 | 31.9–32.2 | 34.0–35.0 | 3.38 → 4.29 |
| explain | 18.2 | 20.3–20.5 | 18.3–18.5 | 2.17 → 2.31 |
| story | 17.6 | 18.5–18.9 | 16.8 | 1.97 → 2.07 |
| story, temperature 0.7 | — | 16.7–17.4 | 14.2–14.5 | 1.96 → 1.92 |
| step time | 88 ms | ~106 ms | ~124 ms | |

- Depth 5 makes every step ~16% slower, and pays that back only where the drafter is right.
- Depth 1's step is 18% faster, but its 2-token cap loses on every workload. It was measured
  with a single sample and rejected.
- Depth 7 was not measured. It lies past the drafter's trained horizon of 5; see
  `docs/gotchas.md` on the wider verify block.

## How

1. The drafter always drafts the deeper block. The checkpoint was trained at 5.
2. The verifier checks the first `depth` drafts. Each verify width has its own static buffers
   and CUDA graphs (`FastDecoder._bind`), and the graph key is (parity, bucket, width).
3. A width's graphs are captured only at the first step that uses it. Capturing a wider width
   ahead of time would be wrong: its warm-up forward writes KV past the step, and in the
   128-slot window ring that overwrites entries the current step still attends to.
4. `engine/spec_depth.py` holds the policy. It re-decides at most once per
   `DSV41_BLOCK_DYNAMIC_TOKENS` (default 60) output tokens and needs a 3% win to switch.
   - **At depth 5**, it knows the depth-3 alternative exactly: `min(accepted, 3) + 1` tokens per
     step.
   - **At depth 3**, it estimates depth 5 from saturated steps, using the extra acceptance
     learned in this request.
   - It compares tokens per second, using step times measured on this process.
   - A request starts at `DSV41_BLOCK_DYNAMIC_START` (default 3).
5. Rank 0 decides. `EPDistributed.control()` now always broadcasts `[keep_going, depth]`,
   including on abort paths, so both ranks verify the same width.

Greedy output is identical under any depth schedule, because verification accepts exactly the
target model's argmax. With sampling the output distribution is unchanged, but the random stream
depends on the schedule.

## Full-engine check, 2026-09-23

`tools/bench_decode_dynamic_tp.py`, TP2, results in `results/decode-dynamic-20260923-2300/`.
This run includes the decode projection defaults (`docs/decode-projection-fusion.md`). For each
greedy workload, token-identical output was checked on both ranks across four modes:
alternating every step, pinned 3, pinned 5, and adaptive. All matched.

| Workload | Pinned 3 | Pinned 5 | Adaptive | Adaptive: steps at 3 / 5 |
| --- | ---: | ---: | ---: | --- |
| python | 35.5 | 44.5 | 43.1 | 15 / 81, 1 switch |
| html | 31.0 | 35.0 | 34.3 | 17 / 108, 1 switch |
| explain | 20.4 | 18.2 | 20.6 | 235 / 0 |
| story | 19.1 | 16.8 | 19.1 | 228 / 0 |
| story, temperature 0.7 | 18.1 | — | 19.4 | 250 / 0 (both at depth 3) |

- Adaptive gets about 97% of depth 5 on code. The gap is the first ~60 tokens, before the first
  decision.
- It never leaves depth 3 on prose.
- Drafting 5 and verifying 3 did not lower depth-3 acceptance compared with true fixed-3 runs:
  3.36 vs 3.38, 3.87 vs 3.87, 2.18 vs 2.17.
- Peak allocated memory was 103.147 GB, versus 103.138 GB without the feature. Both widths
  share one graph pool.

**Not measured:** thinking-mode traces (the policy should keep them at 3 if they accept like
prose), long contexts, and requests that switch between code and prose several times.

## Idea, not built: per-step depth from the DSpark confidence head (2026-09-29)

DSpark has a confidence head that predicts the conditional acceptance of each draft position
(V4.1 tech report 2.4.3; DeepSeek's scheduler turns it into a per-step verify length).
TensorFold's CUDA engine does the same for Qwen3.8 Flash Next's MTP head (`--mtp-confidence`,
default 0.30). This engine chooses depth per request only, so the question was whether the head
is good enough to choose it per step.

`DSV41_SPEC_CONF=1` (off by default) makes the draft graph compute the head's logits and
records `(depth, leading accepts, logits[5])` per step in `last_stats["spec_conf"]`. It is
observation only: nothing reads it back, and it adds no collective. Collect with
`tools/bench_spec_conf_tp.py` (verification pinned at depth 5, so no draft's outcome is hidden),
then replay the policies with `tools/analyze_spec_conf.py`. Data: `results/spec-conf-on/`,
image `e47eb4e9152b`.

The head is informative. Its rank AUC against the real conditional acceptance is 0.86-0.97 on
code and 0.61-0.84 on prose. Replaying the logged steps with fixed step costs of 88/106/124 ms
for depths 1/3/5 gives these gains over the better fixed depth per workload. They are rates of
a model, not engine measurements:

| Workload (steps) | Confidence rule, depths 3/5 | Confidence rule, depths 1/3/5 | Perfect predictor, 1/3/5 |
| --- | ---: | ---: | ---: |
| python (99) | +0.6% | +0.7% | +4.8% |
| html (106) | +2.9% | +3.6% | +6.9% |
| explain (221) | +1.0% | +3.2% | +15% |
| story (245) | -0.1% | +5.5% | +17% |
| story, temperature 0.7 (253) | +0.3% | +8.5% | +18% |
| prose with code (181) | +5.1% | +6.0% | +19% |

- With the existing 3/5 graphs, the gain is too small to pay for the extra collective a
  per-step width needs. Today rank 0 broadcasts depth before drafting; a confidence-based depth
  is only known after it.
- Most of the value needs depth 1, which means width-2 graphs as well. The confidence rule gets
  about a third of the perfect-predictor ceiling on prose. Tuning a logit offset on the same
  data barely helps, so the head's ranking accuracy is the limit, not its calibration.
- Reading the logits costs no extra wait: the host already waits for the draft graph at the
  n-gram hash D2H.
- Greedy and seeded sampled output were token-identical with the flag on and off, and on both
  ranks. The on/off decode rates differed by 5-16% in the flag-on run's favour on all six
  prompts. That is run-to-run variation, not an effect of the flag, so its overhead remains
  unmeasured.
- One prompt per workload. A prototype still has to beat adaptive depth in alternating
  full-engine runs, net of the per-step broadcast.

## Free-sibling tree: measured, and it does not beat the current policy (2026-10-01)

A chain's runner-ups (`top2[i][1]`) are candidates already conditioned on the chain's prefix, so a
tree can carry them at no extra drafter cost -- one chain pass yields `a1,b1,c1` and also `a2` and
`b2`. But each sibling is a **leaf**: its own continuation is a different forward, so accepting one
adds a single token and stops.

`DSV41_TREE_PROBE=1` logged the verifier's own argmax and the drafter's top-2 at every position of a
depth-5 chain across the five workloads (816 steps); `tools/sim_tree_probe.py` replays the shapes
against the measured row-cost curve (`59.0 + 6.65 x rows` ms, fit to the BLOCK 3/5/7 sweep):

| workload | chain3 | chain5 | tree3_rb (chain3 + a2 + b2) |
| --- | ---: | ---: | ---: |
| html | 41.40 | **49.87** | 36.81 |
| python | 43.99 | **54.54** | 38.62 |
| explain | **23.55** | 21.17 | 22.69 |
| story | **22.24** | 19.57 | 21.52 |
| mixed | 32.94 | **33.40** | 29.81 |
| all | 29.59 | **30.53** | 27.35 |

tok/s. The tree carries the sibling at the first two levels (`a2`, `b2`); the first version of the
simulation gave it 6 rows but let its token model take a third-level sibling the block did not
contain, which overstated it slightly -- the numbers above are the corrected ones. It gains tokens
per step over chain5 on prose and still loses, to chain3, because the two extra rows cost more than
the rescued tokens: a row is ~2.7 distinct experts and ~6.65 ms, and a tree uses exactly as many rows
as a chain of the same width. The sibling changes *which* tokens are verified, not how many bytes
are read.

A 7-row tree (with the third sibling) and an 8-row tree5 are worse still (26.00 and 28.44 all-mean).
Every free-sibling shape tested is behind the existing `DSV41_BLOCK_DYNAMIC=3,5`.

Acceptance, same log. For a tree the meaningful number is accepted length per verification, not
accepted/proposed -- a tree's branches are parallel hypotheses, so dividing by the number of nodes
serialises them and understates the tree.

| metric | chain3 | chain5 | tree3_rb |
| --- | ---: | ---: | ---: |
| tokens/step | 2.53 | **3.02** | 2.70 |
| accepted drafts/step | 1.53 | 2.02 | 1.70 |
| rescue (top-1 miss, runner-up holds) | -- | -- | 17.2% |

On prose (explain+story, 464 steps), the shape the tree is for:

| | accepted drafts/step | tokens/step |
| --- | ---: | ---: |
| chain3 (`a1,b1,c1`) | 0.963 | 1.963 |
| tree3 (three branches from t0: `a1-b1-c1`, `a1-b2`, `a2`) | 1.190 | 2.190 (+11.5%) |

Branch accepted lengths (drafts/step): `a1-b1-c1` 0.330 (all three), `a1-b2` 0.155, `a2` 0.149. On
code the same tree adds only +3.0%, because the chain is already deep and the runner-up is rare
there (`a2` 0.048, `a1-b2` 0.102).

So the tree does accept more, on prose, by 11.5%. The whole question is then the two extra rows:
converting that to tok/s needs the sibling rows to cost <= ~5 ms each, a ~25% discount against a
fresh row, which is the expert-overlap a replay cannot measure. Per position (chain prefix fixed):

| pos | top-1 | top-2 | union | P(#2 | miss) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 70.7% | 10.5% | 81.2% | 36.0% |
| 1 | 60.3% | 10.8% | 71.1% | 27.2% |
| 2 | 52.5% | 10.4% | 62.9% | 21.9% |

Depth beats breadth here because chain5 spends its rows where top-1 still holds 42--52 % of the
time, while the runner-up holds ~10 %. One caveat this replay cannot settle: the row cost
(6.65 ms/row) is measured on *chains*. A tree's siblings share a parent, so their MoE expert sets
should overlap more than successive chain tokens, which would make the tree's rows cheaper than the
model assumes. Closing the 11 % gap needs that overlap to be large; only a real tree forward can
measure it, and it was not built for the ~0.10 tokens/step the free part offers.

To beat it, a tree needs a **deeper branch**, which means a second draft chain (or a tree-shaped
draft pass, EAGLE-style) plus tree attention on the verify side: a tree buffer instead of the
position-indexed ring (siblings share a position), per-node RoPE positions, and the recurrent
compressor forked per branch. The measured upside for the free part is ~0.10 tokens/step pooled
(0.15 prose, 0.04 code), so that build starts from a negative and was not attempted.

### The sibling row cost, corrected measurement (2026-10-02)

The first version of `tools/bench_sibling_experts.py` was wrong in three ways: it chose `a1/a2`
and `b1/b2` from the target model rather than DSpark, evaluated `a2` after `a1,a2` instead of
on its real sibling prefix, and inferred the block cost from pairwise overlap instead of measuring
the complete six-row union. Its “~6 % discount” result must not be used.

The corrected microbenchmark takes the production FastDecoder's actual DSpark chain and runner-ups,
reproduces the production prefill/replay state, and routes exactly these blocks:

* depth-3 chain: `root,a1,b1,c1` (4 rows)
* depth-5 chain: `root,a1,b1,c1,d1,e1` (6 rows)
* free-sibling tree: `root,a1,b1,c1,a2,b2` (6 rows), with `a2` under `root` and `b2`
  under `root,a1`

32 positions x 40 layers, k=6, production prune (keep 0.61):

| shape | distinct experts/layer | marginal over depth-3 |
| --- | ---: | ---: |
| depth-3 chain (4 rows) | 16.277 | -- |
| depth-5 chain (6 rows) | 21.403 | 5.126 |
| free-sibling tree (6 rows) | **20.522** | **4.245** |

The sibling pair is genuinely cheaper: it adds 17.2 % fewer experts than the two ordinary tail
rows, and the complete tree reads 4.1 % fewer expert weights than the complete depth-5 chain. That
reverses the old explanation that token identity erased nearly all sibling reuse.

It is still not the same memory load as depth 3. The tree adds 4.245 distinct experts/layer, taking
the expert-weight union from 16.277 to 20.522 (+26.1 %) for the measured +11.5 % prose
tokens/step. Even an optimistic estimate that applies the entire 17.2 % expert discount to the
measured `6.65 ms/row` marginal gives `85.6 + 2 * 6.65 * 0.828 = 96.6 ms`: `2.190 / 96.6 ms =
22.67 tok/s`, just below depth 3's `1.963 / 85.6 ms = 22.93 tok/s`. Compute slack at concurrency
1 can hide the added arithmetic, but it cannot hide the measured extra expert-weight traffic.

That estimate is close enough that only an actual tree-forward timing can settle a roughly one-percent
decision, but the route microbenchmark does **not** establish a free lunch. It says the build starts
near break-even, not clearly ahead.

The rerun also exposed an observation bug: on an exact logit tie, `topk(2)` and `argmax()` can
order the same two tokens differently (2 of 160 observed draft positions). `TREE_PROBE` now pins
column 0 to the greedy-selected token and column 1 to the other top-2 token. Existing acceptance
logs predate that fix; recollect them before using a one-percent throughput margin to justify the
tree.
