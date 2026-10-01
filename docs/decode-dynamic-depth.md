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
