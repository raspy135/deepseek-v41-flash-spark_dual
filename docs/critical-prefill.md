# Selective streaming of important prefill misses

Experimental and off by default. A missing expert is eligible when its current
router weight times a calibrated output-norm estimate accounts for at least
`DSV41_CRITICAL_SHARE` of a token's predicted routed contribution. Rank by the
largest token-level share, so many marginal uses cannot outrank one large use.
This measures a residual-contribution proxy, not causal answer importance.

Only experts with at least three calibration observations qualify. Unknown
experts get the layer median in the prediction denominator but cannot qualify;
an unobserved expert is not being declared unimportant. Sparse calibration and
changes in input activation can make this prediction inaccurate.

Rank 0 plans and every backbone prefill layer broadcasts, including empty plans.
Both ranks load the selected experts into the existing transient ring before
computing MoE. The normal resident mask, LUT, and decode graphs remain intact.
A compact routing map includes the temporary slots on rescued calls. A load
failure after the common plan terminates the failing rank rather than allowing
it to serve a different selection. Profile hash and policy settings are guarded
at boot on both ranks.

The prototype requires native FP4 tensor parallelism and concurrency 1. Disable
prefix caching/persistence and prefill graphs: cached prefixes do not carry the
rescue budget or selection history. Decoder SWA prompt replay is part of prefill
and can rescue experts; subsequent generated tokens use the resident graph.

```bash
DSV41_CRITICAL_PREFILL=1
DSV41_CRITICAL_PROFILE=results/expert-tuning-20261005/critical-profile.npz
DSV41_CRITICAL_SHARE=0.25
DSV41_CRITICAL_PER_LAYER=1
DSV41_CRITICAL_BUDGET=8
DSV41_PREFIX_CACHE=0
DSV41_PREFIX_DISK=0
DSV41_PREFILL_GRAPHS=0
```

The per-layer cap applies per prefill invocation and must fit the transient ring.
The request budget bounds all rescue events, including transient cache hits, so
actual new expert reads can be fewer. Early invocations can consume the budget;
it does not yet reserve capacity for late layers. Chunking can change a bounded
selection. Long prompts, vision, sampling, concurrency, and adaptive-swap
interactions have not been qualified for this prototype.

Build profiles with `tools/expert_trace.py --output-norms`, followed by
`tools/build_critical_profile.py --trace <trace-dir> --out <profile.npz>`.
The trace estimates unweighted norms by dividing the actual weighted BF16
contribution norm by the positive gate weight, without another expert forward.
It uses the layer-streamed reference path and existing checkpoint files. This
is approximate because weighting precedes a BF16 rounding boundary.

## Pilot, 2026-10-05

Two public teacher-forced texts, 132 tokens, layers 0–3, activation fake-quant
disabled: 364 layer/expert pairs had three observations. This is intentionally
a small mechanism test. The profile has no late-layer evidence. Layer 0's
per-pick output-norm 10th/90th percentiles were 2.77/8.19; that is variation
between experts, not evidence that layer 0 is more important than other layers.

Held-out MMLU-Pro seed 20261006, one question per category, fixed residency:
baseline and rescue both 9/14, with identical answer letters and no invalid
responses. Rescue issued nine layer/expert events and five actual expert reads.
Median question time was 0.897 versus 0.973 seconds. Short code throughput was
45.50 versus 40.46 decode tok/s with identical 128-token outputs; prose outputs
changed, so their 20.19 versus 21.81 tok/s is not an output-identical speed gain.
No quality improvement was demonstrated; the feature remains disabled.

Artifacts: `results/expert-tuning-20261005/` in this repository and
`/home/ryan/git/llm_benchmark/results/expert-tuning-20261005/`.

Mia's local EXL3 metadata allocates all routed expert matrices in layers 18–22
two bits, and those in the other backbone layers three bits. This is a useful
layer-level quantization-tolerance clue, not an individual-expert importance
ranking or proof that omitted middle-layer contributions are harmless. Test
early/middle/late degradation before making that a retention rule.
