# Predicting the first adaptation pass

`DSV41_PREDICTIVE_PREFILL=shadow` learns a bounded bank of prompt features and
observed prefill routing distributions. It estimates demand before prefill and
compares its proposed promotions with those selected from actual prefill demand.
`apply` additionally executes the predicted plan before prefill; `off` is the
repository default. This is experimental: similar vocabulary does not guarantee
similar routing, and CPU speed is not evidence of selection quality.

## Request flow

1. Restore any reusable prefix cache and identify the uncached suffix.
2. On rank 0, hash the full context, suffix vocabulary, and ordered suffix bigrams.
   Query up to four sufficiently similar prior examples. Cold starts and unfamiliar
   prompts proceed without a prediction.
3. Preview one normalized request vote alongside the existing aged demand history
   and trace prior. Use the normal global planner, layer floor, swap cap and minimum
   gain. In apply mode, broadcast and load that plan on both TP ranks.
4. Execute ordinary resident prefill. Missing experts may still be missed. There is
   no discovery forward pass, temporary expert streaming, or load-every-miss policy.
5. Evaluate the estimate against observed demand before adding the current example
   to the bank. Keep the existing post-prefill adaptation: actual observations are
   folded once and can correct the prediction before decode. This adaptation does
   not rerun prefill. Normal decode and between-request adaptation also remain.

The predicted vote is never written into demand history. Existing cached states
retain their original computation; predictions cannot improve a prefix already
cached. Better early selection could improve newly computed states, but that
quality benefit has not yet been established.

## Configuration and telemetry

Requires dynamic native FP4 TP2, recorded request-unit demand, and post-prefill
adaptation enabled. Streaming and calibrated rescue must be disabled. Prefix RAM
cache is supported. Prediction is skipped for vision and suffixes smaller than the
existing prefill adaptation threshold, including full cache hits.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DSV41_PREDICTIVE_PREFILL` | `off` | `off`, `shadow`, or `apply` |
| `DSV41_PREDICTIVE_DB` | `results/predictive-prefill.npz` | Rank-0 authoritative bank |
| `DSV41_PREDICTIVE_CAPACITY` | `128` | Maximum distinct prompt/prefix examples; limit 256 |
| `DSV41_PREDICTIVE_MIN_SAMPLES` | `8` | Minimum bank size before predicting |
| `DSV41_PREDICTIVE_MIN_SIMILARITY` | `.85` | Cosine threshold, a heuristic rather than calibrated confidence |

Prediction-related control fields join the boot-time rank configuration guard.
Every enabled request reaches the same plan broadcast, even when skipped or when
prediction fails. Errors before broadcast skip prediction; failure while applying
a distributed plan terminates the rank, as with existing adaptation.

The health engine configuration exposes `predictive_prefill`. Logs include status,
similarity, estimated miss rate, planned/applied swaps, and prediction/load/observation
time. Shadow evaluation additionally reports mean total variation of the demand
distribution, promotion precision/recall, and exact swap overlap against the normal
planner using actual prefill demand. Empty promotion sets report null ratios.
Apply mode cannot provide that same counterfactual comparison after changing the
residents, so it reports demand error only.

The bank stores signed hashed token features, prompt hashes and normalized expert
counts/score mass. It stores no raw prompt text or token sequences. These derived
features are still private user data, not a guarantee of anonymization. Keep the
bank local and out of version control. Aggregate historical expert rankings cannot
initialize it: they lack the corresponding prompt features. Repeated identical
prompt/prefix pairs replace their previous entry. A full bank occupies about
16.5 MiB for 40 layers × 384 experts, plus temporary working memory.

The model configuration, tokenizer and guarded numerical/serving policy identify
the bank. Incompatible or corrupt files are preserved without overwriting them;
inspect the reported `database_error` and use a fresh path after changing policy.

## Validation

Run the focused CPU suite:

```sh
.venv/bin/python -m unittest engine.test_predictive_prefill engine.test_decode_adapt \
  engine.test_global_residency engine.test_expert_seed engine.test_adapt_config \
  engine.test_prune_unit
```

Tests cover matching the normal post-observation planner, no duplicate history
votes, rank agreement, skip/error paths, bounded persistence, and long-context
feature discrimination. The live probe uses isolated copies of demand history and
a separate predictor bank so synthetic traffic does not seed normal use.

On the development Spark, feature extraction plus a 128-example nearest-neighbor
query took median 1.715 ms for 10k new tokens, 16.151 ms for 200k context/2k new,
and 22.424 ms for 524,288 context/5,243 new. These synthetic CPU measurements
exclude ranking, weight loading and model execution. No GPU kernel is needed for
this measured portion. See `RESULTS.md` for live validation and limitations.
