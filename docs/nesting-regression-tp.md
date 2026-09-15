# Short nesting regression investigation, 2026-09-15

The acceptance case is the exact `quality_quant2.py` short prompt (63 encoded tokens),
not the 6,082-token padded probe used during initial TP qualification. The historical
1.000 score must not be dismissed because two current modes share a failure.

Controls use temperature zero, the same checkpoint and runtime image
`338bcaee58ea0a4f157eae2d804b22255a85b629f3e1ecbe80fbf03561cbc803`, cache disabled,
no adaptive swaps and no demand DB writes. Frozen keep-0.61 mask SHA-256:
`560d30d99068671d76c21d1b2459b6eb5ee6c0d614e47e8a5ef2e2c24878b4ca`.

| Source / mode | Experts | Depths 4 / 6 / 8 / 10 |
| --- | --- | --- |
| Current EP | frozen keep 0.61 | pass / pass / fail / pass, twice |
| Current TP | identical frozen keep 0.61 | identical tokens to EP, twice |
| Current TP | all experts, streaming | pass / pass / fail / pass |
| Current TP, non-speculative depth 8 | all experts | same extra-brace failure |
| Historical `752c18a` EP | all experts, streaming | pass / pass / pass / pass |
| Historical `752c18a` EP, non-speculative depth 8 | all experts | pass |
| Historical `752c18a` EP, full prefill + non-speculative depth 8 | all experts | pass |
| Current EP | all experts, streaming | pass / pass / pass / pass; same tokens as historical EP |
| Current EP, non-speculative depth 8, with and without bounded replay | all experts | both pass |

The failing depth-8 answer has eight `n` objects and nine closing braces. Its output
token hash is `da67d04a5ddba560b4ba384a844ae0e2302a00d14e73171f4fb9a7b0c4fa7687`.
The historical passing hash is
`697c651d381b4d542fe688c1ef53f1ea3b6ef01600127b965444fa5d368e087a`.

The first full-expert attempt aborted before grading because the pruned configuration's
16-slot transient ring could not hold the unpruned call. The driver now reserves 384
transient slots *within the same total arena* for this control. That aborted run is
not a quality result. Only short probes are used; no full unpruned benchmark.

The current EP/all control matches historical EP/all, whereas current TP/all fails
depth 8. Thus the pruned comparison had hidden a TP-specific quality effect: EP/pruned
and TP/pruned failing identically did not rule it out. Cache and speculation have been excluded as
sole explanations for the current TP/all failure, not certified in every configuration.

`tools/bench_tp_engine.py --short-nesting-only` provides this bounded experiment.
`--compare-eager --compare-full-prefill` isolates speculation and bounded replay.
`tools/run_two_node_gate.sh` can mount an isolated historical source tree via
`GATE_SOURCE_ROOT`; it does not overwrite either live checkout. Production remains
stopped while these mutually exclusive GPU loads run.

## Routed-expert isolation and candidate correction

In a full-expert TP process, reconstructing the head, shared experts and attention
weights separately did not repair depth 8. Reconstructing **all** dense weights together
also failed, leaving only routed experts sharded. These controls used eager decoding
to avoid executing CUDA graphs captured against replaced weight pointers.

Candidate `DSV41_TP_EXPERT_LAYOUT=output`: keep gate/up rows split, gather their BF16
activated intermediates, and split the down projection by **output rows** rather than
input columns. Each down-projection dot product then retains its full K accumulation
before the BF16 boundary. Gather disjoint routed output rows after the expert sum.
The packed expert bytes per node are unchanged. This avoids adding two independently
accumulated FP32 half-dot-products, whose rounding differed from the unsplit reference.

Real layer-0 kernel tests at T=1,6,128,2048, balanced and EP-skewed routing, were bitwise
identical to the unsplit reference in all eight cases. CUDA graph replay also passed.
Balanced T=6: EP 0.887 ms, candidate 0.579 ms; balanced T=2048: EP 27.866 ms,
candidate 22.542 ms. Skewed T=2048: EP 43.803 ms, candidate 22.793 ms. These are
kernel measurements, not whole-engine throughput claims.

Expert output sharding alone still failed depth 8, including eager decoding. With that
correction present, replicating head/shared/attention separately still failed, but
replicating all dense groups together passed with the historical EP token hash. Thus
fixing only the routed reduction was insufficient.

`DSV41_TP_LINEAR_LAYOUT=output` applies the same complete-dot-product arrangement to
shared-expert down projections and the final attention projection: gather BF16 inputs,
compute disjoint output rows with full K, gather BF16 results. The vocabulary head
already gathers disjoint logits and is unchanged. Both output layouts are now defaults;
legacy `intermediate` layouts remain available for diagnostic comparisons.

With **both** output layouts, all-expert TP passed depths 4/6/8/10 twice with speculation
on (1.000 each run), and eager depth 8 passed. Every output hash matched current and
historical unpruned EP. The checkpoint did not change. This is a bounded regression
test, not a claim that TP is bitwise identical for every input or that pruning is lossless.

Dense layer-0 tests were exact at T=1,6,128,2048, including CUDA graph replay. The added
T=63 test exposed a small residual cuBLAS-geometry difference (100/322560 elements,
maximum absolute difference 0.015625), despite full-K accumulation. Its test uses a
tight relative-L2 bound instead of claiming universal bitwise equality. The complete
engine short-nesting tests above pass at this same prompt length.

The same frozen keep-0.61 mask also passed all four depths twice with both corrected
layouts, speculation on and no prefix reuse. Depth 8 changed from the failing hash
above to the historical passing hash without changing expert selection. Pruned depth
10 uses a different whitespace/token sequence but is semantically exact. The frozen
EP/pruned failure therefore cannot be used as an equivalence guarantee or an excuse
for retaining the old TP arithmetic.

Final routed-kernel checks included T=63 and remained bitwise exact (all ten cases);
graph replay passed. Dense T=63 relative L2 was 5.09956e-5; other tested sizes were
exact and its graph replay passed. Seven CPU sharding/adaptation tests passed.

## Integration and performance

With arena 90 GB, keep 0.61 and frozen placement, the corrected TP run passed both
6,082-token padded nesting probes (depths 8/10), two real adaptive swaps with route/LUT
checks, and depth 10 after those swaps. Saving that prefix, processing another prompt,
clearing RAM/KV and restoring from disk reproduced the exact output hash on both ranks.
Disk load took 0.089 s; restored prefill/replay took 0.320 s. The run ended with
`TP_ENGINE_INTEGRATION_PASSED tp known_quality_failures=[]`.

| 7,709-token README, no prefix reuse | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| First run | 488.92 | 16.04 |
| Repeat 1 | 575.96 | 17.84 |
| Repeat 2 | 954.25 | 18.64 |

Padded nesting prefill reached 1003/1073 tok/s. These are bounded measurements, not
a matched EP comparison or proof of universal performance parity. Resident expert
bytes remain identical; measured allocation after README was 106.844 GB. All 16
CPU tests covering TP layouts/adaptation and prefix disk passed.

## Production API qualification

Built and shipped one identical image to both nodes:
`sha256:6b0da554d31c65cd370c52e1bab868e498b96b3ad1ba0604e594597adeedc427`.
Restarted normal serving with arena 90 GB, keep 0.61, all four TP groups, both output
layouts, speculation, adaptation and disk prefixes enabled. No official API key was used.

At 19:02 UTC on September 15, the public chat API passed all four exact short prompts
twice (1.000 each run, temperature zero, non-thinking, cap 128). First-run prefix hits
were all zero. The second run restored all four 63-token prefixes from disk, with
0.027–0.029 s load times, and all answers remained correct. Normal request-boundary
adaptation applied 127,52,65,107,23,41,52,42 expert swaps over eight generations. Final
health was `ok`, idle, with both output layouts and speculation confirmed active.

This qualifies the reported nesting regression in these tested configurations. It
does not prove universal TP/EP bitwise equality, lossless pruning, or resolution of
the separate constraint/character-count quality gaps.
