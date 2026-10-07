# Immediate temporary streaming for selected layers

Experimental; disabled unless `DSV41_STREAM_LAYERS` lists backbone layer IDs.
Selected layers use the original unpruned router top-k in both prefill and decode.
Missing weights are loaded before that layer's MoE computes the current tokens.
Other layers retain the normal pruned routing and resident CUDA graph segments.

The resident keep masks and LUTs remain intact. Selected layers bypass their
resident LUT lookup, resolve current expert IDs through the transient ring, and
then compute MoE with the original gate weights. Decode graph segments split
into attention/router and MoE at selected layers so host loading can run between
them. Their MoE uses the existing separate routed/shared kernels; the merged
kernel assumes every selected expert has a resident LUT entry and is bypassed.
Both ranks take these fixed, boot-guarded boundaries unconditionally. There is
no rank-local collective decision. A load failure terminates the failing rank.

Both phases deliberately call `store.resolve(..., prefill=True)` at streamed
layers: this uses temporary slots and prevents eviction or promotion from
changing the resident addresses used by other layers' graphs. These decode reads
therefore appear in the existing **prefill** miss counter as well; NVMe bytes and
request time are the useful cost measures. Temporary cache hits can avoid reads.

Boot checks require enough transient slots for **all nonresident experts of each
selected layer**, because one prefill invocation could request all of them. With
235 residents/layer, this is 149 slots. On the current pair, 160 temporary slots
leave 9,423 LRU slots in the existing 90.1 GB arena, enough for all 9,400 residents.
The arena size and resident budget do not increase.

```bash
TRANSIENT_SLOTS=160
DSV41_STREAM_LAYERS=3,4,5,6,7,15,16,17,18,19
DSV41_PREFIX_CACHE=0
DSV41_PREFIX_DISK=0
DSV41_PREFILL_GRAPHS=0
DSV41_CRITICAL_PREFILL=0
```

Requires concurrency 1, native FP4 output-layout TP, resident fast-decode LUTs,
and no prefill replicas. Long-context performance, concurrent requests, vision,
sampling, and adaptive-swap interactions have not been qualified. Keep disabled
until controlled comparisons demonstrate a useful quality/cost tradeoff. This
is a separate policy from calibrated per-expert prefill rescue.

Layer-group probes use the same saved history and resident budget. They compare
early 0–4, middle 18–22, and late 35–39 against unrestricted routing on the two
pruning-sensitive questions and one correct-answer control. These cases guide
selection; they cannot establish a general layer-importance ranking.

The subsequent eight-block ablation found that pruning 0–4 in an otherwise fully
routed model broke math 7867; pruning 5–9 broke philosophy 11054. Pruning each
five-layer block from 10 through 39 separately preserved both answers. These
post-hoc cases show early-block necessity under the control. They do not prove
that pruning all later blocks together is safe. See RESULTS.md for the full table.

## Fine-grained diagnostics without reloading weights

`DSV41_LAYER_ABLATION_FILE` is an optional diagnostic control file, containing
`{"pruned_layers": [0]}` to prune one layer while streaming the rest. It requires
all 40 layers in `DSV41_STREAM_LAYERS`, and every adaptive swap trigger disabled.
Rank 0 reads and validates this JSON at request entry. Both ranks unconditionally
broadcast the resulting masks or error before computation. The peer does not
need its own copy of the file.

Routing masks occupy persistent tensors; changes are in place, preserving the
addresses used by captured graphs. Graph topology remains split at every layer,
including those currently pruned. This is for controlled short ablations, not a
performance mode. Write the file atomically between requests and verify returned
`layer_ablation.pruned_layers`. An empty list restores full routing. The flag and
policy version are in the boot guard; with it unset there is no new collective.

The local benchmark helper `../llm_benchmark/layer_ablation.py` saves before/after
controls, current masks, prompts, full responses and timings. The two cases were
selected after observing pruning damage, so use separate cases for validation.

## Measured policy limits

The example ten-layer policy recovered the two selected failures: 11/14 versus
9/14 on the original diagnostic. It gave **9/14 versus 9/14** on a separate
14-question manifest, with every answer letter identical to the baseline. Full
routing scored 12/14 on that separate manifest, correcting business, engineering,
and history cases the ten-layer policy did not recover. Keep it disabled for
normal use; the selected layers are not a general importance ranking.

On a short identical 39-token counting answer, one warmup and two measured trials
gave median total times 1.219 s resident versus 2.184 s with ten layers streamed;
decode throughput was 45.35 versus 39.58 tok/s. The resident budget and arena
size were identical, and no prefixes were reused. These timings qualify only
that short workload. See RESULTS.md for the manifests and raw artifact paths.

The benchmark helper accepts `--seed` and `--question-ids` with an explicit
`--arms-file` for further small diagnostics. Selecting cases after a full-routing
control makes those subsequent ablations post-hoc; preserve the original
validation result rather than relabeling a retuned policy as held out.

Further post-hoc ablations localized the new history failure to block 5–9,
engineering to 10–14 and 15–19, and business to 30–34, with all other layers
fully routed. Streaming 0–19 recovered engineering/history but not business;
streaming 20–39 recovered business but not engineering/history. Position alone
therefore cannot choose a safe subset. These are three specific questions, not
an importance ranking for entire subject areas.
