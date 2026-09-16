# Draft experts and input embedding TP

Two independent, opt-in switches reduce replicated weights on a native MXFP4 TP2 pair:

| Switch | Saved per node | Implementation |
| --- | ---: | --- |
| `DSV41_TP_DRAFT_EXPERTS=1` | 3,609,722,880 bytes (3.362 GiB) | The existing output-layout FP4 expert arena, also used for the three draft layers. |
| `DSV41_TP_EMBED=1` | 661,913,600 bytes (0.616 GiB) | Each rank keeps half the embedding feature columns and gathers lookup results. |

Combined: 4,271,636,480 bytes (3.978 GiB) per node. This is a weight-storage
saving, not a throughput or quality improvement claim. Temporary collective buffers
and CUDA graphs consume a little of that room. It does not automatically enlarge the
main expert arena or change the pruning mask.

Embedding columns, rather than vocabulary rows, avoid masked additions: reconstructing
a lookup only concatenates stored BF16 values. The CPU weight is sliced into independent
storage before GPU upload. Both ranks must perform the same lookup, including drafting.

Draft TP requires native FP4 TP2 and `DSV41_TP_EXPERT_LAYOUT=output`. The MoE kernel
gathers intermediate activations, computes complete down-projection dot products, and
gathers disjoint output columns. Draft shared experts remain replicated. There must not
be another all-reduce after adding the shared-expert result.

Both switches participate in the startup rank-configuration check and appear in
`/health`. Restart both ranks together when changing them.

## Validation

- `engine.test_tp_embedding`: half-sized independent storage and exact scalar/vector/
  matrix lookups, including signed zero, with a mocked collective.
- `tools/test_tp_draft_embedding_cuda.py`: two-node real-checkpoint lookups through
  1,024 tokens and dynamic CUDA-graph replay; sampled experts from all three draft
  layers at 1, 3, 4 and 16 tokens, compared exactly with the replicated FP4 kernel.
- `tools/bench_tp_memory.py`: paired full-engine runs on the same captured 14K input,
  fixed expert mask, no adaptation or prefix reuse; output hashes, acceptance, memory
  allocation and timing. Captured prompt contents are never printed.

The distributed test drivers are disposable processes. They use the existing TP gates'
synchronized process-exit convention, avoiding NCCL teardown with live CUDA graphs
that retain collective resources.

## Measured on the dual Spark pair

Three runs per mode, 14,435 input tokens and 128 output tokens, 768K context
allocation, packed KV, K=3, 88 GB arena, keep=0.60, concurrency=1. No prefix reuse
or adaptive swaps. Both ranks used the same image and selected-expert mask.

| Mode | Prefill seconds, runs 1/2/3 | Decode tok/s, runs 1/2/3 | Allocated GiB/node |
| --- | --- | --- | ---: |
| Replicated draft + embedding | 24.762 / 20.004 / 17.124 | 15.90 / 19.07 / 19.55 | 98.0383 |
| Both sharded | 21.629 / 17.954 / 17.321 | 16.59 / 18.82 / 19.45 | 94.0601 |

All six output-token hashes matched, and mean acceptance was 2.78 in every run.
Final warmed decode throughput was 0.51% lower; prefill duration was 1.15% longer.
Warm-up was substantial, so these three-run results do not establish a noise range
or a speedup. The measured allocation saving was exactly 4,271,636,480 bytes/node.
This is a memory tradeoff with a small observed timing cost on this workload.

Raw local logs: `results/tp-memory-{baseline,enabled,kernel}/rank{0,1}.log`.
The saved weight bytes correspond to about 454 additional main expert slots, before
allowing headroom; increasing capacity requires raising both arena size and keep rate.

After the fixed-mask comparison, the user's live profile was raised from 88 GB /
keep=0.60 to 92 GB / keep=0.63. That selects 242 experts per layer, 9,680 total
(440 more than before). Arena allocation grows by 4 decimal GB/node, below the
4.272 GB/node freed by sharding. This larger mask was not used for the timing A/B
above and may change outputs and acceptance.

One production HTTP smoke test completed after this restart: 14,435 input / 128
output tokens, prefill 28.928 s, decode 14.40 tok/s, acceptance 2.63. This was a
cold first request with the live adaptive mask and swaps enabled, not a matched
performance comparison. It completed without an allocation failure; adaptation
also ran successfully. Results: `results/tp-memory-capacity-http.json`.
