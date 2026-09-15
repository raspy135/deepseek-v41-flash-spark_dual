# TP and persistent prefixes: implementation and acceptance gates

Branch: `codex/tp-adaptive-prefix-cache`. Baseline checkpoint: `f380019`.

The user requires TP prefill **and** decode performance at least as good as EP,
with adaptive expert loading retained. EP remains the default until measured gates pass.
CPU tests and a working kernel are not a whole-model performance or quality pass.

## Persistent prefixes

Opt in with `DSV41_PREFIX_DISK=1`, `DSV41_PREFIX_SNAPSHOTS=8`. Budget defaults to
`DSV41_PREFIX_DISK_GB=20` (decimal GB per node, approved by the user). Directory defaults
to `results/prefix-cache/rank-N`, or set `DSV41_PREFIX_DISK_DIR` for its parent.

Each prompt bundle stores compressed KV/index rows once, alongside small replay,
window and compressor snapshots at the retained chunk boundaries. The full prompt
is also a boundary. A token SHA-256 index finds the longest matching prefix. An
independent file checksum verifies the payload before weights-only deserialization;
exact token comparison and tensor-layout validation follow. Restores copy into the
existing allocations so captured CUDA graph pointers remain valid.

Both ranks intersect available `(bundle UUID, prefix length)` entries and agree on
successful loading before skipping any prefill. Missing/corrupt/local-I/O failures
are shared misses, not mismatched collective counts. Files use mode 0600 in mode
0700 rank directories. Never share a rank directory between independent servers.

Files are fsynced and atomically published before the SQLite index is committed.
Least-recently-used bundles are evicted across namespaces within each rank directory.
The published-file budget excludes one in-flight temporary bundle and index overhead.
Only one CPU-staged write is queued; the next request joins it before lookup. CPU
staging is charged to prefill; disk restore latency is also included in prefill.
No model weights or original plaintext prompt are stored, but token IDs and KV are
sensitive and may reveal the prompt. Caches/captures are excluded from Git and images.

Compatibility hashes include engine/kernel source, model config/index/tokenizer,
checkpoint file sizes and mtimes, parallel layout, precision settings and pruning
fraction. They are conservative; code/config changes can invalidate previous entries.
Checkpoint identity does not require reading hundreds of GB at startup, and assumes
weights are not replaced while preserving all file metadata.

`DSV41_PREFIX_DISK_STRICT=1` additionally requires the same expert-selection mask.
Default 0 follows the existing RAM cache's historical-prefix semantics: a cached
prefix is not recomputed merely because adaptation changed the resident experts.
That is NOT equivalent to fresh computation under the new mask. Strict and historical
caches occupy separate compatibility namespaces. Vision requests bypass persistence.

Initial GPU trial (before Engram-history repair): 7,696 tokens, 61,450,719-byte bundle,
0.58 s staging, 0.11 s background write, 0.10 s restore. Immediate full replay matched;
replay after another prompt did NOT. The full-hit path had skipped Engram hashing,
leaving compressed-token history from the intervening prompt. Full hits now rebuild
that history too. Do not cite this initial trial as a correctness pass.

After the Engram fix, the two-node gate passed full restore, 6,144-token partial restore,
switching prompts and returning, and a fresh-process restart. All full-restored 7,696-token
outputs matched the original output hash. In the restart trial, loading was 0.190 s and
prefill/replay including that load was 0.729 s. Same-process full restores were 0.36 s.
This validates the bounded test, not arbitrary model/config combinations.

CPU gate: `python -m unittest engine.test_prefix_disk`.
GPU/restart gate: `tools/test_prefix_persistence_engine.py --phase save|load` on both
nodes, same code/config, serving stopped. Uses synthetic/public text and output hashes.

## TP architecture (experimental, gates pending)

- `DSV41_TP_EXPERTS=1`: both nodes hold the same resident expert IDs, but each holds
  half the packed FP4 intermediate dimension. Scale boundaries stay aligned. Adaptive
  swaps update both shards, with no even/odd ownership restriction.
- The up/gate outputs keep their BF16 rounding. Down projections produce FP32 partials;
  reconstruct each expert across ranks BEFORE BF16 rounding and the routed sum. This
  initial correctness implementation communicates more than the EP routed-total sum.
- `DSV41_TP_DENSE=1`: shared-expert TP. The down-projection reduction now happens before
  BF16 rounding, not after separately rounded rank outputs. Unsharded draft experts
  are not accidentally reduced twice.
- `DSV41_TP_ATTN=1`: main-model query heads and output-LoRA groups are sharded, with
  one reduction at the final attention output projection. Compressed KV/indexing,
  small projections and DSpark attention remain replicated.
- `DSV41_TP_HEAD=1`: vocabulary rows are sharded and disjoint logits gathered. Sampling
  still sees the whole vocabulary on both ranks.

Thus TP does not promise removal of every replicated tensor or every performance
imbalance. It removes expert-popularity-based arithmetic imbalance. GPU clocks, I/O,
host scheduling and collectives still matter. Memory savings must be measured before
increasing arena/pruning capacity.

The old shared-only TP negative measurements remain in `v41_ref.tp_dense`; changing
the implementation does not erase those results. The new implementation needs fresh
quality, graph/speculation and paired speed gates.

Initial two-node routed-kernel gate passed real-weight numerical checks and CUDA graph
replay (relative L2 error at most 2.13e-5 for T=1,6,128,2048). Balanced T=6: EP 0.671 ms,
TP 0.658 ms; all-six-experts-on-one-EP-owner T=6: EP 0.928 ms, TP 0.752 ms. But balanced
T=2048: EP 22.73 ms, TP 32.99 ms; skewed T=2048: EP 38.48 ms, TP 32.90 ms. The initial
per-expert reduction removes skew but adds prefill communication. Whole-engine results
and optimization are still required; this does not meet the user's performance gate.

`DSV41_TP_EXPERT_REDUCE=scatter` writes down-projection partials in output-column-owner
order, reduce-scatters each expert, rounds/sums locally, then gathers only routed totals.
It preserves the same per-expert rounding boundary. `auto` uses it above 64 token-expert
pairs and uses the original all-reduce at decode sizes. Default remains `all_reduce`
until whole-engine qualification. The scatter kernel gate passed including graph replay;
balanced T=2048 was 28.73 ms versus the initial all-reduce's 32.99 ms in separate trials.

Initial full TP README runs reached 22.67 decode tok/s and 789.11 prefill tok/s, but are
not a matched EP comparison. One driver failed because it tried to read the private
capture on both nodes; it now reads on rank 0 and broadcasts IDs through the existing
node link. The direct-engine nesting grader also differed from serving: it filtered
all EOS IDs instead of truncating at the first EOS in a speculative burst. The corrected
grader is pending validation; do not attribute that initial failure to TP without it.
