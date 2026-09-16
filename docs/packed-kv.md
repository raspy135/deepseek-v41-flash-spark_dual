# Packed long-context KV

`DSV41_PACKED_KV=1` stores the already-quantized global KV history as FP4 codes
and E4M3 scales. It requires `DSV41_KV_CACHE_QDQ=1`. Restart both ranks when
changing storage mode. Window rings and index keys remain BF16.

Each 512-value row occupies 36 int64 words: 32 words containing 16 FP4 codes
each, followed by four words containing eight scale bytes each. That is 288
bytes rather than 1,024 bytes. The kernels issue 64-bit cache loads/stores;
the int64 tensor is a bit container, not integer-valued attention arithmetic.

The write kernel applies the existing QDQ rounding directly into packed storage.
The gather kernel loads packed groups and reconstructs only selected rows to
BF16. Attention itself is unchanged, including its reduction order. Midpoint
ties and signed zero match the original QDQ implementation. This avoids adding
another quantization error; it does not undo the model's existing FP4 rounding.

At 393,216 tokens, primary KV/index/window storage falls from about 1.340 GiB
to 0.666 GiB per node per request lane. Scratch, graph pools, positional state,
and prefix snapshots are additional. This is not a guarantee of proportional
maximum-context or concurrency growth.

Disk bundles store the packed int64 tensors without expanding them. Existing
namespace hashing separates modes, and restore validates shape and dtype before
copying into graph-visible allocations. Old BF16 bundles remain on disk but
are not reused in packed mode.

## Validation

- `python -m unittest engine.test_packed_kv`: CUDA bitwise reconstruction at
  multiple scales, rounding boundaries, signed zeros, random gathers, indexed
  and contiguous writes, and graph replay with changing positions.
- `python -m unittest engine.test_prefix_disk`: packed-word disk round-trip,
  stable destination pointers, and rejection of incompatible BF16 layouts.
- `python -m engine.bench_packed_kv`: interleaved graph timings for writes and
  indexed reads. This is a component benchmark, not full-model throughput.
- `bench/packed_kv_http.py`: fixed captured token IDs, 128 greedy output tokens,
  full response-length check, timing, and output hash without logging content.

The full-engine comparison uses TP2, K=3, a 384K allocation, the same frozen
expert set, and no prefix reuse. Adaptation and production prefix settings must
be restored after testing. User captures and result JSON are local, not fixtures
to commit.

## Rejected first gather layout

The first flat-element gather addressed each packed word repeatedly while
extracting its values. It passed bitwise tests and full-engine output checks,
but a warmed 14,435-token prompt had median prefill 17.969 s versus baseline
17.189 s (baseline recheck 17.217 s). Decode was essentially unchanged around
20 tok/s. Do not describe that prefill difference as established noise.

The revised gather loads a group word and broadcasts it across its 16 register
values. The write offset is a runtime scalar to avoid compiling a separate
kernel for each prefill chunk position.

September 16 controlled HTTP measurements (14,435 input tokens, 128 greedy
output tokens; median of the last three of five runs, initial two discarded
consistently for warm-up):

| Mode | Prefill seconds | Decode seconds | Decode tok/s |
| --- | ---: | ---: | ---: |
| BF16-storage baseline | 17.189 | 6.338 | 20.04 |
| First packed gather | 17.969 | 6.338 | 20.04 |
| BF16-storage recheck | 17.217 | 6.378 | 19.91 |
| Grouped packed gather | 17.819 | 6.607 | 19.22 |

All 20 runs had the same output hash and acceptance length (2.78). This is one
workload, not a broad quality benchmark. The final prefill median is 3.5% slower
than the baseline recheck and decode duration 3.6% longer. It does **not** establish
below-noise overhead or a decode speedup. Packing is enabled for its memory
savings; use `DSV41_PACKED_KV=0` to return to BF16 storage. Original result files
are under local `results/packed-kv-*.json` (not committed).

Production smoke testing with adaptation restored also saved and reloaded the
14,435-token packed prefix after an unrelated prompt displaced the in-memory
entry. Disk restore reported 0.0767 s; the eight-token follow-up completed in
1.09 s with all response bytes received. This used a 917,504-token allocation,
but exercised only the 14K prefix, not full-length 896K generation.

## Wider-load audit

FP4 expert tensors are labelled uint8 but already contain two values per byte.
The inspected compiled MoE kernel includes `ld.global.v4.b32` (128-bit loads),
then hardware FP4 conversion in registers. A dtype label does not reveal memory
transaction width, and adjacent narrow accesses may also coalesce.

Engram's FP8 path still expands rows with several PyTorch operations before
gathering them (`engine/engram.py`, `to_device`). A fused dequantize-and-gather
is a possible follow-up, not a measured speedup or part of this change.
