# Decode shared-expert overlap experiment

`DSV41_DECODE_SHARED_OVERLAP=1` forks the backbone's replicated shared expert
onto a separate CUDA stream while the current stream runs routed experts and
their TP collectives. Default is **off**. The flag is included in the two-rank
boot agreement and engine configuration report. This is not a public tuning
recommendation until measured on representative workloads.

Both branches consume the already-produced layer input. The side stream first
waits for the producer, and the main stream joins it before adding the shared
output or changing that input for the next layer. The shared tensor records its
use on the consuming stream for allocator safety. No collective is moved to the
side stream; no expert selection, dtype, or addition order is changed. Draft
computation and prefill remain unchanged.

This fork/join also fits [PyTorch's multi-stream graph capture rules](https://docs.pytorch.org/docs/stable/notes/cuda.html#usage-with-multiple-streams).
CUDA graphs may reschedule independent nodes, so merely using two streams does
not establish that communication latency was hidden or that execution is faster.
Shared and routed projections can compete for memory bandwidth and SMs.

`tools/bench_decode_shared_overlap_tp.py` runs the disposable TP2 gate with the
same short HTML prompt as the decode timeline experiment. It loads the saved
expert ranking, freezes swaps and disables prefix reuse/persistence. Separate
graph pools retain serial and overlapped variants without cross-variant storage
aliasing. It warms both, alternates A/B/B/A 512-token generations, compares every
output token, compares fixed-input logits and hidden states, alternates fixed
verification timings, then checks depth-8 nesting on both variants. The small
nesting case is a regression probe, not a comprehensive quality benchmark.

`python3 -m unittest tools.test_decode_shared_overlap` checks host-side ordering,
the unchanged serial path, and the exception join without requiring CUDA. Actual
stream ordering and graph replay correctness require the two-node GPU gate.

Raw results are local, private and ignored under
`results/decode-shared-overlap-20260916/`; no user request text is recorded.

## Initial measurements — September 16

TP2, HC32, draft block 3, 90 GB arena and keep=0.61, with 512 output tokens on
the HTML workload. After one warm-up generation per variant, the order-balanced
unprofiled generations on rank 0 were:

| Order | Serial tok/s | Overlapped tok/s |
| --- | ---: | ---: |
| A then B | 25.94 | 28.31 |
| B then A | 27.56 | 29.35 |

Mean throughput rose from 26.75 to 28.83 tok/s (+7.8%). The serial baseline also
sped up over the run, so retain the individual samples rather than attributing
every difference to overlap. All six 512-token outputs, including the warm-ups,
were identical. Both ranks' fixed-input logits and hidden states were bit-exact
(maximum absolute delta zero). Depth-8 nesting scored 1.0 in both modes with
identical 29-token outputs. These checks are not a general quality guarantee.

Ten-step timing batches alternated A/B, B/A, A/B, B/A. Rank-0 serial means were
104.16, 104.19, 104.28 and 105.57 ms per verification; overlap means were 141.03,
98.49, 98.40 and 98.32 ms. The first overlapped batch is an unexplained slow
outlier, **not discarded**. Medians are 104.23 versus 98.44 ms (5.8 ms less),
but the mean across all four batches is not an improvement. The repeated
end-to-end generations provide separate evidence of a benefit on this prompt.
The fixed-step test excludes draft and Engram fetching and must not be equated
to the earlier profiled 129 ms whole-iteration interval.

Peak allocated CUDA memory with both A/B graph pools retained was 103.14 GB on
rank 0. Production retains just the selected variant. Normal precision, TP
collective order and shared-expert addition order were unchanged.

For a bounded longer-context follow-up, `--capture` reuses a trusted local
capture's prompt IDs only: sampling is greedy with seed 42 and EOS enabled,
not necessarily the capture's original sampling settings. `--max-tokens 128
--quick` runs a warm-up and one measured generation per variant, then the same
fixed-step and nesting checks. No pickle capture is accepted from an untrusted
source; vision and grammar captures require their full server path.

## Longer-context follow-up

The existing 14,435-token text-only capture, capped at 128 generated tokens,
also matched exactly across serial and overlapped runs. Warm-up throughput was
18.06 / 21.09 tok/s; the subsequent measured pair was **21.96 / 23.98 tok/s**
(serial / overlap, +9.2%). Both accepted 2.87 tokens per speculative iteration.
This is only one measured pair, not a confidence interval or a universal gain.

All four alternating fixed-step timing batches improved: serial 102.22, 102.24,
102.18, 101.75 ms; overlap 95.66, 95.62, 95.67, 95.36 ms. Median saving was
6.56 ms per verification. Both ranks again had bit-exact logits and hidden states
and identical depth-8 nesting output, scoring 1.0. The short-test outlier did not
reproduce in this follow-up. Peak allocated memory with both pools was 103.52 GB.

Results: `results/decode-shared-overlap-long-20260916-b/`. An earlier attempt in
the sibling folder without `-b` stopped before generation because the capture
path did not exist on rank 1; no performance results came from that attempt. The
retry used the previously synchronized private copy after verifying matching
file hashes. Capture loading now precedes expensive engine initialization.

The local serving configuration enables the experiment; the code default remains
off. Roll back by setting `DSV41_DECODE_SHARED_OVERLAP=0` and restarting both
ranks together. The dual-node launcher forwards the same value to both nodes.
