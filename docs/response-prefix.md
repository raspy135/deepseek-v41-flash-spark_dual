# Post-response prefix preparation

`DSV41_PREFIX_RESPONSE=1` prepares the generated answer for reuse by
the next request. It is enabled by default; set `0` to disable. It requires normal prefix caching and
encoder/decoder bounded replay; vision and prefill replicas are excluded.

The HTTP handler flushes the complete JSON response or final SSE chunk before
requesting preparation. The engine restores the saved input boundary and runs
only the response tokens through encoder prefill. This reconstructs the replay
tail, sliding-window rings and compressor state without trying to reuse rejected
speculative tokens or an unconsumed final token. It does not sample another answer.
An exact input-boundary snapshot is retained as a fallback, even with zero chunk
snapshots configured. New chunk snapshots are bounded by the existing limit.

The extended prefix remains in RAM and uses the existing disk persistence when
enabled. Both ranks agree on eligibility before replay and execute the same token
spans. Strict routing mode skips preparation if the input snapshot's expert mask
no longer matches. Ordinary mode retains the existing historical-prefix semantics.
Replay does not contribute routing demand a second time.

Only successful plain-text responses qualify. Thinking, parsed tool calls, vision,
and stop-string-truncated responses are skipped because the next chat serialization
may not preserve their raw generated tokens. The next request must still match the
token prefix exactly; this does not add semantic matching or predict the next user
message. Final turn separators are normally handled on the next request.

## Scheduling and cost

At concurrency=1 the request lock remains held during preparation, after the HTTP
body has been flushed. An immediately arriving request can wait for this work.
At concurrency=2 HTTP threads only enqueue optional jobs; the GPU-owner scheduler
processes them when both lanes are idle and no inference request is queued. The
queue holds at most two jobs. Stale jobs whose input snapshot has been replaced
are skipped. A request arriving after preparation begins can still wait; the
operation is not preemptible.

The normal request latency counters exclude this work. Each rank logs a separate
`prefix_response` record with input/response/cached token counts and elapsed time;
`/health` exposes the most recent record for the primary lane. Disk writes retain
the existing shared per-node budget and asynchronous writer.

This shifts prefill work into the gap between requests, rather than eliminating
it. Cache restoration and decoder replay still cost time. There is no guarantee
of a benefit for short responses or changed conversation history.

## Validation

- 26 CPU checks: default enablement and opt-out, response-only spans, input fallback, rank disagreement, strict
  mask changes, demand suppression, delivery/flush ordering, bounded idle queue,
  existing scheduler lifecycle and disk-prefix tests.
- 16 mock HTTP server tests passed with the switch enabled.
- `tools/test_response_prefix_cuda.py`: real native MXFP4 TP2, packed KV, 768K
  allocation, 92 GB arena, frozen keep=0.63. A 12-token prompt generated 89 visible
  answer tokens. Preparation saved all 101 tokens. After an unrelated request
  displaced RAM state, the next 114-token chat restored 101 tokens from disk and
  matched the fresh-prefill output exactly. A second-lane preparation/reuse check
  also passed, including the shared disk writer.

Measured cold preparation: 1.414 s. Disk load: 0.043 s. Cached next-turn prefill:
1.558 s versus 0.500 s fresh. These are short, non-interleaved functional probes,
not a speedup demonstration; the cold cache path was slower. Do not advertise
near-zero prefill or below-noise cost from these results.

Live HTTP check with adaptation restored and concurrency=1: a non-streamed answer
returned 55 tokens after a 12-token input. Preparation then saved 67 tokens in
1.451 s. The next streamed chat reused all 67 out of 77 prompt tokens (only ten
new tokens), completed normally, and prepared its own four-token answer afterward
in 0.269 s. Next-turn prefill still took 1.227 s, including replay overhead; the
test demonstrates reuse and response completion, not near-zero latency.

Local logs: `results/response-prefix-gate/rank{0,1}.log`. No captured user prompts
or runtime prefix bundles are checked into Git.
