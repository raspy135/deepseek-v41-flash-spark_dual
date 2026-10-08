# Runtime decode probe

The native engine can measure decode operations while the model stays loaded. Deploying
this code requires one initial restart; subsequent arm/run/stop commands use HTTP.

Open `/decode-probe` after launching this server version. The already running server
uses `/expert-map?view=decode`, which can receive a static page update without reloading
the model. The dashboard uses the APIs below; scripts can continue using JSON directly.
It retains the latest run report in the browser when probing is stopped.
The default chart shows local operations and communication, hiding enclosing wrappers.
Use **Show nesting** or **Call tree** to expand inclusive totals and compare both nodes
at each level. Parent and child measurements repeat the same work; different streams
can overlap, so leaf times are not a total layer cost either. Communication includes
transfer and peer waiting, not a separately measured pure wait time.
New captures record `metadata.parent`. Older reports recover known dispatch chains
and scope nesting from capture order; the tree labels these links as inferred.
The parent-ID change takes effect after launching the updated engine; the static
dashboard can display older reports immediately without restarting the model.
After editing `server/decode_probe.html`, maintainers should run
`python tools/sync_decode_dashboard.py` to update the query-view bootstrap.

`python tools/decode_probe.py arm`, `run --out /tmp/decode.json`, `status`, and
`stop` control the probe and print compact reports. Use `--url` for another host,
`--names '*dense.fp8*'` for specific operations, or `show --input /tmp/decode.json`
to inspect saved results. Selecting another group requires graph recapture, not a
model reload. Timings cover all recorded groups; private isolation has a bounded budget.

```sh
curl -X POST http://localhost:8000/v1/decode-probe \
  -H 'Content-Type: application/json' -d '{"action":"arm"}'
# Run a normal generation to collect decode cases, then profile between requests:
curl -X POST http://localhost:8000/v1/decode-probe \
  -H 'Content-Type: application/json' -d '{"action":"run"}'
curl http://localhost:8000/v1/decode-probe
curl -X POST http://localhost:8000/v1/decode-probe \
  -H 'Content-Type: application/json' -d '{"action":"stop"}'
```

Arming recaptures instrumented graphs. Decode timings cover recorded operations; safe
local operations matching `names` also retain bounded input snapshots in RAM for idle
replay. Stateful operations and collectives are timed only. Prompt text and tensors are
never returned by the API. Stop releases snapshots and instrumented graphs.
Snapshots occupy disjoint slices of one arena allocated outside graph capture.
`snapshot_bytes` reports logical usage, `snapshot_consumed_bytes` includes alignment,
and `snapshot_reserved_bytes` reports the allocated arena capacity. Isolation adds
temporary working copies and the configured cache-pressure buffer.

Defaults: `names=["*dense*","*head*","*markov*"]`, `max_cases=256`,
`snapshot_budget_mb=64`, `warmup=3`, `repeats=6`, `calls=16`, `flush_mb=64`.
Names accept up to 64 glob patterns of 128 characters each. Limits are 1024 cases,
256 MiB snapshots/flush, 16 warmups/repeats and 64 calls; repeats/calls must be positive.
Run accepts the same bounded overrides; omitted fields retain the armed settings.
GET returns cached status without CUDA work.

Commands return reports from both TP ranks, including diagnostic failures (`ok=false`).
HTTP 409 means inference is busy, 501 means the engine or concurrency=2 is unsupported,
and 503 means the engine pair is faulted. Nested operation times can overlap; idle
replays describe the loaded process and its cache controls, not hardware DRAM counters.

`loop_phases` also reports the latest generation's host phases: draft, block, hash,
hash readback, row submission, target step, optional grammar, verification, rollback,
emission and consumer wait. Default-stream means retain at most 64 spans per phase;
host totals and allocator counters cover every recorded interval. Stream spans include
GPU idle gaps, while consumer time includes the caller's time between yields.

For buffer reuse, compare a warmed generation with `capture_intervals=0`.
`allocation_requests` counts PyTorch requests, including allocations satisfied from
its pool. The legacy `requested_bytes` field contains allocator-accounted allocated
bytes, rounded to allocator bins; the dashboard calls this allocation bytes per round.
`device_allocations`/`device_frees` count pool growth/release.
They do not count all external allocations or GPU memory traffic. Helper allocations
inside captured graphs happen during capture, not every replay. Probe events and
snapshot copies add diagnostic overhead, so use unarmed runs for final throughput.

Validated on the two-Spark TP2 engine with a 33-token prompt and 96-token greedy
completion: 4,520 replayed timing spans and 488 bit-exact isolated comparisons per
rank (512 snapshot candidates; the unused sampled branch is excluded). A subsequent
48-token warmed run captured no new graphs and made zero CUDA device allocations
or frees. It recorded 49 hashing, approximately 23 target-step, and 7 verification
allocation requests per round, satisfied from existing pools. Its output matched
the 48-token run after stopping diagnostics. These are instrumentation/reuse checks,
not a throughput or general quality benchmark.

The initial full-engine screen rejected 56 isolated comparisons per rank. Moving
snapshots out of the shared graph pool increased exact matches from 432/488 to
488/488. Keep output validation enabled; rejected comparisons have no idle latency.
