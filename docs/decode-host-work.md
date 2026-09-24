# Decode host-side work

The target is exposed host delay, not every Python loop. A Python call waiting
for a CUDA stream or a page fault is not evidence that its bytecode is expensive.
Keep kernel settings and expert ranking fixed while comparing host changes.

## Engram gather candidate

The original `_gather_rows()` schedules up to 64 Python tasks per table. Each
constructs slices, runs two NumPy advanced-index gathers, and copies their
temporary outputs into one result. Basic NumPy slices are already views; the
advanced-index gathers are the allocating copies. Replacing a basic slice with
`memoryview` alone would not remove task dispatch or the indirect gather.

`engine/engram_native.py` and `engine/engram_gather.cpp` provide an optional
decode-sized path (`DSV41_ENGRAM_NATIVE=1`, default off):

- One GIL-releasing ctypes call per gather, with persistent sleeping C++ workers.
- Direct 256-byte weight and 8-byte scale copies into the output; no per-task
  Python closure, future, slice, or temporary NumPy gather.
- Four reusable buffers per table, with ownership retained through the returned
  array's ctypes/memoryview base. NumPy slices and Torch CPU aliases keep the
  lease alive. If every buffer is still held by a future/consumer, allocate an
  independent output instead of blocking or overwriting a live buffer.
- Native workers validate every row ID before copying. No value conversion,
  quantization, expert selection or TP collective is changed.
- Only gathers of at most 384 unique rows take this path. Large prefill gathers
  retain the old row-split and Python-pool implementation.

`DSV41_ENGRAM_NATIVE_THREADS` defaults to 64. This is an experimental CPU path,
not a promise that fewer workers are better. The helper needs a C++17 compiler
(`g++` exists in the serving image), builds once into the local Triton cache,
and is keyed by source, compiler version and machine architecture. No downloads,
Python/NumPy extension ABI or CUDA build are involved. Native source participates
in disk-prefix compatibility hashing. Roll back with `DSV41_ENGRAM_NATIVE=0` on
both ranks and restart. The normal launch script forwards the same setting.

### Small CPU measurements

September 16, local Spark. Resident synthetic arrays, 40 samples after warm-up:
Python-pool median ~1.4–2.1 ms for 96–384 rows; native pool generally ~0.1–0.2 ms.
Output allocation alone was ~0.0005 ms. Reusing the output buffer is not the
main source of potential savings.

Actual checkpoint tables, randomized fresh row IDs followed immediately by a
warm repeat, 12 samples per arm in alternating order. No global page-cache flush.
Fresh reads incurred roughly two major page faults per row, comparably in both
arms. Each output was checked byte-for-byte against NumPy indexing:

| Layer / unique rows | Python 64 fresh / warm ms | Native 64 fresh / warm ms |
| --- | ---: | ---: |
| 1 / 96 | 6.18 / 2.25 | 1.61 / 0.47 |
| 1 / 144 | 6.69 / 2.20 | 2.13 / 0.56 |
| 14 / 96 | 6.27 / 2.11 | 1.54 / 0.54 |
| 14 / 144 | 7.03 / 2.17 | 1.86 / 0.46 |

Native **8 workers was rejected**: fresh gathers took ~7–11 ms, slower than the
existing path. Native 16/32 improved as parallelism increased but remained slower
on fresh rows than 64 workers. Do not choose the fastest RAM-only microbenchmark
and accidentally serialize storage faults.

These are per-gather times, not guaranteed decode savings. The two table reads
overlap each other and GPU work. End-to-end A/B is the deciding test.

Reproduce CPU checks with `tools/bench_engram_host.py --native`,
`tools/bench_engram_storage.py`, and `tools/test_engram_native.py` inside the
serving image. The storage test needs a read-only model mount. It does not load
the model onto a GPU. Unit tests cover bounds, duplicate/shuffled rows, empty and
large requests, concurrent calls, pool exhaustion, ownership through NumPy/Torch
views, and idempotent shutdown.

## Other candidates to measure

| Area | Source | Why / current disposition |
| --- | --- | --- |
| Pending compressor buffers and rollback | `FastDecoder.prepare_pending_buffers`, `Caches.rollback` | Per-iteration dictionary loops and small tensor views/copies. Measure CPU time before changing load-bearing cache state. |
| Verification/result handling | `V41Engine._generate` greedy accept path | Already GPU-vectorized with one compact result transfer. Small Python list/stop loops remain; do not confuse GPU wait with Python execution. |
| Loop control broadcast | `EPDistributed.control` | Prior trace ~1 ms on rank 0, mostly coordination. Required for cancellation and stop agreement; removing it can deadlock the pair. |
| Hash readback / row submission | `V41Engine` before `FastDecoder.step` | Host readback also waits for the preceding draft. Earlier measurement found hash work itself ~0.17 ms; asynchronous spelling does not remove the dependency. |
| Detokenization and output routing | `server.app.IncrementalDetokenizer`, `OutputRouter` | Python slicing and marker loops, but detokenization uses a bounded recent window. HTTP-side cost is not covered by a direct-engine profile. |
| Backbone/draft layer loops | `FastDecoder.capture`, `_draft` | In the warmed graph path these execute at capture, not once per output iteration. Not a Python hot-loop rewrite target. |

The diagnostic runner's `--profile-host` takes a separate warmed sample using
paired thread-CPU and wall clocks around selected calls. Each CPU interval uses
the same thread for start and finish. It includes native CPU calls made by that
thread but excludes time blocked on GPU/I/O and work on other threads. Nested
scopes overlap; do not add them. Normal serving does not import these timers.

The first attempt used cProfile with `time.thread_time`. Its results included
negative cumulative times and impossible durations in this threaded PyTorch
process. That profile is invalid and was rejected, not used to rank candidates.
The corresponding end-to-end A/B runs preceded profiling and remain valid.
