# Prefill MoE subphase timing

`DSV41_PREFILL_MOE_TIMING=1` enables an EP2 software-FP4 diagnostic on both ranks.
The flag is checked by the boot configuration guard. It does not change arithmetic,
weight placement, allocation sizes, or collective order. It adds host timestamps and
CUDA events and therefore has measurement overhead; compare similarly instrumented runs.
Decode graph replay is not instrumented.

Each request logs `prefill_moe_timing` on both ranks; rank 0 also returns it in engine
stats. The report includes aggregate totals and per-layer calls, with token count:

- `router_and_lookup`: gate, pruning/demand accounting, top-k, weights and slot maps.
- `grouping`: compact routing/block construction and mapping back to arena slots.
- `scratch_alloc`: weight formatting plus intermediate/output allocations.
- `up_gemm`: launching and executing the grouped gate/up/SwiGLU kernel.
- `down_gemm`: launching and executing the grouped down-projection kernel.
- `reduce`: top-k reduction and output conversion.

`host_ms` measures elapsed time on the dispatching thread, including stalls inside
CUDA/Triton calls. `gpu_envelope_ms` measures between CUDA events on the stream,
including time idle while the host prepares later launches. Neither measures pure
CPU execution or pure GPU kernel busy time. Do not subtract the two as an idle-time
estimate. A device timeline is still required to separate those precisely.

There are no added in-prefill synchronizations: the final event is synchronized only
when reporting after generation. Existing outer phase timers are preserved so the
new breakdown can be compared against `moe_kernel`. The test uses fake events to
verify deferred synchronization and per-call/aggregate accounting:

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest engine.test_prefill_timing
```

The fixed-placement README experiment motivating this diagnostic measured
540 -> 864 -> 720 -> 915 tok/s with zero prefix reuse, zero expert-weight reads, and
no expert swaps. Thus adaptation is not necessary for the variability. On the worker,
the outer expert/routing envelope fell from 4.601 to 2.957 seconds between the 720
and 915 runs; on the head, combine/wait fell from 2.480 to 0.254 seconds. These are
different ranks' overlapping intervals, not additive savings. Engram read time on
the head was 1.432 versus 1.484 seconds (it can overlap computation and is separate
from the expert NVMe counter). These observations localize the investigation without
establishing a kernel or allocator cause.

## Initial serving measurements

The initial deployment missed forwarding `stage_mark` through the engine's arena-format
dispatcher. The first long request failed with a TypeError and the pair was restarted with
that wrapper repaired. A regression test now extracts and exercises the actual nested
dispatcher without constructing model weights; it verifies both the enabled callback and
the unchanged no-callback case. Do not use the failed request's partial timing report.

Repaired image `15660b8d84adb0649cdc1db9fa429f4598190012d32e3bd1b65dc5babf9df717` ran
on both nodes with adaptation on, replicas off, arena=88 GB. Three exact 7,751-token README
requests, zero prefix reuse, measured 16.588 / 14.570 / 8.674 s (467 / 532 / 894 tok/s).
The subsequent 6,082-token depth-10 nesting control passed exact JSON grading after swaps.

The first request had 4.15/4.48 s of head/worker host grouping time versus 44/46 ms of GPU
grouping envelopes. On the second request host grouping fell to 47/49 ms, but prefill was
still slow. Thus that first-request host cost is not the entire explanation.

Comparing the second and third requests on the worker:

| phase | slow host ms | fast host ms | slow GPU envelope ms | fast GPU envelope ms |
|---|---:|---:|---:|---:|
| grouping | 49.393 | 30.027 | 8.353 | 7.947 |
| scratch allocation | 2.289 | 1.377 | 0.103 | 0.124 |
| up GEMM | 21.347 | 20.397 | 4448.277 | 1495.624 |
| down GEMM | 4.421 | 5.176 | 2133.803 | 1223.783 |

Both reports contain 103 MoE calls: 4 encoder chunks x 21 layers plus 19 decoder-tail
layers. Several slow-run worker up-GEMM intervals are 248-364 ms despite host durations of
0.02-0.05 ms. The fast run's largest worker up interval is 19.332 ms. This localizes large
outliers to the GPU timeline around expert computation, not ordinary Python dispatch or
the measured scratch allocations. These remain event envelopes: distinguishing actual
kernel execution, GPU scheduling/other-stream interference and memory-system stalls needs
a device timeline. It does not establish the cause or prove that memory pressure is absent.
Allocating additional persistent scratch is not supported as a seconds-scale speed fix by
these measurements; the measured allocation cost was only milliseconds.
