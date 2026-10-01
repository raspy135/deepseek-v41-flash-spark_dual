# DeepSeek decode speed vs TensorFold: findings (2026-09-30)

Decode is now about 11–14% faster than this morning, with identical output tokens. The remaining
gap to TensorFold is data moved per step, not kernel launches or kernel speed. Detailed
measurements, including null results: [decode-launches.md](decode-launches.md).

## Why TensorFold was faster

The September trace put our decode step at about 129 ms against TensorFold's 53 ms (GLM-5.3, W11).
The gap wasn't one thing:

| Cause | Gap vs TensorFold |
|---|---:|
| ~19 ms of tiny kernels (10,242 per step against their 1,853) | +15 ms |
| FP8 dense weights against their 4-bit | +15 ms |
| GPU idle time, mostly waiting on Engram reads | +13 ms |
| Larger experts and a wider verify step | +10 ms |
| FP32 matmuls on CUDA cores | +8 ms |
| BF16 LM head, read twice per step | +7 ms |
| NCCL calls between the nodes | +5 ms |

## What changed (all bit-identical)

| Step | Kernels per step | ms per step | Speedup |
|---|---:|---:|---:|
| Start (today's build) | 8,457 | ~113 | |
| Fused norm, mixing, RoPE, router, softmax and SwiGLU code; buffer copies removed | 4,115 | −6 to −9 | +6–9% |
| NCCL calls between nodes cut from 4 to 2 per layer | 3,715 | ~−1 | within noise |
| Attention inputs gathered straight into fp32 | 3,557 | 0 | none (null result) |
| Expert kernel rewritten to load data in larger chunks (v2) | 3,557 | −2.5 to −3 | +2.4–2.9% |

- **Net:** about 102 ms per step. html, python and explain went from about 40.7 / 43.3 / 21.5 to
  46.3 / 48.2 / 24.0 tok/s. The start and end figures come from different runs, and throughput
  drifts a few percent over a day; each change's own effect was measured in alternating A/B runs.
- **Checks:** output tokens were identical in every A/B run, and 21 unit tests compare each new
  kernel to the original with `torch.equal` (`tools/test_decode_lean.py`, `tools/test_fp4_moe_v2.py`).
- **Off switches:** every change has one in the boot guard (`DSV41_DECODE_LEAN*`, `DSV41_FP4_CUDA_V2`).

## What was learned

- **Kernel launches were a smaller lever than first claimed.** Each removed kernel saved only about
  2.2 µs, because CUDA graphs already hide the launch gaps. The estimate was +15 ms; about 9 ms was
  recovered.
- **The expert matmuls now run as fast as memory allows.** In the engine they move about 208 GB/s,
  counting the shared expert reading memory at the same time. Their ~46 ms per step comes from the
  data itself: about 22 experts per layer × 9.4 MB × 40 layers.
- **The NCCL calls cost waiting, not data.** Merging them saved little because each remaining call
  waits longer for the other node.
- **Two traps that would have silently changed outputs:**
  - Triton flushes tiny float values (denormals) to zero by default, where PyTorch keeps them.
    Fix: `enable_reflect_ftz=False`.
  - PyTorch's complex multiply rounds in a specific fused-multiply-add order. That only shows up
    when checking at fp32 precision; at bf16, three wrong variants looked correct.

## What's left

The step is now set by data moved per step, so further gains mean reading less:

- **Expert data:** the ~46 ms above, set by how many experts each step reads.
- **FP8 dense weights:** about 30 ms. Moving to 4-bit was measured 2026-10-01 and is **not**
  faster: at M=6 these projections are latency-bound, not byte-bound, and the profiled dense family
  did not move in any arm. It also changes tokens, so it is cost without benefit; stays off. The
  table and the one code fix it needed (fp4 `shard()` under TP attention) are in
  [decode-launches.md](decode-launches.md#dense-fp4-dsv41_dense_fp4-no-decode-gain-on-this-build-null-result).
- **BF16 LM head:** about 7.7 ms. Giving the draft its own cheaper head saves about 3 ms without
  changing output.
- **FP32 matmuls and torch attention:** about 10 ms. Speeding these up changes numerics, so it needs
  the quality checks.
- **NCCL calls:** about 8 ms, mostly waiting.

## State

- **Committed:** `d90481f` on branch `decode-lean` (rounds 1–2).
- **Uncommitted:** round 3, v2 and this write-up.
- **Not yet checked end to end on the new build:** sampling at temperature > 0, long contexts,
  vision requests, `DSV41_MAX_CONCURRENCY > 1`.
- **Rollback without rebuilding:** `DSV41_DECODE_LEAN=0` and `DSV41_FP4_CUDA_V2=0` in `.env` on both
  nodes, then restart the pair.
