# Batched sampled speculative verification

`DSV41_BATCHED_VERIFY=1` enables an experimental verifier for speculative decoding at
temperature > 0 and 0 < top_p <= 1. Default is **0**. The implementation is in
`engine/spec_sampling.py`; it batches temperature/softmax/nucleus calculations across
the B+1 target rows, computes leading acceptance and stop positions on the GPU, then
returns one compact `[accepted_count, bonus_or_minus_one, *drafts]` result to the CPU.
It uses PyTorch CUDA operations, not a new fused CUDA kernel. Full vocabulary sorting
still occurs when top_p < 1.

The existing greedy verifier, default sequential sampled verifier, penalties and grammar
masking, cache rollback, and final output-token budget handling remain in place. The new
branch shares the existing rollback/output handling after the decision. Both the flag
and sampler source digest participate in the EP2 boot configuration agreement and appear
in health/engine statistics as `batched_verify` and `batched_verify_source`. There are no
new collectives. The dual launcher already forwards every DSV41 variable to both ranks.

## Sampling contract

The implementation retains the rejection algorithm:

- Accept proposal d at position i with probability min(1, p_i[d]/q_i[d]), keeping the
  existing 1e-20 denominator clamp.
- Stop at the first rejection and draw from positive(p_i-q_i), normalized. Retain the
  old target-distribution fallback when that residual has zero mass.
- If all drafts are accepted, sample the bonus from the final target row.
- An accepted stop token ends the burst with no bonus. A stop after the first rejection
  has no effect.

The categorical sampler uses an inverse CDF in FP32. Its threshold is capped below the
total mass to avoid selecting a trailing zero-probability token when multiplication rounds
up. This has ordinary floating-point cumulative-sum rounding and is not bit-equivalent to
torch.multinomial. Temperature/nucleus rules are preserved; tie-support tests cover both
small inputs and the 129,280-token model vocabulary on CPU and CUDA.

Random consumption **changes**: one call draws B acceptance uniforms and one categorical
uniform, even when the first draft rejects or an early stop is accepted. Repeated runs with
the same seed and new configuration reproduce each other, but do not reproduce the old
sequential RNG stream. Both ranks must use the same implementation/configuration. This
is why the experiment has an explicit switch rather than replacing sampled decoding by default.

## Measurements (2026-09-22)

Local GB10, 129,280 vocabulary entries, FP32 probability calculations, temperature 0.6,
five warmups and 100 alternating A/B samples per workload. Times are median host wall
milliseconds including CUDA completion and final result transfer. The serving model stayed
loaded; this separate microbenchmark did not issue serving requests or change its settings.
Inputs are synthetic logits/proposals, not a captured real request.

| Drafts | top_p | Proposal case | Sequential ms | Batched ms |
|---:|---:|---|---:|---:|
| 3 | 0.95 | Mixed acceptance | 0.855 | 0.719 |
| 3 | 0.95 | All accepted | 0.870 | 0.730 |
| 3 | 0.95 | First rejected | 0.344 | 0.720 |
| 5 | 0.95 | Mixed acceptance | 1.245 | 0.917 |
| 5 | 0.95 | All accepted | 1.251 | 0.916 |
| 5 | 0.95 | First rejected | 0.356 | 0.903 |
| 3 | 1.0 | Mixed acceptance | 0.376 | 0.191 |
| 3 | 1.0 | First rejected | 0.220 | 0.190 |
| 5 | 1.0 | Mixed acceptance | 0.529 | 0.194 |
| 5 | 1.0 | First rejected | 0.234 | 0.191 |

Mixed trials use different RNG consumption, so acceptance counts are not identical:
three-draft top_p=0.95 averages were 2.54/2.50, and five-draft averages 3.28/3.88. The
all-accepted and first-rejected trials control this difference; the benchmark reports
observed counts for every arm. It does not force a stochastic rejection to occur.

Batched probability calculation alone for four target rows at top_p=0.95 took 0.552 ms
versus 0.644 ms for four separate calls. With top_p=1 (no sorting), it took 0.047 versus
0.150 ms. These measure equal numbers of probability rows; the sequential complete verifier
can skip unused rows. That explains why early rejection reverses the top_p=0.95 result.

The roughly 16% sampler saving in the three-draft mixed case is only 0.14 ms per decode
iteration. It is **not** a 16% model-throughput gain. No full-engine throughput, two-node
generation, or language-quality comparison has run for this option. The early-rejection
regression and small absolute saving keep it default-off. The README serving throughput
numbers still describe the existing verifier.

## Validation and reproduction

17 tests pass, covering fixed-random decisions against an independent sequential oracle,
every rejection position at B=1/3/5/15, accepted stops and ignored later stops, zero residual
fallback, categorical boundaries, masked tokens and nucleus ties, random decisions, a
small-vocabulary distribution check, CUDA graph replay with changed decisions, seeded
repeatability/fixed RNG consumption, and the actual engine branch's rollback and output
budget handling with lightweight model fixtures. These are sampler/integration checks,
not model quality gates.

```sh
.venv/bin/python -m unittest tools.test_spec_sampling -v
.venv/bin/python tools/bench_spec_sampling.py --iters 100
.venv/bin/python tools/bench_spec_sampling.py --iters 100 --accept-pattern all
.venv/bin/python tools/bench_spec_sampling.py --iters 100 --accept-pattern first-reject
```

To test serving, rebuild/copy the same image on both nodes and start the pair with
`DSV41_BATCHED_VERIFY=1`; verify both health fields above before measuring. Revert with
`DSV41_BATCHED_VERIFY=0` and restart the pair. A useful next measurement keeps the image,
pruning/adaptation policy, prompts, temperature/top_p, and output budget matched, and
reports acceptance and milliseconds per iteration along with tokens/sec. Different
seeded outputs mean a single tokens/sec pair can be dominated by acceptance differences.
