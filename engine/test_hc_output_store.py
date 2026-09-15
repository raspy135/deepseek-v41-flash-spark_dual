"""Compare final BF16 store against the original FP32-store-then-cast path.

Small synthetic tensors only; no model weights or user input. CUDA event timings.
"""
import statistics
import torch
from engine.hc_ops import _hc_pre_norm_kernel, hc_pre_rmsnorm


def main():
    torch.manual_seed(17)
    for t in (1, 6, 60, 64, 128, 512, 2048, 2108):
        d, hc = 5120, 4
        x = torch.randn(t, hc, d, device='cuda', dtype=torch.bfloat16)
        pre = torch.rand(t, hc, device='cuda')
        w = torch.randn(d, device='cuda')

        def baseline():
            scratch = torch.empty(t, d, device='cuda', dtype=torch.float32)
            _hc_pre_norm_kernel[(t,)](x, pre, w, scratch, scratch, d, 1e-6,
                                     HC=hc, BD=1024, num_warps=8)
            return scratch.to(torch.bfloat16)

        def candidate():
            return hc_pre_rmsnorm(x, pre, w, 1e-6, direct_store=True)

        ref = baseline()
        assert torch.equal(ref, candidate()), t
        assert torch.equal(ref, hc_pre_rmsnorm(x, pre, w, 1e-6)), t
        # Alternate arms after compilation to limit clock/warm-up bias.
        samples = [[], []]
        for _ in range(20):
            for j, fn in enumerate((baseline, candidate)):
                a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                a.record()
                got = fn()
                b.record()
                b.synchronize()
                assert torch.equal(ref, got), t
                samples[j].append(a.elapsed_time(b))
        old, new = map(statistics.median, samples)
        print(f'T={t} bit_exact=True baseline_ms={old:.4f} candidate_ms={new:.4f} speedup={old/new:.3f}', flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
