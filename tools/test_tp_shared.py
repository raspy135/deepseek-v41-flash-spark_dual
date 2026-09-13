"""A tensor-parallel shared expert must equal the unsharded one.

w1/w3 split by output rows, w2 by input columns, partial outputs summed -- the textbook
column-then-row parallel MLP. What this checks is that FP8Weight.shard slices the 32x32 block
scales on the same boundary as the weights: a scale block that straddles a shard boundary still
produces plausible numbers, just wrong ones.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import torch
from fp8_linear import FP8Weight, quantize_to_fp8
import v41_ref as R

torch.manual_seed(0)
DIM, INTER, T, WORLD = 5120, 2304, 7, 2
dev = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(T, DIM, dtype=torch.bfloat16, device=dev)
w1 = quantize_to_fp8(torch.randn(INTER, DIM, dtype=torch.bfloat16, device=dev) * 0.02)
w3 = quantize_to_fp8(torch.randn(INTER, DIM, dtype=torch.bfloat16, device=dev) * 0.02)
w2 = quantize_to_fp8(torch.randn(DIM, INTER, dtype=torch.bfloat16, device=dev) * 0.02)
LIMIT = 10.0

full = R.expert_ffn(x, w1, w2, w3, LIMIT).float()
parts = []
for r in range(WORLD):
    parts.append(R.expert_ffn(x, w1.shard(0, r, WORLD), w2.shard(1, r, WORLD),
                              w3.shard(0, r, WORLD), LIMIT).float())
tp = sum(parts)

d = (tp - full).abs()
rel = d.norm().item() / full.norm().item()
print(f"full {tuple(full.shape)}  tp {tuple(tp.shape)}  on {dev}")
print(f"  max|abs diff| {d.max().item():.3e}   relative Frobenius {rel:.3e}")
print(f"  per-row cosine min {torch.nn.functional.cosine_similarity(tp, full, dim=-1).min().item():.8f}")
# each rank must do real work, or a broken shard could pass by one side being zero
for r, p in enumerate(parts):
    print(f"  rank {r} partial norm {p.norm().item():.3f}")
assert all(p.norm().item() > 0 for p in parts), "a rank contributed nothing"
assert rel < 5e-3, f"sharded result differs: {rel}"
print("\nTP SHARED EXPERT MATCHES THE UNSHARDED ONE")
