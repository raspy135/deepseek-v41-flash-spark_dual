"""Compare a full routed sum with the two null-masked EP halves on one GPU."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_fp4_moe import load_arena, random_routing
from fp4_moe import DIM, moe_forward


def main():
    arena = load_arena(32, "cuda")
    null = 31
    for name in ("w1", "s1", "w2", "s2", "w3", "s3"):
        getattr(arena, name)[null].zero_()
    gen = torch.Generator().manual_seed(1234)
    for n in (1, 6, 63, 512):
        x = torch.randn(n, DIM, generator=gen).bfloat16().cuda()
        ids, weights = random_routing(n, 30, gen, "cuda")
        full = moe_forward(x, ids, weights, arena, out_dtype=torch.float32)
        halves = []
        for rank in range(2):
            local = torch.where(ids % 2 == rank, ids, torch.full_like(ids, null))
            halves.append(moe_forward(x, local, weights, arena, out_dtype=torch.float32,
                                      slots_repeat=True, null_slot=null))
        split = halves[0] + halves[1]
        shared = torch.randn(n, DIM, generator=gen).bfloat16().cuda().float()
        equal = torch.equal((full + shared).bfloat16(), (split + shared).bfloat16())
        print(f"T={n}: max_delta={float((full-split).abs().max()):.9g} "
              f"fp32_equal={torch.equal(full, split)} final_bf16_equal={equal}", flush=True)
        assert equal, f"EP regrouping changed the BF16 output at T={n}"


if __name__ == "__main__":
    main()
