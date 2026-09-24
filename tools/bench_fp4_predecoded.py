"""Benchmark-only FP16 weight expansion; never changes the serving expert arena.

Run with MODEL_DIR set and the serving worker stopped. The adjacent experiment patch creates
a temporary ABI-compatible CUDA kernel with FP16 loads instead of FP4 unpacking. Expansion
is outside timing: these are optimistic steady-state numbers, not cache-fill/serving costs.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import subprocess
import tempfile

import torch

import fp4_moe as K
import fp4_moe_cuda as CUDA
from bench_fp4_moe_cuda import compare_sources
from test_fp4_moe import load_arena, routing_with_distinct


def expand(arena, scaled):
    result = copy.copy(arena)
    values = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                           -0., -.5, -1, -1.5, -2, -3, -4, -6],
                          device="cuda", dtype=torch.float16)
    for name in ("w1", "w2", "w3"):
        packed = getattr(arena, name)
        shape = (*packed.shape[:-1], packed.shape[-1] * 2)
        unpacked = torch.empty(shape, device=packed.device, dtype=torch.float16)
        scales = getattr(arena, "s" + name[1:])
        for expert in range(packed.shape[0]):
            p = packed[expert]
            v = torch.stack((values[(p & 15).long()], values[(p >> 4).long()]), -1)
            v = v.reshape(shape[1:])
            if scaled:
                factor = torch.exp2(scales[expert].float() - 127).repeat_interleave(32, -1)
                v = (v.float() * factor).half()
            unpacked[expert].copy_(v)
        setattr(result, name, unpacked)
        if scaled:
            setattr(result, "s" + name[1:], torch.full_like(scales, 127))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--scaled", action="store_true",
                        help="Fold scales into FP16 weights (changes arithmetic order)")
    args = parser.parse_args()
    if args.iters < 1 or K.DOT_SCALED:
        parser.error("positive iterations and DSV41_FP4_DOT_SCALED=0 required")
    original = CUDA.SOURCE
    arena = load_arena(32, "cuda")
    expanded = expand(arena, args.scaled)
    packed_bytes = sum(getattr(arena, w).nbytes for w in ("w1", "w2", "w3"))
    print(f"packed_weight_bytes={packed_bytes} expanded_weight_bytes={packed_bytes * 4} "
          f"scaled={args.scaled}", flush=True)
    gen = torch.Generator().manual_seed(20260922)
    with tempfile.TemporaryDirectory(prefix="fp4-predecoded-") as folder:
        source = Path(folder) / "predecoded.cu"
        patch = Path(__file__).with_name("experiments") / "fp4_predecoded.patch"
        subprocess.run(["patch", "--batch", "--fuzz=0", "-o", str(source), str(original),
                        str(patch)], check=True)
        if args.scaled:
            scaled_source = Path(folder) / "prescaled.cu"
            scaled_patch = patch.with_name("fp4_prescaled.patch")
            subprocess.run(["patch", "--batch", "--fuzz=0", "-o", str(scaled_source),
                            str(source), str(scaled_patch)], check=True)
            source = scaled_source
        CUDA.SOURCE = source
        try:
            expanded_lib = CUDA._library()
            for tokens, distinct in ((1, 6), (6, 30), (8, 32)):
                x = torch.randn((tokens, K.DIM), generator=gen).bfloat16().cuda()
                slots, weights = routing_with_distinct(tokens, distinct, gen, "cuda")

                def run():
                    use_expanded = K.CUDA_DECODE and CUDA._LIB._name == expanded_lib._name
                    return K.moe_forward(x, slots, weights, expanded if use_expanded else arena,
                                         out_dtype=torch.float32)

                times, errors, exact = compare_sources(run, args.iters, original, True)
                print(f"tokens={tokens} experts={distinct} ms={times} "
                      f"relative_error={errors} exact_previous={exact}", flush=True)
                if not all(error < 1e-3 for error in errors.values()):
                    raise AssertionError("predecoded candidate failed kernel error threshold")
        finally:
            CUDA.SOURCE = original


if __name__ == "__main__":
    with torch.inference_mode():
        main()
