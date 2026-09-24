"""Reproduce casting/pair-reuse experiments without modifying serving kernels.

Half2 candidates report failed numerical gates instead of aborting, so every workload is measured.
Successful process exit for those experiments is NOT an accuracy or promotion gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile

import torch

import fp4_moe as K
import fp4_moe_cuda as CUDA
from bench_fp4_moe_cuda import compare_sources
from test_fp4_moe import load_arena, routing_with_distinct


def stress_down(original_source, candidate_source):
    """Deliberately large activations: a range stress test, not a serving input distribution."""
    saved_source, saved_lib = CUDA.SOURCE, CUDA._LIB
    try:
        libs = {}
        for name, source in (("baseline", original_source), ("candidate", candidate_source)):
            CUDA.SOURCE = source
            libs[name] = CUDA._library()
        slots = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
        w = torch.full((1, 8, 64), 0x77, dtype=torch.uint8, device="cuda")  # FP4 = +6
        s = torch.full((1, 8, 4), 120, dtype=torch.uint8, device="cuda")  # scale = 1/128
        for activation in (1, 1024, 8192):
            h = torch.full((1, 128), activation, dtype=torch.bfloat16, device="cuda")
            results = {}
            for name, lib in libs.items():
                CUDA._LIB = lib
                block_slot, block_pair, _, nb = CUDA.build_routing_small(slots, 16, 1)
                out = torch.empty((1, 8), device="cuda")
                CUDA.down(h, w, s, out, block_slot, block_pair, 1, 8, 128, 1, nb, -1, True, 1)
                results[name] = dict(value=float(out[0, 0]),
                                     finite=bool(torch.isfinite(out).all()))
            print(f"range_stress activation={activation} expected={activation * 6} "
                  f"results={results}", flush=True)
    finally:
        CUDA.SOURCE, CUDA._LIB = saved_source, saved_lib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("packed", "precast", "pair2", "decode4",
                                               "half2_group", "half2_local"))
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--stress", action="store_true", help="Also check large-activation overflow")
    args = parser.parse_args()
    if args.iters < 1 or K.DOT_SCALED:
        parser.error("positive iterations and DSV41_FP4_DOT_SCALED=0 required")
    original_source, original_lib = CUDA.SOURCE, CUDA._LIB
    original_up, original_down = CUDA.up, CUDA.down
    arena = load_arena(32, "cuda")
    gen = torch.Generator().manual_seed(20260922)
    with tempfile.TemporaryDirectory(prefix="fp4-casting-") as folder:
        source = Path(folder) / "candidate.cu"
        patch = Path(__file__).with_name("experiments") / f"fp4_{args.experiment}.patch"
        subprocess.run(["patch", "--batch", "--fuzz=0", "-o", str(source),
                        str(original_source), str(patch)], check=True)
        try:
            CUDA.SOURCE = source
            if args.experiment == "precast":
                # Include conversions in capture/timing, but only for the candidate library.
                def wrap(fn):
                    def converted(x, *pos, **kw):
                        if hasattr(CUDA._lib(), "fp4_moe_cuda_activation_half"):
                            x = x.to(torch.float16)
                        return fn(x, *pos, **kw)
                    return converted
                CUDA.up, CUDA.down = wrap(original_up), wrap(original_down)
            for tokens, distinct in ((1, 6), (6, 30), (8, 32)):
                x = torch.randn((tokens, K.DIM), generator=gen).bfloat16().cuda()
                slots, weights = routing_with_distinct(tokens, distinct, gen, "cuda")
                times, errors, exact = compare_sources(
                    lambda: K.moe_forward(x, slots, weights, arena, out_dtype=torch.float32),
                    args.iters, original_source, True)
                print(f"experiment={args.experiment} tokens={tokens} experts={distinct} "
                      f"ms={times} relative_error={errors} exact_previous={exact}", flush=True)
                if args.experiment.startswith("half2_"):
                    print(f"kernel_error_gate_passed={all(error < 1e-3 for error in errors.values())}",
                          flush=True)
                else:
                    assert exact and all(error < 1e-3 for error in errors.values())
            if args.stress:
                stress_down(original_source, source)
        finally:
            CUDA.SOURCE, CUDA._LIB = original_source, original_lib
            CUDA.up, CUDA.down = original_up, original_down


if __name__ == "__main__":
    with torch.inference_mode():
        main()
