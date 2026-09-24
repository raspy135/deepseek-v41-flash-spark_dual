"""Real-weight A/B benchmark for the opt-in native CUDA decode MoE path.

Run only with the serving process stopped: the 32-expert arena is about 0.6 GB.  The result is not a
promotion gate by itself; the TP parity and generation gates still apply.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
import torch

import fp4_moe as K
from test_fp4_moe import load_arena, routing_with_distinct


def compare_sources(fn, iters, baseline_source=None, graph_mode=False, include_relaxed=False):
    """Compare the same tensors, optionally replaying graphs to exclude host launch gaps."""
    import fp4_moe_cuda as CUDA
    original_source, original_lib = CUDA.SOURCE, CUDA._LIB
    original_relaxed = CUDA.RELAXED_REDUCE
    original_native = K.CUDA_DECODE
    arms, outputs, graphs = {}, {}, {}
    try:
        sources = {"triton": None}
        if baseline_source:
            sources["previous_cuda"] = Path(baseline_source).resolve()
        sources["cuda"] = original_source
        if include_relaxed:
            sources["cuda_relaxed"] = original_source
        for name, source in sources.items():
            native = source is not None
            lib = None
            if native:
                CUDA.SOURCE = source
                CUDA.RELAXED_REDUCE = name == "cuda_relaxed" if include_relaxed else original_relaxed
                lib = CUDA._library()
            def run(native=native, lib=lib):
                K.CUDA_DECODE = native
                CUDA._LIB = lib
                return fn()
            for _ in range(4):
                output = run()
            torch.cuda.synchronize()
            outputs[name] = output
            if graph_mode:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs[name] = run()
                graph.replay()
                graphs[name] = graph
                arms[name] = graph.replay
            else:
                arms[name] = run
        torch.cuda.synchronize()
        values = {name: [] for name in arms}
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        names = list(arms)
        for repeat in range(iters):
            for name in (names if repeat % 2 == 0 else names[::-1]):
                start.record()
                arms[name]()
                end.record()
                end.synchronize()
                values[name].append(start.elapsed_time(end))
        reference = outputs["triton"]
        errors = {name: float((output - reference).norm() / reference.norm().clamp_min(1e-30))
                  for name, output in outputs.items()}
        previous = outputs.get("previous_cuda")
        exact = None if previous is None else torch.equal(outputs["cuda"], previous)
        return {name: statistics.median(times) for name, times in values.items()}, errors, exact
    finally:
        CUDA.SOURCE, CUDA._LIB = original_source, original_lib
        CUDA.RELAXED_REDUCE = original_relaxed
        K.CUDA_DECODE = original_native


def timed_pair(fn, iters):
    values = {False: [], True: []}
    for native in (False, True) * 2:
        K.CUDA_DECODE = native; fn()
    torch.cuda.synchronize()
    # A/B/B/A controls for clock and competing-process drift better than two sequential runs.
    for repeat in range(iters):
        for native in ((False, True) if repeat % 2 == 0 else (True, False)):
            K.CUDA_DECODE = native
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); fn(); end.record(); end.synchronize()
            values[native].append(start.elapsed_time(end))
    return statistics.median(values[False]), statistics.median(values[True])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--baseline-source", help="Saved CUDA source with the same C ABI")
    parser.add_argument("--graph", action="store_true", help="Time CUDA graph replay")
    parser.add_argument("--compare-relaxed", action="store_true", help="Compare both CUDA reduction modes")
    args = parser.parse_args()
    if args.iters < 1:
        parser.error("--iters must be positive")
    if K.DOT_SCALED:
        parser.error("unset DSV41_FP4_DOT_SCALED to compare the native CUDA path")
    gen = torch.Generator().manual_seed(20260922)
    arena = load_arena(32, "cuda")
    rows = []
    try:
        for tokens, distinct in ((1, 6), (6, 30), (8, 32)):
            x = torch.randn((tokens, K.DIM), generator=gen).bfloat16().cuda()
            slots, weights = routing_with_distinct(tokens, distinct, gen, "cuda")

            if args.baseline_source or args.graph or args.compare_relaxed:
                times, errors, exact = compare_sources(
                    lambda: K.moe_forward(x, slots, weights, arena, out_dtype=torch.float32),
                    args.iters, args.baseline_source, args.graph, args.compare_relaxed)
                print(f"tokens={tokens} experts={distinct} graph={args.graph} "
                      f"ms={times} relative_error={errors} exact_previous={exact}", flush=True)
                assert all(error < 1e-3 for error in errors.values())
                continue

            K.CUDA_DECODE = False
            baseline = K.moe_forward(x, slots, weights, arena, out_dtype=torch.float32)
            K.CUDA_DECODE = True
            native = K.moe_forward(x, slots, weights, arena, out_dtype=torch.float32)
            base_ms, native_ms = timed_pair(
                lambda: K.moe_forward(x, slots, weights, arena, out_dtype=torch.float32),
                args.iters)
            torch.cuda.synchronize()
            rel = float((native - baseline).norm() / baseline.norm())
            weight_bytes = distinct * arena.bytes_per_slot
            row = dict(tokens=tokens, experts=distinct, relative_error=rel,
                       triton_ms=base_ms, cuda_ms=native_ms, speedup=base_ms / native_ms,
                       triton_gbs=weight_bytes / base_ms / 1e6,
                       cuda_gbs=weight_bytes / native_ms / 1e6)
            rows.append(row)
            print(" ".join(f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
                           for key, value in row.items()), flush=True)
        assert all(row["relative_error"] < 1e-3 for row in rows)
    finally:
        K.CUDA_DECODE = False


if __name__ == "__main__":
    with torch.inference_mode():
        main()
