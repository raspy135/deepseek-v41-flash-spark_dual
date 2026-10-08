"""Cold actual-weight screen of v2 ordered-chain pair batching; no serving defaults change.

Run in the native image with serving stopped. The arena holds four independent copies of 32
real layer-0 experts, split using the serving TP output layout. Captured calls rotate disjoint
expert sets through those copies; each reuse is separated by more than the 24 MiB L2 capacity.
Rows=1 only has six routed pairs, and rows=4 cannot reach U=26; impossible cells are omitted.
Both arms retain native FP4 weights, FP32 chains and every original rounding boundary.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from safetensors import safe_open
import fp4_moe as K
import fp4_moe_cuda as CUDA

TOPK = 6


def candidate_library(baseline, build_dir):
    target = Path(build_dir) / "pairbatch2.so"
    nvcc = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
    command = [nvcc, "-std=c++17", "-O3", "-DNDEBUG", "-Xcompiler=-fPIC", "-shared",
               "-DFP4_ROWS_PER_WARP=1", "-DFP4_RELAXED_REDUCE=1", "-DFP4_V2_FMA_SCALE=0",
               "-DFP4_V2_PAIR_BATCH=2", "-gencode", "arch=compute_121a,code=sm_121a",
               str(CUDA.SOURCE), "-o", str(target)]
    subprocess.run(command, check=True)
    lib = ctypes.CDLL(str(target))
    for name in ("fp4_moe_cuda_up", "fp4_moe_cuda_down", "fp4_moe_cuda_up_v2",
                 "fp4_moe_cuda_down_v2", "fp4_moe_cuda_route_small", "fp4_moe_cuda_round_reduce"):
        getattr(lib, name).argtypes = getattr(baseline, name).argtypes
        getattr(lib, name).restype = ctypes.c_int
    return lib


def load_actual_weights():
    os.environ["DSV41_TP_EXPERT_LAYOUT"] = "output"
    arena = K.ExpertArena(128, "cuda", tp_rank=0, tp_world=2)
    root = Path(os.environ["MODEL_DIR"])
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}
    for expert in range(32):
        prefix = f"layers.0.ffn.experts.{expert}."
        values = []
        for matrix in ("w1", "w2", "w3"):
            for field in ("weight", "scale"):
                name = prefix + matrix + "." + field
                filename = index[name]
                if filename not in handles:
                    handles[filename] = safe_open(str(root / filename), "pt", device="cpu")
                values.append(handles[filename].get_tensor(name))
        arena.load_slot(expert, *values)
    for tensor in (arena.w1, arena.s1, arena.w3, arena.s3, arena.w2, arena.s2):
        for copy in range(1, 4):
            tensor[copy * 32:(copy + 1) * 32].copy_(tensor[:32])
    torch.cuda.synchronize()
    return arena


def make_case(arena, rows, distinct, call, generator, null=False):
    # Consecutive calls select disjoint sets, wrapping only after hundreds of MB of reads.
    pool = (torch.arange(distinct) + call * distinct) % (arena.slots - (1 if null else 0))
    slots = pool[torch.arange(rows * TOPK) % distinct].reshape(rows, TOPK).to(torch.int32)
    null_slot = arena.slots - 1 if null else -1
    if null:
        slots[:, -1] = null_slot
    route = slots.clone()
    if null:
        pair = torch.arange(rows * TOPK).reshape(rows, TOPK)
        route = torch.where(slots == null_slot, null_slot + 1 + pair, slots)
    route = route.to(dtype=torch.int32, device="cuda").contiguous()
    bs, bp, br, nb = CUDA.build_routing_small(route, 16, TOPK)
    if null:
        bs = torch.where(bs > null_slot, torch.full_like(bs, null_slot), bs)
    x = (torch.randn(rows, K.DIM, generator=generator) * .5).bfloat16().cuda()
    hf = (torch.randn(rows * TOPK, 2304, generator=generator) * .5).bfloat16().cuda()
    weights = torch.rand(rows, TOPK, generator=generator).cuda()
    return dict(rows=rows, bs=bs, bp=bp, br=br, nb=nb, x=x, hf=hf, weights=weights,
                null_slot=null_slot)


def up(arena, case, out):
    CUDA.up(case["x"], arena.w1, arena.s1, arena.w3, arena.s3, out,
            case["weights"].reshape(-1), case["bs"], case["br"], 10.0, TOPK,
            arena.w1.shape[1], K.DIM, case["nb"], case["null_slot"], v2=True)


def down(arena, case, out, h=None, partial=False):
    CUDA.down(case["hf"] if h is None else h, arena.w2, arena.s2, out, case["bs"], case["bp"],
              TOPK, arena.w2.shape[1], 2304, case["rows"], case["nb"], case["null_slot"],
              partial, 1, v2=True)


def same_bits(a, b):
    return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


def check_case(arena, case, arms):
    results = []
    for _, lib in arms:
        CUDA._LIB = lib
        h = torch.zeros(case["rows"] * TOPK, arena.w1.shape[1], dtype=torch.bfloat16, device="cuda")
        up(arena, case, h)
        # Same synthetic peer half for both arms, actual local native expert-up activations.
        full_h = torch.cat((h, case["hf"][:, 1152:]), dim=1).contiguous()
        values = [h]
        for partial in (False, True):
            parts = torch.empty(case["rows"] * TOPK, arena.w2.shape[1], device="cuda")
            down(arena, case, parts, h=full_h, partial=partial)
            final = parts.view(TOPK, case["rows"], -1).sum(dim=0)
            values.extend((parts, final, final.bfloat16()))
        results.append(values)
    torch.cuda.synchronize()
    same = [same_bits(a, b) for a, b in zip(*results)]
    assert all(same), f"pairbatch intermediate/final parity failed: {same}"
    return len(same)


def measure(arena, cases, arms, repeats):
    graphs = {}
    rows = cases[0]["rows"]
    outputs = {
        "up": torch.empty(rows * TOPK, arena.w1.shape[1], dtype=torch.bfloat16, device="cuda"),
        "down": torch.empty(rows * TOPK, arena.w2.shape[1], device="cuda"),
    }
    for name, lib in arms:
        CUDA._LIB = lib
        for operation, fn in (("up", up), ("down", down)):
            for case in cases:
                fn(arena, case, outputs[operation])
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for case in cases:
                    fn(arena, case, outputs[operation])
            graphs[(name, operation)] = graph
    values = {key: [] for key in graphs}
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for repeat in range(repeats + 2):
        keys = list(graphs)
        if repeat % 2:
            keys.reverse()
        for key in keys:
            # Complete another arena sweep before timing, so the previous arm's last call
            # cannot leave this arm's first small-U down matrix hot.
            graphs[key].replay()
            start.record()
            graphs[key].replay()
            end.record()
            end.synchronize()
            if repeat >= 2:
                values[key].append(start.elapsed_time(end) * 1000 / len(cases))
    return {f"{name}_{op}_us": statistics.median(times)
            for (name, op), times in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--null-only", action="store_true", help="only additional mixed-null parity cells")
    args = parser.parse_args()
    assert CUDA.RELAXED_REDUCE and CUDA.V2_FMA_SCALE == 0
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    baseline = CUDA._library()
    with tempfile.TemporaryDirectory(prefix="fp4-pairbatch-") as build_dir:
        candidate = candidate_library(baseline, build_dir)
        arms = [("baseline", baseline), ("pairbatch2", candidate)]
        CUDA._LIB = baseline
        arena = load_actual_weights()
        report = dict(note=__doc__, source_sha256=CUDA.source_digest(),
                      calls=args.calls, repeats=args.repeats, arena_bytes=arena.bytes_per_slot * arena.slots,
                      device=torch.cuda.get_device_name(), cells=[], exact_checks=0)
        generator = torch.Generator().manual_seed(20261007)
        for rows in (() if args.null_only else (1, 4, 6)):
            for distinct in (6, 8, 14, 20, 26):
                if distinct > rows * TOPK:
                    continue
                CUDA._LIB = baseline
                cases = [make_case(arena, rows, distinct, call, generator)
                         for call in range(args.calls)]
                report["exact_checks"] += check_case(arena, cases[0], arms)
                times = measure(arena, cases, arms, args.repeats)
                before = times["baseline_up_us"] + times["baseline_down_us"]
                after = times["pairbatch2_up_us"] + times["pairbatch2_down_us"]
                cell = dict(rows=rows, distinct=distinct, **times, total_us_before=before,
                            total_us_after=after, speedup=before / after)
                report["cells"].append(cell)
                print(json.dumps(cell), flush=True)
        # Odd pair tails, mixed nulls and partial/full FP32 accumulation boundaries.
        for rows, distinct in ((4, 8), (6, 14)):
            CUDA._LIB = baseline
            case = make_case(arena, rows, distinct, 0, generator, null=True)
            report["exact_checks"] += check_case(arena, case, arms)
        report["all_exact"] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"all_exact": True, "exact_checks": report["exact_checks"],
                          "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
