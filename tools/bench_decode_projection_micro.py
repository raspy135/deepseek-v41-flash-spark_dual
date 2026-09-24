"""Single-GPU graph-replay timings for the decode projection candidates. Synthetic weights at the
served TP2-local shapes; M=4 (DSV41_BLOCK=3). Kernel-level only -- not a decode-rate claim; the
full-engine A/B is tools/bench_decode_projection_tp.py.

    python3 tools/bench_decode_projection_micro.py [--rows 4] [--reps 5]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                os.path.dirname(os.path.abspath(__file__))]
import fp8_linear as F8  # noqa: E402
import v41_ref as R  # noqa: E402
from engine import model as M  # noqa: E402


LAYERS = 40


def graph_ms(fn, inner=LAYERS, reps=5):
    """Median ms per call over `reps` replays of a graph holding `inner` calls fn(0..inner-1).
    Each index names a different weight copy, as the 40 layers do: one weight replayed 40 times
    stays in L2 and reported >2x GB10's DRAM bandwidth in the first version of this script."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(inner):
            fn(i)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(inner):
            fn(i)
    out = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        graph.replay()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) / inner)
    return statistics.median(out)


def weight(n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return F8.quantize_to_fp8(torch.randn(n, k, generator=gen, device="cuda") * 0.02)


def layers(n, k, seed):
    """LAYERS distinct copies: identical values in separate storage, which is all DRAM sees."""
    base = weight(n, k, seed)
    return [F8.FP8Weight(base.w.clone(), base.s.clone()) for _ in range(LAYERS)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    m = args.rows
    report = {"device": torch.cuda.get_device_name(), "rows": m, "fp8": [], "merged": [], "act_qdq": [], "prune_miss": []}

    def emit(kind, item):
        report[kind].append(item)
        print(kind.upper() + " " + json.dumps(item), flush=True)

    # 1. BLOCK_N / warps per projection, arms interleaved in alternating order.
    shapes = (("wq_a", 1280, 5120), ("wkv", 512, 5120), ("sh_w1_local", 1152, 5120),
              ("sh_w2_local", 5120, 1152), ("wq_b_local", 16384, 1280), ("wo_b_local", 2560, 8192),
              ("wqkv_a_merged", 1792, 5120), ("sh_w13_merged", 2304, 5120))
    arms = [("128", 4), ("64", 4), ("32", 4), ("32", 2), ("16", 4), ("16", 2), ("auto", 4)]
    for i, (name, n, k) in enumerate(shapes):
        w = layers(n, k, i)
        x = (torch.randn(m, k, device="cuda") * 0.5).to(torch.bfloat16)
        times = {a: [] for a in arms}
        for rnd in range(4):
            for arm in (arms if rnd % 2 == 0 else arms[::-1]):
                F8.DECODE_BLOCK_N, F8.DECODE_WARPS = arm
                times[arm].append(graph_ms(lambda j: F8.fp8_linear(x, w[j]), reps=args.reps))
        base = statistics.median(times[("128", 4)])
        for arm, t in times.items():
            ms = statistics.median(t)
            emit("fp8", {"name": name, "N": n, "K": k, "block_n": arm[0], "warps": arm[1],
                         "resolved_block_n": F8.decode_block_n(n, 1, x.device, arm[0]),
                         "ms": round(ms, 4), "GBps": round(n * k / ms / 1e6, 1),
                         "vs_128": round(ms / base, 3)})
        del w, x
    F8.DECODE_BLOCK_N, F8.DECODE_WARPS = "128", 4

    # 2. Merged vs separate, including the activation quantization qlinear does per call.
    for i, (name, (na, nb)) in enumerate((("wq_a+wkv", (1280, 512)), ("sh_w1+w3_local", (1152, 1152)))):
        a, b = layers(na, 5120, 50 + i), layers(nb, 5120, 60 + i)
        both = [F8.concat_rows(p, q)[0] for p, q in zip(a, b)]
        x = (torch.randn(m, 5120, device="cuda") * 0.5).to(torch.bfloat16)
        for bn in ("128", "auto"):
            F8.DECODE_BLOCK_N = bn
            sep, mer = [], []
            for rnd in range(4):
                order = (0, 1) if rnd % 2 == 0 else (1, 0)
                for o in order:
                    if o == 0:
                        sep.append(graph_ms(lambda j: (R.qlinear(x, a[j]), R.qlinear(x, b[j])), reps=args.reps))
                    else:
                        mer.append(graph_ms(lambda j: R.qlinear(x, both[j]), reps=args.reps))
            s, g = statistics.median(sep), statistics.median(mer)
            emit("merged", {"name": name, "block_n": bn, "separate_ms": round(s, 4),
                            "merged_ms": round(g, 4), "saved_ms": round(s - g, 4)})
    F8.DECODE_BLOCK_N = "128"

    # 3. Activation quantization inside the GEMM vs ~13 torch kernels before it.
    for i, (name, n, k) in enumerate((("wq_a", 1280, 5120), ("wkv", 512, 5120), ("wq_b_local", 16384, 1280),
                                      ("wqkv_a_merged", 1792, 5120), ("sh_w13_merged", 2304, 5120))):
        w = layers(n, k, 80 + i)
        x = (torch.randn(m, k, device="cuda") * 0.5).to(torch.bfloat16)
        for bn in ("128", "auto"):
            F8.DECODE_BLOCK_N = bn
            t = {False: [], True: []}
            for rnd in range(4):
                for fused in ((False, True) if rnd % 2 == 0 else (True, False)):
                    R.ACT_QDQ_FUSED = fused
                    t[fused].append(graph_ms(lambda j: R.qlinear(x, w[j]), reps=args.reps))
            a, b = statistics.median(t[False]), statistics.median(t[True])
            emit("act_qdq", {"name": name, "block_n": bn, "torch_qdq_ms": round(a, 4),
                             "fused_qdq_ms": round(b, 4), "saved_ms": round(a - b, 4)})
        del w, x
    R.ACT_QDQ_FUSED = False
    F8.DECODE_BLOCK_N = "128"

    # 4. Prune-miss accounting, one layer's call.
    import types
    fake = types.SimpleNamespace(args=types.SimpleNamespace(n_layers=1), _want_counts=None)
    fake.alloc_prune_miss = lambda n, dev: M.Model.alloc_prune_miss(fake, n, dev)
    fake.alloc_prune_miss(384, "cuda")
    scores = torch.nn.functional.softplus(torch.randn(m, 384, device="cuda")).sqrt()
    logits = scores + torch.randn(384, device="cuda") * 0.1
    keep = torch.rand(384, device="cuda") < 0.61
    from engine import prune_miss_fused as PMF
    arms = ["torch", 1, 2, 4, 8]
    t = {a: [] for a in arms}
    for rnd in range(4):
        for arm in (arms if rnd % 2 == 0 else arms[::-1]):
            M.PRUNE_MISS_FUSED = arm != "torch"
            if arm != "torch":
                PMF.NUM_WARPS = arm
            t[arm].append(graph_ms(lambda j: M.Model._record_prune_miss(fake, logits, scores, keep, 0, 6,
                                                                          decode=True), reps=args.reps))
    base = statistics.median(t["torch"])
    for arm in arms:
        ms = statistics.median(t[arm])
        emit("prune_miss", {"arm": arm if arm == "torch" else f"fused_warps{arm}", "ms_per_layer": round(ms, 4),
                            "saved_ms_per_verify_40_layers": round(40 * (base - ms), 3)})
    print("MICRO_REPORT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
