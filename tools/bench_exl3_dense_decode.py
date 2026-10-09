"""Can the EXL3 grouped kernel replace the FP8 decode GEMVs? One rank's dense shapes at 5 bits.

    python3 tools/bench_exl3_dense_decode.py [--rows 4] [--calls 40] [--splits 1,2,4,8,16]

Single GPU, serving stopped. Each dense matrix is one "expert" (or wo_a's four groups) driven
through exl3m_grouped with T member rows -- the GEMV a decode step needs -- over random 5-bit
trellis words (K2 = 10; timing does not depend on the values). `--calls` distinct copies of each
matrix rotate through one CUDA graph so consecutive calls do not share L2, as in
bench_fp4_moe_decode.py. Reports the best split-K per shape against the bytes the FP8 weight of
the same shape reads, and the FP8 kernel times measured in the engine trace (2026-10-08,
results/exl3-tune-20261008/kern-exl3-merged).
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys

import torch

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import exl3_moe_cuda as XC  # noqa: E402
from bench_fp4_moe_decode import graph_ms  # noqa: E402

V = ctypes.c_void_p
# name: (K, N, groups, FP8 us in the engine trace). TP2 rank shapes (engine/tensor_parallel.py):
# wq_a/wkv replicated, wq_b split on heads, wo_a 4 of 8 groups, wo_b output rows, shared w1/w3
# output columns (two matrices, one launch), shared w2 output rows.
SHAPES = {
    "wq_a": (5120, 1280, 1, None),
    "wkv": (5120, 512, 1, None),
    "wq_b": (1280, 16384, 1, 104.0),
    "wo_a": (4096, 1024, 4, 75.0),
    "wo_b": (8192, 2560, 1, 120.0),
    "sh_w13": (5120, 1152, 2, 65.0),
    "sh_w2": (2304, 2560, 1, 30.0),
}
K2 = 10   # 5 bits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--calls", type=int, default=24)
    ap.add_argument("--splits", default="1,2,4,8,16")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    args = ap.parse_args()
    T = args.rows
    lib = XC._lib()
    print(f"rows {T}, 5-bit trellis, {args.calls} L2-cold copies per shape")
    total_fp8, total_x3 = 0.0, 0.0
    for name in args.shapes.split(","):
        K, N, G, fp8_us = SHAPES[name]
        words = (K // 16) * (N // 16) * 2 * (4 * K2)         # int16 words: a tile is 4*K2 uint32
        mats = 2 if name == "sh_w13" else 1
        E = G if name != "sh_w13" else 1
        n_mat = args.calls * E * mats
        store = torch.randint(-32768, 32767, (n_mat, words), dtype=torch.int16, device="cuda")
        ptrs = [store[i].data_ptr() for i in range(n_mat)]
        k2 = torch.full((n_mat,), K2, dtype=torch.int32, device="cuda")
        # rows: wo_a's groups each read their own T rows; X is [T * E, K] with row t * E + g
        P = T * E
        x0 = (torch.randn(P, K) * 0.1).half().cuda()
        x1 = (torch.randn(P, K) * 0.1).half().cuda()
        uids = torch.arange(E, dtype=torch.int32, device="cuda")
        ucount = torch.tensor([E], dtype=torch.int32, device="cuda")
        maxm = 16
        members = torch.full((E, maxm), -1, dtype=torch.int32, device="cuda")
        for g in range(E):
            members[g, :T] = torch.arange(T, dtype=torch.int32, device="cuda") * 32 + g
        # per call: its own copy's pointers (slot table indexed by uid = group)
        tp0 = [torch.tensor([ptrs[(c * E + g) * mats] for g in range(E)], dtype=torch.int64, device="cuda")
               for c in range(args.calls)]
        tp1 = [torch.tensor([ptrs[(c * E + g) * mats + mats - 1] for g in range(E)], dtype=torch.int64,
                            device="cuda") for c in range(args.calls)]
        fp8_bytes = K * N * G * mats
        x3_bytes = words * 2 * E * mats
        best = None
        for sk in (int(s) for s in args.splits.split(",")):
            if (K // 16) % (sk * 4) or N % 128:
                continue
            z = torch.empty((mats, sk, P, N), dtype=torch.float32, device="cuda")

            def call(i, sk=sk, z=z):
                lib.exl3m_grouped(V(x0.data_ptr()), V(x1.data_ptr()), V(tp0[i].data_ptr()), V(tp1[i].data_ptr()),
                                  V(k2.data_ptr()), V(k2.data_ptr()), V(uids.data_ptr()), V(ucount.data_ptr()),
                                  V(members.data_ptr()), V(z.data_ptr()), K, N, P, sk, maxm, E, E, mats,
                                  8, 4, 1, 2, 10, XC.CB_MUL1, XC._stream())

            us = graph_ms(call, args.calls) * 1e3
            if best is None or us < best[1]:
                best = (sk, us)
            print(f"  {name:7s} K={K:5d} N={N:5d} G={G} SK={sk:2d}: {us:7.1f} us  {x3_bytes / us / 1e3:5.0f} GB/s")
        sk, us = best
        fp8_txt = f"FP8 in engine {fp8_us:.0f} us -> {us / fp8_us:.2f}x" if fp8_us else ""
        print(f"* {name:7s} best SK={sk}: {us:.1f} us, {x3_bytes / 1e6:.1f} MB ({x3_bytes / fp8_bytes:.3f} of FP8) {fp8_txt}")
        if fp8_us:
            total_fp8 += fp8_us
            total_x3 += us
    if total_fp8:
        print(f"\nshapes with an engine FP8 time: FP8 {total_fp8:.0f} us vs EXL3 {total_x3:.0f} us per layer "
              f"-> {(total_fp8 - total_x3) * 40 / 1e3:.1f} ms per 40-layer step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
