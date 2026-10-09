"""Achieved bandwidth of the EXL3 decode expert kernels at TP2 shapes, against a plain read.

    python3 tools/bench_exl3_moe_decode.py [--slots 256] [--calls 40] [--rows 4] [--u 12,17,21,26]
        [--variants 8x4x1x1,8x4x2x1]

Single GPU, serving stopped. The method is tools/bench_fp4_moe_decode.py's: a TP-rank-0
`Exl3Arena` (output layout: w1/w3 [5120 -> 1152] and w2 [2304 -> 2560] per slot, 3-bit trellis)
filled with random words; each call routes `--rows` tokens x 6 over U distinct slots drawn fresh
from the arena, so consecutive calls do not share L2; `--calls` calls are captured into one CUDA
graph and replayed. Timed per stage: the grouping kernel, the gate/up grouped kernel and the down
grouped kernel (the rotations and epilogues are a few us and read no weights).

A variant is NTxWARPSxPFxSK (n tiles per warp, warps per program, k steps prefetched, K splits);
it must be one `exl3m_grouped` was compiled for. Bytes per call are what the kernel must read at
least once: U x (t1 + t3) for gate/up and U x t2 for down. "read" is a Triton kernel reading
exactly those bytes with 16-byte loads -- the ceiling for this access pattern.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys

import torch

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import exl3_moe as X3  # noqa: E402
import exl3_moe_cuda as XC  # noqa: E402
from bench_fp4_moe_decode import graph_ms, touch  # noqa: E402

TOPK = 6
V = ctypes.c_void_p


def routing(T, U, gen, n_slots):
    pool = torch.randperm(n_slots, generator=gen)[:U]
    slots = torch.stack([pool[torch.arange(t * TOPK, (t + 1) * TOPK) % U] for t in range(T)])
    return slots.to(torch.int32).cuda().contiguous()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slots", type=int, default=256)
    ap.add_argument("--calls", type=int, default=40)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--u", default="12,17,21,26")
    ap.add_argument("--variants", default="8x4x1x1,8x4x2x1,4x4x2x1")
    args = ap.parse_args()
    T, P = args.rows, args.rows * TOPK
    arena = X3.Exl3Arena(args.slots, "cuda", tp_rank=0, tp_world=2, bits=3.0)
    for t in (arena.t1, arena.t3, arena.t2):
        t.random_(-32768, 32767)
    for n in ("suh1", "suh3", "suh2", "svh1", "svh3", "svh2"):
        getattr(arena, n).fill_(1.0)
    arena.bits_gpu.fill_(6)
    p = XC._ptrs(arena)
    lib = XC._lib()
    dim, inter, down_n = 5120, arena.inter, arena.down_n
    gen = torch.Generator().manual_seed(0)
    xh0 = (torch.randn(P, dim, generator=gen) * 0.1).half().cuda()
    xh1 = (torch.randn(P, dim, generator=gen) * 0.1).half().cuda()
    xd = (torch.randn(P, 2 * inter, generator=gen) * 0.1).half().cuda()
    up_bytes = sum(t[0].numel() * t.element_size() for t in (arena.t1, arena.t3))
    down_bytes = arena.t2[0].numel() * arena.t2.element_size()
    print(f"rows {T}, pairs {P}; per slot: gate/up {up_bytes / 1e6:.2f} MB, down {down_bytes / 1e6:.2f} MB; "
          f"L2-cold calls: {args.calls}")
    variants = [tuple(int(v) for v in s.split("x")) for s in args.variants.split(",")]
    for U in (int(u) for u in args.u.split(",")):
        routes = []
        for _ in range(args.calls):
            s = routing(T, U, gen, args.slots)
            uids = torch.empty(P, dtype=torch.int32, device="cuda")
            ucount = torch.empty(1, dtype=torch.int32, device="cuda")
            members = torch.empty(P, 16, dtype=torch.int32, device="cuda")
            routes.append((s, uids, ucount, members))

        def group(i):
            s, uids, ucount, members = routes[i]
            lib.exl3m_group(V(s.data_ptr()), V(uids.data_ptr()), V(ucount.data_ptr()), V(members.data_ptr()),
                            T, TOPK, args.slots, 16, XC._stream())

        for i in range(args.calls):
            group(i)
        uniqs = [torch.unique(r[0]).to(torch.int32) for r in routes]
        out = torch.empty(U * 65536, dtype=torch.int64, device="cuda")

        def read_up(i):
            for t in (arena.t1, arena.t3):
                touch(t, uniqs[i], out)

        def read_down(i):
            touch(arena.t2, uniqs[i], out)

        r_up, r_down = graph_ms(read_up, args.calls), graph_ms(read_down, args.calls)
        t_group = graph_ms(group, args.calls)
        bu, bd = U * up_bytes, U * down_bytes
        print(f"\nU={U:>2}  group {t_group * 1e3:6.1f} us | read ceiling: gate/up {bu / r_up / 1e6:5.0f} GB/s, "
              f"down {bd / r_down / 1e6:5.0f} GB/s")
        for nt, w, pf, sk in variants:
            z = torch.empty((2, sk, P, inter), dtype=torch.float32, device="cuda")
            zd = torch.empty((1, sk, P, down_n), dtype=torch.float32, device="cuda")

            def up(i):
                _, uids, ucount, members = routes[i]
                lib.exl3m_grouped(V(xh0.data_ptr()), V(xh1.data_ptr()), V(p["t1p"].data_ptr()),
                                  V(p["t3p"].data_ptr()), V(p["k2"].data_ptr()), V(p["k2"].data_ptr()),
                                  V(uids.data_ptr()), V(ucount.data_ptr()), V(members.data_ptr()), V(z.data_ptr()),
                                  dim, inter, P, sk, 16, TOPK, P, 2, nt, w, pf, 2, 10, XC.CB_MUL1, XC._stream())

            def down(i):
                _, uids, ucount, members = routes[i]
                lib.exl3m_grouped(V(xd.data_ptr()), V(xd.data_ptr()), V(p["t2p"].data_ptr()), V(p["t2p"].data_ptr()),
                                  V(p["k2"].data_ptr()), V(p["k2"].data_ptr()), V(uids.data_ptr()),
                                  V(ucount.data_ptr()), V(members.data_ptr()), V(zd.data_ptr()),
                                  2 * inter, down_n, P, sk, 16, TOPK, P, 1, nt, w, pf, 2, 10, XC.CB_MUL1,
                                  XC._stream())

            try:
                t_up, t_down = graph_ms(up, args.calls), graph_ms(down, args.calls)
            except Exception as ex:  # noqa: BLE001 - an uncompiled variant
                print(f"  {nt}x{w}x{pf}x{sk}: {type(ex).__name__} {ex}")
                continue
            print(f"  {nt}x{w}x{pf}x{sk}: gate/up {t_up * 1e3:6.1f} us {bu / t_up / 1e6:5.0f} GB/s | "
                  f"down {t_down * 1e3:6.1f} us {bd / t_down / 1e6:5.0f} GB/s | "
                  f"layer {(t_up + t_down) * 1e3:6.1f} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
