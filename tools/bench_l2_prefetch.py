"""Can a prefetch during an idle-bus window hide the next read? Single GPU, seconds.

The engine's all-gathers are NCCL spin loops: the GPU is busy, DRAM is idle (170 x ~45 us = 7.6 ms/step).
If a side-stream read of the *next* weights during that window is served from L2 instead of DRAM, those
bytes come out of idle bandwidth. This probe emulates the window with `torch.cuda._sleep` (pure compute
busy-wait, no DRAM) and measures a cold read of a 1..16 MB tensor with and without a concurrent prefetch.

    python tools/bench_l2_prefetch.py [--mb 4] [--spin-us 45]

Reports, per size: cold read, warm read, and the spin+read iteration total with and without a prefetch on a
side stream forked *before* the window (forking it after serializes it behind the window and shows nothing --
that was the first version's bug).

Measured on GB10 (48 SMs, 24 MiB L2), window = torch.cuda._sleep, prefetch = a Triton read with
`eviction_policy="evict_last"`:

  window 12.4 us:  2 MB +2.5 us, 4 MB +5.8, 8 MB +9.5, 16 MB -4.6 (the prefetch outruns the window)
  window 30.8 us:  4 MB +22.6, 8 MB +18.9, 16 MB +18.8

(positive = the spin+read iteration got faster with the prefetch)

So the mechanism is real and the rule is: **prefetch only what fits in the idle window.** A budget larger
than window x DRAM rate does not just fail to help, it delays the read behind the join. At ~230 GB/s and our
RoCE window (~15 us) that is ~3 MB per exchange; at NCCL's current ~45 us it is ~8 MB. For the engine's 170
exchanges that projects to roughly 0.5-1 ms/step after RoCE, more before it.

It cannot show a loss on a DRAM-bound window (that needs the real kernel), so it only decides whether the
idea is worth pursuing.
"""
from __future__ import annotations

import argparse
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _read(X, OUT, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + i, mask=i < n, other=0)
    tl.store(OUT + pid, tl.sum(v, 0))


@triton.jit
def _touch(X, OUT, n, BLOCK: tl.constexpr):
    """The prefetch: same bytes as _read but with evict_last, so the lines survive in L2 (2019/0040's .cg)."""
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + i, mask=i < n, other=0, eviction_policy="evict_last")
    tl.store(OUT + pid, tl.sum(v, 0))


def _read_ms(tensors, sinks, reps):
    blocks = triton.cdiv(tensors[0].numel(), 4096)
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(reps):
        for x, s in zip(tensors, sinks):
            _read[(blocks,)](x, s, x.numel(), BLOCK=4096, num_warps=4)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (reps * len(tensors))


def _sleep_ms(cycles, iters):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        torch.cuda._sleep(cycles)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mb", type=int, default=0, help="sizes to sweep (MB); 0 = 1,2,4,8,16")
    ap.add_argument("--spin-us", type=float, default=45.0)
    ap.add_argument("--copies", type=int, default=16)
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    sizes = [args.mb] if args.mb else [1, 2, 4, 8, 16]

    # calibrate _sleep to the target window
    cycles = 100_000
    ms = _sleep_ms(cycles, 20)
    cycles = max(1, int(cycles * (args.spin_us / 1e3) / ms))
    spin_ms = _sleep_ms(cycles, 20)
    print(f"window: torch.cuda._sleep({cycles}) = {spin_ms * 1e3:.1f} us (DRAM-idle busy wait)")

    spin_us = spin_ms * 1e3
    print(f"{'MB':>4} {'cold_us':>8} {'warm_us':>8} | {'spin+read':>10} {'spin+pref+read':>15} "
          f"{'saved/iter':>11}")
    print(f"(window spin = {spin_us:.1f} us per iteration)")
    for mb in sizes:
        n = mb * 1024 * 1024 // 4
        copies = [torch.randint(0, 1 << 30, (n,), dtype=torch.int32, device="cuda")
                  for _ in range(args.copies)]
        sinks = [torch.zeros(triton.cdiv(n, 4096), dtype=torch.int32, device="cuda") for _ in copies]
        cold = _read_ms(copies, sinks, args.reps)
        warm = _read_ms([copies[0]], [sinks[0]], args.reps * args.copies)

        # graph A: spin ; read           graph B: spin ; (side) prefetch ; read
        def build(prefetch):
            side = torch.cuda.Stream()

            def body():
                for x, k in zip(copies, sinks):
                    if prefetch:                                  # fork BEFORE the window, or the prefetch waits it out
                        side.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(side):
                            _touch[(triton.cdiv(x.numel(), 4096),)](x, k, x.numel(), BLOCK=4096, num_warps=4)
                    torch.cuda._sleep(cycles)                     # the idle window, concurrent with the prefetch
                    if prefetch:
                        torch.cuda.current_stream().wait_stream(side)
                    _read[(triton.cdiv(x.numel(), 4096),)](x, k, x.numel(), BLOCK=4096, num_warps=4)

            warm = torch.cuda.Stream()
            warm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warm):                        # warm the exact body on a side stream
                for _ in range(3):
                    body()
            torch.cuda.current_stream().wait_stream(warm)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body()
            return g

        ga, gb = build(False), build(True)
        for g in (ga, gb):
            for _ in range(3):
                g.replay()
        torch.cuda.synchronize()

        def time_graph(g):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(args.reps):
                g.replay()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / args.reps

        ta, tb = time_graph(ga) * 1000 / args.copies, time_graph(gb) * 1000 / args.copies
        # ta/tb are per-iteration totals: spin + read, with and without the concurrent prefetch
        print(f"{mb:>4} {cold * 1e3:8.2f} {warm * 1e3:8.2f} | {ta:10.2f} {tb:15.2f} "
              f"{ta - tb:12.2f}")
        del copies, sinks
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
