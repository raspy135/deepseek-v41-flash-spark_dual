#!/usr/bin/env python3
"""Gate G2 -- all-reduce BANDWIDTH across the pair, at the sizes prefill actually sends.

G0 measured 123 KB (one decode verify block) and answered a latency question: ~60 us, fine. Prefill
is a different regime entirely. The EP2 combine all-reduces the routed partial for a whole chunk:
2048 tokens x 5120 hidden x fp32 = 42 MB per layer, 40 layers per chunk, ~5 GB for a 5.5k prompt.
Nothing has ever measured this link above 123 KB, and dual prefill is SLOWER than a single box
(226 vs 337 tok/s), so the first question is whether the fabric is the reason.

Reports the bus bandwidth an all-reduce achieves: a ring all-reduce moves ~2*(n-1)/n * size per
rank, so bus GB/s = size / elapsed for world=2 (not send+receive combined).
"""
from __future__ import annotations
import datetime, os, statistics, time
import torch, torch.distributed as dist

RANK, WORLD = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
SIZES_MB = [0.123, 1, 4, 16, 42, 64]      # 0.123 = the G0 decode payload; 42 = one prefill layer

def main() -> int:
    torch.cuda.set_device(0)
    dist.init_process_group(backend=os.environ.get("G0_BACKEND", "nccl"), rank=RANK,
                            world_size=WORLD, timeout=datetime.timedelta(seconds=180))
    if RANK == 0:
        print(f"{'size':>10} {'median':>10} {'bus GB/s':>10}   (world={WORLD})")
    for mb in SIZES_MB:
        n = max(1, int(mb * 1e6 / 4))
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        reps = 50 if mb < 16 else 20
        samples = []
        for _ in range(reps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            dist.all_reduce(x)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)
        med = statistics.median(samples)
        if RANK == 0:
            gbs = 2 * (WORLD - 1) / WORLD * (n * 4) / med / 1e9
            unit = f"{med*1e6:.0f} us" if med < 1e-3 else f"{med*1e3:.2f} ms"
            print(f"{mb:>8.3f}MB {unit:>10} {gbs:>10.2f}")
        del x
    if RANK == 0:
        print("\nwhat it means for prefill: 40 layers x 42 MB per 2048-token chunk")
    dist.destroy_process_group()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
