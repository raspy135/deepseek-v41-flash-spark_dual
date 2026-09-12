#!/usr/bin/env python3
"""Gate G0 -- 2-rank all-reduce latency on the dual Spark link.

Kill criterion (docs/dual-spark-plan.md): if a 123 KB fp32 all-reduce exceeds
~150 us median, EP2 adds >~240 ms/token and we stop to reconsider.

Env:
  RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT  (required)
  G0_BACKEND=nccl|gloo|cpu:gloo,cuda:nccl    (default nccl)
"""
from __future__ import annotations

import os
import statistics
import sys
import time

print(f"[G0] rank={os.environ.get('RANK')} world={os.environ.get('WORLD_SIZE')} "
      f"master={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')} "
      f"backend={os.environ.get('G0_BACKEND', 'nccl')}", flush=True)

import torch
import torch.distributed as dist

print(f"[G0] torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 2
    backend = os.environ.get("G0_BACKEND", "nccl")

    print(f"[G0] init_process_group({backend}) ...", flush=True)
    t_init = time.perf_counter()
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world,
        timeout=__import__("datetime").timedelta(seconds=120),
    )
    print(f"[G0] process group up in {time.perf_counter() - t_init:.2f}s", flush=True)

    use_cuda = backend != "gloo" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(0)
        device = "cuda"
    else:
        device = "cpu"

    # Payload size of one MoE combine: 6 tokens x 5120 hidden x fp32 ~= 123 KB
    n = 6 * 5120
    x = torch.randn(n, device=device, dtype=torch.float32)

    print(f"[G0] warmup 20 iters on {device} ...", flush=True)
    for _ in range(20):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    if use_cuda:
        torch.cuda.synchronize()

    samples_us: list[float] = []
    print("[G0] measuring 200 iters ...", flush=True)
    for _ in range(200):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        if use_cuda:
            torch.cuda.synchronize()
        samples_us.append((time.perf_counter() - t0) * 1e6)

    samples_us.sort()
    med = statistics.median(samples_us)
    p95 = samples_us[int(0.95 * (len(samples_us) - 1))]
    mean = statistics.mean(samples_us)
    per_step_ms = 40 * med / 1000.0
    per_token_ms = per_step_ms / 3.0

    if rank == 0:
        print(f"G0 all_reduce 123KB fp32  world={world} backend={backend} device={device}")
        print(f"  n=200  median={med:.1f} us  mean={mean:.1f} us  p95={p95:.1f} us")
        print(f"  projected: {per_step_ms:.1f} ms/step (40 layers)  ~{per_token_ms:.1f} ms/token @ acc=3")
        kill = 150.0
        # gloo/cpu is expected slower; only enforce kill for nccl path
        if backend == "nccl" and med > kill:
            print(f"  VERDICT: FAIL  median {med:.1f} us > {kill} us kill criterion")
            dist.destroy_process_group()
            raise SystemExit(2)
        if backend != "nccl":
            print(f"  VERDICT: INFO  backend={backend} (kill criterion applies to nccl only)")
        else:
            print(f"  VERDICT: PASS  median {med:.1f} us <= {kill} us")
    dist.destroy_process_group()
    print("[G0] done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[G0] EXCEPTION: {type(e).__name__}: {e}", flush=True)
        raise
