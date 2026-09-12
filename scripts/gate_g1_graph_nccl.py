#!/usr/bin/env python3
"""Gate G1 -- can a 2-rank NCCL all-reduce be captured into a CUDA graph and replayed?

This is the hard dependency for Mode A on the pair. With every routed expert resident the
engine's fast path captures a WHOLE layer as one CUDA graph (engine/fastdecode.py::_layer_ab,
taken when `self.lut is not None`), which removes the per-layer host round-trip that Gate G0's
follow-up measurements showed eating ~58 % of decode. Under EP2 that graph now contains the
routed-expert all-reduce, so if collectives cannot be captured, Mode A cannot be graphed on two
boxes and the whole plan reduces to "run Mode A on one box".

Three things are checked, in order of how badly they bite:

  1. capture succeeds at all. NCCL has supported stream capture for years, but it needs the
     communicator to already exist -- a cold comm tries to bootstrap (host sync, allocations)
     inside the capture and fails. Hence the eager warm-up before capturing.
  2. TORCH_NCCL_BLOCKING_WAIT. scripts/dual-up.sh currently exports it =1, and blocking wait
     works by calling cudaStreamSynchronize on the collective's stream -- which is ILLEGAL
     during capture. This gate runs whatever the environment says so the failure is attributed
     to the flag rather than to the hardware. G1_TEST_BOTH=1 runs it both ways in one go.
  3. replay actually reduces. A graph that captured nothing still replays happily and returns
     stale buffer contents, which would look like a pass and produce subtly wrong logits --
     the same failure mode as the null-slot bug. So each replay writes NEW values into the
     static input and checks the output against the arithmetic answer.

Env: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT (as G0).
"""
from __future__ import annotations

import datetime
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
N = 6 * 5120          # the EP2 combine payload: 6-token verify block x 5120 hidden, fp32


def log(msg):
    print(f"[G1 r{RANK}] {msg}", flush=True)


def check(t: torch.Tensor, want: float, label: str) -> bool:
    """Every element must equal `want`; all_reduce(SUM) of a constant is exactly representable."""
    got = t.min().item(), t.max().item()
    ok = got == (want, want)
    if not ok:
        log(f"FAIL {label}: expected all {want}, got min/max {got}")
    return ok


def run(blocking_wait: bool) -> bool:
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1" if blocking_wait else "0"
    log(f"--- TORCH_NCCL_BLOCKING_WAIT={os.environ['TORCH_NCCL_BLOCKING_WAIT']}")
    dev = torch.device("cuda")
    x = torch.zeros(N, device=dev, dtype=torch.float32)

    # 1. eager warm-up. Creates the communicator and its buffers OUTSIDE the capture; a comm
    #    that bootstraps during capture takes a host lock and the capture dies.
    for _ in range(5):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # 2. capture on a side stream, as CUDA graph capture requires.
    g = torch.cuda.CUDAGraph()
    x.fill_(float(RANK + 1))
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(g):
                dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
    except Exception as e:
        log(f"CAPTURE FAILED: {type(e).__name__}: {e}")
        return False
    log("capture ok")

    # 3. replay, twice, with different inputs -- proves the reduce runs on every replay instead
    #    of the graph having captured nothing and returning whatever is in the buffer.
    want_sum = float(sum(r + 1 for r in range(WORLD)))     # ranks contribute 1..WORLD
    ok = True
    for trial in (1, 2):
        x.fill_(float(RANK + 1) * trial)
        g.replay()
        torch.cuda.synchronize()
        ok &= check(x, want_sum * trial, f"replay {trial}")
    if not ok:
        return False
    log("replay ok, values correct")

    # 4. latency of the replayed collective vs the eager one.
    def bench(fn, n=200):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1e6)
        return statistics.median(sorted(out))

    eager = bench(lambda: dist.all_reduce(x, op=dist.ReduceOp.SUM))
    graphed = bench(g.replay)
    if RANK == 0:
        log(f"median eager {eager:.1f} us | graph replay {graphed:.1f} us "
            f"({eager - graphed:+.1f} us)")
    return True


def main() -> int:
    log(f"torch {torch.__version__} nccl {torch.cuda.nccl.version()}")
    torch.cuda.set_device(0)
    # Pure NCCL by default, NOT the engine's mixed "cpu:gloo,cuda:nccl". This gate asks one
    # question -- can a CUDA collective be captured -- and the CUDA half of the mixed group is
    # the same ProcessGroupNCCL either way, so gloo adds only a second rendezvous that can fail
    # on its own (it did, on the first run of this gate: rank 0 timed out constructing
    # ProcessGroupGloo while rank 1 had already moved on to the NCCL comm).
    backend = os.environ.get("G0_BACKEND", "nccl")
    dist.init_process_group(backend=backend, rank=RANK, world_size=WORLD,
                            timeout=datetime.timedelta(seconds=120))
    log("process group up")
    results = {}
    modes = [False, True] if os.environ.get("G1_TEST_BOTH", "1") == "1" \
        else [os.environ.get("TORCH_NCCL_BLOCKING_WAIT", "0") == "1"]
    for bw in modes:
        results[bw] = run(bw)
    dist.destroy_process_group()
    if RANK == 0:
        print("\n=== G1 VERDICT ===")
        for bw, ok in results.items():
            print(f"  TORCH_NCCL_BLOCKING_WAIT={int(bw)}: {'PASS' if ok else 'FAIL'}")
        if results.get(False) or results.get(True):
            print("  => graph-captured collectives WORK; EP2 can use the graphed fast path.")
            if results.get(False) and not results.get(True, True):
                print("     NOTE: only with BLOCKING_WAIT=0 -- scripts/dual-up.sh must stop exporting it =1.")
        else:
            print("  => collectives CANNOT be captured here; Mode A on the pair needs another design.")
    return 0 if any(results.values()) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        print(f"[G1] EXCEPTION: {type(e).__name__}: {e}", flush=True)
        raise
