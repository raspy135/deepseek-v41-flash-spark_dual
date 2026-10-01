"""NCCL vs the one-shot RoCE all-gather (tools/roce/, ported from b12x "RoCEnante" via TensorFold's 0230).

    # fast, one node, both ranks in this process over the NIC loopback (no second Spark needed):
    python tools/bench_roce_gather.py loopback --sizes 16k,48k,96k,256k
    # two nodes, NCCL vs RoCE at our shard sizes (both ranks, same args but --rank):
    python tools/bench_roce_gather.py bench --rank 0 --master 10.0.0.1
    python tools/bench_roce_gather.py bench --rank 1 --master 10.0.0.1

Sizes are bytes ONE rank sends. Our decode shards are ~48 KiB (a [6,4096] bf16 output-parallel shard)
to ~97 KiB ([42,1152] merged MoE intermediates), so the defaults cover them.

The loopback warm-ups interleave the two ranks' collectives (TensorFold 0350): a rank's collectives only
complete once its peer's of the same sequence run, so warming one rank alone times out.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "roce")]
import roce  # noqa: E402  tools/roce/roce.py


def _size(tok: str) -> int:
    tok = tok.strip().lower()
    mult = {"k": 1024, "m": 1024 * 1024}.get(tok[-1:], 1)
    return int(float(tok[:-1] if tok[-1:] in "km" else tok) * mult)


def _pattern(rank, n, salt):
    return roce.pattern(rank, n, salt)


def _expected(world, n, salt):
    return torch.cat([_pattern(r, n, salt) for r in range(world)])


def _time_eager(fn, iters):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


def _graph(fn, ops, tail=None):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for _ in range(ops):
                fn()
                if tail is not None:
                    tail()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(ops):
            fn()
            if tail is not None:
                tail()
    return g


def _graph_pair(fns, ops, tails=None):
    """One process driving several runtimes (loopback): interleave the eager warm-ups op by op, then capture each
    graph (a capture executes nothing)."""
    tails = tails or [None] * len(fns)
    streams = [torch.cuda.Stream() for _ in fns]
    for st in streams:
        st.wait_stream(torch.cuda.current_stream())
    for _ in range(2):
        for _ in range(ops):
            for fn, tail, st in zip(fns, tails, streams):
                with torch.cuda.stream(st):
                    fn()
                    if tail is not None:
                        tail()
    for st in streams:
        torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    graphs = []
    for fn, tail in zip(fns, tails):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(ops):
                fn()
                if tail is not None:
                    tail()
        graphs.append(g)
    return graphs


def _time_graph(g, reps, ops):
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (reps * ops)


class NCCL:
    """Minimal two-node communicator for the microbench (the engine uses EPDistributed)."""

    def __init__(self, rank, world, master, port):
        import torch.distributed as dist
        os.environ["MASTER_ADDR"] = master        # override: the shell may carry serving's MASTER_*
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("nccl", rank=rank, world_size=world)
        self.rank, self.world, self.device = rank, world, "cuda"

    def all_gather(self, send, recv):
        import torch.distributed as dist
        dist.all_gather_into_tensor(recv, send)

    def barrier(self):
        import torch.distributed as dist
        dist.barrier()


def loopback(args):
    torch.cuda.set_device(0)
    s = dataclasses.replace(roce.settings(), max_bytes=max(args.sizes), timeout_s=args.timeout or 10.0)
    hcas = roce.detect(s.hca_spec)[: s.hcas]
    if not hcas:
        raise SystemExit("no ACTIVE RoCE port with an IPv4-mapped RoCE v2 GID")
    print("HCAs:", [(h.name, h.gid_index, h.ipv4) for h in hcas], flush=True)
    rts = [roce.Runtime(rank=r, world=2, hcas=hcas, s=s) for r in (0, 1)]
    blobs = [rt.blob() for rt in rts]
    for rt in rts:
        rt.connect(blobs)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    print(f"{'bytes/rank':>11} {'bits':>5} {'eager_pair_us':>13} {'graph_us/op':>11}")
    ok_all = True
    for salt, n in enumerate(args.sizes):
        xs = [_pattern(r, n, salt) for r in (0, 1)]
        ys = [torch.empty(2 * n, dtype=torch.uint8, device="cuda") for _ in (0, 1)]
        torch.cuda.synchronize()

        def both():
            for r in (0, 1):
                with torch.cuda.stream(streams[r]):
                    rts[r].gather(xs[r], ys[r])

        both()
        torch.cuda.synchronize()
        for rt in rts:
            rt.check()
        exp = _expected(2, n, salt)
        ok = all(torch.equal(y, exp) for y in ys)
        ok_all &= ok
        us = _time_eager(both, args.iters)
        gs = _graph_pair([lambda r=r: rts[r].gather(xs[r], ys[r]) for r in (0, 1)], args.ops)
        for _ in range(3):
            for r in (0, 1):
                with torch.cuda.stream(streams[r]):
                    gs[r].replay()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(args.reps):
            for r in (0, 1):
                with torch.cuda.stream(streams[r]):
                    gs[r].replay()
        torch.cuda.synchronize()
        g_us = (time.perf_counter() - t) * 1e6 / (args.reps * args.ops)
        for rt in rts:
            rt.check()
        print(f"{n:11d} {str(ok):>5} {us:13.2f} {g_us:11.2f}", flush=True)
    print("rank0 stats:", json.dumps(rts[0].snapshot()), flush=True)
    torch.cuda.synchronize()
    for rt in rts:
        rt.close()
    if not ok_all:
        sys.exit(1)


def bench(args):
    torch.cuda.set_device(0)
    base = NCCL(args.rank, 2, args.master, args.port)
    base.barrier()
    s = dataclasses.replace(roce.settings(), max_bytes=max(args.sizes), timeout_s=args.timeout or 120.0)
    comm = roce.connect(base, s)
    rt = comm.rt
    if args.rank == 0:
        print(json.dumps({"hcas": [dataclasses.asdict(h) for h in rt.hcas], "slot_bytes": rt.slot_bytes}),
              flush=True)
    rows = []
    for salt, n in enumerate(args.sizes):
        x = _pattern(args.rank, n, salt)
        ref = torch.empty(2 * n, dtype=torch.uint8, device="cuda")
        mine = torch.empty(2 * n, dtype=torch.uint8, device="cuda")
        base.all_gather(x, ref)
        rt.gather(x, mine)
        torch.cuda.synchronize()
        rt.check()
        same = bool(torch.equal(mine, ref))
        res = {"bytes": n, "bits_equal": same}
        for name, fn in (("nccl", lambda: base.all_gather(x, ref)), ("roce", lambda: rt.gather(x, mine))):
            base.barrier()
            res[f"{name}_eager_us"] = round(_time_eager(fn, args.iters), 2)
            base.barrier()
            res[f"{name}_graph_us"] = round(_time_graph(_graph(fn, args.ops), args.reps, args.ops), 2)
        rows.append(res)
        if args.rank == 0:
            print(json.dumps(res), flush=True)
    if args.rank == 0:
        print("\n| bytes/rank | bits | NCCL eager | RoCE eager | NCCL graph | RoCE graph |")
        print("| ---: | :---: | ---: | ---: | ---: | ---: |")
        for r in rows:
            print(f"| {r['bytes']} | {'ok' if r['bits_equal'] else 'DIFF'} | {r['nccl_eager_us']} | "
                  f"{r['roce_eager_us']} | {r['nccl_graph_us']} | {r['roce_graph_us']} |")
        print("saved on a 170-exchange step (graph): " +
              ", ".join(f"{r['bytes']} B: {(r['nccl_graph_us'] - r['roce_graph_us']) * 170 / 1e3:.2f} ms"
                        for r in rows))
    if not all(r["bits_equal"] for r in rows):
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="loopback", choices=["loopback", "bench"])
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--master", default=os.environ.get("HEAD_IP", "10.0.0.1"))
    ap.add_argument("--port", type=int, default=29561)
    ap.add_argument("--sizes", type=lambda v: [_size(t) for t in v.split(",")], default="16k,48k,96k,256k")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--ops", type=int, default=90)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=None)
    args = ap.parse_args()
    if isinstance(args.sizes, str):
        args.sizes = [_size(t) for t in args.sizes.split(",")]
    {"loopback": loopback, "bench": bench}[args.mode](args)


if __name__ == "__main__":
    main()
