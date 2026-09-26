"""Two-node TP collective gate: values, graph replay, physical rails, timing.

Run with the serving network environment and DSV41_PREFILL_DUAL_RAIL=1.
Does not load weights. rx_write_requests are per-HCA hardware counters; a test
host must have no other RDMA workload. NCCL INFO channel logs are a second check.
"""
import datetime
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from engine.dist import EPDistributed
from engine import collective_rails as rails
from engine.tensor_parallel import FeatureParallelEmbedding


def counters():
    return [int(Path(f'/sys/class/infiniband/{h}/ports/1/hw_counters/rx_write_requests').read_text())
            for h in ('rocep1s0f1', 'roceP2p1s0f1')]


def main():
    ep = EPDistributed()
    ep.init('cuda')
    rank, world = ep.rank, ep.world
    assert world == 2 and ep.prefill_dual_rail
    local = torch.arange(128, device='cuda', dtype=torch.float32).view(16, 8) + rank*128
    embed = FeatureParallelEmbedding(local, world)
    for prefill in (False, True, False):
        with rails.phase(prefill):
            # Exercise an actual TP wrapper, including a one-token prefill.
            got = embed[torch.tensor([2], device='cuda')]
            want = torch.cat((local[2]-rank*128, local[2]+(1-rank)*128))
            torch.testing.assert_close(got[0], want, rtol=0, atol=0)
            for kind in ('all_reduce', 'all_gather', 'reduce_scatter'):
                x = torch.full((1024,), rank+1., device='cuda')
                if kind == 'all_reduce':
                    dist.all_reduce(x, group=rails.group())
                    assert (x == 3).all().item()
                elif kind == 'all_gather':
                    out = torch.empty(2048, device='cuda')
                    dist.all_gather_into_tensor(out, x, group=rails.group())
                    assert (out[:1024] == 1).all().item() and (out[1024:] == 2).all().item()
                else:
                    out = torch.empty(512, device='cuda')
                    dist.reduce_scatter_tensor(out, x, group=rails.group())
                    assert (out == 3).all().item()

    # The captured graph must keep using decode's group after a prefill, with fresh values.
    x = torch.ones(30720, device='cuda')
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5): dist.all_reduce(x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        dist.all_reduce(x, group=rails.group())
    for trial in range(1, 4):
        with rails.phase(True):
            y = torch.ones(1024, device='cuda')
            dist.all_reduce(y, group=rails.group())
            assert (y == 2).all().item()
        x.fill_((rank+1)*trial)
        g.replay()
        torch.cuda.synchronize()
        assert (x == 3*trial).all().item()
    del g

    rows = []
    for prefill in (False, True, False):
        for n in (30720, 10500000):
            with rails.phase(prefill):
                x = torch.zeros(n, device='cuda')
                for _ in range(10): dist.all_reduce(x, group=rails.group())
                torch.cuda.synchronize()
                time.sleep(1)  # let cached hardware counters settle before/after the batch
                before = counters()
                dist.barrier()  # CPU/Gloo, outside the timed CUDA collectives
                ts = []
                for _ in range(100):
                    t = time.perf_counter()
                    dist.all_reduce(x, group=rails.group())
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter()-t)
                time.sleep(1)
                after = counters()
                # Neither rank may start the next phase until both read their counters.
                dist.barrier()
                delta = [b-a for a,b in zip(before,after)]
                # Every rank must use the same phase; counters prove physical rail usage.
                ok = all(d > 0 for d in delta) if prefill else delta[0] > 0 and delta[1] == 0
                checks = ep.gather_objects((ok, delta))
                assert all(c[0] for c in checks), checks
                med = statistics.median(ts)
                item = dict(rank=rank, prefill=prefill, bytes=n*4, median_us=med*1e6,
                            bus_gbs=n*4/med/1e9, rx_writes=delta)
                rows.append(item)
                print('RAILS '+json.dumps(item), flush=True)
    print('RAILS_PASS '+json.dumps(dict(rank=rank, correctness=True, graph=True)), flush=True)
    ep.destroy()


if __name__ == '__main__':
    main()
