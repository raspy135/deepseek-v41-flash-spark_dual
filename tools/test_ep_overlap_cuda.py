"""Two-node CUDA race regression; run with the serving pair stopped.

The legacy control intentionally consumes an unfinished reduction, but never uses
that value as an index. The corrected path must produce exactly the expected sums.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.distributed as dist
from engine.dist import EPDistributed


def main():
    ep = EPDistributed()
    ep.init('cuda:0')
    comm = torch.cuda.Stream()
    # Warm communicator before the intentional timing asymmetry.
    ep.combine(torch.ones(1, device='cuda'))
    torch.cuda.synchronize()
    results = []
    for legacy in (True, False):
        failures = 0
        for tokens in (1, 16, 128, 512, 2048):
            for trial in range(3):
                x = torch.full((tokens, 5120), float(ep.rank + 1), device='cuda')
                torch.cuda.synchronize()
                dist.barrier()
                # Rank 0 can consume its partial long before rank 1 offers its own.
                if ep.rank == 1:
                    torch.cuda._sleep(100_000_000)
                work = ep.combine_async(x, comm)
                shared = torch.full_like(x, 7.)
                if legacy:
                    torch.cuda.current_stream().wait_stream(comm)
                else:
                    ep.finish_combine(work)
                out = x + shared
                # Final wait cannot repair an already computed wrong `out`.
                work.wait()
                torch.cuda.synchronize()
                failures += int(not torch.equal(out, torch.full_like(out, 10.)))
                del x, shared, out, work
                # Exercise allocator reuse between trials.
                scratch = torch.empty((tokens, 5120), device='cuda')
                scratch.fill_(-99.)
                del scratch
        n = torch.tensor([failures], device='cuda')
        dist.all_reduce(n)
        results.append(dict(legacy=legacy, mismatches_pair_total=int(n.item()), trials_per_rank=15))
    if ep.rank == 0:
        print(json.dumps(results), flush=True)
    assert results[0]['mismatches_pair_total'] > 0, 'legacy race did not reproduce'
    assert results[1]['mismatches_pair_total'] == 0, 'corrected join failed'
    ep.destroy()


if __name__ == '__main__':
    main()
