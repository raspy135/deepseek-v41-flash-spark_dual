"""Bounded synchronous RoCE remote-hit prototype, not a serving cache protocol.

Requests are CUDA int64 IDs sent over NCCL/RoCE; peer reads its host row cache,
then replies with raw bytes. Includes host/device handoffs and synchronization.
The peer is already waiting. This omits production arbitration, cache directory,
remote misses and contention, so a win establishes feasibility, not decode gain.
"""
import hashlib
import json
import resource
import statistics
import time

import numpy as np
import torch
import torch.distributed as dist


def measure_remote(e):
    table = next(iter(e.tables.values()))
    assert table.row_cache is not None and e.ep.world == 2
    records = []
    rng = np.random.default_rng(93821)
    for n in (24, 144, 384):
        request = torch.empty(n, device='cuda', dtype=torch.int64)
        reply = torch.empty((n, 264), device='cuda', dtype=torch.uint8)
        for iteration in range(12):
            ids = rng.integers(0, table.n_rows, n, dtype=np.int64)
            # Only the peer reads these new random rows before the measurement.
            if e.ep.rank == 1:
                table._gather_rows(ids)
            e.ep.gather_objects(True)
            expected = None
            local_ms = local_warm_ms = major_faults = None
            if e.ep.rank == 0:
                faults = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
                start = time.perf_counter()
                expected = table._gather_rows_uncached(ids)
                local_ms = (time.perf_counter() - start) * 1000
                major_faults = resource.getrusage(resource.RUSAGE_SELF).ru_majflt - faults
                start = time.perf_counter()
                table._gather_rows_uncached(ids)
                local_warm_ms = (time.perf_counter() - start) * 1000
            e.ep.gather_objects(True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            if e.ep.rank == 0:
                request.copy_(torch.from_numpy(ids))
            dist.broadcast(request, src=0)
            if e.ep.rank == 1:
                incoming = request.cpu().numpy()
                raw = table._gather_rows(incoming)
                reply.copy_(torch.from_numpy(raw))
            dist.broadcast(reply, src=1)
            torch.cuda.synchronize()
            remote_ms = (time.perf_counter() - start) * 1000
            # Validation is outside timing; consumption normally stays on GPU.
            signature = hashlib.sha256(reply.cpu().numpy().tobytes()).hexdigest()
            expected_signature = hashlib.sha256(expected.tobytes()).hexdigest() if expected is not None else signature
            assert all(e.ep.gather_objects(signature == expected_signature))
            if iteration >= 2:
                records.append(dict(rows=n, iteration=iteration, local_first_ms=local_ms,
                                    local_warm_ms=local_warm_ms, major_faults=major_faults,
                                    remote_ms=remote_ms))
    summary = {}
    for n in (24, 144, 384):
        rows = [r for r in records if r['rows'] == n]
        summary[n] = {key: statistics.median(r[key] for r in rows)
                      for key in ('local_first_ms', 'local_warm_ms', 'major_faults', 'remote_ms')
                      if rows[0][key] is not None}
    report = dict(note=__doc__, samples=records, median=summary)
    print('ENGRAM_REMOTE ' + json.dumps(summary), flush=True)
    return report
