"""CPU-only gather costs. Resident arrays isolate dispatch from storage latency."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import statistics
import sys
import time
sys.path[:0] = ['/app', '/app/tools']
import numpy as np
from engine.engram import EngramTable


def measure(fn, count=40):
    for _ in range(5): fn()
    samples = []
    for _ in range(count):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns()-start)/1e6)
    return dict(median_ms=statistics.median(samples), p95_ms=float(np.percentile(samples, 95)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    rng = np.random.default_rng(731)
    table = EngramTable.__new__(EngramTable)
    table.w_mm = rng.integers(0, 256, (8192, 256), dtype=np.uint8)
    table.s_mm = rng.integers(0, 256, (8192, 8), dtype=np.uint8)
    table.gather_min_parallel = 32
    table.native_gather = None
    table.pool = ThreadPoolExecutor(64)
    try:
        for n in (96, 144, 384):
            ids = np.sort(rng.integers(0, 8192, n, dtype=np.int64))
            expected = np.concatenate((table.w_mm[ids], table.s_mm[ids]), axis=1)
            for workers in (1, 4, 8, 16, 32, 64):
                table.gather_threads = workers
                np.testing.assert_array_equal(table._gather_rows(ids), expected)
                print(json.dumps(dict(kind='python_pool', rows=n, workers=workers,
                                      **measure(lambda: table._gather_rows(ids)))), flush=True)
            print(json.dumps(dict(kind='allocation_only', rows=n,
                                  **measure(lambda: np.empty((n, 264), np.uint8)))), flush=True)
            if args.native:
                from engine.engram_native import NativeGather
                for workers in (1, 4, 8, 16, 32, 64):
                    with NativeGather(table.w_mm, table.s_mm, workers=workers) as gather:
                        np.testing.assert_array_equal(gather.gather(ids), expected)
                        print(json.dumps(dict(kind='native_pool', rows=n, workers=workers,
                                              **measure(lambda: gather.gather(ids)))), flush=True)
    finally:
        table.pool.shutdown()


if __name__ == '__main__':
    main()
