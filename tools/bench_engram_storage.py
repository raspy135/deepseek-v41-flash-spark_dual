"""Small local-checkpoint gather comparison, no GPU and no page-cache eviction."""
import json
import os
import resource
import statistics
import sys
import time
sys.path[:0] = ['/app', '/app/tools']
import numpy as np
from engine.engram import EngramTable
from engine.engram_native import NativeGather


def main():
    root = os.environ['MODEL_DIR']
    with open(os.path.join(root, 'model.safetensors.index.json')) as f: index = json.load(f)
    rng = np.random.default_rng(51821)
    for layer in (1, 14):
        table = EngramTable(root, index, layer, 'cpu')
        natives = {n: NativeGather(table.w_mm, table.s_mm, n) for n in (8, 16, 32, 64)}
        methods = [('python64', table._gather_rows)] + [(f'native{n}', g.gather) for n, g in natives.items()]
        try:
            for count in (96, 144):
                times = {name: {'fresh_ms': [], 'warm_ms': [], 'major_faults': []} for name, _ in methods}
                for repeat in range(12):
                    for name, fn in (methods if repeat % 2 == 0 else methods[::-1]):
                        ids = np.sort(rng.integers(0, table.n_rows, count, dtype=np.int64))
                        before = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
                        start = time.perf_counter_ns(); got = fn(ids)
                        fresh = (time.perf_counter_ns()-start)/1e6
                        faults = resource.getrusage(resource.RUSAGE_SELF).ru_majflt-before
                        start = time.perf_counter_ns(); again = fn(ids)
                        warm = (time.perf_counter_ns()-start)/1e6
                        np.testing.assert_array_equal(got, again)
                        np.testing.assert_array_equal(got[:, :256], table.w_mm[ids])
                        np.testing.assert_array_equal(got[:, 256:], table.s_mm[ids])
                        for key, value in (('fresh_ms', fresh), ('warm_ms', warm), ('major_faults', faults)):
                            times[name][key].append(value)
                for name, values in times.items():
                    print(json.dumps(dict(layer=layer, rows=count, method=name,
                        **{k:dict(median=statistics.median(v), p95=float(np.percentile(v,95)))
                           for k,v in values.items()})), flush=True)
        finally:
            for native in natives.values(): native.close()
            table.pool.shutdown(); os.close(table.fd)


if __name__ == '__main__': main()
