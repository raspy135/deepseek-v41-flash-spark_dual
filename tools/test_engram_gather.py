"""The engram row gather must be parallel at decode sizes, and byte-identical to the pread path.

A `max(1024, ...)` floor on the task size used to make any gather under 1024 rows a single task,
which is every decode step (a verify block asks for ~144 unique rows). It ran serially, one
synchronous page fault at a time, and cost 83 ms of a 196 ms decode step. The floor is invisible
in output -- the rows are correct either way -- so this test asserts the *behaviour*: that the
thread pool is actually used, and that using it does not change a byte.

Needs the checkpoint; run it in the serving image.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import v41_ref as R
from engine.engram import EngramTable

MD = os.environ.get("MODEL_DIR") or os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
a = R.Args.from_json(os.path.join(MD, "config.json"))
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))
t = EngramTable(MD, idx, a.engram_layer_ids[0], "cpu")
print(f"layer {a.engram_layer_ids[0]}: {t.n_rows:,} rows, {t.gather_threads} gather threads")

rng = np.random.default_rng(0)

# --- the pool is really used at the sizes decode asks for -------------------------------------
class CountingPool:
    def __init__(self, inner): self.inner, self.maps, self.tasks = inner, 0, 0
    def map(self, fn, it):
        it = list(it); self.maps += 1; self.tasks += len(it)
        return self.inner.map(fn, it)

real = t.pool
t.pool = CountingPool(real)
T_VERIFY, N_HASH = 6, 24
for n in (T_VERIFY * N_HASH, 288, 1024):
    t.pool.maps = t.pool.tasks = 0
    ids = np.sort(rng.integers(0, t.n_rows, size=n, dtype=np.int64))
    t._gather_rows(ids)
    assert t.pool.maps == 1, f"n={n}: gather ran serially ({t.pool.maps} pool calls)"
    assert t.pool.tasks > 1, f"n={n}: one task only -- the task-size floor is back"
    print(f"  n={n:5d} -> {t.pool.tasks} parallel tasks")
t.pool = real

# small gathers stay on the calling thread on purpose
t.pool = CountingPool(real)
t._gather_rows(np.sort(rng.integers(0, t.n_rows, size=8, dtype=np.int64)))
assert t.pool.maps == 0, "tiny gather should not pay for the pool"
print("  n=    8 -> serial, as intended")
t.pool = real

# --- and the rows are the same bytes the pread path returns -----------------------------------
for n in (1, 5, 144, 700):
    ids = np.sort(rng.integers(0, t.n_rows, size=n, dtype=np.int64))
    got, want = t._gather_rows(ids), t._read_rows(ids)
    assert got.shape == want.shape == (n, 264), (got.shape, want.shape)
    assert np.array_equal(got, want), f"n={n}: gather disagrees with pread"
print("gather == pread for n in (1, 5, 144, 700)")

print("\nENGRAM GATHER IS PARALLEL AT DECODE SIZE AND BYTE-IDENTICAL")
