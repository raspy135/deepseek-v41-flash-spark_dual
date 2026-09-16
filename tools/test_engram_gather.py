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
import torch
import v41_ref as R
from engine.engram import EngramTable

MD = os.environ.get("MODEL_DIR") or os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
a = R.Args.from_json(os.path.join(MD, "config.json"))
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))
t = EngramTable(MD, idx, a.engram_layer_ids[0], "cpu")
print(f"layer {a.engram_layer_ids[0]}: {t.n_rows:,} rows, {t.gather_threads} gather threads")
native = t.native_gather
t.native_gather = None  # Check the legacy fallback's parallelism even when native is enabled.

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
t.native_gather = native
for n in (1, 5, 144, 700):
    ids = np.sort(rng.integers(0, t.n_rows, size=n, dtype=np.int64))
    got, want = t._gather_rows(ids), t._read_rows(ids)
    assert got.shape == want.shape == (n, 264), (got.shape, want.shape)
    assert np.array_equal(got, want), f"n={n}: gather disagrees with pread"
print("gather == pread for n in (1, 5, 144, 700)")

# --- the EP row split is size-gated, and both ranks decide identically ------------------------
# The split trades half the rows for a stream-synchronising H2D and an all-reduce per layer --
# worth it for a 49k-row prefill chunk, not for a 144-row decode block. The danger is a threshold
# that could differ between ranks: one rank reaching a collective alone hangs the pair until the
# process-group timeout. So the decision may only look at `n`, which both ranks derive from
# identical hashes.
class FakeEP:
    def __init__(self, rank): self.rank, self.world, self.active = rank, 2, True

t.row_split = True
DECODE_N, PREFILL_N = T_VERIFY * N_HASH, 2048 * N_HASH
for rank in (0, 1):
    t.ep = FakeEP(rank)
    assert not t._split_call(DECODE_N), f"rank {rank}: decode block ({DECODE_N} rows) still splits"
    assert t._split_call(PREFILL_N), f"rank {rank}: prefill chunk ({PREFILL_N} rows) stopped splitting"
    # the two ranks must agree at every size, including right at the boundary
    for n in (0, 1, DECODE_N, t.split_min_rows - 1, t.split_min_rows, PREFILL_N):
        assert t._split_call(n) == (n >= t.split_min_rows), (rank, n)
t.ep = FakeEP(0)
a0 = [t._split_call(n) for n in range(0, 8192, 97)]
t.ep = FakeEP(1)
a1 = [t._split_call(n) for n in range(0, 8192, 97)]
assert a0 == a1, "the two ranks disagree about which gathers to split -- this wedges the pair"
t.ep, t.row_split = None, False
print(f"split gate: decode {DECODE_N} rows -> local, prefill {PREFILL_N} rows -> split, "
      f"both ranks agree at every size")

# --- pinned staging must not change a single row ---------------------------------------------
# to_device's pinned path copies through a reused double buffer and transfers non_blocking, so a
# missing event wait would let the next call overwrite bytes the previous H2D had not read yet --
# which shows up as plausible-looking wrong rows, never as an error. Needs a free GPU: run this
# while the server is down.
if torch.cuda.is_available():
    try:
        tt = EngramTable(MD, idx, a.engram_layer_ids[0], "cuda")
        h = rng.integers(0, tt.n_rows, size=(T_VERIFY, N_HASH), dtype=np.int64)
        raw, inv, shape = tt.read_raw(h)
        tt.pinned = False
        want = tt.to_device(raw, inv, shape).clone()
        tt.pinned = True
        # Allocate the staging buffers the way the engine does -- inside inference_mode. Buffers
        # created there are inference tensors and cannot be written from outside it, which took the
        # server down the first time this path ran for real.
        with torch.inference_mode():
            tt.to_device(raw, inv, shape)
        for k in range(4):      # >2 calls, so both stage buffers are reused at least once
            got = tt.to_device(raw, inv, shape).clone()
            torch.cuda.synchronize()
            assert torch.equal(got, want), f"pinned call {k} differs from pageable"
        with torch.inference_mode():
            got = tt.to_device(raw, inv, shape).clone()
        assert torch.equal(got, want), "pinned differs when called under inference_mode"
        print(f"pinned == pageable over 6 calls, in and out of inference_mode, "
              f"{tuple(want.shape)} {want.dtype}")
    except torch.OutOfMemoryError:
        print("pinned check SKIPPED: no free GPU memory (is the server up?)")
else:
    print("pinned check skipped: no CUDA")

print("\nENGRAM FALLBACK IS PARALLEL AT DECODE SIZE; SELECTED PATH IS BYTE-IDENTICAL")
