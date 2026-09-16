"""Optional CPU gather. One GIL-releasing call, persistent native workers.

Returned arrays lease reusable buffers through their ctypes/memoryview base, so
slices, Torch CPU aliases and outstanding read-ahead futures retain ownership.
Exhausting the small reuse pool allocates an independent buffer; never blocks a
prefetch worker while its consumer is waiting for a different future.
"""
import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import threading
import weakref

import numpy as np

SOURCE = Path(__file__).with_name('engram_gather.cpp')


def source_digest():
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def _library():
    # No downloads, Python headers, NumPy ABI, CUDA or torch extension build.
    compiler = os.environ.get('CXX', 'g++')
    version = subprocess.check_output([compiler, '--version'])
    key = hashlib.sha256(SOURCE.read_bytes() + version + platform.machine().encode()).hexdigest()[:24]
    folder = Path(os.environ.get('TRITON_CACHE_DIR', '/tmp/dsv41-native')) / 'engram-cpu'
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = folder / f'gather-{key}.so'
    with (folder / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            with tempfile.TemporaryDirectory(prefix='build-', dir=folder) as tmp:
                built = Path(tmp) / 'gather.so'
                subprocess.run([compiler, '-std=c++17', '-O3', '-DNDEBUG', '-fPIC', '-shared',
                                '-pthread', str(SOURCE), '-o', str(built)], check=True)
                os.replace(built, target)
    lib = ctypes.CDLL(str(target))  # CDLL releases the GIL for the complete native call.
    ptr = ctypes.c_void_p
    lib.engram_gather_create.argtypes = [ptr, ptr, ctypes.c_int64, ctypes.c_int]
    lib.engram_gather_create.restype = ptr
    lib.engram_gather_run.argtypes = [ptr, ptr, ctypes.c_size_t, ptr]
    lib.engram_gather_run.restype = ctypes.c_int
    lib.engram_gather_destroy.argtypes = [ptr]
    lib.engram_gather_destroy.restype = None
    return lib


class RowBuffers:
    def __init__(self, rows=384, slots=4):
        self.rows = rows
        self.lock = threading.Lock()
        self.free = [np.empty((rows, 264), dtype=np.uint8) for _ in range(slots)]

    def acquire(self, n):
        with self.lock:
            storage = self.free.pop() if n <= self.rows and self.free else None
        if storage is None:
            return np.empty((n, 264), dtype=np.uint8)
        # as_array's memoryview retains this ctypes owner, including through views.
        # Do NOT finalize just the ndarray: a slice may outlive the original array.
        owner = (ctypes.c_uint8 * (n * 264)).from_buffer(storage)
        weakref.finalize(owner, self.release, storage)
        return np.ctypeslib.as_array(owner).reshape(n, 264)

    def release(self, storage):
        with self.lock:
            self.free.append(storage)


class NativeGather:
    def __init__(self, weights, scales, workers=16):
        for array, width in ((weights, 256), (scales, 8)):
            if array.dtype != np.uint8 or array.ndim != 2 or array.shape[1] != width or not array.flags.c_contiguous:
                raise ValueError('native gather requires contiguous uint8 row tables')
        if weights.shape[0] != scales.shape[0] or not 1 <= workers <= 128:
            raise ValueError('table row counts or worker count invalid')
        self.weights, self.scales = weights, scales  # Own the mappings while C borrows pointers.
        self.lib = _library()
        self.lock = threading.Lock()
        self.buffers = RowBuffers()
        self.ctx = self.lib.engram_gather_create(weights.ctypes.data, scales.ctypes.data,
                                                weights.shape[0], workers)
        if not self.ctx:
            raise RuntimeError('cannot create native gather worker pool')
        self._cleanup = weakref.finalize(self, self.lib.engram_gather_destroy, self.ctx)

    def gather(self, ids):
        if ids.dtype != np.int64 or ids.ndim != 1 or not ids.flags.c_contiguous:
            raise ValueError('row IDs must be contiguous int64 vector')
        out = self.buffers.acquire(len(ids))
        # Also protects native context destruction. Different tables still run in parallel.
        with self.lock:
            if not self.ctx:
                raise RuntimeError('native gather is closed')
            result = self.lib.engram_gather_run(self.ctx, ids.ctypes.data, len(ids), out.ctypes.data)
        if result == 1:
            raise IndexError('Engram row ID out of bounds')
        if result:
            raise RuntimeError(f'native gather failed: {result}')
        return out

    def close(self):
        with self.lock:
            if self.ctx:
                self._cleanup()
                self.ctx = None

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
