"""GB10 experiment: CUDA graph replay with device-resident vs mapped pinned-host FP4 weights.

Uses identical weights and the native CUDA path. This measures steady-state GPU read cost only;
it excludes SSD reads, initial copies, and allocation cost. It does not change the serving arena.
Run with the serving worker stopped and MODEL_DIR set to the checkpoint directory.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import statistics
from types import SimpleNamespace

import torch

import fp4_moe as K
from test_fp4_moe import load_arena, routing_with_distinct


class MappedWeight:
    """Retain the host allocation and expose its CUDA-mapped address to the pointer wrapper."""

    def __init__(self, gpu, runtime):
        self.host = gpu.cpu().pin_memory()
        self.shape = self.host.shape
        pointer = ctypes.c_void_p()
        status = runtime.cudaHostGetDevicePointer(ctypes.byref(pointer), self.host.data_ptr(), 0)
        if status:
            raise RuntimeError(f"cudaHostGetDevicePointer failed with CUDA error {status}")
        self.pointer = pointer.value

    def data_ptr(self):
        return self.pointer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=60)
    args = parser.parse_args()
    if args.iters < 1:
        parser.error("--iters must be positive")
    if K.DOT_SCALED:
        parser.error("unset DSV41_FP4_DOT_SCALED to measure native CUDA reads")
    path = ctypes.util.find_library("cudart") or os.path.join(
        os.environ.get("CUDA_HOME", "/usr/local/cuda"), "lib64", "libcudart.so")
    runtime = ctypes.CDLL(path)
    runtime.cudaHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
    runtime.cudaHostGetDevicePointer.restype = ctypes.c_int
    arena = load_arena(32, "cuda")
    mapped = SimpleNamespace(**arena.__dict__)
    for key in ("w1", "s1", "w2", "s2", "w3", "s3"):
        setattr(mapped, key, MappedWeight(getattr(arena, key), runtime))
    original_native = K.CUDA_DECODE
    K.CUDA_DECODE = True
    gen = torch.Generator().manual_seed(20260922)
    try:
        for tokens, experts in ((1, 6), (6, 30), (8, 32)):
            x = torch.randn((tokens, K.DIM), generator=gen).bfloat16().cuda()
            slots, weights = routing_with_distinct(tokens, experts, gen, "cuda")
            graphs, outputs = {}, {}
            for name, weights_arena in (("device", arena), ("mapped", mapped)):
                for _ in range(4):
                    K.moe_forward(x, slots, weights, weights_arena, out_dtype=torch.float32)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs[name] = K.moe_forward(x, slots, weights, weights_arena,
                                                  out_dtype=torch.float32)
                graph.replay()
                graphs[name] = graph
            times = {name: [] for name in graphs}
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            names = list(graphs)
            for repeat in range(args.iters):
                for name in (names if repeat % 2 == 0 else names[::-1]):
                    start.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    times[name].append(start.elapsed_time(end))
            medians = {name: statistics.median(values) for name, values in times.items()}
            exact = torch.equal(outputs["device"], outputs["mapped"])
            print(f"tokens={tokens} experts={experts} ms={medians} exact={exact}", flush=True)
            assert exact, "mapped and device weight outputs differ"
    finally:
        K.CUDA_DECODE = original_native


if __name__ == "__main__":
    with torch.inference_mode():
        main()
