"""P3 target: decode-shaped MoE time for FP4 vs the EXL3 reference, one layer, no engine boot.

Drives `tools/fp4_moe.py` (the FP4 kernel the EXL3 one must beat) and
`tools/exl3_moe.py::moe_forward_exl3_ref` at the same TP-world-1 shapes (DIM=5120, INTER=2304),
on real EXL3 experts from the source and synthetic routing.  This is the number the fused EXL3
kernel is measured against; it is NOT a serving benchmark and does not touch the checkpoint's
FP4 weights beyond their shapes.

Run:  .venv/bin/python tools/bench_exl3_decode.py [T ...]
"""

import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exl3_moe as X3  # noqa: E402
import fp4_moe as F4  # noqa: E402
from pack_exl3_experts import SourceCheckpoint  # noqa: E402

try:
    import exl3_moe_cuda as XC  # noqa: E402
except Exception as _e:  # noqa: BLE001 - nvcc/toolchain missing is not fatal for the reference column
    XC = None
    print(f"exl3_moe_cuda unavailable ({_e!r}); skipping the kernel column")

SRC = os.environ.get("EXL3_SOURCE", os.path.expanduser("~/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw"))
E = int(os.environ.get("EXPERTS", "8"))
TOPK = 6
ITERS = int(os.environ.get("ITERS", "5"))


def timed(fn, iters=ITERS):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    dev = "cuda"
    src = SourceCheckpoint(SRC)
    bits = src.layer_bits()

    # real EXL3 experts, one layer, from the source's world=1 (full-N) slice
    x3 = X3.Exl3Arena(E, device=dev, tp_rank=0, tp_world=1, bits=3.0)
    for e in range(E):
        x3.load_slot(e, src.record(0, e, 0, 1), bits[0])

    # FP4 arena: timing only, shapes are what matters. 18.80 MB/slot.
    f4 = F4.ExpertArena(E, device=dev, tp_rank=0, tp_world=1)
    for name in ("w1", "s1", "w2", "s2", "w3", "s3"):
        getattr(f4, name).copy_(torch.randint(0, 256, getattr(f4, name).shape, dtype=torch.uint8, device=dev))

    print(f"FP4 slot {f4.bytes_per_slot / 1e6:.2f} MB | EXL3 slot {x3.bytes_per_slot / 1e6:.2f} MB "
          f"| EXL3/FP4 = {x3.bytes_per_slot / f4.bytes_per_slot:.3f}x")
    print(f"{ 'T':>3} {'P':>4} | {'FP4 ms':>9} {'EXL3 cu ms':>10} {'cu/FP4':>7} {'EXL3 ref ms':>12}")
    for T in (int(v) for v in (sys.argv[1:] or ["1", "2", "4", "6"])):
        P = T * TOPK
        slots = torch.randint(0, E, (T, TOPK), dtype=torch.int32, device=dev)
        w = torch.rand(T, TOPK, device=dev)
        x = (torch.randn(T, X3.DIM, dtype=torch.float32, device=dev) * 0.1).to(torch.bfloat16)
        fp4 = timed(lambda: F4.moe_forward(x, slots, w, f4, 10.0))
        cu = timed(lambda: XC.moe_forward(x, slots, w, x3, 10.0)) if XC is not None else float("nan")
        ref = timed(lambda: X3.moe_forward_exl3_ref(x, slots, w, x3, 10.0, out_dtype=torch.bfloat16),
                    iters=2)
        print(f"{T:>3} {P:>4} | {fp4:>9.3f} {cu:>10.3f} {cu / fp4:>6.2f}x {ref:>12.2f}")

    # where the reference spends it: full fp64 dequant of one expert
    t = timed(lambda: x3.dequant_slot(0), iters=3)
    print(f"\nEXL3 dequant_slot(one expert, fp64, full matrices): {t:.2f} ms")
    import exl3_ref as R
    t = timed(lambda: R.unpack(x3.read_slot(0)["t1"], float(x3.bits[0]), "mul1"), iters=5)
    print(f"EXL3 unpack W_q (one w1, fp16, no rotation):        {t:.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
