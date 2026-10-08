"""EXPERIMENTAL / CUDA-UNQUALIFIED exact decode softmax prototype.

Prepared only; CPU geometry tests passed, no CUDA compilation, numerical gate,
or timing has been run.  The user clarified that the intended optimization was
dense weights, so this separate softmax experiment was stopped before GPU work.

This is intentionally not connected to production.  PyTorch 2.13 Reduce.cuh
uses four accumulators per lane, a left fold, then descending warp shuffles.
For vectorized rows (N >= 128), the four streams are adjacent components of
float4 loads; otherwise they are stride-separated.  Misaligned row starts put
the short header and tail in accumulator zero.  A normal tl.sum(p) therefore
has a different rounding tree even though the mathematical answer is equal.

The installed geometry is emulated on the host by reduction_config.  Cases
needing a block-y or global reduction are rejected rather than guessed.  The
decoder's usual rows >= 32, N <= 4096 use one 32-lane reduction per row.
"""
from __future__ import annotations

from dataclasses import dataclass
import struct


@dataclass(frozen=True)
class ReductionConfig:
    n: int
    rows: int
    width: int
    height: int
    vectorized: bool
    split_y: bool


def _last_pow2(value: int) -> int:
    return 1 << (max(1, value).bit_length() - 1)


def reduction_config(n: int, rows: int) -> ReductionConfig:
    """Reduce.cuh setReduceConfig<float,float,4,4> for contiguous last axis."""
    if n <= 0 or rows <= 0:
        raise ValueError("positive reduction width and output count required")
    vectorized = n >= 128
    dim0 = n // 4 if vectorized else n
    d0 = min(_last_pow2(dim0), 512)
    d1 = min(_last_pow2(rows), 512)
    width = min(d0, 32)
    height = min(d1, 512 // width)
    width = min(d0, 512 // height)
    split_y = (n + width - 1) // width >= min(height * 16, 256)
    return ReductionConfig(n, rows, width, height, vectorized, split_y)


def lane_streams(config: ReductionConfig, row: int) -> list[list[list[int]]]:
    """Indices visited by each lane's four independent FP32 accumulators."""
    if config.split_y and config.height > 1:
        raise ValueError("prototype does not support block-y reductions")
    n, width = config.n, config.width
    streams = [[[] for _ in range(4)] for _ in range(width)]
    if config.vectorized:
        misalignment = (row * n) % 4  # fresh torch allocation starts 16B aligned
        header = 4 - misalignment if misalignment else 0
        if header:
            for lane in range(misalignment, 4):
                streams[lane][0].append(lane - misalignment)
        end = n - header
        for lane in range(width):
            vector = lane
            while 4 * vector + 3 < end:
                for component in range(4):
                    streams[lane][component].append(header + 4 * vector + component)
                vector += width
        tail = end - end % 4
        for lane in range(end % 4):
            streams[lane][0].append(header + tail + lane)
    else:
        for lane in range(width):
            index = lane
            accumulator = 0
            while index < n:
                streams[lane][accumulator].append(index)
                index += width
                accumulator = (accumulator + 1) % 4
    return streams


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def sum_row_cpu(values: list[float], rows: int, row: int = 0) -> float:
    """Independent scalar derivation, not torch's CPU sum implementation."""
    config = reduction_config(len(values), rows)
    lanes = []
    for streams in lane_streams(config, row):
        acc = [0.0] * 4
        for component, indices in enumerate(streams):
            for index in indices:
                acc[component] = _f32(acc[component] + values[index])
        lanes.append(_f32(_f32(_f32(acc[0] + acc[1]) + acc[2]) + acc[3]))
    offset = config.width // 2
    while offset:
        previous = lanes[:]
        for lane in range(offset):
            lanes[lane] = _f32(previous[lane] + previous[lane + offset])
        offset //= 2
    return lanes[0]


def _kernels():
    # Lazy import keeps CPU geometry tests runnable without CUDA/PyTorch.
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def gather_valid(p, indices, valid, BN: tl.constexpr):
        # Mask indices before gathering; no OOB even on a short last chunk.
        indices = tl.where(valid, indices, 0)
        result = tl.gather(p, indices, axis=0)
        return tl.where(valid, result, 0.0)

    @triton.jit
    def ordered_sum(p, row, N: tl.constexpr, BN: tl.constexpr,
                    BW: tl.constexpr, VECTORIZED: tl.constexpr):
        lane = tl.arange(0, BW)
        a0 = tl.full((BW,), 0.0, tl.float32)
        a1 = tl.full((BW,), 0.0, tl.float32)
        a2 = tl.full((BW,), 0.0, tl.float32)
        a3 = tl.full((BW,), 0.0, tl.float32)
        if VECTORIZED:
            if N % 4 == 0:
                # Constant register rearrangements avoid the general gather's
                # dynamic row-alignment path for every serving cache width.
                even, odd = tl.split(tl.reshape(p, (BN // 2, 2)))
                p0, p2 = tl.split(tl.reshape(even, (BN // 4, 2)))
                p1, p3 = tl.split(tl.reshape(odd, (BN // 4, 2)))
                for k in tl.static_range(triton.cdiv(N, 4 * BW)):
                    vector = lane + k * BW
                    valid = vector * 4 + 3 < N
                    a0 = a0 + gather_valid(p0, vector, valid, BN // 4)
                    a1 = a1 + gather_valid(p1, vector, valid, BN // 4)
                    a2 = a2 + gather_valid(p2, vector, valid, BN // 4)
                    a3 = a3 + gather_valid(p3, vector, valid, BN // 4)
            else:
                misalignment = (row * N) % 4
                header = tl.where(misalignment != 0, 4 - misalignment, 0)
                head_index = lane - misalignment
                head_valid = (misalignment != 0) & (lane >= misalignment) & (lane < 4)
                a0 = gather_valid(p, head_index, head_valid, BN)
                end = N - header
                for k in tl.static_range(triton.cdiv(N, 4 * BW)):
                    vector = lane + k * BW
                    first = header + vector * 4
                    valid = vector * 4 + 3 < end
                    a0 = a0 + gather_valid(p, first, valid, BN)
                    a1 = a1 + gather_valid(p, first + 1, valid, BN)
                    a2 = a2 + gather_valid(p, first + 2, valid, BN)
                    a3 = a3 + gather_valid(p, first + 3, valid, BN)
                tail_start = end - end % 4
                tail_index = header + tail_start + lane
                a0 = a0 + gather_valid(p, tail_index, lane < end % 4, BN)
        else:
            for k in tl.static_range(triton.cdiv(N, 4 * BW)):
                first = lane + k * 4 * BW
                a0 = a0 + gather_valid(p, first, first < N, BN)
                a1 = a1 + gather_valid(p, first + BW, first + BW < N, BN)
                a2 = a2 + gather_valid(p, first + 2 * BW, first + 2 * BW < N, BN)
                a3 = a3 + gather_valid(p, first + 3 * BW, first + 3 * BW < N, BN)
        value = ((a0 + a1) + a2) + a3
        # Explicit descending tree; do not let a generic reduction reassociate.
        for stage in tl.static_range(9):
            offset: tl.constexpr = 256 >> stage
            if offset < BW:
                other_lane = tl.where(lane + offset < BW, lane + offset, lane)
                other = tl.gather(value, other_lane, axis=0)
                value = value + other
        return tl.sum(tl.where(lane == 0, value, 0.0), axis=0)

    @triton.jit
    def sum_kernel(P, OUT, N: tl.constexpr, BN: tl.constexpr,
                   BW: tl.constexpr, VECTORIZED: tl.constexpr):
        row = tl.program_id(0)
        n = tl.arange(0, BN)
        p = tl.load(P + row * N + n, n < N, other=0.0)
        total = ordered_sum(p, row, N, BN, BW, VECTORIZED)
        tl.store(OUT + row, total)

    @triton.jit
    def softmax_kernel(S, MASK, SINK, OUT, PSUM, MX, H: tl.constexpr, N: tl.constexpr,
                       ST: tl.constexpr, SH: tl.constexpr, SN: tl.constexpr,
                       SCALE: tl.constexpr, BN: tl.constexpr, BW: tl.constexpr,
                       VECTORIZED: tl.constexpr, DEBUG: tl.constexpr):
        row = tl.program_id(0)
        t, h = row // H, row % H
        n = tl.arange(0, BN)
        valid = n < N
        score = tl.load(S + t * ST + h * SH + n * SN, valid, other=0.0) * SCALE
        keep = tl.load(MASK + t * N + n, valid, other=0) != 0
        score = tl.where(keep, score, float("-inf"))
        mx = tl.maximum(tl.max(tl.where(valid, score, float("-inf")), axis=0), -1e30)
        p = libdevice.exp(score - mx)
        total = ordered_sum(p, row, N, BN, BW, VECTORIZED)
        denom = total + libdevice.exp(tl.load(SINK + h) - mx)
        tl.store(OUT + row * N + n, libdevice.div_rn(p, denom), valid)
        if DEBUG:
            tl.store(PSUM + row, total)
            tl.store(MX + row, mx)

    return sum_kernel, softmax_kernel


_cached_kernels = None


def _get_kernels():
    global _cached_kernels
    if _cached_kernels is None:
        _cached_kernels = _kernels()
    return _cached_kernels


def _supported_config(n, rows):
    config = reduction_config(n, rows)
    if config.split_y and config.height > 1:
        raise ValueError(f"block-y/global reduction unsupported: {config}")
    if n > 8192:
        raise ValueError("prototype limited to N <= 8192")
    return config


def sum_rows(p):
    """Diagnostic isolated sum; input must have fresh aligned contiguous rows."""
    import torch
    import triton
    if p.dtype != torch.float32 or not p.is_cuda or not p.is_contiguous() or p.data_ptr() % 16:
        raise ValueError("aligned contiguous CUDA FP32 input required")
    n = p.shape[-1]
    rows = p.numel() // n
    config = _supported_config(n, rows)
    out = torch.empty(p.shape[:-1] + (1,), device=p.device, dtype=torch.float32)
    _get_kernels()[0][(rows,)](p, out, n, triton.next_power_of_2(n), config.width,
        config.vectorized, num_warps=4, enable_fp_fusion=False, enable_reflect_ftz=False)
    return out


def attn_probs(s, mask, sink, scale, *, out=None, debug=False):
    """Prototype equivalent of LeanOps.attn_probs, with no intermediate p buffer."""
    import torch
    import triton
    if s.dtype != torch.float32 or s.ndim != 3 or not s.is_cuda:
        raise ValueError("scores must be CUDA FP32 [T,H,N]")
    t, h, n = s.shape
    if mask.shape != (t, n) or sink.shape != (h,) or sink.dtype != torch.float32:
        raise ValueError("invalid mask/sink shapes or sink dtype")
    if mask.device != s.device or sink.device != s.device or not sink.is_contiguous():
        raise ValueError("mask/sink must share score device; sink must be contiguous")
    if mask.dtype not in (torch.bool, torch.uint8):
        raise ValueError("mask must be bool or uint8")
    config = _supported_config(n, t * h)
    mask = mask.contiguous()
    if out is None:
        out = torch.empty(s.shape, device=s.device, dtype=s.dtype)
    if out.shape != s.shape or out.dtype != s.dtype or out.device != s.device or not out.is_contiguous():
        raise ValueError("out must be contiguous CUDA FP32 with score shape")
    psum = torch.empty((t, h, 1), device=s.device, dtype=s.dtype) if debug else out
    mx = torch.empty_like(psum) if debug else out
    _get_kernels()[1][(t * h,)](s, mask, sink, out, psum, mx, h, n, *s.stride(),
        float(scale), triton.next_power_of_2(n), config.width, config.vectorized, debug,
        num_warps=4, enable_fp_fusion=False, enable_reflect_ftz=False)
    return (out, psum, mx) if debug else out
