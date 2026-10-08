"""Lossless BF16 vocabulary-head storage experiment (not quantization).

Sign/mantissa bytes and four-bit exponent deltas retain the original 16 bits.
Each 128-value group stores its maximum exponent; rare wider exponent ranges
use an exact exponent escape block. The GEMM still rounds FP32 accumulation
to BF16 before widening. cuBLAS accumulation equivalence must be qualified.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _classify(W, HEADER, GROUPS: tl.constexpr, G: tl.constexpr = 128):
    group = tl.program_id(0)
    bits = tl.load(W + group * G + tl.arange(0, G)).to(tl.uint16, bitcast=True)
    exponent = (bits >> 7) & 255
    maximum = tl.max(exponent, 0)
    raw = maximum - tl.min(exponent, 0) > 15
    tl.store(HEADER + group, maximum | (raw.to(tl.uint32) << 8))


@triton.jit
def _encode(W, LOW, DELTA, HEADER, PREFIX, ESCAPE, G: tl.constexpr = 128):
    group = tl.program_id(0)
    ks = tl.arange(0, G)
    bits = tl.load(W + group * G + ks).to(tl.uint16, bitcast=True)
    exponent = (bits >> 7) & 255
    header = tl.load(HEADER + group)
    maximum, raw = header & 255, header >> 8 != 0
    slot = tl.where(raw, tl.load(PREFIX + group), 0)
    tl.store(HEADER + group, maximum | (slot.to(tl.uint32) << 8))
    low = (bits & 127) | ((bits >> 8) & 128)
    tl.store(LOW + group * G + ks, low.to(tl.uint8))
    delta = tl.minimum(maximum - exponent, 15).to(tl.uint8)
    even, odd = tl.split(delta.reshape((G // 2, 2)))
    tl.store(DELTA + group * (G // 2) + tl.arange(0, G // 2), even | (odd << 4))
    tl.store(ESCAPE + (slot - 1) * G + ks, exponent.to(tl.uint8), raw)


@triton.jit
def _decode(LOW, DELTA, HEADER, ESCAPE, OUT, COUNT: tl.constexpr,
            B: tl.constexpr = 256):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    low = tl.load(LOW + idx, idx < COUNT, 0).to(tl.uint16)
    delta = tl.load(DELTA + idx // 2, idx < COUNT, 0).to(tl.uint16)
    delta = (delta >> ((idx & 1) * 4)) & 15
    header = tl.load(HEADER + idx // 128, idx < COUNT, 0).to(tl.uint32)
    slot = header >> 8
    escaped = tl.load(ESCAPE + (slot - 1) * 128 + idx % 128,
                      (idx < COUNT) & (slot != 0), 0).to(tl.uint16)
    exponent = tl.where(slot != 0, escaped, (header & 255) - delta).to(tl.uint16)
    bits = (low & 127) | ((low & 128) << 8) | (exponent << 7)
    tl.store(OUT + idx, bits.to(tl.bfloat16, bitcast=True), idx < COUNT)


class PackedHead:
    """Storage may replace the original matrix after lossless reconstruction tests."""
    def __init__(self, weight):
        if (weight.dtype != torch.bfloat16 or weight.ndim != 2
                or not weight.is_contiguous() or weight.shape[1] % 128):
            raise ValueError('requires a contiguous BF16 head with K divisible by 128')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('pack native head outside graph capture')
        self.shape, self.device, self.dtype = weight.shape, weight.device, weight.dtype
        n, k = self.shape
        groups = n * (k // 128)
        if groups >= 2**23:
            raise ValueError('escape slots must fit the signed 32-bit packed header')
        self.low = torch.empty((n, k), device=weight.device, dtype=torch.uint8)
        self.delta = torch.empty((n, k//2), device=weight.device, dtype=torch.uint8)
        self.header = torch.empty((n, k//128), device=weight.device, dtype=torch.int32)
        _classify[(groups,)](weight, self.header, groups, num_warps=4)
        # Prefix numbers are temporary. Header zero in the high bits means no
        # escape; one-based slots let the decode kernel mask its rare extra load.
        flags = self.header.flatten() >> 8
        prefix = flags.cumsum(0, dtype=torch.int32)
        self.escape_groups = int(prefix[-1].item())
        self.escape = torch.empty((self.escape_groups, 128), device=weight.device, dtype=torch.uint8)
        _encode[(groups,)](weight, self.low, self.delta, self.header, prefix, self.escape,
                            num_warps=4)

    @property
    def stored_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.low, self.delta, self.header, self.escape))

    def dequant(self):
        out = torch.empty(self.shape, dtype=torch.bfloat16, device=self.device)
        _decode[(triton.cdiv(out.numel(), 256),)](self.low, self.delta, self.header,
            self.escape, out, out.numel(), num_warps=4)
        return out


@triton.jit
def _project(X, LOW, DELTA, HEADER, ESCAPE, OUT,
             M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             XM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr = 128):
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    ms = tl.arange(0, 16)
    ks = tl.arange(0, BK)
    acc = tl.zeros((16, BN), tl.float32)
    for start in range(0, K, BK):
        x = tl.load(X + ms[:, None]*XM + start + ks[None, :], ms[:, None] < M, 0)
        low = tl.load(LOW + ns[None, :]*K + start + ks[:, None],
                      ns[None, :] < N, 0).to(tl.uint16)
        delta = tl.load(DELTA + ns[None, :]*(K//2) + (start+ks[:, None])//2,
                        ns[None, :] < N, 0).to(tl.uint16)
        delta = (delta >> ((ks[:, None] & 1) * 4)) & 15
        header = tl.load(HEADER + ns*(K//128) + start//128,
                         ns < N, 0).to(tl.uint32)
        slot = header >> 8
        escaped = tl.load(ESCAPE + (slot[None, :]-1)*128 + ks[:, None],
                          (ns[None, :] < N) & (slot[None, :] != 0), 0).to(tl.uint16)
        exponent = tl.where(slot[None, :] != 0, escaped,
                            (header[None, :] & 255) - delta).to(tl.uint16)
        bits = (low & 127) | ((low & 128) << 8) | (exponent << 7)
        w = bits.to(tl.bfloat16, bitcast=True)
        acc = tl.dot(x, w, acc)
    out = acc.to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + ms[:, None]*N + ns[None, :], out,
             (ms[:, None] < M) & (ns[None, :] < N))


def project(x, weight, *, bn=64, bk=128, split=1, warps=4, stages=3):
    if not isinstance(weight, PackedHead):
        raise TypeError('prepare PackedHead before capture')
    if (x.dtype != torch.bfloat16 or x.ndim != 2 or not 0 < x.shape[0] <= 16
            or x.shape[1] != weight.shape[1] or x.stride(1) != 1
            or x.device != weight.device):
        raise ValueError('expected BF16 decode activations matching the packed head')
    if bk != 128 or split != 1:
        raise ValueError('initial packed head supports BK128 and no split reduction')
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    _project[(triton.cdiv(n, bn),)](x, weight.low, weight.delta, weight.header,
        weight.escape, out, m, n, k, x.stride(0), BN=bn, BK=bk,
        num_warps=warps, num_stages=stages)
    return out
