"""Lossless BF16 head with K-group-major code tables and coalesced headers.

LOW[K/128,N,128], DELTA[K/128,N,64], HEADER[K/128,N] place the
current K tile's vocabulary rows together. The row-major representation already
coalesces bytes within each 128-value group; this variant reduces gaps between
vocabulary rows and makes each header vector contiguous. No speed or cuBLAS
accumulation-equivalence claim is implied by lossless weight reconstruction.
"""
from __future__ import annotations

import weakref

import torch
import triton
import triton.language as tl

G = 128


def pack_bits_cpu(codes):
    """Reference layout for all BF16 bit patterns, including NaNs and -0."""
    if codes.device.type != 'cpu' or codes.dtype != torch.int16 or codes.ndim != 2 or codes.shape[1] % G:
        raise ValueError('expected CPU int16 [N,K], K divisible by128')
    n, k = codes.shape
    bits = (codes.to(torch.int32) & 65535).reshape(n, k // G, G).permute(1, 0, 2).contiguous()
    exponent = (bits >> 7) & 255
    maximum = exponent.amax(-1)
    escaped = maximum - exponent.amin(-1) > 15
    prefix = escaped.flatten().cumsum(0, dtype=torch.int32).reshape(k // G, n)
    low = ((bits & 127) | ((bits >> 8) & 128)).to(torch.uint8)
    delta = (maximum[:, :, None] - exponent).clamp_max(15).to(torch.uint8)
    delta = delta[:, :, 0::2] | (delta[:, :, 1::2] << 4)
    header = maximum | (torch.where(escaped, prefix, 0) << 8)
    escape = exponent[escaped].to(torch.uint8)
    return low, delta, header, escape


def unpack_bits_cpu(low, delta, header, escape):
    """Return native row-major int16 bits from the exact tiled tables."""
    slots = header >> 8
    nibble = torch.stack((delta & 15, delta >> 4), -1).reshape(*header.shape, G).to(torch.int32)
    exponent = (header & 255)[:, :, None] - nibble
    raw = slots != 0
    if raw.any():
        exponent[raw] = escape[slots[raw] - 1].to(torch.int32)
    lo = low.to(torch.int32)
    bits = (lo & 127) | ((lo & 128) << 8) | (exponent << 7)
    return bits.permute(1, 0, 2).reshape(header.shape[1], -1).to(torch.int16)


@triton.jit
def _classify(W, HEADER, N: tl.constexpr, KG: tl.constexpr,
              GROUP: tl.constexpr = 128):
    source_group = tl.program_id(0)
    row, kg = source_group // KG, source_group % KG
    destination = kg * N + row
    bits = tl.load(W + source_group * GROUP + tl.arange(0, GROUP)).to(tl.uint16, bitcast=True)
    exponent = (bits >> 7) & 255
    maximum = tl.max(exponent, 0)
    raw = maximum - tl.min(exponent, 0) > 15
    tl.store(HEADER + destination, maximum | (raw.to(tl.uint32) << 8))


@triton.jit
def _encode(W, LOW, DELTA, HEADER, PREFIX, ESCAPE,
            N: tl.constexpr, KG: tl.constexpr, GROUP: tl.constexpr = 128):
    source_group = tl.program_id(0)
    row, kg = source_group // KG, source_group % KG
    destination = kg * N + row
    ks = tl.arange(0, GROUP)
    bits = tl.load(W + source_group * GROUP + ks).to(tl.uint16, bitcast=True)
    exponent = (bits >> 7) & 255
    header = tl.load(HEADER + destination)
    maximum, raw = header & 255, header >> 8 != 0
    slot = tl.where(raw, tl.load(PREFIX + destination), 0)
    tl.store(HEADER + destination, maximum | (slot.to(tl.uint32) << 8))
    low = (bits & 127) | ((bits >> 8) & 128)
    tl.store(LOW + destination * GROUP + ks, low.to(tl.uint8))
    delta = tl.minimum(maximum - exponent, 15).to(tl.uint8)
    even, odd = tl.split(delta.reshape((GROUP // 2, 2)))
    tl.store(DELTA + destination * (GROUP // 2) + tl.arange(0, GROUP // 2), even | (odd << 4))
    tl.store(ESCAPE + (slot - 1) * GROUP + ks, exponent.to(tl.uint8), raw)


@triton.jit
def _decode(LOW, DELTA, HEADER, ESCAPE, OUT, N: tl.constexpr,
            K: tl.constexpr, FIRST: tl.constexpr, COUNT: tl.constexpr,
            B: tl.constexpr = 256):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    row = FIRST + idx // K
    kg, ks = (idx % K) // 128, idx % 128
    group = kg * N + row
    low = tl.load(LOW + group * 128 + ks, idx < COUNT, 0).to(tl.uint16)
    delta = tl.load(DELTA + group * 64 + ks // 2, idx < COUNT, 0).to(tl.uint16)
    delta = (delta >> ((ks & 1) * 4)) & 15
    header = tl.load(HEADER + group, idx < COUNT, 0).to(tl.uint32)
    slot = header >> 8
    escaped = tl.load(ESCAPE + (slot - 1) * 128 + ks,
                      (idx < COUNT) & (slot != 0), 0).to(tl.uint16)
    exponent = tl.where(slot != 0, escaped, (header & 255) - delta).to(tl.uint16)
    bits = (low & 127) | ((low & 128) << 8) | (exponent << 7)
    tl.store(OUT + idx, bits.to(tl.bfloat16, bitcast=True), idx < COUNT)


class PackedHead:
    """Own the tiled tables only; the original BF16 matrix is never retained."""
    def __init__(self, weight):
        if (weight.dtype != torch.bfloat16 or weight.ndim != 2 or not weight.is_contiguous()
                or weight.shape[0] <= 0 or weight.shape[1] <= 0 or weight.shape[1] % G):
            raise ValueError('requires contiguous BF16 [N,K], positive N and K divisible by128')
        self.shape, self.device, self.dtype = weight.shape, weight.device, weight.dtype
        n, k = self.shape
        groups = n * (k // G)
        if groups >= 2**23:
            raise ValueError('escape slots must fit the signed32-bit header')
        if weight.device.type == 'cpu':
            self.low, self.delta, self.header, self.escape = pack_bits_cpu(weight.view(torch.int16))
            self.escape_groups = self.escape.shape[0]
            return
        if weight.device.type != 'cuda':
            raise ValueError('packed head supports CPU reference or CUDA')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('pack native head outside graph capture')
        self.low = torch.empty((k // G, n, G), device=self.device, dtype=torch.uint8)
        self.delta = torch.empty((k // G, n, G // 2), device=self.device, dtype=torch.uint8)
        self.header = torch.empty((k // G, n), device=self.device, dtype=torch.int32)
        _classify[(groups,)](weight, self.header, n, k // G, num_warps=4)
        prefix = (self.header.flatten() >> 8).cumsum(0, dtype=torch.int32)
        self.escape_groups = int(prefix[-1].item())
        self.escape = torch.empty((self.escape_groups, G), device=self.device, dtype=torch.uint8)
        _encode[(groups,)](weight, self.low, self.delta, self.header, prefix, self.escape,
                          n, k // G, num_warps=4)

    @property
    def stored_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.low, self.delta, self.header, self.escape))

    def dequant(self, first=0, last=None):
        n, k = self.shape
        last = n if last is None else last
        if not 0 <= first <= last <= n:
            raise ValueError('invalid head row range')
        if self.device.type == 'cpu':
            return unpack_bits_cpu(self.low, self.delta, self.header, self.escape)[first:last].view(torch.bfloat16)
        out = torch.empty((last - first, k), dtype=torch.bfloat16, device=self.device)
        if out.numel():
            _decode[(triton.cdiv(out.numel(), 256),)](self.low, self.delta, self.header, self.escape,
                out, n, k, first, out.numel(), num_warps=4)
        return out

    def dequant_rows(self, first, last):
        return self.dequant(first, last)


@triton.jit
def _project(X, LOW, DELTA, HEADER, ESCAPE, OUT,
             M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             XM: tl.constexpr, BN: tl.constexpr, WORD_LOADS: tl.constexpr,
             BK: tl.constexpr = 128):
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    ms, ks = tl.arange(0, 16), tl.arange(0, BK)
    acc = tl.zeros((16, BN), tl.float32)
    for start in range(0, K, BK):
        x = tl.load(X + ms[:, None] * XM + start + ks[None, :], ms[:, None] < M, 0)
        group = (start // 128) * N + ns
        header = tl.load(HEADER + group, ns < N, 0).to(tl.uint32)
        slot = header >> 8
        if WORD_LOADS:
            # Preserve the same stored bits. Uint16 source loads steer dot's
            # operand layout toward native BF16 kWidth2. The GPU cancellation
            # gate returns native 640 / byte 0 / word 640; seeded widths
            # 1/2/4/6/16 and graph replay match the native projection.
            # results/head-native-20261008/tiled-word-qualification.log
            low = tl.load(LOW + group[None, :] * 64 + ks[:, None] // 2,
                          ns[None, :] < N, 0).to(tl.uint16)
            low = (low >> ((ks[:, None] & 1) * 8)) & 255
            delta = tl.load(DELTA + group[None, :] * 32 + ks[:, None] // 4,
                            ns[None, :] < N, 0).to(tl.uint16)
            delta = (delta >> ((ks[:, None] & 3) * 4)) & 15
            escaped = tl.load(ESCAPE + (slot[None, :] - 1) * 64 + ks[:, None] // 2,
                              (ns[None, :] < N) & (slot[None, :] != 0), 0).to(tl.uint16)
            escaped = (escaped >> ((ks[:, None] & 1) * 8)) & 255
        else:
            # Negative kept: actual head M4, BN32/stages2 measured 2.956ms vs
            # row-major packed 4.624 / native 2.953, but 72/77 of 258560 logits
            # changed after BF16 rounding. Exact storage is not exact GEMM.
            # results/head-native-20261008/tiled-screen-row4.json
            low = tl.load(LOW + group[None, :] * 128 + ks[:, None], ns[None, :] < N, 0).to(tl.uint16)
            delta = tl.load(DELTA + group[None, :] * 64 + ks[:, None] // 2, ns[None, :] < N, 0).to(tl.uint16)
            delta = (delta >> ((ks[:, None] & 1) * 4)) & 15
            escaped = tl.load(ESCAPE + (slot[None, :] - 1) * 128 + ks[:, None],
                              (ns[None, :] < N) & (slot[None, :] != 0), 0).to(tl.uint16)
        exponent = tl.where(slot[None, :] != 0, escaped, (header[None, :] & 255) - delta).to(tl.uint16)
        bits = (low & 127) | ((low & 128) << 8) | (exponent << 7)
        w = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
        acc = tl.dot(x, w, acc)
    value = acc.to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + ms[:, None] * N + ns[None, :], value,
             (ms[:, None] < M) & (ns[None, :] < N))


def project(x, weight, *, bn=64, bk=128, split=1, warps=4, stages=3, word_loads=False):
    if not isinstance(weight, PackedHead):
        raise TypeError('prepare tiled PackedHead outside capture')
    if (x.dtype != torch.bfloat16 or x.ndim != 2 or not 0 < x.shape[0] <= 16
            or x.shape[1] != weight.shape[1] or x.stride(1) != 1 or x.device != weight.device):
        raise ValueError('expected BF16 decode activations matching tiled head')
    if x.device.type != 'cuda' or bk != G or split != 1:
        raise ValueError('initial tiled kernel requires CUDA, BK128 and split1')
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    low, delta, escape = ((t.view(torch.uint16) for t in (weight.low, weight.delta, weight.escape))
                          if word_loads else (weight.low, weight.delta, weight.escape))
    _project[(triton.cdiv(n, bn),)](x, low, delta, weight.header, escape,
        out, m, n, k, x.stride(0), BN=bn, BK=bk, WORD_LOADS=bool(word_loads),
        num_warps=warps, num_stages=stages)
    return out


_packed = {}


def clear_packed_cache():
    _packed.clear()


def project_native(x, weight, **kwargs):
    """Microbenchmark adapter: weakly identify the unchanged native tensor."""
    key = id(weight)
    entry = _packed.get(key)
    if entry is None or entry[0]() is not weight:
        def release(ref):
            found = _packed.get(key)
            if found is not None and found[0] is ref:
                _packed.pop(key, None)
        packed = PackedHead(weight)
        entry = _packed[key] = (weakref.ref(weight, release), packed)
    return project(x, entry[1], **kwargs)
