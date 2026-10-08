"""Load the lossless packed head through 16-bit words to retain native MMA geometry.

Triton inferred kWidth=4 from byte loads in the first packed kernel, changing
K-fragment summation order and doubling registers. Word loads retain the same
storage bytes while testing native kWidth=2 without a rematerialization buffer.
"""
import torch
import triton
import triton.language as tl
from head_packed import PackedHead


@triton.jit
def _project(X, LOW16, DELTA16, HEADER, ESCAPE16, OUT,
             M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             XM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr = 128):
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    ms = tl.arange(0, 16)
    ks = tl.arange(0, BK)
    acc = tl.zeros((16, BN), tl.float32)
    for start in range(0, K, BK):
        x = tl.load(X + ms[:, None]*XM + start + ks[None, :], ms[:, None] < M, 0)
        low_word = tl.load(LOW16 + ns[None, :]*(K//2) + (start+ks[:, None])//2,
                           ns[None, :] < N, 0).to(tl.uint16)
        low = (low_word >> ((ks[:, None] & 1)*8)) & 255
        delta_word = tl.load(DELTA16 + ns[None, :]*(K//4) + (start+ks[:, None])//4,
                             ns[None, :] < N, 0).to(tl.uint16)
        delta = (delta_word >> ((ks[:, None] & 3)*4)) & 15
        header = tl.load(HEADER + ns*(K//128) + start//128, ns < N, 0).to(tl.uint32)
        slot = header >> 8
        escape_word = tl.load(ESCAPE16 + (slot[None, :]-1)*64 + ks[:, None]//2,
                              (ns[None, :] < N) & (slot[None, :] != 0), 0).to(tl.uint16)
        escaped = (escape_word >> ((ks[:, None] & 1)*8)) & 255
        exponent = tl.where(slot[None, :] != 0, escaped,
                            (header[None, :] & 255)-delta).to(tl.uint16)
        bits = (low & 127) | ((low & 128) << 8) | (exponent << 7)
        w = bits.to(tl.bfloat16, bitcast=True)
        acc = tl.dot(x, w, acc)
    tl.store(OUT + ms[:, None]*N + ns[None, :], acc.to(tl.bfloat16).to(tl.float32),
             (ms[:, None] < M) & (ns[None, :] < N))


def project(x, weight, *, bn=64, bk=128, split=1, warps=4, stages=2):
    if not isinstance(weight, PackedHead):
        raise TypeError('requires a prepared lossless PackedHead')
    if (x.dtype != torch.bfloat16 or x.ndim != 2 or not 0 < x.shape[0] <= 16
            or x.shape[1] != weight.shape[1] or x.stride(1) != 1 or x.device != weight.device):
        raise ValueError('invalid native head activations')
    if bk != 128 or split != 1:
        raise ValueError('initial word-load kernel requires BK128 and split1')
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    _project[(triton.cdiv(n,bn),)](x, weight.low.view(torch.uint16),
        weight.delta.view(torch.uint16), weight.header, weight.escape.view(torch.uint16),
        out, m, n, k, x.stride(0), BN=bn, BK=bk,
        num_warps=warps, num_stages=stages)
    return out
