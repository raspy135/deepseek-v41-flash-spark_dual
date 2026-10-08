"""Packed-head experiment: materialize BF16 tiles in a bounded CTA workspace.

Byte-derived dot operands made Triton choose kWidth=4 rather than the native
BF16 load's kWidth=2, doubled registers and altered rare rounded logits. A
volatile BF16 reload tests whether preserving the native operand geometry can
recover exact arithmetic and bandwidth. The temporary is bounded by grid size,
not vocabulary size; never allocate another full head.
"""
import torch
import triton
import triton.language as tl
from head_packed import PackedHead


@triton.jit
def _project(X, LOW, DELTA, HEADER, ESCAPE, SCRATCH, OUT,
             M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             XM: tl.constexpr, BN: tl.constexpr, GRID: tl.constexpr,
             BK: tl.constexpr = 128):
    pid = tl.program_id(0)
    ms = tl.arange(0, 16)
    ks = tl.arange(0, BK)
    ni = tl.arange(0, BN)
    scratch = SCRATCH + pid * BN * BK + ni[None, :] * BK + ks[:, None]
    for first in range(pid * BN, N, GRID * BN):
        ns = first + ni
        acc = tl.zeros((16, BN), tl.float32)
        for start in range(0, K, BK):
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
                                (header[None, :] & 255)-delta).to(tl.uint16)
            bits = (low & 127) | ((low & 128) << 8) | (exponent << 7)
            tl.store(scratch, bits.to(tl.bfloat16, bitcast=True))
            tl.debug_barrier()
            # Volatile prevents forwarding the byte-derived register value.
            w = tl.load(scratch, volatile=True)
            x = tl.load(X + ms[:, None]*XM + start + ks[None, :], ms[:, None] < M, 0)
            acc = tl.dot(x, w, acc)
            tl.debug_barrier()
        tl.store(OUT + ms[:, None]*N + ns[None, :], acc.to(tl.bfloat16).to(tl.float32),
                 (ms[:, None] < M) & (ns[None, :] < N))


def project(x, weight, *, bn=64, bk=128, split=1, warps=4, stages=2, grid=128):
    if not isinstance(weight, PackedHead):
        raise TypeError('requires a prepared lossless PackedHead')
    if (x.dtype != torch.bfloat16 or x.ndim != 2 or not 0 < x.shape[0] <= 16
            or x.shape[1] != weight.shape[1] or x.stride(1) != 1 or x.device != weight.device):
        raise ValueError('invalid native head activations')
    if bk != 128 or split != 1 or grid < 1:
        raise ValueError('requires BK128, split1 and a positive bounded grid')
    m, k = x.shape
    n = weight.shape[0]
    grid = min(grid, triton.cdiv(n, bn))
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    scratch = torch.empty((grid, bn, bk), device=x.device, dtype=torch.bfloat16)
    _project[(grid,)](x, weight.low, weight.delta, weight.header, weight.escape,
        scratch, out, m, n, k, x.stride(0), BN=bn, BK=bk, GRID=grid,
        num_warps=warps, num_stages=stages)
    return out
