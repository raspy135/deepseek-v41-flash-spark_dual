"""Native BF16 vocabulary-head kernel experiments.

These retain every checkpoint weight and BF16 activation value, accumulate in
FP32, then round logits to BF16 before widening, as the existing head does.
Accumulation order may differ from cuBLAS; benchmark exactness and acceptance
before integrating. No engine dispatch or default is changed by this module.
"""
import torch
import triton
import triton.language as tl


_packed = {}


def clear_packed_cache():
    _packed.clear()


@triton.jit
def _tensor_core(X, W, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 XM: tl.constexpr, WN: tl.constexpr, BN: tl.constexpr,
                 BK: tl.constexpr, SPLIT: tl.constexpr):
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    part = tl.program_id(1)
    ms = tl.arange(0, 16)
    ks = tl.arange(0, BK)
    chunk: tl.constexpr = triton.cdiv(K, SPLIT * BK) * BK
    acc = tl.zeros((16, BN), tl.float32)
    for start in range(part * chunk, tl.minimum((part + 1) * chunk, K), BK):
        x = tl.load(X + ms[:, None] * XM + start + ks[None, :],
                    (ms[:, None] < M) & (start + ks[None, :] < K), 0)
        w = tl.load(W + ns[None, :] * WN + start + ks[:, None],
                    (ns[None, :] < N) & (start + ks[:, None] < K), 0)
        acc = tl.dot(x, w, acc)
    if SPLIT == 1:
        value = acc.to(tl.bfloat16).to(tl.float32)
    else:
        value = acc
    tl.store(OUT + (part * M + ms[:, None]) * N + ns[None, :], value,
             (ms[:, None] < M) & (ns[None, :] < N))


@triton.jit
def _reduce(PARTS, OUT, SIZE: tl.constexpr, SPLIT: tl.constexpr,
            B: tl.constexpr = 256):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    acc = tl.load(PARTS + idx, idx < SIZE, 0)
    for part in range(1, SPLIT):
        acc += tl.load(PARTS + part * SIZE + idx, idx < SIZE, 0)
    tl.store(OUT + idx, acc.to(tl.bfloat16).to(tl.float32), idx < SIZE)


@triton.jit
def _simt(X, W, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          XM: tl.constexpr, WN: tl.constexpr, BN: tl.constexpr,
          BK: tl.constexpr, SPLIT: tl.constexpr):
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    part = tl.program_id(1)
    ks = tl.arange(0, BK)
    chunk: tl.constexpr = triton.cdiv(K, SPLIT * BK) * BK
    # Each CTA reads a weight tile once for all decode rows. Static token loops
    # let the compiler reuse loaded weights instead of issuing M independent GEMVs.
    accs = tl.full((M, BN, BK), 0, tl.float32)
    for start in range(part * chunk, tl.minimum((part + 1) * chunk, K), BK):
        w = tl.load(W + ns[:, None] * WN + start + ks[None, :],
                    (ns[:, None] < N) & (start + ks[None, :] < K), 0).to(tl.float32)
        ms = tl.arange(0, triton.next_power_of_2(M))
        # M is power-of-two here; inactive rows are masked by the wrapper.
        x = tl.load(X + ms[:, None] * XM + start + ks[None, :],
                    (ms[:, None] < M) & (start + ks[None, :] < K), 0).to(tl.float32)
        accs = tl.fma(x[:, None, :], w[None, :, :], accs)
    acc = tl.sum(accs, 2)
    ms = tl.arange(0, M)
    if SPLIT == 1:
        value = acc.to(tl.bfloat16).to(tl.float32)
    else:
        value = acc
    tl.store(OUT + (part * M + ms[:, None]) * N + ns[None, :], value,
             ns[None, :] < N)


def project(x, weight, *, backend='tc', bn=64, bk=128, split=1,
            warps=4, stages=3):
    """BF16 [1..16,K] x BF16 [N,K] -> BF16-rounded FP32 [M,N]."""
    if backend in ('packedtc', 'packed-remat', 'packed-word'):
        from head_packed import PackedHead, project as packed_project
        if backend == 'packed-remat':
            from head_packed_remat import project as packed_project
        elif backend == 'packed-word':
            from head_packed_word import project as packed_project
        if id(weight) not in _packed:
            # Experimental driver cache only. A serving integration should
            # retain the PackedHead instead of this original matrix reference.
            _packed[id(weight)] = (weight, PackedHead(weight))
        return packed_project(x, _packed[id(weight)][1], bn=bn, bk=bk,
                              split=split, warps=warps, stages=stages)
    if (x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]
            or not 0 < x.shape[0] <= 16):
        raise ValueError('expected decode activations and a matching vocabulary shard')
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError('native head requires BF16 input and checkpoint weight')
    if x.stride(1) != 1 or weight.stride(1) != 1 or x.device != weight.device:
        raise ValueError('head matrices require unit K stride on the same device')
    if backend not in ('tc', 'simt') or split not in (1, 2, 4, 8):
        raise ValueError('unsupported experimental head backend or split')
    m, k = x.shape
    n = weight.shape[0]
    if backend == 'simt' and m & (m - 1):
        # Preserve fixed row-independent arithmetic for the tensor-core path;
        # SIMT screens only need power-of-two input counts.
        padded = torch.zeros((triton.next_power_of_2(m), k), device=x.device, dtype=x.dtype)
        padded[:m].copy_(x)
        return project(padded, weight, backend=backend, bn=bn, bk=bk,
                       split=split, warps=warps, stages=stages)[:m]
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    partials = out if split == 1 else torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    kernel = _tensor_core if backend == 'tc' else _simt
    kernel[(triton.cdiv(n, bn), split)](x, weight, partials, m, n, k,
        x.stride(0), weight.stride(0), BN=bn, BK=bk, SPLIT=split,
        num_warps=warps, num_stages=stages)
    if split != 1:
        _reduce[(triton.cdiv(m*n, 256),)](partials, out, m*n, SPLIT=split,
                                        num_warps=4, enable_fp_fusion=False)
    return out
