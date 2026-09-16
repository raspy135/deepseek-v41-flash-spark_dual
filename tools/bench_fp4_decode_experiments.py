"""Benchmark-only alternatives; never imported by serving or included in cache identity."""
import torch
import triton
import triton.language as tl
from fp4_moe import _quad_dot


@triton.jit
def _separate_up(x, w1, s1, w3, s3, tmp, block_slot, block_pair,
                 SX: tl.constexpr, TOPK: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 P: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                 NULL_SLOT: tl.constexpr):
    mb, nb, projection = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(block_slot + mb)
    if slot < 0 or slot == NULL_SLOT:
        return
    slot = slot.to(tl.int64)
    pairs = tl.load(block_pair + mb * BM + tl.arange(0, BM))
    valid = pairs >= 0
    pairs = tl.where(valid, pairs, 0)
    ns = nb * BN + tl.arange(0, BN)
    wp = tl.where(projection == 0, w1, w3)
    sp = tl.where(projection == 0, s1, s3)
    wt = wp + slot * (N * (K // 2)) + ns[:, None] * (K // 2) + tl.arange(0, 64)[None, :]
    st = sp + slot * (N * (K // 32)) + ns[:, None] * (K // 32) + tl.arange(0, 4)[None, :]
    xb = x + (pairs // TOPK).to(tl.int64)[:, None] * SX
    xk = 2 * tl.arange(0, 16)[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for q in range(K // 128):
        acc += _quad_dot(xb + q * 128, xk, valid[:, None], wt + q * 64, st + q * 4, BN)
    tl.store(tmp + projection * P * N + pairs[:, None] * N + ns[None, :],
             acc.to(tl.bfloat16), valid[:, None])


@triton.jit
def _activate(tmp, weights, out, SIZE: tl.constexpr, N: tl.constexpr,
              LIMIT: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    g = tl.load(tmp + i, i < SIZE, 0).to(tl.float32)
    u = tl.load(tmp + SIZE + i, i < SIZE, 0).to(tl.float32)
    g = tl.minimum(g, LIMIT)
    u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
    w = tl.load(weights + i // N, i < SIZE, 0)
    h = g * tl.sigmoid(g) * u * w
    tl.store(out + i, h.to(tl.bfloat16), i < SIZE)


class SeparateUp:
    """Same launcher interface as _moe_up_kernel, only for non-null TP benchmarks."""
    def __getitem__(self, grid):
        def launch(x, w1, s1, w3, s3, h, weights, slots, pairs, sx, sh, limit, **kw):
            assert not kw['SCALED'] and kw['NULL_SLOT'] == -1
            n = kw['N']
            p = h.shape[0]
            tmp = torch.empty((2, p, n), dtype=torch.bfloat16, device=x.device)
            _separate_up[(*grid, 2)](
                x, w1, s1, w3, s3, tmp, slots, pairs, SX=sx, P=p,
                **{k: v for k, v in kw.items() if k != 'SCALED'})
            _activate[(triton.cdiv(p*n, 512),)](tmp, weights, h, p*n, n, limit, 512)
        return launch


@triton.jit
def _route_pairs(slots, block_slots, pairs, P: tl.constexpr, BM: tl.constexpr, B: tl.constexpr):
    """One block per input pair; only its expert's first occurrence is active."""
    p = tl.program_id(0)
    i = tl.arange(0, B)
    target = tl.load(slots + p)
    values = tl.load(slots + i, i < P, 2147483647)
    same = (i < P) & (values == target)
    leader = tl.min(tl.where(same, i, P), 0)
    tl.store(block_slots + p, tl.where(leader == p, target, -1))
    if leader == p:
        ordered = tl.sort(tl.where(same, i, P), descending=False)
        tl.store(pairs + p * BM + i, tl.where(ordered < P, ordered, -1), i < BM)
    else:
        tl.store(pairs + p * BM + i, -1, i < BM)


def fused_routing(slots, bm):
    p = slots.numel()
    assert p <= 64 and slots.dtype == torch.int32 and slots.is_contiguous()
    block_slots = torch.empty((p,), dtype=torch.int32, device=slots.device)
    pairs = torch.empty((p * bm,), dtype=torch.int32, device=slots.device)
    _route_pairs[(p,)](slots, block_slots, pairs, p, bm, max(triton.next_power_of_2(p), bm),
                       num_warps=4)
    return block_slots, pairs, p


def narrow_fp8_linear(x, weight, out_dtype=torch.bfloat16):
    """Scheduling-only candidate for measured narrow projections; baseline otherwise."""
    from fp8_linear import fp8_linear, _fp8_linear_kernel
    if weight.K != 5120 or weight.N not in (512, 1280, 2304) or x.numel() // weight.K > 16:
        return fp8_linear(x, weight, out_dtype)
    x2 = x.reshape(-1, weight.K).to(torch.bfloat16).contiguous()
    m = x2.shape[0]
    y = torch.empty((m, weight.N), dtype=out_dtype, device=x.device)
    _fp8_linear_kernel[(triton.cdiv(weight.N, 64), 1)](
        x2, weight.w, weight.s, y, m, weight.N, weight.K,
        x2.stride(0), weight.w.stride(0), weight.s.stride(0), y.stride(0),
        BLOCK_M=16, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=2)
    return y.view(*x.shape[:-1], weight.N)
