"""Decode attention GEMMs with tensor cores and the existing FP32 softmax.

Unlike decode_attn's online softmax, this keeps the complete score matrix and
normalizes before PV. Q/K are BF16 values (possibly stored in FP32 by LeanOps).
PV uses TF32x3: probabilities stay FP32, without the older two-BF16-term truncation.
Accumulation order still differs from cuBLAS SIMT; this is an experimental
arithmetic path, not a bit-exact replacement or a serving default.

DSV41_ATTN_STAGED=1 uses FP32 key staging; =2 retains the same key values in
BF16 storage. Both keep FP32 probabilities/outputs and native projection weights.
The BF16X3, split-PV and direct-gather variants below are microbenchmark-only
negative results, retained to prevent repeating the same experiments.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _qk(Q, K, S, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
        QT: tl.constexpr, QH: tl.constexpr, KT: tl.constexpr, KN: tl.constexpr,
        BH: tl.constexpr = 16, BN: tl.constexpr = 32, BK: tl.constexpr = 64):
    h = tl.program_id(0) * BH + tl.arange(0, BH)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    t = tl.program_id(2)
    k = tl.arange(0, BK)
    acc = tl.zeros((BH, BN), tl.float32)
    for start in range(0, D, BK):
        q = tl.load(Q + t * QT + h[:, None] * QH + start + k[None, :],
                    h[:, None] < H, 0).to(tl.bfloat16)
        kv = tl.load(K + t * KT + n[:, None] * KN + start + k[None, :],
                     n[:, None] < N, 0).to(tl.bfloat16)
        acc = tl.dot(q, tl.trans(kv), acc)
    tl.store(S + (t * H + h[:, None]) * N + n[None, :], acc,
             (h[:, None] < H) & (n[None, :] < N))


@triton.jit
def _pv(P, K, O, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
        KT: tl.constexpr, KN: tl.constexpr,
        BH: tl.constexpr = 16, BD: tl.constexpr = 32, BK: tl.constexpr = 32,
        BF16X3: tl.constexpr = False, SPLIT: tl.constexpr = 1):
    h = tl.program_id(0) * BH + tl.arange(0, BH)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    t = tl.program_id(2) // SPLIT
    part = tl.program_id(2) % SPLIT
    n = tl.arange(0, BK)
    acc = tl.zeros((BH, BD), tl.float32)
    lo = tl.zeros((BH, BD), tl.float32)
    tail = tl.zeros((BH, BD), tl.float32)
    chunk: tl.constexpr = triton.cdiv(N, BK * SPLIT) * BK
    for start in range(part * chunk, tl.minimum((part + 1) * chunk, N), BK):
        p = tl.load(P + (t * H + h[:, None]) * N + start + n[None, :],
                    (h[:, None] < H) & (start + n[None, :] < N), 0)
        kv = tl.load(K + t * KT + (start + n[:, None]) * KN + d[None, :],
                     (start + n[:, None] < N) & (d[None, :] < D), 0).to(tl.float32)
        if BF16X3:
            # Rejected: no speed gain and 3.72e-6 relative error vs FP64 at
            # 3,200 keys, versus 6.96e-7 for TF32x3. Not used by serving.
            hi = p.to(tl.bfloat16)
            rem = p - hi.to(tl.float32)
            low = rem.to(tl.bfloat16)
            last = (rem - low.to(tl.float32)).to(tl.bfloat16)
            kb = kv.to(tl.bfloat16)
            acc = tl.dot(hi, kb, acc)
            lo = tl.dot(low, kb, lo)
            tail = tl.dot(last, kb, tail)
        else:
            acc = tl.dot(p, kv, acc, input_precision="tf32x3")
    if BF16X3:
        acc = (acc + lo) + tail
    tl.store(O + ((t * SPLIT + part) * H + h[:, None]) * D + d[None, :], acc,
             (h[:, None] < H) & (d[None, :] < D))


@triton.jit
def _pv_sum(P, O, HD: tl.constexpr, SPLIT: tl.constexpr, B: tl.constexpr = 256):
    i = tl.program_id(0) * B + tl.arange(0, B)
    t = tl.program_id(1)
    acc = tl.load(P + t * SPLIT * HD + i, i < HD, 0)
    for s in range(1, SPLIT):
        acc = acc + tl.load(P + (t * SPLIT + s) * HD + i, i < HD, 0)
    tl.store(O + t * HD + i, acc, i < HD)


def scores(q, kv):
    t, h, d = q.shape
    n = kv.shape[1]
    if kv.shape != (t,n,d) or q.dtype not in (torch.float32,torch.bfloat16) or kv.dtype not in (torch.float32,torch.bfloat16):
        raise ValueError("Q/K must be BF16 values with matching token/head dimensions")
    if d % 64 or q.stride(-1) != 1 or kv.stride(-1) != 1:
        raise ValueError("staged attention needs aligned, unit-stride head dimensions")
    out = torch.empty((t, h, n), device=q.device, dtype=torch.float32)
    _qk[(triton.cdiv(h, 16), triton.cdiv(n, 32), t)](
        q, kv, out, h, n, d, q.stride(0), q.stride(1), kv.stride(0), kv.stride(1),
        num_warps=4, num_stages=2)
    return out


def values(p, kv, bf16x3=False, split=1):
    t, h, n = p.shape
    d = kv.shape[-1]
    if kv.shape[:2] != (t,n) or kv.stride(-1) != 1 or split not in (1,4):
        raise ValueError("PV needs matching keys and split 1 or 4")
    if not p.is_contiguous() or p.dtype != torch.float32:
        raise ValueError("staged attention needs contiguous FP32 probabilities")
    out = torch.empty((t, h, d), device=p.device, dtype=torch.float32)
    parts = out if split == 1 else torch.empty((t,split,h,d),device=p.device,dtype=torch.float32)
    _pv[(triton.cdiv(h, 16), triton.cdiv(d, 32), t * split)](
        p, kv, parts, h, n, d, kv.stride(0), kv.stride(1), BF16X3=bf16x3, SPLIT=split,
        num_warps=4, num_stages=2)
    if split != 1:
        _pv_sum[(triton.cdiv(h*d,256),t)](parts,out,h*d,split)
    return out


def attention(q, kv, mask, sink, scale, lean, bf16x3=False, split=1):
    return values(lean.attn_probs(scores(q, kv), mask, sink, scale), kv, bf16x3, split)


@triton.jit
def _key_tile(Ring, Slots, Cache, Ids, t, rows, cols,
              N1: tl.constexpr, N2: tl.constexpr, D: tl.constexpr,
              CS: tl.constexpr):
    """Gather BF16 window / packed cache keys directly into a tensor-core tile.

    Match packed_kv._gather, including the BF16 rounding and signed zero. No
    context-sized FP32 key block is written and read back from device memory.
    """
    win = rows < N1
    src = tl.load(Slots + t * N1 + rows, win, 0)
    value = tl.load(Ring + src[:, None] * D + cols[None, :],
                    win[:, None] & (cols[None, :] < D), 0)
    if N2 > 0:
        valid = (rows >= N1) & (rows < N1 + N2)
        idx = tl.load(Ids + t * N2 + rows - N1, valid, 0)
        group = cols // 16
        word = tl.load(Cache + idx[:, None] * CS + group[None,:],
                       valid[:,None] & (cols[None,:] < D), 0).to(tl.uint64)
        code = ((word >> ((cols[None,:] % 16) * 4)) & 15).to(tl.int32)
        mag = code & 7
        v = tl.where(mag < 4, mag * .5,
                     tl.where(mag < 6, mag - 2., 2. * mag - 8.))
        sw = tl.load(Cache + idx[:,None] * CS + D // 16 + group[None,:] // 8,
                     valid[:,None] & (cols[None,:] < D), 0).to(tl.uint64)
        sb = ((sw >> ((group[None,:] % 8) * 8)) & 255).to(tl.uint8)
        scale = sb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        bits = (v * scale).to(tl.bfloat16).to(tl.uint16, bitcast=True)
        bits = bits | ((code >= 8).to(tl.uint16) << 15)
        value = tl.where(valid[:,None], bits.to(tl.bfloat16,bitcast=True), value)
    return value


@triton.jit
def _qk_direct(Q, Ring, Slots, Cache, Ids, S,
               H: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr, D: tl.constexpr,
               QT: tl.constexpr, QH: tl.constexpr, CS: tl.constexpr):
    h = tl.program_id(0) * 16 + tl.arange(0,16)
    n = tl.program_id(1) * 32 + tl.arange(0,32)
    t = tl.program_id(2)
    d = tl.arange(0,64)
    acc = tl.zeros((16,32),tl.float32)
    for start in range(0,D,64):
        q = tl.load(Q + t * QT + h[:,None] * QH + start + d[None,:],
                    h[:,None] < H, 0).to(tl.bfloat16)
        kv = _key_tile(Ring,Slots,Cache,Ids,t,n,start+d,N1,N2,D,CS)
        acc = tl.dot(q,tl.trans(kv),acc)
    tl.store(S + (t * H + h[:,None]) * (N1+N2) + n[None,:],acc,
             (h[:,None] < H) & (n[None,:] < N1+N2))


@triton.jit
def _pv_direct(P, Ring, Slots, Cache, Ids, O,
               H: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr, D: tl.constexpr,
               CS: tl.constexpr):
    h = tl.program_id(0) * 16 + tl.arange(0,16)
    d = tl.program_id(1) * 32 + tl.arange(0,32)
    t = tl.program_id(2)
    n = tl.arange(0,32)
    acc = tl.zeros((16,32),tl.float32)
    for start in range(0,N1+N2,32):
        p = tl.load(P + (t * H + h[:,None]) * (N1+N2) + start + n[None,:],
                    (h[:,None] < H) & (start+n[None,:] < N1+N2), 0)
        kv = _key_tile(Ring,Slots,Cache,Ids,t,start+n,d,N1,N2,D,CS).to(tl.float32)
        acc = tl.dot(p,kv,acc,input_precision="tf32x3")
    tl.store(O + (t * H + h[:,None]) * D + d[None,:],acc,
             (h[:,None] < H) & (d[None,:] < D))


def attention_direct(q, ring, slots, cache, ids, mask, sink, scale, lean):
    """Microbenchmark-only direct gather, slower for compressed attention.

    October 7: 236 us vs 47.7 us BF16 staging at T=4, H=32, N=1152.
    Repeated tile-local unpacking dominates the saved scratch traffic. Keeping
    the same mathematical decomposition does not ensure identical FP32 bits:
    BF16-typed keys also change the compiler's TF32x3 lowering slightly.
    """
    t,h,d = q.shape
    n1 = slots.shape[1]
    n2 = 0 if ids is None else ids.shape[1]
    if not ring.is_contiguous() or ring.dtype != torch.bfloat16 or d%64:
        raise ValueError("direct attention requires the contiguous BF16 window")
    if slots.shape[0] != t or not slots.is_contiguous():
        raise ValueError("direct attention requires contiguous per-token window indices")
    if n2 and (cache.dtype != torch.int64 or not cache.is_contiguous() or not ids.is_contiguous()):
        raise ValueError("direct attention requires packed cache and contiguous indices")
    cs = cache.shape[1] if n2 else 0
    cache = cache if n2 else ring
    ids = ids if n2 else slots
    s = torch.empty((t,h,n1+n2),device=q.device,dtype=torch.float32)
    out = torch.empty((t,h,d),device=q.device,dtype=torch.float32)
    _qk_direct[(triton.cdiv(h,16),triton.cdiv(n1+n2,32),t)](
        q,ring,slots,cache,ids,s,h,n1,n2,d,q.stride(0),q.stride(1),cs,
        num_warps=4,num_stages=2)
    p = lean.attn_probs(s,mask,sink,scale)
    _pv_direct[(triton.cdiv(h,16),triton.cdiv(d,32),t)](
        p,ring,slots,cache,ids,out,h,n1,n2,d,cs,num_warps=4,num_stages=2)
    return out
