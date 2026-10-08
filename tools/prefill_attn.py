"""
prefill_attn.py -- the prefill attention of tools/decode_attn.py, reading its keys where they live.

decode_attention takes [T, n, d] key tensors, so prefill used to materialise two of them per layer:
the 128 window rows of every query out of the layer's ring (`ring[wpos % RING]`) and the 512
compressed rows the indexer selected (`packed_kv.gather`, which also expands the FP4 cache to
BF16). At a 2,048-row chunk the second one is [2048, 512, 512] BF16 = 1 GB written and read back
per layer -- 7.7 ms of `_gather` per call, ~0.9 s of an 8K prefill, before attention reads it again.

This kernel takes the ring, the window positions, the cache and the selected indices instead, and
loads each key tile by index: window rows straight from the ring, compressed rows straight from
the cache, dequantised in registers with packed_kv._gather's exact formula (E2M1 code x E4M3 group
scale, rounded once to BF16, sign set afterwards). The values reaching `tl.dot` are therefore the
same BF16 values the gathered tensors held, the key tiles keep decode_attention's boundaries
(window [0, NW), compressed [NW, NW + NC), BLOCK_N at a time) and the online softmax is the same
code, so the output is bit-identical to gather + decode_attention(split=1). Masked rows still load
row `max(idx, 0)` / `max(pos, 0) % RING`, as the gathers did.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_NEG = tl.constexpr(-1e30)


@triton.jit
def _keys_ring(RING, WPOS, offs_n, nm, stride_ring, RING_N, d):
    pos = tl.load(WPOS + offs_n, mask=nm, other=0)
    row = (tl.maximum(pos, 0) % RING_N).to(tl.int64)
    return tl.load(RING + row[:, None] * stride_ring + d[None, :], mask=nm[:, None], other=0.0)


@triton.jit
def _keys_cache(CK, CIDX, offs_c, nm, stride_ck, D: tl.constexpr, BLOCK_N: tl.constexpr,
                PACKED: tl.constexpr):
    idx = tl.maximum(tl.load(CIDX + offs_c, mask=nm, other=0), 0).to(tl.int64)
    if PACKED:
        # packed_kv layout: D // 16 code words (16 x 4-bit E2M1 codes each), then D // 128 words of
        # eight E4M3 group scales. Element w * 16 + j is nibble j of word w, scaled by byte w % 8
        # of scale word w // 8.
        G: tl.constexpr = D // 16
        w = tl.arange(0, G)
        word = tl.load(CK + idx[:, None] * stride_ck + w[None, :], mask=nm[:, None], other=0).to(tl.uint64)
        j = tl.arange(0, 16)
        code = ((word[:, :, None] >> (j[None, None, :] * 4)) & 15).to(tl.int32)
        mag = code & 7
        value = tl.where(mag < 2, mag * .5,
                         tl.where(mag < 4, mag * .5,
                                  tl.where(mag < 6, mag - 2., 2. * mag - 8.)))
        sw = tl.load(CK + idx[:, None] * stride_ck + G + w[None, :] // 8, mask=nm[:, None], other=0).to(tl.uint64)
        sb = ((sw >> ((w[None, :] % 8) * 8)) & 255).to(tl.uint8)
        scale = sb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        bits = (value * scale[:, :, None]).to(tl.bfloat16).to(tl.uint16, bitcast=True)
        bits = bits | ((code >= 8).to(tl.uint16) << 15)
        k = tl.reshape(bits.to(tl.bfloat16, bitcast=True), (BLOCK_N, D))
        return tl.where(nm[:, None], k, 0.0)
    else:
        d = tl.arange(0, D)
        return tl.load(CK + idx[:, None] * stride_ck + d[None, :], mask=nm[:, None], other=0.0)


@triton.jit
def _step(q, k, s_mask, m_i, l_i, acc, scale, PV_SPLIT: tl.constexpr):
    s = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
    s = s * scale
    s = tl.where(s_mask[None, :], s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    m_new = tl.maximum(m_new, _NEG)
    alpha = tl.exp(m_i - m_new)
    p = tl.exp(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None]
    ph = p.to(tl.bfloat16)
    acc += tl.dot(ph, k, out_dtype=tl.float32)
    if PV_SPLIT:
        pl = (p - ph.to(tl.float32)).to(tl.bfloat16)
        acc += tl.dot(pl, k, out_dtype=tl.float32)
    return m_new, l_i, acc


@triton.jit
def _pattn_kernel(Q, RING, WPOS, CK, CIDX, MSK, SINK, OUT,
                  H, NW, NC, RING_N,
                  stride_qt, stride_qh, stride_ring, stride_wt, stride_ck, stride_ct, stride_mt,
                  stride_ot, stride_oh, scale,
                  D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
                  PV_SPLIT: tl.constexpr, PACKED: tl.constexpr, HAS_C: tl.constexpr):
    pid_h = tl.program_id(0)
    t = tl.program_id(1).to(tl.int64)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = offs_h < H
    d = tl.arange(0, D)
    q = tl.load(Q + t * stride_qt + offs_h[:, None] * stride_qh + d[None, :], mask=h_mask[:, None], other=0.0)
    m_i = tl.full((BLOCK_H,), _NEG, tl.float32)
    l_i = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, D), tl.float32)
    mb = MSK + t * stride_mt
    for n0 in range(0, NW, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nm = offs_n < NW
        k = _keys_ring(RING, WPOS + t * stride_wt, offs_n, nm, stride_ring, RING_N, d)
        keep = (tl.load(mb + offs_n, mask=nm, other=0) != 0) & nm
        m_i, l_i, acc = _step(q, k, keep, m_i, l_i, acc, scale, PV_SPLIT)
    if HAS_C:
        for c0 in range(0, NC, BLOCK_N):
            offs_c = c0 + tl.arange(0, BLOCK_N)
            nm = offs_c < NC
            k = _keys_cache(CK, CIDX + t * stride_ct, offs_c, nm, stride_ck, D, BLOCK_N, PACKED)
            keep = (tl.load(mb + NW + offs_c, mask=nm, other=0) != 0) & nm
            m_i, l_i, acc = _step(q, k, keep, m_i, l_i, acc, scale, PV_SPLIT)
    sink = tl.load(SINK + offs_h, mask=h_mask, other=0.0).to(tl.float32)
    denom = l_i + tl.exp(sink - m_i)
    ob = OUT + t * stride_ot + offs_h[:, None] * stride_oh
    tl.store(ob + d[None, :], (acc / denom[:, None]).to(tl.bfloat16), mask=h_mask[:, None])


def prefill_attention_indexed(q: torch.Tensor, ring: torch.Tensor, wpos: torch.Tensor,
                              cache: torch.Tensor | None, cidx: torch.Tensor | None,
                              mask: torch.Tensor, sink: torch.Tensor, scale: float,
                              block_h: int = 16, block_n: int = 32, pv_split: int = 1,
                              num_warps: int = 4, num_stages: int = 2) -> torch.Tensor:
    """q bf16 [T, H, D]; ring bf16 [RING, D]; wpos int64 [T, NW] (window positions, -1 = none);
    cache: packed int64 [n, D*9/128 // 8] or bf16 [n, D]; cidx int64 [T, NC] (-1 = none);
    mask bool [T, NW + NC] -> o bf16 [T, H, D].

    Same output bits as decode_attention(q, ring[wpos.clamp_min(0) % RING],
    packed_kv.gather(cache, cidx.clamp_min(0)), mask, sink, scale, split=1)."""
    T, H, D = q.shape
    NW = wpos.shape[1]
    NC = 0 if cidx is None else cidx.shape[1]
    assert D & (D - 1) == 0 and D % 128 == 0, D
    assert mask.shape == (T, NW + NC) and q.stride(-1) == 1 and ring.stride(-1) == 1
    assert q.dtype == torch.bfloat16 and ring.dtype == torch.bfloat16 and ring.shape[1] == D
    assert block_h in (1, 2, 4, 8, 16, 32) and block_h <= H
    assert NW % block_n == 0, 'window tiles must keep decode_attention boundaries'
    packed = cache is not None and cache.dtype == torch.int64
    if cache is not None:
        assert cache.stride(-1) == 1
        assert packed or (cache.dtype == torch.bfloat16 and cache.shape[1] == D)
        if packed:
            assert cache.shape[1] == D // 16 + D // 128, cache.shape
    wpos = wpos.contiguous()
    msk = mask.contiguous()
    msk = msk if msk.dtype == torch.int8 else msk.view(torch.int8)
    c = cache if cache is not None else ring
    ci = cidx.contiguous() if cidx is not None else wpos
    o = torch.empty(T, H, D, dtype=torch.bfloat16, device=q.device)
    if T == 0:
        return o
    grid = (triton.cdiv(H, block_h), T)
    _pattn_kernel[grid](q, ring, wpos, c, ci, msk, sink, o,
                        H, NW, NC, ring.shape[0],
                        q.stride(0), q.stride(1), ring.stride(0), wpos.stride(0), c.stride(0), ci.stride(0),
                        msk.stride(0), o.stride(0), o.stride(1), scale,
                        D=D, BLOCK_H=block_h, BLOCK_N=block_n, PV_SPLIT=pv_split, PACKED=packed,
                        HAS_C=NC > 0, num_warps=num_warps, num_stages=num_stages)
    return o
