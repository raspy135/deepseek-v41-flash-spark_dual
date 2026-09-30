"""Decode-sized spellings of rmsnorm and the hyper-connection coefficients with fewer launches.

Both are meant to be bit-identical to the torch spellings they replace (v41_ref.rmsnorm and
Model._hc_mixes at M <= 16). What sets their bits are two row-count-dependent torch ops, and those
stay torch ops on the same shapes:

  * the RMS reduction `t.square().mean(-1)` over a [MM_TILE, D] fp32 tensor (v41_ref.tiled_rows),
  * the HC projection, one cuBLAS fp32 GEMM over [HC_MM_TILE, 20480] rows (v41_ref.mm).

What goes is everything around them. The torch spelling re-creates the zero padding on every call
(`new_zeros` + `cat`), slices the result back out (`cat` + copy), converts the weight to fp32, and
runs `+ eps`, `rsqrt` and two or three multiplies as separate kernels: 12 launches per rmsnorm and
15 per HC site (18 with the three copies into the decoder's static buffers). Here the padding lives
in a static buffer whose tail rows are never written, and the elementwise tail is one Triton kernel:
4 launches per rmsnorm, 6 per HC site.

The Triton tail reproduces the torch elementwise ops one for one: an fp32 add of eps, `rsqrt`, and
IEEE multiplies in the same order, compiled without FMA contraction. tools/test_decode_lean.py
checks torch.equal against the torch spellings on real shapes, in and out of CUDA graphs.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _rms_apply_kernel(X, MEAN, W, OUT, D, EPS, BD: tl.constexpr, OUT_BF16: tl.constexpr):
    """out[t, :] = (w * (x[t, :] * rsqrt(mean[t] + eps))) as v41_ref.rmsnorm computes it."""
    t = tl.program_id(0)
    db = tl.program_id(1) * BD + tl.arange(0, BD)
    m = db < D
    rs = tl.math.rsqrt(tl.load(MEAN + t) + EPS)
    x = tl.load(X + t * D + db, mask=m, other=0.0)
    y = tl.load(W + db, mask=m, other=0.0) * (x * rs)
    if OUT_BF16:
        tl.store(OUT + t * D + db, y.to(tl.bfloat16), mask=m)
    else:
        tl.store(OUT + t * D + db, y, mask=m)


@triton.jit
def _hc_rms_kernel(MM, MEAN, S, B, PRE, POST, COMB, EPS_RMS,
                   iters: tl.constexpr, eps: tl.constexpr, HC: tl.constexpr):
    """engine/hc_sinkhorn._hc_kernel with its input formed in-kernel: mixes = mm * rsqrt(mean + eps).
    The body below the first three lines is that kernel's, unchanged."""
    row = tl.program_id(0)
    MIX: tl.constexpr = (2 + HC) * HC
    rs = tl.math.rsqrt(tl.load(MEAN + row) + EPS_RMS)
    j = tl.arange(0, HC)
    s0 = tl.load(S + 0)
    s1 = tl.load(S + 1)
    s2 = tl.load(S + 2)
    m_pre = tl.load(MM + row * MIX + j) * rs
    b_pre = tl.load(B + j)
    pre = tl.sigmoid(m_pre * s0 + b_pre) + eps
    tl.store(PRE + row * HC + j, pre)
    m_post = tl.load(MM + row * MIX + HC + j) * rs
    b_post = tl.load(B + HC + j)
    post = 2.0 * tl.sigmoid(m_post * s1 + b_post)
    tl.store(POST + row * HC + j, post)
    r = tl.arange(0, HC)[:, None]
    c = tl.arange(0, HC)[None, :]
    off = 2 * HC + r * HC + c
    comb = (tl.load(MM + row * MIX + off) * rs) * s2 + tl.load(B + off)
    mx = tl.max(comb, axis=1)[:, None]
    e = tl.exp(comb - mx)
    comb = e / tl.sum(e, axis=1)[:, None] + eps
    comb = comb / (tl.sum(comb, axis=0)[None, :] + eps)
    for _ in range(iters - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + eps)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + eps)
    tl.store(COMB + row * HC * HC + r * HC + c, comb)


@triton.jit
def _swiglu_kernel(GU, H, N, LIMIT, BN: tl.constexpr, CLAMP: tl.constexpr):
    """h[t, :] = bf16(silu(min(gate, L)) * clamp(up, -L, L)), gate = gu[t, :N], up = gu[t, N:2N], as
    v41_ref.expert_ffn computes it: fp32 from bf16, torch's silu x / (1 + exp(-x)) with libdevice's
    exp and an IEEE division (Triton's `/` and tl.exp are approximate), then one fp32 multiply."""
    t = tl.program_id(0)
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    m = j < N
    g = tl.load(GU + t * 2 * N + j, mask=m, other=0.0).to(tl.float32)
    u = tl.load(GU + t * 2 * N + N + j, mask=m, other=0.0).to(tl.float32)
    if CLAMP:
        u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
        g = tl.minimum(g, LIMIT)
    s = libdevice.div_rn(g, 1.0 + libdevice.exp(-g))
    tl.store(H + t * N + j, (s * u).to(tl.bfloat16), mask=m)


@triton.jit
def _rope_kernel(X, F, OUT, H, D: tl.constexpr, RD: tl.constexpr, BD: tl.constexpr,
                 CONJ: tl.constexpr, RE: tl.constexpr, IM: tl.constexpr):
    """One row of cat([x[:D-RD], rotate(x[D-RD:])]) as v41_ref.apply_rotary makes it: fp32 pairs
    times the complex factor, rounded to bf16. RE / IM pick the multiply's FMA contraction
    (0: none, 1: fma on the first product, 2: fma on the second) -- whichever reproduces torch's
    c10::complex<float> operator* as nvcc compiled it (tools/test_decode_lean.py finds it)."""
    row = tl.program_id(0)
    t = row // H
    d = tl.arange(0, BD)
    keep = d < D - RD
    x = tl.load(X + row * D + d, mask=keep, other=0.0)
    tl.store(OUT + row * D + d, x, mask=keep)
    i = tl.arange(0, RD // 2)
    a = tl.load(X + row * D + (D - RD) + 2 * i).to(tl.float32)
    b = tl.load(X + row * D + (D - RD) + 2 * i + 1).to(tl.float32)
    c = tl.load(F + t * RD + 2 * i)
    dd = tl.load(F + t * RD + 2 * i + 1)
    if CONJ:
        dd = -dd
    if RE == 0:
        re = a * c - b * dd
    elif RE == 1:
        re = tl.fma(a, c, -(b * dd))
    else:
        re = tl.fma(-b, dd, a * c)
    if IM == 0:
        im = a * dd + b * c
    elif IM == 1:
        im = tl.fma(a, dd, b * c)
    else:
        im = tl.fma(b, c, a * dd)
    tl.store(OUT + row * D + (D - RD) + 2 * i, re.to(tl.bfloat16))
    tl.store(OUT + row * D + (D - RD) + 2 * i + 1, im.to(tl.bfloat16))


@triton.jit
def _router_pre_kernel(G, BIAS, KEEP, SCORES, LOGITS, MASKED, E, HAS_KEEP: tl.constexpr, BE: tl.constexpr):
    """scores = sqrt(softplus(g)), logits = scores + bias, masked = where(keep, logits, -inf), as
    Model's router computes them: torch's softplus is `x > 20 ? x : log1p(exp(x))` (beta 1) with
    libdevice's log1p/exp, and its sqrt is the IEEE one."""
    t = tl.program_id(0)
    e = tl.arange(0, BE)
    m = e < E
    g = tl.load(G + t * E + e, mask=m, other=0.0)
    sp = tl.where(g > 20.0, g, libdevice.log1p(libdevice.exp(g)))
    sc = libdevice.sqrt_rn(sp)
    lg = sc + tl.load(BIAS + e, mask=m, other=0.0)
    tl.store(SCORES + t * E + e, sc, mask=m)
    tl.store(LOGITS + t * E + e, lg, mask=m)
    if HAS_KEEP:
        keep = tl.load(KEEP + e, mask=m, other=0) != 0
        lg = tl.where(keep, lg, float("-inf"))
    tl.store(MASKED + t * E + e, lg, mask=m)


@triton.jit
def _router_post_kernel(WTS, SUM, OUT, SCALE, K: tl.constexpr, KB: tl.constexpr):
    """route_w = wts / (sum + 1e-20) * route_scale, one IEEE division and one multiply."""
    t = tl.program_id(0)
    k = tl.arange(0, KB)
    m = k < K
    w = tl.load(WTS + t * K + k, mask=m, other=0.0)
    s = tl.load(SUM + t) + 1e-20
    tl.store(OUT + t * K + k, libdevice.div_rn(w, s) * SCALE, mask=m)


@triton.jit
def _softmax_pre_kernel(S, MASK, P, MX, H, N, s_t, s_h, s_n, SCALE, BN: tl.constexpr):
    """x = where(mask, s * scale, -inf); mx = max(amax(x), -1e30); p = exp(x - mx) -- the torch
    decode attention's elementwise ops; amax is exact in any order, exp is libdevice's."""
    row = tl.program_id(0)
    t = row // H
    h = row % H
    n = tl.arange(0, BN)
    m = n < N
    x = tl.load(S + t * s_t + h * s_h + n * s_n, mask=m, other=0.0) * SCALE
    keep = tl.load(MASK + t * N + n, mask=m, other=0) != 0
    x = tl.where(keep, x, float("-inf"))
    mx = tl.maximum(tl.max(tl.where(m, x, float("-inf")), axis=0), -1e30)
    tl.store(P + row * N + n, libdevice.exp(x - mx), mask=m)
    tl.store(MX + row, mx)


@triton.jit
def _softmax_post_kernel(P, PSUM, MX, SINK, OUT, H, N, BN: tl.constexpr):
    """out = p / (psum + exp(sink - mx)), one IEEE division per element."""
    row = tl.program_id(0)
    h = row % H
    n = tl.arange(0, BN)
    m = n < N
    denom = tl.load(PSUM + row) + libdevice.exp(tl.load(SINK + h) - tl.load(MX + row))
    p = tl.load(P + row * N + n, mask=m, other=0.0)
    tl.store(OUT + row * N + n, libdevice.div_rn(p, denom), mask=m)


# The FMA contraction of torch's complex multiply (c10::complex<float> operator* as nvcc compiled
# it in this torch build): re = fma(a, c, -(b*d)), im = fma(b, c, a*d). Measured on 4.2M fp32
# products, 0 mismatches; every other pairing mismatches 24-33 % of them
# (tools/test_decode_lean.py test_rope_fp32_contraction). A torch upgrade must re-run that test.
ROPE_RE, ROPE_IM = 1, 2


class LeanOps:
    """Static buffers plus the two decode ops. One instance per FastDecoder.

    A buffer is keyed by (purpose, width, rows): each verify width, the draft and every norm width
    has its own, so the zero rows past a call's T are never written by anybody. Buffers are
    created on first use, which is the capture warm-up (outside any graph), and then keep their
    addresses for the graphs that bake them in."""

    def __init__(self, device, mm_tile: int, hc_mm_tile: int):
        self.dev = device
        self.mm_tile = int(mm_tile)
        self.hc_mm_tile = int(hc_mm_tile)
        self._bufs: dict = {}
        self._wf: dict = {}

    def _buf(self, key, shape, dtype=torch.float32):
        b = self._bufs.get(key)
        if b is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"decode_lean buffer {key} first requested inside a graph capture")
            b = self._bufs[key] = torch.zeros(shape, dtype=dtype, device=self.dev)
        return b

    def _wf32(self, w):
        """w.float(), made once. A norm weight never changes after load."""
        k = (w.data_ptr(), tuple(w.shape), w.dtype)
        f = self._wf.get(k)
        if f is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("decode_lean fp32 weight copy first requested inside a graph capture")
            f = self._wf[k] = w.detach().float().contiguous()
        return f

    def usable_rms(self, x) -> bool:
        return (self.mm_tile > 0 and x.is_cuda and x.dtype in (torch.bfloat16, torch.float32)
                and 0 < x.numel() // x.shape[-1] < self.mm_tile)

    def rmsnorm(self, x, w, eps: float):
        """v41_ref.rmsnorm(x, w, eps) for fewer than MM_TILE rows: 4 launches instead of 12."""
        D = x.shape[-1]
        T = x.numel() // D
        pad = self._buf(("rms", D, T), (self.mm_tile, D))
        pad[:T].copy_(x.reshape(T, D))
        mean = pad.square().mean(-1, keepdim=True)
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        BD = 1024 if D >= 1024 else triton.next_power_of_2(D)
        _rms_apply_kernel[(T, triton.cdiv(D, BD))](
            pad, mean, self._wf32(w), out, D, float(eps), BD=BD,
            OUT_BF16=x.dtype == torch.bfloat16, num_warps=4, enable_fp_fusion=False, enable_reflect_ftz=False)
        return out

    def swiglu(self, gu, n: int, limit: float):
        """The elementwise middle of v41_ref.expert_ffn with a merged w1||w3 output `gu` [T, 2n]
        bf16: 1 launch instead of 7."""
        T = gu.numel() // gu.shape[-1]
        assert gu.dtype == torch.bfloat16 and gu.shape[-1] == 2 * n and gu.is_contiguous()
        h = torch.empty(*gu.shape[:-1], n, dtype=torch.bfloat16, device=gu.device)
        BN = 1024
        _swiglu_kernel[(T, triton.cdiv(n, BN))](gu, h, n, float(limit), BN=BN, CLAMP=limit > 0,
                                                num_warps=4, enable_fp_fusion=False, enable_reflect_ftz=False)
        return h

    def rope(self, x, fq, rd: int, inverse: bool = False, re: int | None = None, im: int | None = None):
        """torch.cat([x[..., :-rd], v41_ref.apply_rotary(x[..., -rd:], fq, inverse)], -1) for a
        contiguous bf16 x of shape [T, D] or [T, H, D] and fq complex64 [T, rd/2]: 1 launch."""
        assert x.dtype == torch.bfloat16 and fq.dtype == torch.complex64
        x = x.contiguous()
        D = x.shape[-1]
        T = x.shape[0]
        H = x.numel() // (T * D)
        f = torch.view_as_real(fq.contiguous())
        assert f.shape == (T, rd // 2, 2), (f.shape, T, rd)
        out = torch.empty_like(x)
        _rope_kernel[(T * H,)](x, f, out, H, D=D, RD=rd, BD=triton.next_power_of_2(D), CONJ=inverse,
                               RE=ROPE_RE if re is None else re, IM=ROPE_IM if im is None else im,
                               num_warps=4, enable_fp_fusion=False, enable_reflect_ftz=False)
        return out

    def usable_router(self, y, gate_w) -> bool:
        return (self.mm_tile > 0 and y.dtype == torch.bfloat16 and y.dim() == 2
                and 0 < y.size(0) < self.mm_tile and gate_w.dtype == torch.float32 and gate_w.dim() == 2
                and gate_w.shape[0] <= 1024)

    def router(self, y, gate_w, gate_bias, keep, k: int, route_scale: float, idx_out, w_out,
               record=None):
        """Model's decode router for `y` [T, dim] bf16 with the fp32 gate: top-k expert ids into
        `idx_out`, normalized weights into `w_out`. `keep` is the prune mask (or None) and
        `record(logits, scores)` the prune-miss accounting, fed the unmasked logits as before.
        torch keeps what sets bits by order: the padded gate GEMM, topk, and the k-way sum."""
        T = y.size(0)
        E = gate_w.shape[0]
        pad = self._buf(("router", T), (self.mm_tile, gate_w.shape[1]))
        pad[:T].copy_(y)
        g = self._buf(("router_g", T), (self.mm_tile, E))
        torch.mm(pad, gate_w.t(), out=g)
        scores = torch.empty(T, E, dtype=torch.float32, device=y.device)
        logits = torch.empty_like(scores)
        masked = torch.empty_like(scores)
        keep_u8 = keep.view(torch.uint8) if keep is not None else None
        _router_pre_kernel[(T,)](g, gate_bias, keep_u8 if keep is not None else g, scores, logits, masked, E,
                                 HAS_KEEP=keep is not None, BE=triton.next_power_of_2(E), num_warps=4,
                                 enable_fp_fusion=False, enable_reflect_ftz=False)
        if record is not None:
            record(logits, scores)
        vals = self._buf(("router_v", T, k), (T, k))
        torch.topk(masked, k, dim=-1, out=(vals, idx_out))
        wts = scores.gather(1, idx_out)
        _router_post_kernel[(T,)](wts, wts.sum(dim=-1, keepdim=True), w_out, float(route_scale), K=k,
                                  KB=triton.next_power_of_2(k), num_warps=1, enable_fp_fusion=False, enable_reflect_ftz=False)
        return scores

    def attn_probs(self, s, mask, sink, scale: float):
        """The torch decode attention between its two einsums: scores s [T, H, N] fp32 (any
        strides) -> probabilities [T, H, N]. 3 launches (pre, torch's sum, post) instead of ~12."""
        T, H, N = s.shape
        assert s.dtype == torch.float32 and mask.shape == (T, N) and sink.shape == (H,)
        p = torch.empty(T, H, N, dtype=torch.float32, device=s.device)
        mx = torch.empty(T, H, 1, dtype=torch.float32, device=s.device)
        BN = triton.next_power_of_2(N)
        msk = mask if mask.is_contiguous() else mask.contiguous()
        _softmax_pre_kernel[(T * H,)](s, msk.view(torch.uint8), p, mx, H, N, s.stride(0), s.stride(1),
                                      s.stride(2), float(scale), BN=BN, num_warps=4, enable_fp_fusion=False, enable_reflect_ftz=False)
        psum = p.sum(-1, keepdim=True)
        out = torch.empty_like(p)
        _softmax_post_kernel[(T * H,)](p, psum, mx, sink, out, H, N, BN=BN, num_warps=4,
                                       enable_fp_fusion=False, enable_reflect_ftz=False)
        return out

    def usable_hc(self, x, hc_fn) -> bool:
        return (self.mm_tile > 0 and x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 3
                and hc_fn.dtype == torch.float32 and tuple(hc_fn.shape) == (24, 20480)
                and x.shape[1] * x.shape[2] == 20480
                and 0 < x.size(0) < min(self.mm_tile, self.hc_mm_tile))

    def hc_mixes(self, x, hc_fn, hc_scale, hc_base, norm_eps: float, hc: int, iters: int, eps: float,
                 out=None):
        """Model._hc_mixes(x, ...) with HC_FUSED, for fewer than MM_TILE rows: 6 launches instead of
        15, and none for the copies when `out=(pre, post, comb)` names the destination buffers."""
        T = x.size(0)
        rows = max(self.mm_tile, self.hc_mm_tile)
        pad = self._buf(("hc", T), (rows, hc_fn.shape[1]))
        pad[:T].copy_(x.reshape(T, -1))
        mm = self._buf(("hc_mm", T), (self.hc_mm_tile, hc_fn.shape[0]))
        torch.mm(pad[:self.hc_mm_tile], hc_fn.t(), out=mm)
        mean = pad[:self.mm_tile].square().mean(-1, keepdim=True)
        if out is None:
            pre = torch.empty(T, hc, dtype=torch.float32, device=x.device)
            post = torch.empty(T, hc, dtype=torch.float32, device=x.device)
            comb = torch.empty(T, hc, hc, dtype=torch.float32, device=x.device)
        else:
            pre, post, comb = out
        _hc_rms_kernel[(T,)](mm, mean, hc_scale.contiguous(), hc_base.contiguous(), pre, post, comb,
                             float(norm_eps), iters=iters, eps=eps, HC=hc)
        return pre, post, comb
