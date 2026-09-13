"""
fastdecode.py -- CUDA-graph decode path for the 6-token DSpark verify step (and the 5-token draft).

Why: profiled with every expert resident, one verify step of `Model.forward` costs ~436 ms, of which
~360 ms is CPU launch overhead (~10k tiny ops) and the GPU side is slowed by fp32 GEMMs (the fp32 LM
head alone: 37 ms at 72 GB/s). This module runs the same math with static buffers so it can be
captured into CUDA graphs, uses bf16 GEMMs (fp32 accumulation) everywhere the checkpoint stores bf16
or fp8, one fused Triton kernel for the Hyper-Connection Sinkhorn coefficients, and fixed-length
(masked) indexer scoring instead of context-length-dependent slices.

Semantics are those of engine/model.py (window ring, shared compressed KV, Hierarchical Sparse
Indexer with the layer-20 candidate pool, engram rows, mHC single-pass shift, DSpark draft with the
Markov head). Per backbone layer there are two graphs: A = attention + HC + router (ends with the
expert ids), then the host resolves expert slots (LRU / NVMe), then B = MoE + shared expert + HC
residual. With every routed expert resident (`--prune-keep`) the host step is a LUT lookup and the
whole layer could be one graph; that is left for later.

Numerics vs `Model.forward`: identical math, but GEMMs are not issued in fixed 16-row tiles and the
head is bf16, so the two paths can differ in the last bits (the chunk-invariance guarantee of
model.py does not extend across the two paths). `engine/test_fastdecode.py` measures the gap.
"""

from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

from engine import model as M
from engine.model import HC_OPS as _HC_OPS, _hc_post_fused, _hc_pre_rn_fused
from engine.hc_sinkhorn import hc_split_sinkhorn
import v41_ref as R

try:
    from decode_attn import decode_attention  # tools/decode_attn.py (Triton)
except Exception:  # noqa: BLE001
    decode_attention = None

try:
    from fp32_skinny import skinny_linear, wins as skinny_wins, BLOCK_MP as SKINNY_MAX_M  # tools/fp32_skinny.py (Triton)
except Exception:  # noqa: BLE001
    skinny_linear, skinny_wins, SKINNY_MAX_M = None, None, 0

# One fused Triton kernel for the sinked softmax attention instead of two fp32 SIMT batched GEMMs
# and the elementwise passes around them. DSV41_FUSED_ATTN=0 restores the torch path.
# Default OFF since 2026-09-11: with the dense projections in fp4 and the fp8 head, greedy
# decoding through this kernel diverges from the same decode without it and can fall into a
# repetition loop (NOTES 2026-09-11 20:30, engine/test_spec_lossless.py). Each piece is fine
# alone; together they cross the precision the verify step needs. DSV41_FUSED_ATTN=1 re-enables.
FUSED_ATTN = os.environ.get("DSV41_FUSED_ATTN", "0") == "1" and decode_attention is not None
# Split-K Triton kernel for the skinny fp32 projections (the HC mix GEMM: M=6, N=24, K=20480, where
# cuBLAS is latency-bound at ~30 GB/s). DSV41_HC_KERNEL=0 restores F.linear everywhere.
HC_KERNEL = os.environ.get("DSV41_HC_KERNEL", "1") == "1" and skinny_linear is not None
# Capture whole runs of layers into one graph instead of one graph per layer (resident mode only).
# DSV41_GRAPH_SEGMENTS=0 restores one graph per layer.
GRAPH_SEGMENTS = os.environ.get("DSV41_GRAPH_SEGMENTS", "1") == "1"
# Lean step: the same switch engine/v41_engine.py reads. It removes the per-step host work that is
# not the graph replay itself -- the `torch.arange` and the `repeat` of the embedding rebuilt every
# step, the constant pre_mix, the second `prepare_pending_buffers()` (only the capture run needs it)
# and the self-copy of the block ids. DSV41_LEAN_STEP=0 restores the original sequence exactly; the
# math is identical either way (the buffers get the same values).
LEAN_STEP = os.environ.get("DSV41_LEAN_STEP", "1") == "1"
# Context-bucketed sparse-index scoring.  The fallback is intentionally retained for controlled
# A/Bs and emergency rollback; it restores the old full-max-seq graph shapes exactly.
INDEX_BUCKETS = os.environ.get("DSV41_INDEX_BUCKETS", "1") == "1"


def _fp32_lin(x, w):
    """fp32 y = x @ w^T with an fp32 weight. `x` may be bf16: the Triton kernel upcasts the loaded
    tile itself (same values, half the activation bytes); the cuBLAS fallback needs the fp32 copy."""
    # The row bound is part of the guard, not an assert inside the kernel: skinny_linear's partial
    # buffer is BLOCK_MP rows and it asserts M <= BLOCK_MP. DSV41_BLOCK is documented as 1..15, and
    # anything wide enough to push M past 8 used to reach that assert and take the pair out of step
    # mid-request -- a supported knob with a crash behind it. cuBLAS handles the wider M fine; this
    # kernel exists only for the shapes it handles badly (see fp32_skinny.wins).
    if HC_KERNEL and skinny_wins(w.size(0)) and x.size(0) <= SKINNY_MAX_M:
        return skinny_linear(x, w)
    return F.linear(x.float(), w)

# The verify block: one accepted token plus DSV41_BLOCK drafted ones. The checkpoint's DSpark head
# was trained at `dspark_block_size` = 5 (so 6 verify positions), which stays the default; a larger
# block reads the same 40 layers of routed experts for more candidate tokens per step but drafts
# further outside the head's trained horizon, and a smaller one does the reverse. Only even verify
# widths are allowed: the ratio-2 key compressor groups the block's positions in pairs around a
# single pending slot, which is what `capture(S_parity)`'s two parities encode.
def _draft_block() -> int:
    v = os.environ.get("DSV41_BLOCK", "").strip()
    if v in ("", "off", "default"):
        return 5
    try:
        b = int(v)
    except ValueError:
        raise ValueError(f"DSV41_BLOCK: {v!r}; use an odd integer 1..15, or leave it unset") from None
    if not 1 <= b <= 15 or b % 2 == 0:
        raise ValueError(f"DSV41_BLOCK: {b}; must be odd and 1..15 so the verify block (b + 1) is even")
    return b


T_DRAFT = _draft_block()
T_VERIFY = T_DRAFT + 1   # tok + T_DRAFT drafts

# The graphed indexer needs a fixed key-axis shape, but that shape does not need to be the
# configured maximum context.  A small floor avoids capturing a new graph every few hundred
# tokens while powers of two keep the number of resident graph variants logarithmic.
INDEX_BUCKET_MIN = 4096


def _index_bucket(used_tokens: int, max_seq: int) -> int:
    """Smallest power-of-two token bucket covering ``used_tokens``, capped at ``max_seq``."""
    if not 0 < used_tokens <= max_seq:
        raise ValueError(f"index bucket needs 0 < used_tokens <= max_seq ({used_tokens}, {max_seq})")
    target = max(used_tokens, min(INDEX_BUCKET_MIN, max_seq))
    bucket = 1 << (target - 1).bit_length()
    return min(bucket, max_seq)


def _index_cache_rows(token_bucket: int, ratio: int, max_rows: int) -> int:
    """Cache-key rows needed by an uncompressed-token bucket at compression ``ratio``."""
    if token_bucket <= 0 or ratio <= 0 or max_rows <= 0:
        raise ValueError((token_bucket, ratio, max_rows))
    return min(max_rows, (token_bucket + ratio - 1) // ratio)


def _lin(x, w):  # bf16 tensor -> cuBLAS; FP8Weight/FP4Weight -> the Triton kernel for that stored format
    return R.dense(x, w)


class FastDecoder:
    def __init__(self, model: M.Model, engine, use_graphs: bool = True):
        self.m = model
        self.eng = engine
        self.a = model.args
        self.dev = model.dev
        self.W = model.W
        self.c = model.c
        self.use_graphs = use_graphs
        a = self.a
        dev = self.dev
        T = T_VERIFY
        # ---- static buffers (inputs of the graphs)
        self.ids = torch.zeros(T, dtype=torch.long, device=dev)
        self.pos = torch.zeros(T, dtype=torch.long, device=dev)
        self.eg_rows = {L: torch.zeros(T, a.engram_n_heads * (a.engram_max_ngram_size - 1), a.engram_head_dim,
                                       dtype=torch.float32, device=dev) for L in a.engram_layer_ids}
        self.slots = torch.zeros(T, a.n_activated_experts, dtype=torch.int32, device=dev)
        # ---- static state carried between graphs of one step
        self.h = torch.zeros(T, a.hc_mult, a.dim, dtype=torch.bfloat16, device=dev)
        self.pre_mix = torch.zeros(T, a.hc_mult, dtype=torch.float32, device=dev)
        self.attn_pre = torch.zeros(T, a.hc_mult, dtype=torch.float32, device=dev)
        self.ffn_post = torch.zeros(T, a.hc_mult, dtype=torch.float32, device=dev)
        self.ffn_comb = torch.zeros(T, a.hc_mult, a.hc_mult, dtype=torch.float32, device=dev)
        self.ffn_pre = torch.zeros(T, a.hc_mult, dtype=torch.float32, device=dev)
        self.y = torch.zeros(T, a.dim, dtype=torch.bfloat16, device=dev)
        self.route_idx = torch.zeros(T, a.n_activated_experts, dtype=torch.long, device=dev)
        self.route_w = torch.zeros(T, a.n_activated_experts, dtype=torch.float32, device=dev)
        self.topk = torch.full((T, a.index_topk), -1, dtype=torch.long, device=dev)
        n_cand = self._n_cache(1)
        self.candidates = torch.zeros(T, n_cand, dtype=torch.bool, device=dev)
        self.main_hidden = torch.zeros(T, a.dim * len(a.dspark_target_layer_ids), dtype=torch.float32, device=dev)
        self.logits = torch.zeros(T, a.vocab_size, dtype=torch.float32, device=dev)
        # draft
        self.d_tok = torch.zeros(1, dtype=torch.long, device=dev)
        self.d_last = torch.zeros(1, dtype=torch.long, device=dev)   # last main position
        self.d_noise = torch.zeros(T_DRAFT, a.vocab_size, dtype=torch.float32, device=dev)  # gumbel noise
        self.d_temp = torch.zeros(1, dtype=torch.float32, device=dev)
        self.d_out = torch.zeros(T_DRAFT, dtype=torch.long, device=dev)
        self.d_probs = torch.zeros(T_DRAFT, a.vocab_size, dtype=torch.float32, device=dev)
        # Mode-independent draft constants.  They used to be rebuilt by captured fill/arange
        # kernels every replay; only element zero of the ids changes with the accepted token.
        self._d_ids = torch.full((T_DRAFT,), a.dspark_noise_token_id, dtype=torch.long, device=dev)
        self._d_ar = torch.arange(T_DRAFT, device=dev)
        self._d_premix0 = torch.zeros(T_DRAFT, a.hc_mult, dtype=torch.float32, device=dev)
        self._d_premix0[:, 0] = 1.0
        # weights in decode-friendly dtypes (views/copies; small)
        # bf16 tensor, or the FP8Weight / FP4Weight `make_head` produced; `_lin` (= R.dense)
        # dispatches on the object, so the graphs capture the matching kernel either way.
        self.head_bf16 = self.W.head.to(torch.bfloat16) if torch.is_tensor(self.W.head) else self.W.head
        self.gate_bf16 = [w.gate_w.to(torch.bfloat16) for w in self.W.layers]
        self.mtp_gate_bf16 = [w.gate_w.to(torch.bfloat16) for w in self.W.mtp]
        self.markov_embed_bf16 = self.W.mtp[2].markov_embed.to(torch.bfloat16)
        self.markov_head_bf16 = self.W.mtp[2].markov_head.to(torch.bfloat16)
        self.win_off = torch.arange(a.window_size - 1, -1, -1, device=dev)
        # constants the lean step reuses instead of rebuilding them per step
        self._ar_t = torch.arange(T, device=dev)
        self._index_cpos = torch.arange(self._n_cache(1), device=dev)
        self._premix0 = torch.zeros(T, a.hc_mult, dtype=torch.float32, device=dev)
        self._premix0[:, 0] = 1.0
        self.graphs = {}
        self.draft_graphs = None
        self.pool = None
        # resident mode: (layer, expert) -> arena slot as a device table, so the router's expert ids can be
        # turned into slots inside the graph and the whole layer is ONE graph (no host round-trip per layer)
        self.lut = None
        self.lut_version = -1
        self.pend_buf = {L: torch.zeros(2, a.head_dim, dtype=torch.float32, device=dev)
                         for L in a.kv_source_layers if a.compress_ratios[L] > 1}
        self.kvl_buf = {L: torch.zeros(T, a.head_dim, dtype=torch.float32, device=dev) for L in self.pend_buf}
        self.sc_buf = {L: torch.zeros(T, a.head_dim, dtype=torch.float32, device=dev) for L in self.pend_buf}
        R.MM_TILE = 0  # plain GEMMs in this path (and from now on in prefill too); the 16-row tiling was a test aid
        self.stats = {"steps": 0, "graph_s": 0.0, "resolve_s": 0.0, "engram_s": 0.0, "draft_s": 0.0}
        # DSV41_GPU_TIMING=1: CUDA events around each phase of a step. Host timers cannot decompose
        # this path -- every one of them measures the host waiting for, or launching into, an async
        # queue, which is why removing route_s, the host resolve and the pageable engram copy each
        # collapsed its own timer and moved the wall clock not at all. Events sit ON the GPU
        # timeline and measure the work itself. Pairs are recorded per step and only read at the
        # end of the request, so no step pays a synchronise for being measured.
        self.gpu_timing = os.environ.get("DSV41_GPU_TIMING", "0") == "1"
        self._ev_pairs: dict[str, list] = {"draft": [], "layers": [], "segments": [], "hash": []}
        # DSV41_ROUTE_STATS=1 counts, per backbone layer, how many DISTINCT routed experts the
        # T_VERIFY tokens of a verify block ask for -- the quantity that sets the expert bytes a step
        # has to read, since one expert is read once however many of the block's tokens route to it.
        # Both ops have static shapes and no host round-trip, so they capture into the layer graphs
        # like everything else; with the flag off nothing is allocated and `_layer_a` runs one
        # `if self.rs_hits is not None` per layer.
        if M.PRUNE_MISS:
            # must exist, and keep their addresses, before any graph capture records into them
            self.m.alloc_prune_miss(a.n_routed_experts, dev)
        self.rs_hits = self.rs_uniq = None
        if os.environ.get("DSV41_ROUTE_STATS", "0") == "1":
            self.rs_hits = torch.zeros(a.n_routed_experts, dtype=torch.int32, device=dev)
            self.rs_uniq = torch.zeros(self.m.args.n_layers, dtype=torch.float64, device=dev)
            self.rs_steps = 0

    # ------------------------------------------------------------------ helpers
    def _n_cache(self, r):
        # rows of the shared compressed cache for ratio r (Caches allocates max_seq // r + 1)
        return self.c.max_seq // r + 1

    def _rope(self, x, fq, inverse=False):
        rd = self.a.rope_head_dim
        return torch.cat([x[..., :-rd], R.apply_rotary(x[..., -rd:], fq, inverse=inverse)], dim=-1)

    def _hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        xb = x.flatten(1)
        rsqrt = torch.rsqrt(xb.float().square().mean(-1, keepdim=True) + self.a.norm_eps)
        mixes = _fp32_lin(xb, hc_fn) * rsqrt
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.a.hc_mult, self.a.hc_sinkhorn_iters, self.a.hc_eps)

    # ------------------------------------------------------------------ attention (decode, T tokens)
    def _attention(self, x, w, L, ring, freqs, pos, sh_state, mtp_last=None):
        a = self.a
        T = x.size(0)
        fq = freqs[pos]
        qr = R.rmsnorm(_lin(x, w.wq_a), w.q_norm, a.norm_eps)
        q = self._rope(_lin(qr, w.wq_b).view(T, a.n_heads, a.head_dim), fq)
        kv = self._rope(R.rmsnorm(_lin(x, w.wkv), w.kv_norm, a.norm_eps), fq)
        # The key set is two pieces: the window rows and (verify) the CSA2 rows / (draft) the draft
        # keys. The fused kernel takes both as base pointers, so they are never cat'ed; only the
        # torch fallback materialises kv_all. The draft's window is a stride-0 broadcast view.
        if mtp_last is None:
            wpos = pos[:, None] - self.win_off[None, :]
            ring[pos % M.RING] = kv
            kv1 = ring[wpos.clamp_min(0) % M.RING]
            kv2 = None
            mask = wpos >= 0
            if w.ratio:
                kv2, cmask = self._compressed(x, qr, w, L, pos, sh_state)
                mask = torch.cat([mask, cmask], dim=1)
        else:
            wpos = mtp_last - self.win_off  # [128]
            kv1 = ring[wpos.clamp_min(0) % M.RING][None].expand(T, -1, -1)
            kv2 = kv[None].expand(T, -1, -1)
            mask = torch.cat([(wpos >= 0)[None].expand(T, -1),
                              torch.ones(T, T, dtype=torch.bool, device=self.dev)], dim=1)
        scale = a.head_dim ** -0.5
        if FUSED_ATTN:
            o = decode_attention(q, kv1, kv2, mask, w.attn_sink, scale)
        else:
            kv_all = kv1 if kv2 is None else torch.cat([kv1, kv2], dim=1)
            scores = torch.einsum("thd,tnd->thn", q.float(), kv_all.float()) * scale
            scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
            mx = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
            p = torch.exp(scores - mx)
            denom = p.sum(-1, keepdim=True) + torch.exp(w.attn_sink[None, :, None] - mx)
            o = torch.einsum("thn,tnd->thd", p / denom, kv_all.float()).to(torch.bfloat16)
        o = self._rope(o, fq, inverse=True).reshape(T, a.o_groups, -1)
        o = R.wo_a_proj(o, w.wo_a)
        return _lin(o.flatten(1), w.wo_b)

    def _compressed(self, x, qr, w, L, pos, st):
        """st: dict with 'parity' (python int, S % 2, fixed per graph), 'ckv', 'ik', 'ratio'."""
        a = self.a
        r = w.ratio
        c = self.c
        T = x.size(0)
        if w.is_kv_source:
            if r > 1:
                kvl, sc = _fp32_lin(x, w.comp_wkv), _fp32_lin(x, w.comp_wgate)
                # static per-layer copies for Caches.rollback (host patches the tuple after the step)
                self.kvl_buf[L].copy_(kvl); self.sc_buf[L].copy_(sc)
                buf = self.pend_buf[L]  # [2, 512]: kv, score of the pending token (host-maintained)
                if st["parity"] == 1:  # S odd: pending + t0, (t1,t2), ...; the last token -> new pending
                    kvl2 = torch.cat([buf[0][None], kvl]); sc2 = torch.cat([buf[1][None], sc])
                    g_kv = kvl2[:T].unflatten(0, (-1, r)); g_sc = sc2[:T].unflatten(0, (-1, r))
                    buf[0].copy_(kvl[T - 1]); buf[1].copy_(sc[T - 1])
                else:  # S even: (t0,t1),(t2,t3),..., no pending
                    g_kv = kvl.unflatten(0, (-1, r)); g_sc = sc.unflatten(0, (-1, r))
                latent = (g_kv * g_sc.softmax(dim=1)).sum(dim=1)
                latent = R.rmsnorm(latent.to(torch.bfloat16), w.comp_norm, a.norm_eps)
                j0 = (pos[0] - st["parity"]) // r
            else:
                latent = R.rmsnorm(_lin(x, w.comp_wkv), w.comp_norm, a.norm_eps)
                j0 = pos[0]
            nj = latent.size(0)
            jidx = j0 + self._ar_t[:nj]
            fj = self.m.freqs_c[jidx * r]
            if L in self.W.indexers:
                iw = self.W.indexers[L]
                k = R.rmsnorm(_lin(latent, iw.wk), iw.k_norm, a.norm_eps)
                c.ik[L][jidx] = self._rope(k, fj)
            c.ckv[L][jidx] = self._rope(latent, fj)
            st["ckv"], st["ik"], st["ratio"] = c.ckv[L], c.ik[L], r
        compress_lens = (pos + 1) // r
        if L in self.W.indexers:
            self.topk.copy_(self._indexer(x, qr, L, pos, compress_lens, st))
        idx = self.topk
        rows = st["ckv"][idx.clamp_min(0)]
        return rows, idx >= 0

    def _indexer(self, x, qr, L, pos, compress_lens, st):
        a = self.a
        iw = self.W.indexers[L]
        T = x.size(0)
        q = self._rope(_lin(qr, iw.wq_b).view(T, a.index_n_heads, a.index_head_dim), self.m.freqs_c[pos])
        wts = _lin(x, iw.weights_proj).float() * (a.index_head_dim ** -0.5 * a.index_n_heads ** -0.5)
        # CUDA graphs require a fixed N, but scoring all max_seq rows makes every request pay for
        # the configured context capacity.  ``index_bucket`` is fixed for one graph variant and
        # expressed in uncompressed token positions; convert it to this layer's cache rows.  The
        # exact per-query visibility mask below is still authoritative, so bucket padding cannot
        # affect which rows are selected.
        token_bucket = st.get("index_bucket", self.c.max_seq)
        ratio = st["ratio"]
        cache_rows = _index_cache_rows(token_bucket, ratio, st["ik"].size(0))
        k = st["ik"][:cache_rows]
        sc = torch.einsum("thd,nd->thn", q, k)
        score = (sc.float().relu_() * wts[:, :, None]).sum(dim=1)  # [T, N]
        cpos = self._index_cpos[:score.size(1)]
        score.masked_fill_(cpos[None, :] >= compress_lens[:, None], float("-inf"))
        if L == a.candidate_source_layer:
            selected = M.Model._select_candidates(score, compress_lens, a.candidate_topk_blocks,
                                                   a.candidate_block_size)
            self.candidates[:, :score.size(1)].copy_(selected)
        elif 0 <= a.candidate_source_layer < L:
            score = score.masked_fill(~self.candidates[:, :score.size(1)], float("-inf"))
        idx = score.topk(a.index_topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idx < compress_lens[:, None], idx, torch.full_like(idx, -1))

    # ------------------------------------------------------------------ layer graphs
    def _tap(self, name, L, t):
        f = getattr(self, 'tap', None)
        if f is not None:
            f(name, L, t)

    def _layer_a(self, L, sh_state):
        """attention + HC + router for backbone layer L, reading self.h/self.pre_mix/self.pos."""
        a = self.a
        w = self.W.layers[L]
        h = self.h
        self._tap('h_in', L, h)
        if L in self.W.engram:
            h = R.engram_forward(h, self.eg_rows[L], self.W.engram[L], a)
        if L in a.dspark_target_layer_ids:
            i = list(a.dspark_target_layer_ids).index(L)
            self.main_hidden[:, i * a.dim:(i + 1) * a.dim] = h.float().mean(dim=1)
        residual = h
        attn_pre, attn_post, attn_comb = self._hc_mixes(h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base)
        # Same fused stream ops the prefill path uses (engine/hc_ops.py). At T=6 the tensors are
        # small, so the win here is launch count rather than bandwidth: these replace ~10 tiny
        # elementwise kernels per site with one, and a graphed step issues ~1,015 such launches.
        y = (_hc_pre_rn_fused(h, self.pre_mix, w.attn_norm, a.norm_eps) if _HC_OPS
             else R.rmsnorm(R.hc_pre(h, self.pre_mix), w.attn_norm, a.norm_eps))
        self._tap('attn_x', L, y)
        y = self._attention(y, w, L, self.c.win[L], self.m.freqs_c if w.ratio else self.m.freqs_w, self.pos, sh_state)
        self._tap('attn_out', L, y)
        h = (_hc_post_fused(y, residual, attn_post, attn_comb) if _HC_OPS
             else R.hc_post(y, residual, attn_post, attn_comb))
        self.h.copy_(h)
        ffn_pre, ffn_post, ffn_comb = self._hc_mixes(h, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base)
        self.ffn_pre.copy_(ffn_pre); self.ffn_post.copy_(ffn_post); self.ffn_comb.copy_(ffn_comb)
        y = R.rmsnorm(R.hc_pre(h, attn_pre), w.ffn_norm, a.norm_eps)
        self.y.copy_(y)
        # fp32, exactly as Model.moe does it. The gate picks 6 of 384 experts and its scores are
        # full of near-ties, so a bf16 GEMM here (which this path used until 2026-09-11) changes
        # 11 % of the picks at layer 0 -- where the inputs are bit-identical -- and up to 31 %
        # deeper in, which is what made the graphed path disagree with the reference at all.
        scores = F.softplus(R.mm(y.float(), self.W.layers[L].gate_w)).sqrt()
        logits = scores + w.gate_bias
        pm = getattr(self.m, "prune_mask", None)
        if pm is not None and L in pm:
            # same accounting as Model.moe: what the router wanted before pruning masked it.
            # Fixed-shape throughout, and the accumulators are allocated before capture, so this
            # records correctly from inside the graph.
            if M.PRUNE_MISS:
                self.m._record_prune_miss(logits, scores, pm[L], L, a.n_activated_experts, decode=True)
            logits = logits.masked_fill(~pm[L], float("-inf"))
        idx = logits.topk(a.n_activated_experts, dim=-1)[1]
        wts = scores.gather(1, idx)
        wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-20) * a.route_scale
        self.route_idx.copy_(idx); self.route_w.copy_(wts)
        if self.rs_uniq is not None:
            # one-hot the block's k*T expert ids into a [n_routed_experts] table and count the rows
            # that were hit; `index_fill_` writes 1 however many tokens name the same expert.
            self.rs_hits.zero_()
            self.rs_hits.index_fill_(0, idx.reshape(-1), 1)
            self.rs_uniq[L] += self.rs_hits.sum()
        self._tap('moe_in', L, y); self._tap('route_idx', L, idx); self._tap('topk', L, self.topk)

    def _layer_b(self, L):
        a = self.a
        w = self.W.layers[L]
        store = self.m.store
        if getattr(store, "null_slot", None) is not None:
            # EP2 (engine/dist.py). This path does NOT go through Model.moe, so the two things
            # that method does for expert parallel have to be done here as well -- and both are
            # load-bearing:
            #   slots_repeat=True + null_slot  every non-owned expert shares one zero slot. The
            #                      decode router gives those pairs temporary unique keys, skips
            #                      their compute, and keeps BM=16 for the real experts. Omitting
            #                      both hooks either overflows build_routing_small or falls back to
            #                      the substantially slower BM=64 path (tools/fp4_moe.py).
            #   fp32 + combine     this rank computed only its half of the k-sum; the all-reduce
            #                      supplies the rest. Rounding to bf16 happens ONCE, after the
            #                      combine, at the same point the single-box path rounds.
            # The all-reduce is captured into this graph -- Gate G1 (scripts/gate_g1_graph_nccl.py)
            # verified NCCL collectives capture and replay correctly on these two boxes.
            out = self.m.moe_fn(self.y, self.slots, self.route_w, store.arena, a.swiglu_limit,
                                out_dtype=torch.float32, slots_repeat=True,
                                null_slot=store.null_slot)
            store.ep.combine(out)
            out = out.to(torch.bfloat16).float()
        else:
            out = self.m.moe_fn(self.y, self.slots, self.route_w, store.arena, a.swiglu_limit).float()
        # shared expert is replicated and added AFTER the combine, so it is counted once
        _sh = R.expert_ffn(self.y, w.sh_w1, w.sh_w2, w.sh_w3, a.swiglu_limit).float()
        if M.TP_DENSE_WORLD > 1:      # partial: this rank holds half the intermediate dim
            torch.distributed.all_reduce(_sh, op=torch.distributed.ReduceOp.SUM)
        out += _sh
        h = R.hc_post(out.to(torch.bfloat16), self.h, self.ffn_post, self.ffn_comb)
        self.h.copy_(h); self.pre_mix.copy_(self.ffn_pre)

    def _final(self):
        a = self.a
        x = R.rmsnorm(R.hc_pre(self.h, self.pre_mix), self.W.norm, a.norm_eps)
        self.logits.copy_(_lin(x, self.head_bf16).float())
        # seed the drafter rings for all 6 positions (harmless beyond the accepted ones)
        m0 = self.W.mtp[0]
        main_x = R.rmsnorm(_lin(self.main_hidden.to(torch.bfloat16), m0.main_proj), m0.main_norm, a.norm_eps)
        fq = self.m.freqs_w[self.pos]
        for k, w in enumerate(self.W.mtp):
            kv = self._rope(R.rmsnorm(_lin(main_x, w.wkv), w.kv_norm, a.norm_eps), fq)
            self.c.mtp_win[k][self.pos % M.RING] = kv

    def _draft(self, greedy: bool):
        """DSpark draft: 5 positions after d_last; gumbel-max sampling with self.d_noise (temperature in d_temp,
        or argmax in the separately captured greedy graph)."""
        a = self.a
        ids = self._d_ids
        ids[0] = self.d_tok[0]
        h = self.W.embed[ids].unsqueeze(1).repeat(1, a.hc_mult, 1)
        pre_mix = self._d_premix0
        pos = self.d_last[0] + 1 + self._d_ar
        for k, w in enumerate(self.W.mtp):
            residual = h
            attn_pre, attn_post, attn_comb = self._hc_mixes(h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base)
            y = R.rmsnorm(R.hc_pre(h, pre_mix), w.attn_norm, a.norm_eps)
            y = self._attention(y, w, 40 + k, self.c.mtp_win[k], self.m.freqs_w, pos, None, mtp_last=self.d_last[0])
            h = (_hc_post_fused(y, residual, attn_post, attn_comb) if _HC_OPS
             else R.hc_post(y, residual, attn_post, attn_comb))
            residual = h
            ffn_pre, ffn_post, ffn_comb = self._hc_mixes(h, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base)
            y = R.rmsnorm(R.hc_pre(h, attn_pre), w.ffn_norm, a.norm_eps)
            scores = F.softplus(R.mm(y.float(), self.W.mtp[k].gate_w)).sqrt()  # fp32, as Model.moe
            idx = (scores + w.gate_bias).topk(3, dim=-1)[1]
            wts = scores.gather(1, idx); wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-20) * a.route_scale
            slots = (idx.to(torch.int32) + k * 128)
            out = self.m.moe_fn(y, slots, wts, self.W.dspark_arena, a.swiglu_limit).float()
            # NOT all-reduced: MTPWeights loads its own shared experts (engine/model.py) and
            # _tp_shard never touches them. The DSpark drafter is deliberately not EP-parallel --
            # it has its own FixedStore arena -- and sharding it would put a collective on a path
            # that runs five times per step to save 3 blocks' worth of weights. All-reducing an
            # UNSHARDED output would simply double it.
            out += R.expert_ffn(y, w.sh_w1, w.sh_w2, w.sh_w3, a.swiglu_limit).float()
            h = R.hc_post(out.to(torch.bfloat16), residual, ffn_post, ffn_comb)
            pre_mix = ffn_pre
        w = self.W.mtp[2]
        x = R.rmsnorm(R.hc_pre(h, pre_mix), w.norm, a.norm_eps)
        logits = _lin(x, self.head_bf16).float()  # [T_DRAFT, V]
        prev = self.d_tok[0]
        for i in range(T_DRAFT):
            bias = _lin(self.markov_embed_bf16[prev.view(1)], self.markov_head_bf16).float()[0]  # 1-d index: a 0-d tensor index syncs
            lg = logits[i] + bias
            if greedy:
                # Greedy verification never reads q=d_probs.  Avoid five vocabulary-wide
                # softmax/log/Gumbel passes that the old dynamic torch.where computed anyway.
                nxt = lg.argmax()
            else:
                temp = self.d_temp[0]
                p = torch.softmax(lg / temp.clamp_min(1e-5), dim=-1)
                nxt = (torch.log(p.clamp_min(1e-30)) + self.d_noise[i]).argmax()
                self.d_probs[i].copy_(p)
            self.d_out[i] = nxt
            prev = nxt

    def _capture_draft_graphs(self):
        """Capture one graph per sampling mode; neither mode pays for the other's dead branch."""
        if self.draft_graphs is not None:
            return self.draft_graphs
        out = []
        for greedy in (True, False):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                self._draft(greedy)
            out.append(graph)
        self.draft_graphs = tuple(out)
        return self.draft_graphs

    def build_lut(self):
        """Device slot table from the store's LRU. Only valid while no expert is evicted/loaded; the
        engine rebuilds it whenever the store reports a miss.

        EP2: this rank's LRU holds only the experts it OWNS, so every other routable expert would
        be left at the -1 fill and gather garbage. The right value for them is the null slot --
        zero-filled weights, an exact 0 contribution, and the peer's all-reduce supplies the real
        one. That is the same substitution ExpertStore.resolve makes on the host path, so the
        graphed and eager paths agree. Experts pruned out of the keep set also land on the null
        slot and never matter: the router masks them to -inf, so they are never routed at all.

        This is what unlocks the segmented capture. `self.lut is not None` gates the branch in
        capture() that records whole runs of layers as one graph -- 41 replays per step become 3,
        with no host round-trip between A and B -- which is where the 40 blocking device->host
        syncs per step go.
        """
        st = self.m.store
        null = getattr(st, "null_slot", None)
        fill = -1 if null is None else int(null)
        lut = torch.full((self.a.n_layers, self.a.n_routed_experts), fill, dtype=torch.int32)
        for (L, e), slot in st.lru.items():
            lut[L, e] = slot
        # Validate ONCE, on the host tensor, before it goes to the device: a -1 left in the table
        # indexes the arena out of bounds inside a captured graph, where there is no bounds check
        # and the symptom would be corrupted experts rather than an error. Checking per call would
        # be a device->host sync per layer, which is what this table exists to remove.
        # Only ROUTABLE experts have to be mapped. A pruned expert is never gathered -- the router
        # masks it to -inf (Model.moe) -- so its -1 is correct and expected, and demanding a slot
        # for it broke the single-box pruned path: at keep 0.31 that is 264 of 384 experts per
        # layer, 10,560 entries, and the assert fired at startup. It never showed under EP2, where
        # the fill value is the null slot and every entry is mapped either way.
        pm = getattr(self.m, "prune_mask", None)
        if pm:
            routable = torch.zeros_like(lut, dtype=torch.bool)
            for L, msk in pm.items():
                routable[L] = msk.to(lut.device)
        else:
            routable = torch.ones_like(lut, dtype=torch.bool)
        bad = int(((lut < 0) & routable).sum())
        assert bad == 0, (f"slot table leaves {bad} ROUTABLE (layer, expert) entries unmapped; "
                          f"every expert the router can pick needs a real slot or the null slot")
        self.lut = lut.to(self.dev)
        self.lut_version = st.stats.get("misses", 0)

    def _layer_ab(self, L, sh_state):
        self._layer_a(L, sh_state)
        self.slots.copy_(self.lut[L][self.route_idx])  # -1 never occurs while the LUT is valid
        self._layer_b(L)

    # ------------------------------------------------------------------ gpu timing
    def _ev_begin(self, phase):
        if not self.gpu_timing:
            return None
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        self._ev_pairs[phase].append((a, b))
        return b

    def gpu_timing_reset(self):
        for v in self._ev_pairs.values():
            v.clear()

    def gpu_timing_report(self):
        """{phase: total ms on the GPU timeline}. Synchronises once, here, not per step."""
        if not self.gpu_timing or not any(self._ev_pairs.values()):
            return None
        torch.cuda.synchronize()
        out = {}
        for phase, pairs in self._ev_pairs.items():
            # A step that raised between the begin and end events leaves the last pair half
            # recorded, and elapsed_time() then throws from inside the error handler -- which is
            # how a skinny-kernel assert reached the log as "Both events must be recorded",
            # twice, with the real traceback buried under it. Measurement must never be the thing
            # that reports the failure.
            total = 0.0
            done = 0
            for a, b in pairs:
                try:
                    total += a.elapsed_time(b)
                except (ValueError, RuntimeError):
                    continue
                done += 1
            out[phase] = round(total, 1)
            out[phase + "_n"] = done
        return out

    # ------------------------------------------------------------------ route stats
    def route_stats_reset(self):
        """Zero the DSV41_ROUTE_STATS accumulator. Call it after the graphs are captured: capture's
        own warm-up and capture runs execute `_layer_a` and would otherwise be counted."""
        if self.rs_uniq is not None:
            self.rs_uniq.zero_(); self.rs_steps = 0

    def route_stats_report(self):
        """Mean distinct routed experts per layer per verify block, over the steps since the last
        reset: {'steps', 'per_layer' (n_layers floats), 'mean', 'total'}."""
        if self.rs_uniq is None or not self.rs_steps:
            return None
        per = (self.rs_uniq / self.rs_steps).tolist()
        return {"steps": self.rs_steps, "per_layer": [round(v, 4) for v in per],
                "mean": sum(per) / len(per), "total": sum(per)}

    # ------------------------------------------------------------------ capture
    def capture(self, S_parity: int, index_bucket: int):
        """Capture graphs for one start parity and fixed indexer context bucket."""
        if not self.use_graphs:
            return
        key = (S_parity, index_bucket)
        if key in self.graphs:
            return
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        st = {"parity": S_parity, "index_bucket": index_bucket,
              "ckv": None, "ik": None, "ratio": 0}
        gA, gB = [], []
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            # warm-up run (allocations, triton compiles) on a scratch copy of the state
            saved = [self.h.clone(), self.pre_mix.clone()]
            for L in range(self.a.n_layers):
                self._layer_a(L, st); self._layer_b(L)
            self._final(); self._draft(True); self._draft(False)
            self.h.copy_(saved[0]); self.pre_mix.copy_(saved[1])
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        st = {"parity": S_parity, "index_bucket": index_bucket,
              "ckv": None, "ik": None, "ratio": 0}
        if self.lut is not None and GRAPH_SEGMENTS:
            # Resident mode: routing is a device LUT lookup, so the ONLY host dependency inside a
            # step is the Engram rows of layers 1 and 14. Capture the layers between those
            # boundaries as single graphs -- 41 replays per step become 3 -- and keep the overlap:
            # a segment is queued asynchronously, so the host blocks on the next boundary's NVMe
            # reads while the GPU is still running the segment before it.
            bounds = sorted({0, self.a.n_layers} | {L for L in self.a.engram_layer_ids if 0 < L < self.a.n_layers})
            segs = []
            for i in range(len(bounds) - 1):
                lo, hi = bounds[i], bounds[i + 1]
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool):
                    for L in range(lo, hi):
                        self._layer_ab(L, st)
                    if hi == self.a.n_layers:
                        self._final()  # the head + drafter seeding ride in the last segment
                segs.append((lo, g))
            gD = self._capture_draft_graphs()
            self.graphs[key] = (None, None, None, gD, segs)
            torch.cuda.synchronize()
            return
        for L in range(self.a.n_layers):
            if self.lut is not None:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool):
                    self._layer_ab(L, st)
                gA.append(g); gB.append(None)
                continue
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._layer_a(L, st)
            gA.append(g)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._layer_b(L)
            gB.append(g)
        gF = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gF, pool=self.pool):
            self._final()
        gD = self._capture_draft_graphs()
        self.graphs[key] = (gA, gB, gF, gD, None)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ run
    def _resolve(self, L):
        # host: expert ids -> arena slots (loads misses from NVMe)
        idx = self.route_idx
        slots = self.m.store.resolve(L, idx, False)
        self.slots.copy_(slots)

    def step(self, block_ids: torch.Tensor, S: int, engram_rows: dict):
        """Run the 6-token verify block at positions S..S+5. engram_rows: {L: [6,24,256] fp32}.
        Returns (logits [6,V] fp32, main_hidden [6, 15360] fp32) as views of static buffers."""
        a = self.a
        assert block_ids.numel() == T_VERIFY
        assert self.c.len == S, (self.c.len, S)
        if LEAN_STEP:
            if block_ids is not self.ids:
                self.ids.copy_(block_ids)
            torch.add(self._ar_t, S, out=self.pos)
        else:
            self.ids.copy_(block_ids)
            self.pos.copy_(S + torch.arange(T_VERIFY, device=self.dev))
        rows_fn = engram_rows if callable(engram_rows) else None
        if rows_fn is None:
            for L, rows in engram_rows.items():
                self.eg_rows[L].copy_(rows)
        if LEAN_STEP:
            # expand, not repeat: copy_ reads the stride-0 view directly, nothing is materialised
            self.h.copy_(self.W.embed[self.ids].unsqueeze(1).expand(-1, a.hc_mult, -1))
            self.pre_mix.copy_(self._premix0)
        else:
            self.h.copy_(self.W.embed[self.ids].unsqueeze(1).repeat(1, a.hc_mult, 1))
            self.pre_mix.zero_(); self.pre_mix[:, 0] = 1.0
        parity = S % 2
        index_bucket = (_index_bucket(S + T_VERIFY, self.c.max_seq)
                        if INDEX_BUCKETS else self.c.max_seq)
        graph_key = (parity, index_bucket)
        self.prepare_pending_buffers()
        if not LEAN_STEP or graph_key not in self.graphs:
            self.capture(parity, index_bucket)
            self.prepare_pending_buffers()  # capture's warm-up/capture runs overwrite the buffers
        t0 = time.perf_counter()
        futs = rows_fn() if rows_fn is not None else None  # {layer: Future} -- reads already in flight
        _ev_layers = self._ev_begin("layers")
        if self.use_graphs:
            gA, gB, gF, gD, segs = self.graphs[graph_key]
            if segs is not None:
                for lo, g in segs:
                    if futs is not None and lo in futs:
                        # this boundary's rows were read while the previous segment ran on the GPU
                        t0r = time.perf_counter()
                        fut, finish = futs[lo]
                        self.eg_rows[lo].copy_(finish(*fut.result()))
                        self.stats["engram_s"] += time.perf_counter() - t0r
                    # Unlike the outer `layers` pair, this pair surrounds only one graph replay.
                    # Summing it excludes CPU waits before Engram boundaries; layers-segments is
                    # therefore the exposed GPU-idle/H2D gap rather than assumed kernel work.
                    _ev_segment = self._ev_begin("segments")
                    g.replay()
                    if _ev_segment is not None:
                        _ev_segment.record()
            else:
                for L in range(a.n_layers):
                    if futs is not None and L in futs:
                        # this layer's rows were read while the previous layers ran on the GPU
                        t0r = time.perf_counter()
                        fut, finish = futs[L]
                        self.eg_rows[L].copy_(finish(*fut.result()))
                        self.stats["engram_s"] += time.perf_counter() - t0r
                    _ev_segment = self._ev_begin("segments")
                    gA[L].replay()
                    if gB[L] is not None:
                        self._resolve(L)
                        gB[L].replay()
                    if _ev_segment is not None:
                        _ev_segment.record()
                gF.replay()
            if self.lut is not None:
                # bookkeeping the host resolve would have done: LRU touch is irrelevant while resident
                self.m.store.stats["hits"] += int(a.n_layers * self.route_idx.numel())
        else:
            if futs is not None:
                for LL, (f, finish) in futs.items():
                    self.eg_rows[LL].copy_(finish(*f.result()))
            st = {"parity": parity, "index_bucket": index_bucket,
                  "ckv": None, "ik": None, "ratio": 0}
            for L in range(a.n_layers):
                self._layer_a(L, st); self._resolve(L); self._layer_b(L)
            self._final()
        # host-side bookkeeping for Caches.rollback (same tuple layout as model.py)
        for L in self.kvl_buf:
            before = self.c.pending.get(L)
            self.c._chunk_inputs[L] = (S, self.kvl_buf[L], self.sc_buf[L], before)
            if parity == 1:
                self.c.pending[L] = (self.kvl_buf[L][T_VERIFY - 1].clone(),
                                     self.sc_buf[L][T_VERIFY - 1].clone())  # the last token is unpaired
            else:
                self.c.pending[L] = None
        if _ev_layers is not None:
            _ev_layers.record()
        self.c.len = S + T_VERIFY
        self.stats["steps"] += 1
        self.stats["graph_s"] += time.perf_counter() - t0
        if self.rs_uniq is not None:
            self.rs_steps += 1
        return self.logits, self.main_hidden

    def draft(self, tok: int, last_main_pos: int, temperature: float):
        self.d_tok.fill_(tok); self.d_last.fill_(last_main_pos); self.d_temp.fill_(temperature)
        greedy = temperature <= 0
        if not greedy:
            u = torch.rand_like(self.d_noise).clamp_min(1e-30)
            self.d_noise.copy_(-torch.log(-torch.log(u)))
        t0 = time.perf_counter()
        _ev_draft = self._ev_begin("draft")
        if self.use_graphs and self.graphs:
            self.graphs[next(iter(self.graphs))][3][0 if greedy else 1].replay()
        else:
            self._draft(greedy)
        if _ev_draft is not None:
            _ev_draft.record()
        self.stats["draft_s"] += time.perf_counter() - t0
        return self.d_out, self.d_probs

    def prepare_pending_buffers(self):
        """Copy Caches.pending (host truth) into the static buffers the graphs read."""
        for L, buf in self.pend_buf.items():
            p = self.c.pending.get(L)
            if p is not None:
                buf[0].copy_(p[0]); buf[1].copy_(p[1])
