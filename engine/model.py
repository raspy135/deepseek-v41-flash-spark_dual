"""
model.py -- DeepSeek-V4.1-Flash text model for one-box serving: chunked prefill, multi-token
decode blocks (DSpark verification) and cache rollback, batch size 1.

Ported from the reference `inference/model.py`; the tilelang kernels are replaced by torch ops
and the routed experts by `engine.experts.ExpertStore` (+ the Triton FP4 grouped-MoE kernel in
`tools/fp4_moe.py`). Deviations from the reference, all towards MORE precision:
  * activations are not fake-quantized to fp8 (optional flag),
  * window KV and compressed KV caches are kept in bf16 instead of fp8 / FP4-E4M3,
  * the indexer's Q/K are not FP4-quantized.
Position semantics are the reference's: a chunk of T tokens at absolute start position S.
Any (S, T) with T <= 512 works, which is what chunked prefill and 6-token verify blocks need.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import v41_ref as R  # noqa: E402

# Window ring slots. Must exceed window_size + the longest chunk a single forward sees, because
# `attention` gathers a query's window out of the ring AFTER writing the whole chunk into it
# (128 + 2048 here). 4096 slots x 512 dims x bf16 x 40 layers = 167 MB.
RING = int(os.environ.get("DSV41_RING", 4096))
# Longest prefill chunk. Bigger chunks are strictly cheaper on this recipe: a prefill chunk streams
# nearly every expert of every layer through the transient ring whatever its length (a 512-token
# chunk already touches ~370 of 384), so the NVMe traffic of a prompt is ~chunks x layers x 384
# experts and quadrupling the chunk quarters it. The ceiling is activation memory: at T=2048 the
# gathered window+compressed KV of one layer is ~2.7 GB.
MAX_CHUNK = int(os.environ.get("DSV41_PREFILL_CHUNK", 2048))

# Chunk invariance requires every GEMM to give the same row whatever the batch length M. cuBLAS
# picks split-K kernels for small M and, with this flag on, reduces the K-splits in bf16, so
# F.linear(x[:6], w) != F.linear(x, w)[:6] by ~2.4e-3 for the N=512 wkv projection -- which the
# attention softmax then amplifies ~2x per layer. fp32 reduction cuts that to ~9e-5.
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

# ... and the same GEMM must be issued with the same M whatever the chunk length, or cuBLAS
# switches tiling/split-K and a row comes out a few ulps different. See v41_ref.mm().
# 16 and not something larger: for several of these shapes (wq_a, the expert w1/w3) cuBLAS gives a
# row a slightly different result depending on its OFFSET inside the tile, and a token's offset is
# chunk-relative. 8 and 16 are offset-invariant for every shape the model uses; 32/64/128 are not.
MM_TILE = 16
# The attention softmax is a *batched* GEMM (one independent problem per token), which is already
# offset-invariant, so it can use a bigger tile.
ATTN_TILE = 64
# DSV41_ATTN_TIMING=1: split a prefill layer into GPU-timeline phases using CUDA events (see
# _mark). Cheap enough to leave on, and unlike the host-timer version it replaced it does not
# reattribute the work: that one synchronised, so the first phase in the function swallowed the
# whole queued backlog and reported q_proj at 72 % of attention. Several prefill rewrites were
# proposed off that bad number and measured at 1.0x or worse before being written.
ATTN_TIMING = os.environ.get("DSV41_ATTN_TIMING", "0") == "1"
ATTN_PHASES: dict = {}
_PHASE_MARKS: list = []   # [(name, cuda_event)] in stream order; read once, at report time


def _use_fused(mtp_extra, T: int) -> bool:
    """Whether this attention call takes the fused kernel (tools/decode_attn.py).

    Decided in one place because the key tensors are BUILT differently for it: the fused path takes
    the window and the compressed rows as two base pointers and walks them as one key axis, so it
    never wants the concatenated [T, 640, d] tensor at all. At a 2,048-token chunk that tensor is
    ~1.5 GB written and re-read per layer.

    T > 16 keeps the decode-sized calls (verify blocks, drafts) on the torch path, where the split
    tuning and the graph capture live. mtp_extra is the DSpark draft, whose second key block is a
    stride-0 broadcast; left alone for now.
    """
    return (mtp_extra is None and T > 16 and _prefill_attn is not None
            and os.environ.get("DSV41_PREFILL_FUSED_ATTN", "1") == "1")


def _mark(name):
    """Record a CUDA event on the current stream. NOT a host timestamp.

    The first version of this used torch.cuda.synchronize() + perf_counter and it lied: a sync
    drains everything queued, so whichever phase is first in the function absorbs the whole
    backlog. That is how q_proj came out at 72 % of attention, and why merely enabling the flag
    moved attn_s from 8.93 s to 25.91 s -- the same work measured three times larger. Host
    timestamps cannot attribute asynchronous work: without a sync they time the launch, with a
    blocking op they time the wait for unrelated kernels queued earlier.

    A CUDA event is a timestamp ON the stream, between kernels. The gap between consecutive events
    is the GPU time of the work queued between them. A few microseconds to record, so the ~700 of
    them in a prefill are free.
    """
    if ATTN_TIMING:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        _PHASE_MARKS.append((name, e))


def phase_report():
    """{phase: total GPU ms}. One synchronise, here, after everything has already run."""
    if not ATTN_TIMING or len(_PHASE_MARKS) < 2:
        return None
    torch.cuda.synchronize()
    out: dict = {}
    for (n0, e0), (n1, e1) in zip(_PHASE_MARKS, _PHASE_MARKS[1:]):
        if n1.endswith("_begin"):
            # The gap from the last attention mark to the next layer's region is the rest of the
            # block: output projection, the HC residual mixes, and anything not marked below.
            out["layer_rest"] = out.get("layer_rest", 0.0) + e0.elapsed_time(e1)
            continue
        out[n1] = out.get(n1, 0.0) + e0.elapsed_time(e1)
    _PHASE_MARKS.clear()
    return {k: round(v, 1) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


def _aph(name, t0):
    """Call-site compatible with the old host-timer helper; records an event instead."""
    _mark(name)
    return t0
KEY_BLOCK = 512  # indexer score tile along the compressed-key axis (= index_topk)
R.MM_TILE = MM_TILE

# Fused sinked-softmax attention for prefill-sized calls (T > 16). Same kernel the decode path
# uses (tools/decode_attn.py). Microbench at fixed keys=640 looks ~6x faster than the torch tile
# path, but on the live EP2 prefill (variable topk/compressed width, 40 layers) it measured
# attn_s 3.16 -> 13.3 s and sank overall tok/s. Default OFF until that is fixed; set
# DSV41_PREFILL_FUSED_ATTN=1 to re-enable. Independent of DSV41_FUSED_ATTN (decode graphs).
try:
    from decode_attn import decode_attention as _prefill_attn  # noqa: E402
except Exception:  # noqa: BLE001
    _prefill_attn = None

# Fused Hyper-Connection coefficients. engine/hc_sinkhorn.py replaces ~160 tiny torch ops per call
# with one launch, and the decode path has used it all along -- prefill did not, so it ran the
# reference torch version twice per layer (attention and FFN mixes), i.e. ~160 launches x 2 x 40
# layers x every chunk. That is what `layer_rest` was.
#
# It is also the better shape for chunk invariance: one program per token row, so a row's result
# cannot depend on how many rows are in the call, which is exactly what v41_ref.tiled_rows exists
# to fake for the torch version. DSV41_HC_FUSED=0 restores the torch path.
try:
    from engine.hc_sinkhorn import hc_split_sinkhorn as _hc_sinkhorn_fused  # noqa: E402
except Exception:  # noqa: BLE001
    _hc_sinkhorn_fused = None
HC_FUSED = os.environ.get("DSV41_HC_FUSED", "1") == "1" and _hc_sinkhorn_fused is not None

# Fused hyper-connection stream ops (engine/hc_ops.py). The torch versions materialise large fp32
# temporaries over the hc-wide residual -- [T, 4, 5120] is 84 MB at a 2,048-token chunk -- and the
# four HC sites of a layer measured 33 % of prefill on the GPU timeline, more than the MoE kernel,
# while doing almost no arithmetic. Fused: hc_post 9.6x, hc_pre+rmsnorm 6.6x.
try:
    from engine.hc_ops import hc_post as _hc_post_fused, hc_pre_rmsnorm as _hc_pre_rn_fused  # noqa: E402
except Exception:  # noqa: BLE001
    _hc_post_fused = _hc_pre_rn_fused = None
HC_OPS = os.environ.get("DSV41_HC_OPS", "1") == "1" and _hc_post_fused is not None


# ----------------------------------------------------------------------------- weights
class IndexerWeights:
    def __init__(self, get, p: str, owns_k: bool, device: str):
        w, sc = get(p + "indexer.wq_b.weight").to(device), get(p + "indexer.wq_b.scale").to(device)
        self.wq_b = R.FP8Weight(w, sc) if (R.FP8Weight is not None and os.environ.get("DSV41_DENSE_FP8", "1") == "1") else R.dequant_fp8_block(w, sc)
        self.weights_proj = get(p + "indexer.weights_proj.weight").to(device).to(torch.bfloat16)
        self.owns_k = owns_k
        if owns_k:
            self.wk = get(p + "indexer.wk.weight").to(device).to(torch.bfloat16)
            self.k_norm = get(p + "indexer.k_norm.weight").to(device).to(torch.bfloat16)


class Weights:
    """All non-routed-expert weights on the GPU, bf16 (fp32 where the reference uses fp32)."""

    def __init__(self, model_dir: str, index: dict, args: R.Args, device: str, log=print, act_quant: bool = False,
                 n_layers: int | None = None, load_mtp: bool = True, engram_dir: str | None = None):
        self.args, self.device = args, device
        n_load = args.n_layers if n_layers is None else n_layers
        wm = index["weight_map"]
        handles = {}

        def get(name):
            f = wm[name]
            if f not in handles:
                path = os.path.join(model_dir, f)
                if not os.path.exists(path) and ".engram." in name:
                    # engram shard not (yet) on disk: the small non-table tensors fetched by tools/engram_rows.py
                    L = name.split(".")[1]
                    path = os.path.join(engram_dir or "engram_rows", f"layer{L}_weights.safetensors")
                handles[f] = safe_open(path, "pt", device="cpu")
            return handles[f].get_tensor(name)

        t0 = time.time()
        self.embed = get("embed.weight").to(device).to(torch.bfloat16)
        # bf16 (the stored dtype) unless DSV41_HEAD_FP32=1. The reference keeps the LM head in fp32
        # ("so the logits come out in fp32 directly"); the fast decode path already ran a bf16 copy
        # (fp32 accumulate, logits rounded to bf16), so with a bf16 head here the two paths use the
        # same weights and the 2.65 GB fp32 copy disappears (= ~140 more expert slots).
        # ... and, with DSV41_HEAD_FMT, in fp8 or fp4 instead: the head is read in full on every
        # decode step, so its stored format is worth as much as a dense projection group's.
        self.head = R.make_head(get("head.weight").to(device))
        self.norm = get("norm.weight").to(device).to(torch.bfloat16)
        self.layers = []
        self.indexers = {}
        self.engram = {}
        for L in range(n_load):
            self.layers.append(R.LayerWeights(get, L, args, device))
            if L in args.index_source_layers:
                self.indexers[L] = IndexerWeights(get, f"layers.{L}.attn.", L in args.kv_source_layers, device)
            if L in args.engram_layer_ids:
                self.engram[L] = R.EngramWeights(get, L, device)
            if L % 10 == 9:
                log(f"weights: layer {L} loaded ({time.time() - t0:.0f}s)")
            for f in list(handles):
                if f.endswith(f"{L + 3:05d}-of-00048.safetensors"):
                    del handles[f]
        # DSpark blocks
        self.mtp = []
        for k in range(3 if load_mtp else 0):
            self.mtp.append(MTPWeights(get, k, args, device))
        self.dspark_experts = None  # filled by the engine (arena of 3 x 128 experts)
        log(f"weights: all non-expert weights on GPU in {time.time() - t0:.0f}s")


class MTPWeights:
    """One DSpark block (`mtp.k.*`): a Block with a 128-expert MoE, plus stage-specific heads."""

    def __init__(self, get, k: int, args: R.Args, device: str):
        p = f"mtp.{k}."
        self.k = k

        class _A(R.Args):
            pass

        a = R.Args(**{f: getattr(args, f) for f in R.Args.__dataclass_fields__})
        a.n_routed_experts = 128
        a.n_activated_experts = 3
        self.args = a
        # LayerWeights expects "layers.{L}." prefix; build the same fields by hand
        dev = device

        def bf(name):
            return get(p + name).to(dev).to(torch.bfloat16)

        def f32(name):
            return get(p + name).to(dev).to(torch.float32)

        fp4_groups = R.dense_fp4_groups()

        def fp8lin(name):
            w, sc = get(p + name + ".weight").to(dev), get(p + name + ".scale").to(dev)
            if R.FP8Weight is not None and os.environ.get("DSV41_DENSE_FP8", "1") == "1":
                return R.maybe_fp4(R.FP8Weight(w, sc), name, fp4_groups)
            return R.dequant_fp8_block(w, sc)

        self.attn_norm = bf("attn_norm.weight"); self.ffn_norm = bf("ffn_norm.weight")
        self.attn_sink = f32("attn.attn_sink"); self.q_norm = bf("attn.q_norm.weight"); self.kv_norm = bf("attn.kv_norm.weight")
        self.wq_a = fp8lin("attn.wq_a"); self.wq_b = fp8lin("attn.wq_b"); self.wkv = fp8lin("attn.wkv")
        self.wo_a = R.make_wo_a(get(p + "attn.wo_a.weight").to(dev), get(p + "attn.wo_a.scale").to(dev), args, fp4_groups); self.wo_b = fp8lin("attn.wo_b")
        self.hc_attn_fn = f32("hc_attn_fn"); self.hc_ffn_fn = f32("hc_ffn_fn")
        self.hc_attn_base = f32("hc_attn_base"); self.hc_ffn_base = f32("hc_ffn_base")
        self.hc_attn_scale = f32("hc_attn_scale"); self.hc_ffn_scale = f32("hc_ffn_scale")
        self.gate_w = f32("ffn.gate.weight"); self.gate_bias = f32("ffn.gate.bias")
        self.sh_w1 = fp8lin("ffn.shared_experts.w1"); self.sh_w2 = fp8lin("ffn.shared_experts.w2"); self.sh_w3 = fp8lin("ffn.shared_experts.w3")
        self.ratio = 0
        self.is_kv_source = False
        self.layer = args.n_layers + k
        if k == 0:
            self.main_proj = fp8lin("main_proj")
            self.main_norm = bf("main_norm.weight")
        if k == 2:
            self.norm = bf("norm.weight")
            self.markov_embed = bf("markov_head.embed.weight")
            # fp32 once, for the same reason as the LM head: this one is applied once per drafted
            # token, i.e. five times per DSpark step.
            self.markov_head = get(p + "markov_head.head.weight").to(dev).to(torch.bfloat16)
            self.conf_proj = get(p + "confidence_head.proj.weight").to(dev).float()


# ----------------------------------------------------------------------------- caches
class Caches:
    def __init__(self, args: R.Args, max_seq: int, device: str):
        self.args, self.max_seq, self.device = args, max_seq, device
        d = args.head_dim
        self.win = [torch.zeros(RING, d, dtype=torch.bfloat16, device=device) for _ in range(args.n_layers)]
        self.mtp_win = [torch.zeros(RING, d, dtype=torch.bfloat16, device=device) for _ in range(3)]
        self.ckv = {}
        self.ik = {}
        self.pending = {}  # ratio-2 sources: (kv fp32 [512], score fp32 [512]) of an unpaired position, or None
        for L in args.kv_source_layers:
            r = args.compress_ratios[L]
            n = max_seq // r + 1
            self.ckv[L] = torch.zeros(n, d, dtype=torch.bfloat16, device=device)
            self.ik[L] = torch.zeros(n, args.index_head_dim, dtype=torch.bfloat16, device=device)
            self.pending[L] = None
        self.len = 0  # number of valid positions
        # per-chunk memory for rollback of the compressor state
        self._chunk_inputs = {}  # L -> (S, kv [T,512] fp32, score [T,512] fp32, pending_before)

    def rollback(self, n: int):
        """Discard everything at positions >= n.

        Only the compressor carries state across positions, so only `pending` has to be restored:
        the window ring and the compressed/index caches are append-only and every slot at or after
        n is rewritten by the next forward before anything can read it.
        """
        assert n <= self.len
        for L, (S, kv, sc, before) in self._chunk_inputs.items():
            r = self.args.compress_ratios[L]
            if r == 1:
                continue
            if n % r == 0:
                self.pending[L] = None  # n positions = n/r whole groups, nothing left over
                continue
            p = n - 1  # the position left unpaired at n
            if p >= S:
                self.pending[L] = (kv[p - S], sc[p - S])
            elif p == S - 1:
                self.pending[L] = before  # exactly back to the start of the last chunk
            else:
                raise ValueError(f"rollback({n}) reaches before the last chunk (start {S}); the "
                                 f"compressor input for position {p} is no longer kept")
        self.len = n


class Shared:
    def __init__(self):
        self.ckv = None
        self.ik = None
        self.ratio = 0
        self.topk = None  # [T, k] absolute compressed positions or -1
        self.candidates = None  # [T, n_c] bool


# ----------------------------------------------------------------------------- model
class Model:
    def __init__(self, W: Weights, store, caches: Caches, moe_fn, act_quant: bool = False):
        self.W, self.store, self.c, self.moe_fn = W, store, caches, moe_fn
        self.args = W.args
        self.dev = W.device
        a = self.args
        self.freqs_c = R.precompute_freqs_cis(a.rope_head_dim, caches.max_seq + 8, a.original_seq_len, a.compress_rope_theta,
                                              a.rope_factor, a.beta_fast, a.beta_slow, self.dev)
        self.freqs_w = R.precompute_freqs_cis(a.rope_head_dim, caches.max_seq + 8, 0, a.rope_theta, a.rope_factor,
                                              a.beta_fast, a.beta_slow, self.dev)
        self.tap = None  # optional diagnostic hook: callable(name, L, tensor)
        self.engram_rows = None  # callable (layer, hashes [T,24]) -> [T,24,256] float32
        self.hash_state = None  # reference NgramHashState
        if not act_quant:
            R.act_qdq_fp8 = lambda x, block=32: x.to(torch.bfloat16)
        self.stats = {"attn_s": 0.0, "moe_s": 0.0, "engram_s": 0.0, "ep_s": 0.0, "ep_calls": 0,
                      "tokens": 0}
        self.begin_prompt()

    def _tap(self, name, L, t):
        if self.tap is not None:
            self.tap(name, L, t)

    # ------------------------------------------------------------------ attention
    def _hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        """(pre, post, comb) for one hyper-connection site. Same arithmetic as v41_ref.hc_mixes;
        only the Sinkhorn balancing runs fused. R.mm stays for the projection because chunk
        invariance depends on its fixed-row tiling."""
        a = self.args
        xf = x.flatten(1).float()
        mixes = R.mm(xf, hc_fn) * R.rms_rsqrt(xf, a.norm_eps)
        if HC_FUSED:
            return _hc_sinkhorn_fused(mixes, hc_scale, hc_base, a.hc_mult,
                                      a.hc_sinkhorn_iters, a.hc_eps)
        return R.tiled_rows(lambda t: R.hc_split_sinkhorn(t, hc_scale, hc_base, a.hc_mult,
                                                          a.hc_sinkhorn_iters, a.hc_eps), mixes)

    def _window_positions(self, pos: torch.Tensor):
        """[T, 128] absolute positions each query may see in its sliding window, -1 if none."""
        w = self.args.window_size
        p = pos[:, None] - torch.arange(w - 1, -1, -1, device=self.dev)[None, :]
        return torch.where(p >= 0, p, torch.full_like(p, -1))

    def attention(self, x: torch.Tensor, w, L: int, S: int, sh: Shared, ring: torch.Tensor,
                  freqs: torch.Tensor, mtp_extra=None, win_lo: int = 0):
        """x: [T, d] normed input. Returns [T, d]. `ring` is this layer's window KV ring.

        `win_lo` is the first position whose window KV this ring actually holds. It is 0 everywhere
        except in the decoder replay (SWA Bounded Replay, tech report 3.2.2), where the decoder
        layers have only seen the last 128 prompt tokens and a query near the start of the replay
        would otherwise gather whatever the ring happens to hold below it.
        """
        a = self.args
        T = x.size(0)
        pos = torch.arange(S, S + T, device=self.dev)
        rd = a.rope_head_dim
        fq = freqs[S:S + T]

        self._tap("attn_x", L, x)
        _mark("attn_begin")
        _t = 0.0
        qr = R.rmsnorm(R.qlinear(x, w.wq_a), w.q_norm, a.norm_eps)
        q = R.qlinear(qr, w.wq_b).view(T, a.n_heads, a.head_dim)
        q = torch.cat([q[..., :-rd], R.apply_rotary(q[..., -rd:], fq)], dim=-1)
        _t = _aph("q_proj", _t)
        self._tap("q", L, q)

        kv = R.rmsnorm(R.qlinear(x, w.wkv), w.kv_norm, a.norm_eps)
        kv = torch.cat([kv[:, :-rd], R.apply_rotary(kv[:, -rd:], fq)], dim=-1)
        _t = _aph("kv_proj", _t)
        self._tap("kv_new", L, kv)
        if mtp_extra is None:
            # gather the window BEFORE writing (a chunk may overwrite slots older queries still need)
            wpos = self._window_positions(pos)  # [T, 128]
            ring[pos % RING] = kv
            wkv = ring[wpos.clamp_min(0) % RING]  # [T, 128, d]
            wmask = wpos >= win_lo if win_lo else wpos >= 0
            self._tap("win_kv", L, wkv); self._tap("win_mask", L, wmask)
            kv_all, mask, ckv_rows = wkv, wmask, None
            _t = _aph("window_gather", _t)
            if w.ratio:
                ckv_rows, cmask = self._compressed(x, qr, w, L, S, T, pos, sh)
                _t = _aph("compressed", _t)
                self._tap("ckv_rows", L, ckv_rows); self._tap("c_mask", L, cmask)
                mask = torch.cat([wmask, cmask], dim=1)   # [T, 640] of bool: 3.8 kB, always built
                if not _use_fused(mtp_extra, T):
                    kv_all = torch.cat([wkv, ckv_rows], dim=1)   # only the torch path needs it
        else:
            # DSpark draft attention: window from the main stream's ring (positions <= S-1) + all draft kvs
            main_last = mtp_extra  # position of the last main token in the ring
            wpos = main_last - torch.arange(a.window_size - 1, -1, -1, device=self.dev)
            wpos = torch.where(wpos >= 0, wpos, torch.full_like(wpos, -1))  # [128]
            wkv = ring[wpos.clamp_min(0) % RING][None].expand(T, -1, -1)
            kv_all = torch.cat([wkv, kv[None].expand(T, -1, -1)], dim=1)
            mask = torch.cat([(wpos >= 0)[None].expand(T, -1), torch.ones(T, T, dtype=torch.bool, device=self.dev)], dim=1)

        # The softmax runs on fixed-size token tiles for the same reason the GEMMs do: the batched
        # score/PV products are not invariant to the number of query rows in the call.
        # fp32 PV product, like tools/v41_ref: rounding the probabilities to bf16 first is a
        # cliff that turns 1e-7 fp32 GEMM jitter into 1e-4 output jitter, which is enough to flip
        # a borderline router top-k and make the MoE output depend on the chunk length.
        _t = time.perf_counter() if ATTN_TIMING else _t
        if _use_fused(mtp_extra, T):
            # Same math as _softmax_attn (sinked softmax, fp32 scores, split-precision PV), fused:
            # no [T, H, N] score tensor, no concatenated key tensor. Measured 3.6x faster than the
            # torch path at T=512 once decode_attn picks SPLIT from the available parallelism --
            # the old constant SPLIT=2 is tuned for a 6-token verify block and is a slowdown here.
            o = _prefill_attn(q, wkv, ckv_rows, mask, w.attn_sink, a.head_dim ** -0.5)
        else:
            o = self._softmax_attn(q, kv_all, mask, w.attn_sink)
        _t = _aph("softmax_attn", _t)
        o = torch.cat([o[..., :-rd], R.apply_rotary(o[..., -rd:], fq, inverse=True)], dim=-1)
        o = o.reshape(T, a.o_groups, -1)
        # grouped output projection: "sgd,grd->sgr" is a GEMM with M = number of tokens, so it too
        # has to run on fixed-size token tiles (it differs most visibly at a 1-token chunk).
        o = R.wo_a_proj(o, w.wo_a, tiled=True)
        out = R.qlinear(o.flatten(1), w.wo_b)
        _mark("o_proj")      # wo_a (grouped) + wo_b; was being charged to the moe gap
        self._tap("attn_out", L, out)
        return out

    def _softmax_attn(self, q, kv_all, mask, sink):
        """Sinked softmax attention over [T, n, d] KV, in fixed-size query tiles.

        Padding rows are all-masked: their scores are -inf, so m clamps to -1e30, p is 0 and the
        sink term makes the denominator +inf -- 0/inf = 0, no NaN.
        """
        T = q.size(0)
        scale = self.args.head_dim ** -0.5
        B = ATTN_TILE if ATTN_TILE > 0 else T

        def tile(qt, kvt, mt):
            scores = torch.einsum("thd,tnd->thn", qt, kvt) * scale
            scores = scores.masked_fill(~mt[:, None, :], float("-inf"))
            mx = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
            p = torch.exp(scores - mx)
            denom = p.sum(-1, keepdim=True) + torch.exp(sink[None, :, None] - mx)
            return torch.einsum("thn,tnd->thd", p / denom, kvt)

        outs = []
        for i in range(0, T, B):
            j = min(i + B, T)
            qt, kvt, mt = q[i:j].float(), kv_all[i:j].float(), mask[i:j]
            n = j - i
            if n < B:  # pad the last tile so every call sees exactly B query rows
                qt = torch.cat([qt, qt.new_zeros(B - n, *qt.shape[1:])])
                kvt = torch.cat([kvt, kvt.new_zeros(B - n, *kvt.shape[1:])])
                mt = torch.cat([mt, mt.new_zeros(B - n, mt.size(1))])
            outs.append(tile(qt, kvt, mt)[:n])
        return torch.cat(outs).to(torch.bfloat16)

    def _compressed(self, x, qr, w, L, S, T, pos, sh: Shared):
        """Produce/read the shared compressed KV for this chunk; run/reuse the indexer; return the
        gathered rows [T, k, d] and their mask [T, k]."""
        a = self.args
        r = w.ratio
        c = self.c
        rd = a.rope_head_dim
        if w.is_kv_source:
            # latent for every position of the chunk (plus the pending unpaired one)
            if r > 1:
                xf = x.float()
                kvl, sc = R.mm(xf, w.comp_wkv), R.mm(xf, w.comp_wgate)
                before = c.pending[L]
                c._chunk_inputs[L] = (S, kvl, sc, before)
                if before is not None:
                    kvl = torch.cat([before[0][None], kvl]); sc = torch.cat([before[1][None], sc])
                    first = S - 1
                else:
                    first = S
                n_tok = kvl.size(0)
                cut = n_tok - n_tok % r
                if n_tok % r:
                    c.pending[L] = (kvl[-1], sc[-1])
                else:
                    c.pending[L] = None
                if cut > 0:
                    g_kv = kvl[:cut].unflatten(0, (-1, r)); g_sc = sc[:cut].unflatten(0, (-1, r))
                    latent = (g_kv * g_sc.softmax(dim=1)).sum(dim=1)
                    latent = R.rmsnorm(latent.to(torch.bfloat16), w.comp_norm, a.norm_eps)
                    j0 = first // r
                else:
                    latent, j0 = None, first // r
            else:
                latent = R.rmsnorm(R.mm(x, w.comp_wkv), w.comp_norm, a.norm_eps)
                j0 = S
            if latent is not None:
                nj = latent.size(0)
                jpos = (j0 + torch.arange(nj, device=self.dev)) * r
                fj = self.freqs_c[jpos]
                if L in self.W.indexers:  # index key from the pre-RoPE latent
                    iw = self.W.indexers[L]
                    k = R.rmsnorm(R.mm(latent, iw.wk), iw.k_norm, a.norm_eps)
                    k = torch.cat([k[:, :-rd], R.apply_rotary(k[:, -rd:], fj)], dim=-1)
                    c.ik[L][j0:j0 + nj] = k
                lat = torch.cat([latent[:, :-rd], R.apply_rotary(latent[:, -rd:], fj)], dim=-1)
                c.ckv[L][j0:j0 + nj] = lat
                self._tap("latent", L, (j0, lat))
            sh.ckv, sh.ik, sh.ratio = c.ckv[L], c.ik[L], r
        assert sh.ratio == r, (L, sh.ratio, r)
        compress_lens = (pos + 1) // r  # visible compressed positions per query
        n_c = int((S + T) // r)
        if L in self.W.indexers:
            sh.topk = self._indexer(x, qr, L, pos, compress_lens, n_c, sh)
        idx = sh.topk  # [T, k] absolute compressed positions, -1 = none
        self._tap("topk", L, idx); self._tap("n_c", L, n_c)
        rows = sh.ckv[idx.clamp_min(0)]
        return rows, idx >= 0

    def _indexer(self, x, qr, L, pos, compress_lens, n_c, sh: Shared):
        a = self.args
        iw = self.W.indexers[L]
        T = x.size(0)
        rd = a.rope_head_dim
        if n_c == 0:
            return self._pad_topk(torch.full((T, 0), -1, dtype=torch.int64, device=self.dev))
        q = R.qlinear(qr, iw.wq_b).view(T, a.index_n_heads, a.index_head_dim)
        q = torch.cat([q[..., :-rd], R.apply_rotary(q[..., -rd:], self.freqs_c[pos])], dim=-1)
        wts = (R.mm(x, iw.weights_proj).float() * (a.index_head_dim ** -0.5 * a.index_n_heads ** -0.5))  # [T, H]
        # fixed [MM_TILE queries x KEY_BLOCK keys] score tiles: n_c grows with the chunk, so a
        # single GEMM over sh.ik[:n_c] would be a different shape in every chunk. Columns past
        # n_c are masked out below, so the padding cannot be selected.
        NB = KEY_BLOCK
        n_pad = max(NB, (n_c + NB - 1) // NB * NB)
        k = sh.ik[:n_pad]
        if k.size(0) < n_pad:
            k = torch.cat([k, k.new_zeros(n_pad - k.size(0), k.size(1))])
        B = ATTN_TILE if ATTN_TILE > 0 else T
        score = torch.empty(T, n_pad, dtype=torch.float32, device=self.dev)
        for i in range(0, T, B):
            j = min(i + B, T)
            qt, wt = q[i:j], wts[i:j]
            if j - i < B:
                qt = torch.cat([qt, qt.new_zeros(B - (j - i), *qt.shape[1:])])
                wt = torch.cat([wt, wt.new_zeros(B - (j - i), wt.size(1))])
            for jb in range(0, n_pad, NB):
                sc = torch.einsum("thd,nd->thn", qt, k[jb:jb + NB])  # bf16
                sc = sc.float().relu_() * wt[:, :, None]
                score[i:j, jb:jb + NB] = sc.sum(dim=1)[:j - i]
        cpos = torch.arange(n_pad, device=self.dev)
        score.masked_fill_(cpos[None, :] >= compress_lens[:, None], float("-inf"))
        is_cand_src = L == a.candidate_source_layer
        if is_cand_src:
            sh.candidates = self._select_candidates(score, compress_lens, a.candidate_topk_blocks, a.candidate_block_size)
        elif 0 <= a.candidate_source_layer < L and sh.candidates is not None:
            score = score.masked_fill(~sh.candidates, float("-inf"))
        k_ = min(a.index_topk, n_c)
        idx = score.topk(k_, dim=-1, sorted=False).indices.sort(dim=-1).values
        idx = torch.where(idx < compress_lens[:, None], idx, torch.full_like(idx, -1))
        return self._pad_topk(idx)

    def _pad_topk(self, idx: torch.Tensor) -> torch.Tensor:
        """Always hand back exactly index_topk columns, the tail filled with -1 (= masked off).

        The number of compressed rows a chunk can reach, min(index_topk, (S+T)//ratio), grows with
        the chunk, so without this the concatenated KV of the attention softmax would be a
        different width in a short chunk than in a long one. The extra columns are fully masked
        and change nothing mathematically, but a different N makes cuBLAS pick a different kernel
        for the score GEMM, and the resulting ulp differences flip router decisions downstream.
        """
        pad = self.args.index_topk - idx.size(1)
        return idx if pad <= 0 else F.pad(idx, (0, pad), value=-1)

    @staticmethod
    def _select_candidates(logits, compress_lens, topk_blocks, block_size):
        width = logits.size(-1)
        scores = F.pad(logits, (0, -width % block_size), value=float("-inf"))
        scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
        num_blocks = scores.size(-1)
        last = ((compress_lens - 1) // block_size)[:, None]
        scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device)[None, :] == last, float("inf"))
        top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
        keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
        return keep.repeat_interleave(block_size, dim=-1)[..., :width]

    # ------------------------------------------------------------------ blocks
    def moe(self, y: torch.Tensor, w, L: int, prefill: bool, store, arena, n_experts: int):
        a = self.args
        self._tap("moe_in", L, y)
        scores = F.softplus(R.mm(y.float(), w.gate_w)).sqrt()
        k = 3 if n_experts == 128 else a.n_activated_experts
        logits = scores + w.gate_bias
        pm = getattr(self, "prune_mask", None)
        if pm is not None and n_experts != 128 and L in pm:
            # expert pruning experiment: the router may only pick surviving experts (REAP-style drop)
            logits = logits.masked_fill(~pm[L], float("-inf"))
        indices = logits.topk(k, dim=-1)[1]
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * a.route_scale
        self._tap("route_idx", L, indices); self._tap("route_w", L, weights)
        t0 = time.perf_counter()
        # All-resident configurations carry a device slot table (built by the engine from the
        # store's LRU). Using it here turns routing into one GPU gather instead of a host round-trip
        # plus a Python pass over every (layer, expert) pair in the chunk. The table is only valid
        # while nothing is evicted, which is exactly the pruned all-resident case; anything else
        # takes the host path.
        #
        # EP2: the table is built from THIS rank's LRU, which holds only the experts this rank
        # owns, so a non-owned expert would gather -1 rather than the null slot. v41_engine.py
        # refuses to build it under expert parallel for exactly that reason; the assert is the
        # backstop, because the failure mode is silent wrong slots, not an error.
        lut = getattr(self, "slot_lut", None)
        if lut is not None and n_experts != 128:
            # EP2-safe: build_lut fills non-owned entries with the null slot rather than -1, and
            # validates that once at construction -- checking it here would be a device->host sync
            # per layer per chunk, which is the cost this path exists to remove.
            slots = lut[L][indices]
            self.stats["hits"] = self.stats.get("hits", 0) + indices.numel()
        else:
            slots = store.resolve(L, indices, prefill)
        # EP2 combine: an ExpertStore with a null slot is by construction an ep-enabled main model
        # store (the DSpark drafter runs on a FixedStore and never has one), and its `routed` holds
        # only THIS rank's owned subset -- zeros everywhere else. One fp32 all-reduce makes both
        # ranks hold the full routed sum, which is what keeps the two decode loops computing the
        # same tokens (engine/dist.py). shared is added AFTER the combine, and only here, so it is
        # counted once and its single-node numerics are untouched.
        #
        # The rounding point matters as much as the sum. The kernel accumulates the k-sum in fp32
        # and rounds to bf16 on the way out; under EP2 that would round THIS HALF, the peer's half
        # separately, and leave their fp32 total carrying both errors. So EP2 takes the half in
        # fp32, all-reduces, and rounds once, exactly where the single-box path rounds: what
        # remains is purely the REGROUPED fp32 addition engine/dist.py documents.
        if getattr(store, "null_slot", None) is not None:
            route = getattr(self, "prefill_routes", {}).get(L) if prefill else None
            route_args = {}
            if route is not None and os.environ.get("DSV41_PREFILL_FIXED_ROUTING", "1") == "1":
                route_args = {"routing_ids": route[0][indices], "routing_slot_map": route[1]}
            # slots_repeat: every non-owned expert of this call shares the ONE null slot, so the
            # decode block aims ~half its (token, k) pairs at a single slot. The decode-sized
            # routing builder gives each slot one BM-wide block and silently overflows past BM
            # pairs -- with BM=16 and a 6-token verify block that corrupted ~1 token in 10 while
            # staying fluent enough to look like a sampling quirk. tools/fp4_moe.py.
            routed = self.moe_fn(y, slots, weights, arena, a.swiglu_limit, out_dtype=torch.float32,
                                 slots_repeat=True,
                                 null_slot=(store.null_slot if prefill and
                                            os.environ.get("DSV41_PREFILL_SKIP_NULL", "1") == "1" else -1),
                                 **route_args)
            # Split the combine off on the GPU timeline. The host timer below (ep_s) measures the
            # LAUNCH -- dist.all_reduce is async on CUDA -- which is why it reports ~0.25 ms for a
            # collective that G2 measured at 3.5 ms for this payload. The gap between these two
            # marks is the real cost, and it includes waiting for the peer to arrive: with 40
            # rendezvous per chunk, that wait is the prime suspect for the GPU going up and down
            # during prefill.
            _mark("moe_kernel")
            tc = time.perf_counter()
            store.ep.combine(routed)
            _mark("ep_combine")
            self.stats["ep_s"] += time.perf_counter() - tc   # pure network time; moe_s includes it
            self.stats["ep_calls"] += 1                      # ep_s/ep_calls = per-collective cost
            routed = routed.to(torch.bfloat16).float()
        else:
            routed = self.moe_fn(y, slots, weights, arena, a.swiglu_limit).float()
        _mark("moe")         # router + slot resolve + the routed-expert kernel + the EP2 combine
        shared = R.expert_ffn(y, w.sh_w1, w.sh_w2, w.sh_w3, a.swiglu_limit).float()
        _mark("shared_expert")   # the dense FFN every token passes through, replicated on both ranks
        self._tap("moe_routed", L, routed); self._tap("moe_shared", L, shared)
        out = routed + shared
        self.stats["moe_s"] += time.perf_counter() - t0
        return out.to(y.dtype)

    def block(self, h, pre_mix, w, L, S, sh, ring, freqs, prefill, store, arena, n_experts, mtp_extra=None,
              win_lo: int = 0):
        a = self.args
        residual = h
        attn_pre, attn_post, attn_comb = self._hc_mixes(h, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base)
        y = (_hc_pre_rn_fused(h, pre_mix, w.attn_norm, a.norm_eps) if HC_OPS
             else R.rmsnorm(R.hc_pre(h, pre_mix), w.attn_norm, a.norm_eps))
        _mark("hc_attn_mix")
        t0 = time.perf_counter()
        y = self.attention(y, w, L, S, sh, ring, freqs, mtp_extra, win_lo=win_lo)
        self.stats["attn_s"] += time.perf_counter() - t0
        h = (_hc_post_fused(y, residual, attn_post, attn_comb) if HC_OPS
             else R.hc_post(y, residual, attn_post, attn_comb))
        _mark("hc_post_attn")
        residual = h
        ffn_pre, ffn_post, ffn_comb = self._hc_mixes(h, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base)
        y = (_hc_pre_rn_fused(h, attn_pre, w.ffn_norm, a.norm_eps) if HC_OPS
             else R.rmsnorm(R.hc_pre(h, attn_pre), w.ffn_norm, a.norm_eps))
        # Everything above since o_proj is hyper-connection bookkeeping on the 4x-wide residual
        # stream, not MoE -- it was being charged to the moe_kernel gap.
        _mark("hc_ffn_mix")
        y = self.moe(y, w, L, prefill, store, arena, n_experts)
        h = (_hc_post_fused(y, residual, ffn_post, ffn_comb) if HC_OPS
             else R.hc_post(y, residual, ffn_post, ffn_comb))
        _mark("hc_post_ffn")
        return h, ffn_pre

    # ------------------------------------------------------------------ SWA bounded replay
    def begin_prompt(self):
        """Drop whatever the previous prompt left in the replay buffer."""
        self._rep = {"h": [], "pre_mix": [], "topk": [], "cand": []}
        self._rep_end = 0

    def _rep_keep(self, h, pre_mix, sh: Shared, S: int, T: int):
        """Remember the last `window_size` encoder outputs of the prompt so far.

        Only the tail is ever needed, so each chunk contributes at most `window_size` rows and the
        buffer is trimmed as soon as it has more than that.
        """
        w = self.args.window_size
        n = min(w, T)
        sl = slice(T - n, T)
        r = self._rep
        r["h"].append(h[sl]); r["pre_mix"].append(pre_mix[sl])
        # layers 21..23 reuse layer 20's top-k and layers 24..39 search inside layer 20's candidate
        # pool, and both are computed per query -- so they belong to the queries, not to the caches,
        # and the replay has to carry them across from the encoder pass instead of recomputing them.
        r["topk"].append(sh.topk[sl])
        r["cand"].append(sh.candidates[sl] if sh.candidates is not None else None)
        self._rep_end = S + T
        while sum(t.size(0) for t in r["h"]) - r["h"][0].size(0) >= w:
            for k in r:
                r[k].pop(0)

    def _rep_tail(self):
        w = self.args.window_size
        r = self._rep
        h = torch.cat(r["h"])[-w:]
        pre_mix = torch.cat(r["pre_mix"])[-w:]
        topk = torch.cat(r["topk"])[-w:]
        cands = r["cand"]
        cand = None
        if cands and cands[0] is not None:
            width = max(c.size(1) for c in cands)
            # older chunks scored fewer compressed columns; a query can only ever see columns below
            # its own position, all of which are inside its own chunk's width, so padding the rest
            # with False changes nothing that is reachable.
            cand = torch.cat([c if c.size(1) == width else F.pad(c, (0, width - c.size(1)), value=False)
                              for c in cands])[-w:]
        return h, pre_mix, topk, cand, self._rep_end - h.size(0)

    @torch.inference_mode()
    def decoder_replay(self, need_logits: bool = True):
        """Decoder SWA Bounded Replay (tech report 2.2 / 3.2.2).

        Under CED the decoder's global KV is projected from the last encoder layer's hidden state,
        which the encoder pass has already written for every prompt position. The only thing the
        decoder layers still owe the first decode steps is their own sliding-window KV -- so they
        are run over the last `window_size` prompt tokens only, with SWA truncated to that segment,
        instead of over the whole prompt. The prompt's final logits come from this pass.
        """
        a = self.args
        h, pre_mix, topk, cand, S = self._rep_tail()
        T = h.size(0)
        sh = Shared()
        src = a.candidate_source_layer
        sh.ckv, sh.ik, sh.ratio = self.c.ckv[src], self.c.ik[src], self.args.compress_ratios[src]
        sh.topk, sh.candidates = topk, cand
        main_hiddens = []
        for L in range(src + 1, len(self.W.layers)):
            w = self.W.layers[L]
            if L in a.dspark_target_layer_ids:
                main_hiddens.append(h.float().mean(dim=1))
            freqs = self.freqs_c if w.ratio else self.freqs_w
            h, pre_mix = self.block(h, pre_mix, w, L, S, sh, self.c.win[L], freqs, True, self.store,
                                    self.store.arena, a.n_routed_experts, win_lo=S)
            self._tap("h", L, h); self._tap("pre_mix", L, pre_mix)
        self.last_h, self.last_pre_mix = h, pre_mix
        logits = None
        if need_logits:
            x = R.hc_pre(h, pre_mix)
            x = R.rmsnorm(x, self.W.norm, a.norm_eps)
            logits = R.head_logits(x, self.W.head)
        self.stats["replay_tokens"] = self.stats.get("replay_tokens", 0) + T
        return logits, (torch.cat(main_hiddens, dim=-1) if main_hiddens else None), S

    @torch.inference_mode()
    def forward(self, ids: torch.Tensor, S: int, prefill: bool, need_logits: bool = True,
                encoder_only: bool = False):
        """ids: [T] token ids at positions S..S+T-1. Returns (logits [T, V] fp32 or None, main_hidden [T, 15360]).
        Caches must be valid for positions < S (self.c.len == S).

        ``encoder_only`` stops after the candidate-source layer (the last layer that writes global
        KV, layer 20 here): everything above it is replayed once over the prompt tail by
        `decoder_replay`. It returns (None, None) -- there are no logits and no DSpark hidden
        states below layer 37.
        """
        a = self.args
        assert self.c.len == S, (self.c.len, S)
        T = ids.size(0)
        assert T <= MAX_CHUNK, (T, MAX_CHUNK)
        t0 = time.perf_counter()
        hashes = self.hash_state(ids[None], S)[0] if self.hash_state is not None else None  # [T, 2, 24]
        self.stats["engram_s"] += time.perf_counter() - t0
        h = self.W.embed[ids].unsqueeze(1).repeat(1, a.hc_mult, 1)
        pre_mix = torch.zeros(T, a.hc_mult, device=self.dev)
        pre_mix[:, 0] = 1.0
        sh = Shared()
        main_hiddens = []
        n_layers = len(self.W.layers)
        last = a.candidate_source_layer if encoder_only else n_layers - 1
        prefetch = getattr(self, "engram_prefetch", None)
        context = (prefetch(hashes) if prefill and hashes is not None and prefetch is not None
                   and os.environ.get("DSV41_PREFILL_ENGRAM", "1") == "1"
                   else nullcontext(self.engram_rows))
        with context as get_rows:
            for L in range(last + 1):
                w = self.W.layers[L]
                if L in self.W.engram:
                    t0 = time.perf_counter()
                    li = list(a.engram_layer_ids).index(L)
                    rows = get_rows(L, hashes[:, li, :])
                    self._tap("engram_rows", L, rows)
                    h = R.engram_forward(h, rows, self.W.engram[L], a)
                    self._tap("engram_out", L, h)
                    self.stats["engram_s"] += time.perf_counter() - t0
                    _mark("engram")
                if L in a.dspark_target_layer_ids:
                    main_hiddens.append(h.float().mean(dim=1))
                freqs = self.freqs_c if w.ratio else self.freqs_w
                h, pre_mix = self.block(h, pre_mix, w, L, S, sh, self.c.win[L], freqs, prefill, self.store,
                                        self.store.arena, a.n_routed_experts)
                self._tap("h", L, h); self._tap("pre_mix", L, pre_mix)
        self.c.len = S + T
        self.stats["tokens"] += T
        if encoder_only:
            self._rep_keep(h, pre_mix, sh, S, T)
            return None, None
        logits = None
        self.last_h, self.last_pre_mix = h, pre_mix
        if need_logits and n_layers == a.n_layers:
            x = R.hc_pre(h, pre_mix)
            x = R.rmsnorm(x, self.W.norm, a.norm_eps)
            logits = R.head_logits(x, self.W.head)
        return logits, (torch.cat(main_hiddens, dim=-1) if main_hiddens else None)

    # ------------------------------------------------------------------ DSpark
    @torch.inference_mode()
    def dspark_seed(self, main_hidden: torch.Tensor, S: int):
        """Write the drafter's window KV for main positions S..S+M-1 from their main hiddens [M, 15360]."""
        a = self.args
        m0 = self.W.mtp[0]
        main_x = R.rmsnorm(R.qlinear(main_hidden.to(torch.bfloat16), m0.main_proj), m0.main_norm, a.norm_eps)
        M = main_x.size(0)
        pos = torch.arange(S, S + M, device=self.dev)
        rd = a.rope_head_dim
        for k, w in enumerate(self.W.mtp):
            kv = R.rmsnorm(R.qlinear(main_x, w.wkv), w.kv_norm, a.norm_eps)
            kv = torch.cat([kv[:, :-rd], R.apply_rotary(kv[:, -rd:], self.freqs_w[S:S + M])], dim=-1)
            self.c.mtp_win[k][pos % RING] = kv

    @torch.inference_mode()
    def dspark_draft(self, tok: int, last_main_pos: int, temperature: float):
        """Draft block: returns (draft ids [B], draft probs [B, V] fp32 at the given temperature,
        confidence [B]) for B = the DSpark block size. Queries sit at last_main_pos+1 .. +B."""
        a = self.args
        from engine.fastdecode import T_DRAFT as B   # DSV41_BLOCK, default the checkpoint's 5
        ids = torch.full((B,), 128799, dtype=torch.long, device=self.dev)
        ids[0] = tok
        h = self.W.embed[ids].unsqueeze(1).repeat(1, a.hc_mult, 1)
        pre_mix = torch.zeros(B, a.hc_mult, device=self.dev); pre_mix[:, 0] = 1.0
        S = last_main_pos + 1
        sh = Shared()
        for k, w in enumerate(self.W.mtp):
            h, pre_mix = self.block(h, pre_mix, w, a.n_layers + k, S, sh, self.c.mtp_win[k], self.freqs_w, False,
                                    self.W.dspark_store, self.W.dspark_arena, 128, mtp_extra=last_main_pos)
        w = self.W.mtp[2]
        x = R.hc_pre(h, pre_mix)
        # The reference feeds the UN-normed hc_pre output to the confidence head and the normed one
        # to the LM head (inference/model.py::DSparkBlock.forward_head), so keep both.
        x_pre = x
        x = R.rmsnorm(x, w.norm, a.norm_eps)
        logits = R.head_logits(x, self.W.head)  # [B, V] fp32
        out = torch.empty(B + 1, dtype=torch.long, device=self.dev)
        out[0] = tok
        probs = []
        embeds = []
        for i in range(B):
            e = w.markov_embed[out[i]]  # [256]
            bias = F.linear(e.to(torch.bfloat16)[None], w.markov_head).float()[0]  # [V]
            lg = logits[i] + bias
            if temperature <= 0:
                p = torch.zeros_like(lg); p[lg.argmax()] = 1.0
                nxt = lg.argmax()
            else:
                p = torch.softmax(lg / temperature, dim=-1)
                nxt = torch.multinomial(p, 1)[0]
            out[i + 1] = nxt
            probs.append(p)
            embeds.append(e.float())
        # DSparkConfidenceHead returns the raw projection (no sigmoid); adaptive verification is off
        # in this engine, so it is reported, not acted on.
        conf = (torch.cat([x_pre.float(), torch.stack(embeds)], dim=-1) @ w.conf_proj.T).squeeze(-1)
        return out[1:], torch.stack(probs), conf
