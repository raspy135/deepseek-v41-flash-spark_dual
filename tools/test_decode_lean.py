"""engine/decode_lean.py and the in-place HC outputs must equal the torch spellings bit for bit.

    python3 -m unittest tools.test_decode_lean      (GPU; MODEL_DIR for real HC/norm weights)

Real hc_fn / hc_scale / hc_base / norm weights from one checkpoint layer when MODEL_DIR is set,
seeded random ones otherwise. Activations are seeded bf16 with per-row scales from 1e-3 to 3e2, so
the RMS spans the range the residual stream reaches. Every comparison is torch.equal.
"""
from __future__ import annotations

import json
import os
import sys
import types
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
import v41_ref as R  # noqa: E402
from engine import model as M  # noqa: E402  (sets R.MM_TILE)
from engine.decode_lean import LeanOps  # noqa: E402
from engine.hc_ops import hc_post, hc_pre_rmsnorm  # noqa: E402

DEV = "cuda"
EPS, HC_EPS, ITERS, HC = 1e-6, 1e-6, 20, 4


def _layer_tensors():
    md = os.environ.get("MODEL_DIR")
    g = torch.Generator(device="cpu").manual_seed(7)
    if md and os.path.exists(f"{md}/model.safetensors.index.json"):
        from safetensors import safe_open
        idx = json.load(open(f"{md}/model.safetensors.index.json"))["weight_map"]
        names = {"hc_fn": "layers.3.hc_attn_fn", "hc_scale": "layers.3.hc_attn_scale",
                 "hc_base": "layers.3.hc_attn_base", "q_norm": "layers.3.attn.q_norm.weight",
                 "kv_norm": "layers.3.attn.kv_norm.weight", "attn_norm": "layers.3.attn_norm.weight"}
        out = {}
        for k, n in names.items():
            with safe_open(f"{md}/{idx[n]}", "pt") as f:
                out[k] = f.get_tensor(n).to(DEV)
        return out, True
    return {"hc_fn": (torch.randn(24, 20480, generator=g) * 0.02).to(DEV),
            "hc_scale": torch.tensor([0.7, 1.3, 0.9]).to(DEV),
            "hc_base": torch.randn(24, generator=g).to(DEV),
            "q_norm": (1 + 0.1 * torch.randn(1280, generator=g)).bfloat16().to(DEV),
            "kv_norm": (1 + 0.1 * torch.randn(512, generator=g)).bfloat16().to(DEV),
            "attn_norm": (1 + 0.1 * torch.randn(5120, generator=g)).bfloat16().to(DEV)}, False


def _act(shape, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(*shape, generator=g)
    scale = torch.logspace(-3, 2.5, shape[0])
    return (x * scale.view(-1, *([1] * (len(shape) - 1)))).bfloat16().to(DEV)


def _hc_ref(x, t):
    fake = types.SimpleNamespace(args=types.SimpleNamespace(norm_eps=EPS, hc_mult=HC, hc_sinkhorn_iters=ITERS,
                                                            hc_eps=HC_EPS))
    return M.Model._hc_mixes(fake, x, t["hc_fn"], t["hc_scale"], t["hc_base"])


class LeanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert R.MM_TILE == 16 and M.HC_FUSED, (R.MM_TILE, M.HC_FUSED)
        cls.t, cls.real = _layer_tensors()
        cls.lean = LeanOps(DEV, R.MM_TILE, R.HC_MM_TILE)

    def test_rmsnorm_equal(self):
        for key in ("q_norm", "kv_norm", "attn_norm"):
            w = self.t[key]
            for T in (1, 2, 3, 4, 5, 6, 10, 15):
                for dtype in (torch.bfloat16, torch.float32):
                    x = _act((T, w.numel()), 100 + T).to(dtype)
                    self.assertTrue(torch.equal(self.lean.rmsnorm(x, w, EPS), R.rmsnorm(x, w, EPS)),
                                    (key, T, dtype))

    def test_rmsnorm_noncontiguous_slice(self):
        # the merged wq_a||wkv output: rmsnorm of a column slice
        qkv = _act((6, 1280 + 512), 3)
        q, kv = qkv[:, :1280], qkv[:, 1280:]
        self.assertTrue(torch.equal(self.lean.rmsnorm(q, self.t["q_norm"], EPS),
                                    R.rmsnorm(q.contiguous(), self.t["q_norm"], EPS)))
        self.assertTrue(torch.equal(self.lean.rmsnorm(kv, self.t["kv_norm"], EPS),
                                    R.rmsnorm(kv.contiguous(), self.t["kv_norm"], EPS)))

    def test_hc_mixes_equal(self):
        for T in (1, 3, 4, 5, 6, 10, 15):
            x = _act((T, HC, 5120), 200 + T)
            ref = _hc_ref(x, self.t)
            got = self.lean.hc_mixes(x, self.t["hc_fn"], self.t["hc_scale"], self.t["hc_base"],
                                     EPS, HC, ITERS, HC_EPS)
            for name, a, b in zip(("pre", "post", "comb"), got, ref):
                self.assertTrue(torch.equal(a, b), (T, name, float((a - b).abs().max())))

    def test_hc_mixes_into_buffers(self):
        x = _act((6, HC, 5120), 11)
        bufs = (torch.zeros(6, HC, device=DEV), torch.zeros(6, HC, device=DEV), torch.zeros(6, HC, HC, device=DEV))
        self.lean.hc_mixes(x, self.t["hc_fn"], self.t["hc_scale"], self.t["hc_base"], EPS, HC, ITERS, HC_EPS,
                           out=bufs)
        for a, b in zip(bufs, _hc_ref(x, self.t)):
            self.assertTrue(torch.equal(a, b))

    def test_hc_post_in_place(self):
        T = 6
        res = _act((T, HC, 5120), 21)
        y = _act((T, 5120), 22)
        post = torch.rand(T, HC, device=DEV)
        comb = torch.rand(T, HC, HC, device=DEV)
        ref = hc_post(y, res, post, comb)
        inplace = res.clone()
        hc_post(y, inplace, post, comb, out=inplace)
        self.assertTrue(torch.equal(inplace, ref))

    def test_hc_pre_rmsnorm_direct_and_fp32_weight(self):
        for T in (4, 5, 6):
            h = _act((T, HC, 5120), 31 + T)
            pre = torch.rand(T, HC, device=DEV)
            w = self.t["attn_norm"]
            ref = hc_pre_rmsnorm(h, pre, w, EPS)
            out = torch.empty(T, 5120, dtype=torch.bfloat16, device=DEV)
            hc_pre_rmsnorm(h, pre, self.lean._wf32(w), EPS, out=out)
            self.assertTrue(torch.equal(out, ref), T)

    def test_swiglu_equal(self):
        import torch.nn.functional as F
        for n, T in ((1152, 6), (1152, 4), (2304, 5), (2304, 1)):
            gu = _act((T, 2 * n), 50 + T) * 40  # spans the +-10 clamp and exp overflow of silu
            gu[0, :8] = torch.tensor([-200., -90., -1e-30, 0., 1e-30, 9.99, 10., 10.01], device=DEV).bfloat16()
            for limit in (10.0, 0.0):
                gate, up = gu[:, :n].float(), gu[:, n:].float()
                if limit > 0:
                    up = torch.clamp(up, min=-limit, max=limit)
                    gate = torch.clamp(gate, max=limit)
                ref = (F.silu(gate) * up).to(torch.bfloat16)
                self.assertTrue(torch.equal(self.lean.swiglu(gu, n, limit), ref), (n, T, limit))

    def test_shared_ffn_fp8_equal(self):
        from fp8_linear import concat_rows, quantize_to_fp8
        g = torch.Generator(device="cpu").manual_seed(9)
        w1 = quantize_to_fp8((torch.randn(1152, 5120, generator=g) * 0.02).bfloat16().to(DEV))
        w3 = quantize_to_fp8((torch.randn(1152, 5120, generator=g) * 0.02).bfloat16().to(DEV))
        w2 = quantize_to_fp8((torch.randn(5120, 1152, generator=g) * 0.02).bfloat16().to(DEV))
        w13, (w1v, w3v) = concat_rows(w1, w3)
        for T in (4, 6):
            y = _act((T, 5120), 60 + T)
            ref = R.expert_ffn(y, w1v, w2, w3v, 10.0, w13=w13)
            got = R.qlinear(self.lean.swiglu(R.qlinear(y, w13), w1v.N, 10.0), w2)
            self.assertTrue(torch.equal(got, ref), T)

    def test_hc_post_add_fold(self):
        T = 6
        res = _act((T, HC, 5120), 71)
        routed = _act((T, 5120), 72).float() * 3
        shared = _act((T, 5120), 73)
        post = torch.rand(T, HC, device=DEV)
        comb = torch.rand(T, HC, HC, device=DEV)
        ref = hc_post((routed + shared.float()).to(torch.bfloat16), res, post, comb)
        self.assertTrue(torch.equal(hc_post(routed, res, post, comb, x2=shared), ref))
        inplace = res.clone()
        hc_post(routed, inplace, post, comb, out=inplace, x2=shared)
        self.assertTrue(torch.equal(inplace, ref))

    def _gate(self):
        md = os.environ.get("MODEL_DIR")
        if self.real:
            from safetensors import safe_open
            idx = json.load(open(f"{md}/model.safetensors.index.json"))["weight_map"]
            out = []
            for n in ("layers.3.ffn.gate.weight", "layers.3.ffn.gate.bias"):
                with safe_open(f"{md}/{idx[n]}", "pt") as f:
                    out.append(f.get_tensor(n).to(DEV).float())
            return out
        g = torch.Generator(device="cpu").manual_seed(17)
        return (torch.randn(384, 5120, generator=g) * 0.02).to(DEV), (torch.randn(384, generator=g) * 0.1).to(DEV)

    def test_router_equal(self):
        import torch.nn.functional as F
        gate_w, gate_b = self._gate()
        g = torch.Generator(device="cpu").manual_seed(19)
        for T in (4, 5, 6):
            for pruned in (False, True):
                y = _act((T, 5120), 80 + T) * 0.05 + 0.01
                keep = (torch.rand(384, generator=g) < 0.61).to(DEV) if pruned else None
                # torch spelling (FastDecoder._layer_a before decode_lean)
                scores = F.softplus(R.mm(y.float(), gate_w)).sqrt()
                logits = scores + gate_b
                ref_logits = logits.clone()
                if keep is not None:
                    logits = logits.masked_fill(~keep, float("-inf"))
                idx = logits.topk(6, dim=-1)[1]
                wts = scores.gather(1, idx)
                wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-20) * 2.5
                got_idx = torch.zeros(T, 6, dtype=torch.long, device=DEV)
                got_w = torch.zeros(T, 6, device=DEV)
                seen = {}
                got_scores = self.lean.router(y, gate_w, gate_b, keep, 6, 2.5, got_idx, got_w,
                                              record=lambda lg, sc: seen.update(lg=lg.clone(), sc=sc.clone()))
                self.assertTrue(torch.equal(got_idx, idx), (T, pruned))
                self.assertTrue(torch.equal(got_w, wts), (T, pruned))
                self.assertTrue(torch.equal(got_scores, scores), (T, pruned))
                self.assertTrue(torch.equal(seen["lg"], ref_logits) and torch.equal(seen["sc"], scores))

    def test_attn_probs_equal(self):
        g = torch.Generator(device="cpu").manual_seed(23)
        for T, H, N in ((6, 32, 640), (4, 32, 128), (5, 64, 133), (6, 32, 128)):
            q = _act((T, H, 512), 90 + N) * 0.2
            kv = _act((T, N, 512), 91 + N) * 0.2
            mask = torch.rand(T, N, generator=g).to(DEV) < 0.8
            mask[0] = False  # a query that sees nothing: every p is 0, the sink takes it all
            sink = (torch.randn(H, generator=g) * 2).to(DEV)
            scale = 512 ** -0.5
            s = torch.einsum("thd,tnd->thn", q.float(), kv.float())
            scores = (s * scale).masked_fill(~mask[:, None, :], float("-inf"))
            mx = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
            p = torch.exp(scores - mx)
            ref = p / (p.sum(-1, keepdim=True) + torch.exp(sink[None, :, None] - mx))
            got = self.lean.attn_probs(torch.einsum("thd,tnd->thn", q.float(), kv.float()), mask, sink, scale)
            self.assertTrue(torch.equal(got, ref), (T, H, N, float((got - ref).abs().max())))

    def test_route_prep_and_block_null(self):
        g = torch.Generator(device="cpu").manual_seed(31)
        null = 9573
        lut = torch.randint(0, 9000, (384,), generator=g, dtype=torch.int32).to(DEV)
        lut[torch.randperm(384, generator=g)[:150].to(DEV)] = null  # pruned / not resident
        for T in (4, 6):
            idx = torch.stack([torch.randperm(384, generator=g)[:6] for _ in range(T)]).to(DEV)
            slots = torch.zeros(T, 6, dtype=torch.int32, device=DEV)
            route = self.lean.route_prep(idx, lut, slots, null)
            ref_slots = lut[idx]
            pair = torch.arange(T * 6, dtype=torch.int32, device=DEV).view_as(ref_slots)
            ref_route = torch.where(ref_slots == null, null + 1 + pair, ref_slots)
            self.assertTrue(torch.equal(slots, ref_slots) and torch.equal(route, ref_route), T)
            bs = torch.randint(0, null + 40, (T * 6,), generator=g, dtype=torch.int32).to(DEV)
            ref_bs = torch.where(bs > null, torch.full_like(bs, null), bs)
            self.assertTrue(torch.equal(self.lean.block_null(bs.clone(), null), ref_bs), T)

    def test_hc_post_split_layout(self):
        T, W = 6, 2560
        res = _act((T, HC, 5120), 81)
        gathered = _act((2 * T, W), 82).float() * 7  # [world*T, W] as all_gather_into_tensor leaves it
        post = torch.rand(T, HC, device=DEV)
        comb = torch.rand(T, HC, HC, device=DEV)
        x = gathered.view(2, T, W).transpose(0, 1).reshape(T, 2 * W)
        ref = hc_post(x.to(torch.bfloat16), res, post, comb)
        self.assertTrue(torch.equal(hc_post(gathered, res, post, comb, split_w=W), ref))
        inplace = res.clone()
        hc_post(gathered, inplace, post, comb, out=inplace, split_w=W)
        self.assertTrue(torch.equal(inplace, ref))

    def test_routed_shared_sum_before_gather(self):
        """The merged MoE path sums rank-local column blocks, then gathers; the old one gathered
        both, then summed. Emulate a two-rank gather and compare what reaches hc_post."""
        T, W = 6, 2560
        routed = [_act((T, W), 83 + r).float() * 3 for r in range(2)]
        shared = [_act((T, W), 85 + r) for r in range(2)]
        full_r = torch.stack(routed).transpose(0, 1).reshape(T, 2 * W)
        full_s = torch.stack(shared).transpose(0, 1).reshape(T, 2 * W)
        old = (full_r + full_s.float()).to(torch.bfloat16)
        new = torch.cat([routed[r] + shared[r] for r in range(2)])  # [2*T, W], gather layout
        new = new.view(2, T, W).transpose(0, 1).reshape(T, 2 * W).to(torch.bfloat16)
        self.assertTrue(torch.equal(old, new))

    def test_keys_f32(self):
        from engine.packed_kv import gather, write
        g = torch.Generator(device="cpu").manual_seed(37)
        ring = _act((128, 512), 101)
        cache = torch.zeros(4096, 36, dtype=torch.int64, device=DEV)  # 512 values = 32 words + 4 scale words
        vals = _act((4096, 512), 102)
        vals[5, :7] = -0.0  # negative zeros survive only if the sign is set on the bf16 bits
        write(cache, vals, 0)
        for T, n2 in ((6, 512), (4, 512), (6, 0)):
            slot_r = torch.randint(0, 128, (T, 128), generator=g).to(DEV)
            if n2:
                idx = torch.randint(0, 4096, (T, n2), generator=g).to(DEV)
                idx[0, :3] = 5
                ref = torch.cat([ring[slot_r], gather(cache, idx)], dim=1).float()
                got = self.lean.keys_f32(ring, slot_r, cache, idx)
            else:
                ref = ring[slot_r].float()
                got = self.lean.keys_f32(ring, slot_r)
            self.assertTrue(torch.equal(got, ref), (T, n2))
            self.assertTrue(torch.equal(torch.signbit(got), torch.signbit(ref)), (T, n2))

    def test_rope_f32_modes(self):
        rope = lambda x, f, inv: torch.cat([x[..., :-64], R.apply_rotary(x[..., -64:], f, inverse=inv)], -1)
        for x, f, inv, ref in self._rope_cases():
            self.assertTrue(torch.equal(self.lean.rope(x, f, 64, inv, out_f32=True), ref.float()))
            x32 = x.float() * 1.001  # not bf16-representable: the kernel must round it first
            self.assertTrue(torch.equal(self.lean.rope(x32, f, 64, inv), rope(x32.to(torch.bfloat16), f, inv)))

    def _rope_cases(self):
        fw = R.precompute_freqs_cis(64, 600_000, 0, 10000, 16, 32, 1, DEV)
        fc = R.precompute_freqs_cis(64, 600_000, 65536, 160000, 16, 32, 1, DEV)
        g = torch.Generator(device="cpu").manual_seed(13)
        for freqs in (fw, fc):
            for shape in ((6, 32, 512), (6, 512), (5, 32, 512), (3, 512), (6, 64, 128), (4, 128)):
                pos = torch.randint(0, 590_000, (shape[0],), generator=g).to(DEV)
                x = _act(shape, int(pos[0]) % 1000)
                for inverse in (False, True):
                    ref = torch.cat([x[..., :-64], R.apply_rotary(x[..., -64:], freqs[pos], inverse=inverse)], -1)
                    yield x, freqs[pos], inverse, ref

    def test_rope_contraction_search(self):
        """Report every (RE, IM) contraction that reproduces torch; the configured one must."""
        from engine import decode_lean as DL
        ok = []
        for re_ in (0, 1, 2):
            for im in (0, 1, 2):
                if all(torch.equal(self.lean.rope(x, f, 64, inv, re=re_, im=im), ref)
                       for x, f, inv, ref in self._rope_cases()):
                    ok.append((re_, im))
        print(f"\nrope contractions equal to torch: {ok}")
        self.assertIn((DL.ROPE_RE, DL.ROPE_IM), ok)

    def test_rope_fp32_contraction(self):
        """The contraction decode_lean uses must equal torch's complex multiply in fp32, before the
        bf16 rounding that can hide a difference."""
        import triton
        import triton.language as tl
        from engine import decode_lean as DL

        @triton.jit
        def _mul(A, B, C, D, RO, IO, N, RE: tl.constexpr, IM: tl.constexpr, BS: tl.constexpr):
            i = tl.program_id(0) * BS + tl.arange(0, BS)
            m = i < N
            a = tl.load(A + i, mask=m); b = tl.load(B + i, mask=m)
            c = tl.load(C + i, mask=m); d = tl.load(D + i, mask=m)
            re = a * c - b * d if RE == 0 else (tl.fma(a, c, -(b * d)) if RE == 1 else tl.fma(-b, d, a * c))
            im = a * d + b * c if IM == 0 else (tl.fma(a, d, b * c) if IM == 1 else tl.fma(b, c, a * d))
            tl.store(RO + i, re, mask=m); tl.store(IO + i, im, mask=m)

        N = 1 << 20
        g = torch.Generator(device="cpu").manual_seed(29)
        x = (torch.randn(N, 2, generator=g) * torch.logspace(-2, 2, N)[:, None]).bfloat16().float().to(DEV)
        f = R.precompute_freqs_cis(64, 600_000, 0, 10000, 16, 32, 1, DEV).reshape(-1)
        fc = f[torch.randint(0, f.numel(), (N,), generator=g).to(DEV)]
        ref = torch.view_as_real(torch.view_as_complex(x.contiguous()) * fc)
        cd = torch.view_as_real(fc)
        ro, io = torch.empty(N, device=DEV), torch.empty(N, device=DEV)
        _mul[(triton.cdiv(N, 1024),)](x[:, 0].contiguous(), x[:, 1].contiguous(), cd[:, 0].contiguous(),
                                      cd[:, 1].contiguous(), ro, io, N, RE=DL.ROPE_RE, IM=DL.ROPE_IM, BS=1024,
                                      enable_fp_fusion=False)
        self.assertTrue(torch.equal(ro, ref[:, 0]) and torch.equal(io, ref[:, 1]))

    def test_graph_replay(self):
        T = 6
        w = self.t["q_norm"]
        x_in = torch.zeros(T, 1280, dtype=torch.bfloat16, device=DEV)
        h_in = torch.zeros(T, HC, 5120, dtype=torch.bfloat16, device=DEV)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):  # warm-up creates the buffers outside the capture
            self.lean.rmsnorm(x_in, w, EPS)
            self.lean.hc_mixes(h_in, self.t["hc_fn"], self.t["hc_scale"], self.t["hc_base"], EPS, HC, ITERS, HC_EPS)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = self.lean.rmsnorm(x_in, w, EPS)
            mixes = self.lean.hc_mixes(h_in, self.t["hc_fn"], self.t["hc_scale"], self.t["hc_base"],
                                       EPS, HC, ITERS, HC_EPS)
        for seed in (41, 42, 43):
            x_in.copy_(_act((T, 1280), seed))
            h_in.copy_(_act((T, HC, 5120), seed + 100))
            g.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(y, R.rmsnorm(x_in, w, EPS)), seed)
            for a, b in zip(mixes, _hc_ref(h_in, self.t)):
                self.assertTrue(torch.equal(a, b), seed)


if __name__ == "__main__":
    unittest.main()
