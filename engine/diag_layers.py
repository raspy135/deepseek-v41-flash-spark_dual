"""Per-layer / per-tensor divergence diagnostic.

Two modes:
  * `python engine/diag_layers.py ref [seq]`     -- engine vs tools/v41_ref, single chunk
  * `python engine/diag_layers.py chunk [seq] [c1,c2,...]`
        -- single-chunk vs chunked, comparing EVERY tapped tensor (residual stream, window KV
           rows/mask, compressed rows/mask, compressor latents, indexer top-k, engram rows)
           position by position, so the first deviating (layer, tensor, position) is visible.
"""
import glob, json, os, sys, time, torch, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from engine.model import Caches, Model, Weights, Shared
from engine import experts as EX, moe_fallback as K
from engine.engram import EngramTable, make_hash_state
from safetensors import safe_open
import v41_ref as R

md = os.environ.get("MODEL_DIR") or "./models/DeepSeek-V4.1-Flash"; dev = "cuda"
index = json.load(open(f"{md}/model.safetensors.index.json")); args = R.Args.from_json(f"{md}/inference/config.json")
NL = int(os.environ.get("NL", "4"))
MODE = sys.argv[1] if len(sys.argv) > 1 else "chunk"
if MODE == "nest" and NL != args.n_layers:
    raise ValueError(f"nest logits require the full model: set NL={args.n_layers}, not {NL}")
SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CHUNKS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None

W = Weights(md, index, args, dev, act_quant=True, n_layers=NL, load_mtp=False)
arena = K.ExpertArena(1600, dev); store = EX.ExpertStore(md, index, arena, NL, transient_slots=400, io_threads=8)
caches = Caches(args, 4096, dev); m = Model(W, store, caches, K.moe_forward, act_quant=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(md); m.hash_state = make_hash_state(md, tok, 4096, dev)

# Use the same local-shard row reader as the serving engine.  The old diagnostic depended on
# pre-extracted ``engram_rows/*.npz`` files, which are not part of a normal full-checkpoint
# install and made the reference comparison unusable on the machine it is meant to diagnose.
tables = {L: EngramTable(md, index, L, dev) for L in args.engram_layer_ids}

def rows_from_store(L, hashes):
    return tables[L].rows(hashes)
m.engram_rows = rows_from_store

TRACE = sorted(glob.glob("results/trace-*"))[-1]  # newest results/trace-<name>/
meta = json.load(open(f"{TRACE}/meta.json"))
corpus = {}
for corpus_path in sorted(glob.glob("corpus/trace_corpus*.jsonl")):
    for line in open(corpus_path):
        row = json.loads(line)
        corpus[row["id"]] = row["text"]
if MODE == "nest":
    # Teacher-forced nesting probe (same prompt as llm_benchmark/quality_quant2.py, same chat
    # template the server uses). The sequence is prompt + the CANONICAL correct answer, so every
    # position has a known target: a position where the engine's argmax is not that target is a
    # real fork in the engine's computation, not accumulated free-running drift.
    sys.path.insert(0, os.path.join(md, "encoding"))
    from encoding import encode_messages
    _d = SEQ or 8
    _leaf = 40 + _d
    _prompt = ('Output one JSON object nested exactly %d levels deep and nothing else. '
               'Each level has exactly one key "n" whose value is the next level down. '
               'The innermost "n" is the integer %d. '
               'So depth 2 would be: {"n": {"n": %d}}') % (_d, _leaf, _leaf)
    _answer = '{"n": ' * (_d - 1) + '{"n": %d}' % _leaf + '}' * (_d - 1)
    _pr = encode_messages([{"role": "user", "content": _prompt}], thinking_mode="chat")
    _pr = _pr[0] if isinstance(_pr, tuple) else _pr
    pids = tok.encode(_pr, add_special_tokens=False)
    aids = tok.encode(_answer, add_special_tokens=False)
    ids = torch.tensor(pids + aids, device=dev); T = ids.numel()
    N_PROMPT = len(pids)
    print(f"nest depth={_d} prompt_tokens={N_PROMPT} answer_tokens={len(aids)} total={T}")
    print(f"  teacher-forced answer: {_answer!r}")
else:
    sid = meta["seqs"][SEQ]["id"]
    ids = torch.tensor(tok.encode(corpus[sid], add_special_tokens=False), device=dev); T = ids.numel()
    print(sid, "T", T, "mode", MODE)


def reset():
    caches.len = 0; caches._chunk_inputs.clear()
    for L in caches.pending: caches.pending[L] = None
    for t in caches.win: t.zero_()
    for L in caches.ckv: caches.ckv[L].zero_(); caches.ik[L].zero_()


def run_taps(ids, chunks):
    """Run `ids` in `chunks` and collect every tapped tensor keyed by (name, L), stored with
    absolute token positions on dim 0 (latents keyed by absolute compressed group index)."""
    reset()
    store_ = {}
    s = 0
    for T_ in chunks:
        cur = {"S": s}
        def tap(name, L, t, cur=cur):
            key = (name, L)
            if name == "latent":
                j0, lat = t
                d = store_.setdefault(key, {})
                for i in range(lat.shape[0]): d[j0 + i] = lat[i]
            elif name == "n_c":
                store_.setdefault(key, {})[cur["S"]] = t
            else:
                d = store_.setdefault(key, {})
                for i in range(t.shape[0]): d[cur["S"] + i] = t[i].clone()
        m.tap = tap
        m.forward(ids[s:s + T_], s, prefill=True, need_logits=False)
        m.tap = None
        s += T_
    return store_


def rel(a, b):
    a, b = a.float(), b.float()
    n = b.norm()
    return float((a - b).norm() / n) if n > 0 else float((a - b).norm())


if MODE in ("ref", "nest"):
    handles = {}
    def get(name):
        f = index["weight_map"][name]
        if f not in handles: handles[f] = safe_open(f"{md}/{f}", "pt", device="cpu")
        return handles[f].get_tensor(name)
    hashes = m.hash_state(ids[None], 0)[0]
    h_ref = W.embed[ids].unsqueeze(1).repeat(1, 4, 1); st = R.SeqState(h_ref, torch.zeros(T, 4, device=dev)); st.pre_mix[:, 0] = 1
    a = args; reset()
    h = W.embed[ids].unsqueeze(1).repeat(1, a.hc_mult, 1); pre = torch.zeros(T, a.hc_mult, device=dev); pre[:, 0] = 1.0; sh = Shared()

    def ref_block_taps(st, w, experts, cache):
        """Reference block_forward with diagnostic taps at the engine's observable boundaries."""
        taps = {}
        x = st.h; residual = x
        attn_pre, attn_post, attn_comb = R.hc_mixes(
            x, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, a)
        y = R.rmsnorm(R.hc_pre(x, st.pre_mix), w.attn_norm, a.norm_eps)
        taps["attn_x"] = y
        if not w.ratio:
            freqs = R.precompute_freqs_cis(
                a.rope_head_dim, y.size(0), 0, a.rope_theta, a.rope_factor,
                a.beta_fast, a.beta_slow, str(y.device))
            qr = R.rmsnorm(R.qlinear(y, w.wq_a), w.q_norm, a.norm_eps)
            q = R.qlinear(qr, w.wq_b).view(y.size(0), a.n_heads, a.head_dim)
            q = torch.cat([q[..., :-a.rope_head_dim],
                           R.apply_rotary(q[..., -a.rope_head_dim:], freqs)], dim=-1)
            kv = R.rmsnorm(R.qlinear(y, w.wkv), w.kv_norm, a.norm_eps)
            kv = torch.cat([kv[:, :-a.rope_head_dim],
                            R.apply_rotary(kv[:, -a.rope_head_dim:], freqs)], dim=-1)
            kv = R.act_qdq_fp8(kv)
            taps["q"], taps["kv_new"] = q, kv
            pos = torch.arange(y.size(0), device=y.device)
            mask = ((pos[None, :] <= pos[:, None]) &
                    (pos[None, :] > pos[:, None] - a.window_size))
            scores = torch.einsum("thd,nd->thn", q.float(), kv.float()) * a.head_dim ** -0.5
            scores.masked_fill_(~mask[:, None, :], float("-inf"))
            mx = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
            p = torch.exp(scores - mx)
            denom = p.sum(-1, keepdim=True) + torch.exp(w.attn_sink[None, :, None] - mx)
            o = torch.einsum("thn,nd->thd", p / denom, kv.float()).to(torch.bfloat16)
            taps["attn_core"] = o
            o = torch.cat([o[..., :-a.rope_head_dim],
                           R.apply_rotary(o[..., -a.rope_head_dim:], freqs, inverse=True)], dim=-1)
            o = R.wo_a_proj(o.reshape(y.size(0), a.o_groups, -1), w.wo_a)
            taps["attn_wo_a"] = o
            y = R.qlinear(o.flatten(1), w.wo_b)
        else:
            y = R.attention(y, w, st, a)
        taps["attn_out"] = y
        x = R.hc_post(y, residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb = R.hc_mixes(
            x, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, a)
        y = R.rmsnorm(R.hc_pre(x, attn_pre), w.ffn_norm, a.norm_eps)
        taps["moe_in"] = y
        weights, indices, _ = R.router(y, w, a)
        taps["route_idx"], taps["route_w"] = indices, weights
        routed = torch.zeros_like(y, dtype=torch.float32)
        for e in torch.unique(indices).tolist():
            if e not in cache:
                cache[e] = experts(e)
            w1, w2, w3 = cache[e]
            idx, top = torch.where(indices == e)
            routed[idx] += R.expert_ffn(
                y[idx], w1, w2, w3, a.swiglu_limit, weights[idx, top, None]).float()
        shared = R.expert_ffn(y, w.sh_w1, w.sh_w2, w.sh_w3, a.swiglu_limit).float()
        taps["moe_routed"], taps["moe_shared"] = routed, shared
        st.h = R.hc_post((routed + shared).to(y.dtype), residual, ffn_post, ffn_comb)
        st.pre_mix = ffn_pre
        return taps

    for L in range(NL):
        w = W.layers[L]
        if L in W.engram:
            li = list(a.engram_layer_ids).index(L); rows = rows_from_store(L, hashes[:, li, :])
            st.h = R.engram_forward(st.h, rows, W.engram[L], a); h = R.engram_forward(h, rows, W.engram[L], a)
        rt = ref_block_taps(st, w, R.ExpertLoader(get, L, dev), {})
        freqs = m.freqs_c if w.ratio else m.freqs_w
        et = {}
        m.tap = lambda name, layer, value: et.__setitem__(
            name, value.clone() if torch.is_tensor(value) else value)
        h, pre = m.block(h, pre, w, L, 0, sh, caches.win[L], freqs, True, store, arena, 384)
        m.tap = None
        detail = []
        for name in ("attn_x", "q", "kv_new", "attn_core", "attn_wo_a", "attn_out",
                     "moe_in", "route_w", "moe_routed", "moe_shared"):
            if name in et and name in rt:   # compressor layers have no ref attention-internal taps
                detail.append(f"{name}={rel(et[name], rt[name]):.4f}")
        route_eq = float((et["route_idx"] == rt["route_idx"]).float().mean())
        print(f"L{L} (ratio {w.ratio}) rel err h {rel(h, st.h):.4f} pre_mix {rel(pre, st.pre_mix):.4f}")
        print(f"    {' '.join(detail)} route_pos_eq={route_eq:.3f}")
        if os.environ.get("ISOLATE"): h = st.h.clone(); pre = st.pre_mix.clone()

    if MODE == "nest":
        # Head on both final residuals, then compare next-token distributions position by position.
        x_e = R.rmsnorm(R.hc_pre(h, pre), W.norm, a.norm_eps)
        x_r = R.rmsnorm(R.hc_pre(st.h, st.pre_mix), W.norm, a.norm_eps)
        le = R.head_logits(x_e, W.head).float()
        lr = R.head_logits(x_r, W.head).float()
        print(f"\nhead-input rel err {rel(x_e, x_r):.4f}")
        first_neq = first_eng_wrong = first_ref_wrong = None
        for i in range(N_PROMPT - 1, T - 1):
            tgt = int(ids[i + 1])
            ea, ra = int(le[i].argmax()), int(lr[i].argmax())
            er = int((le[i] > le[i][tgt]).sum()); rr = int((lr[i] > lr[i][tgt]).sum())
            if ea != ra and first_neq is None: first_neq = i
            if ea != tgt and first_eng_wrong is None: first_eng_wrong = i
            if ra != tgt and first_ref_wrong is None: first_ref_wrong = i
            print(f"  pos {i:3d} target={tok.decode([tgt])!r:16s} eng_top1={tok.decode([ea])!r:16s} "
                  f"rank={er} margin={float(le[i].max() - le[i][tgt]):+.3f} | "
                  f"ref_top1={tok.decode([ra])!r:16s} rank={rr} margin={float(lr[i].max() - lr[i][tgt]):+.3f}")
        print(f"\nfirst engine/ref argmax disagreement: pos {first_neq}")
        print(f"first position the ENGINE fails to rank the target first: pos {first_eng_wrong}")
        print(f"first position the REF    fails to rank the target first: pos {first_ref_wrong}")
        for who, lg, first in (("engine", le, first_eng_wrong), ("ref", lr, first_ref_wrong)):
            if first is None: continue
            i = first
            print(f"  {who} top-5 at pos {i}: " + ", ".join(
                f"{tok.decode([int(j)])!r}:{float(lg[i][j]):.2f}" for j in lg[i].topk(5).indices.tolist()))
    sys.exit(0)

# ---------------------------------------------------------------- chunked vs single
chunks = CHUNKS or [T // 3, T - T // 3]
print("chunks", chunks)
A = run_taps(ids, [T])
B = run_taps(ids, chunks)
bound = set(np.cumsum(chunks[:-1]).tolist())
print("chunk boundaries at absolute positions", sorted(bound))

names = ["engram_rows", "engram_out", "attn_x", "q", "kv_new", "win_kv", "win_mask", "attn_out",
         "moe_in", "route_w", "moe_routed", "moe_shared", "h", "pre_mix"]
for L in range(NL):
    # latents first (indexed by compressed group)
    key = ("latent", L)
    if key in A:
        da, db = A[key], B[key]
        common = sorted(set(da) & set(db))
        errs = [(j, rel(db[j], da[j])) for j in common]
        nex = sum(1 for j in common if not torch.equal(da[j], db[j]))
        print(f"  L{L} latent: {len(common)} groups, max err {max(e[1] for e in errs):.2e} at j={max(errs, key=lambda e: e[1])[0]}, "
              f"{nex} not bit-equal")
    # compressed visibility as a set of absolute compressed positions per token
    key = ("topk", L)
    if key in A:
        da, db = A[key], B[key]
        common = sorted(set(da) & set(db))
        diff = [p for p in common if set(da[p][da[p] >= 0].tolist()) != set(db[p][db[p] >= 0].tolist())]
        print(f"  L{L} visible-compressed-set: {len(diff)}/{len(common)} tokens differ {diff[:10]}")
    key = ("route_idx", L)
    if key in A:
        da, db = A[key], B[key]
        common = sorted(set(da) & set(db))
        diff = [p for p in common if not torch.equal(da[p].sort().values, db[p].sort().values)]
        print(f"  L{L} router expert set: {len(diff)}/{len(common)} tokens differ {diff[:10]}")
    for nm in names:
        key = (nm, L)
        if key not in A: continue
        da, db = A[key], B[key]
        common = sorted(set(da) & set(db))
        if nm in ("win_mask", "route_idx"):
            diff = [p for p in common if not torch.equal(da[p], db[p])]
            print(f"  L{L} {nm}: {len(diff)}/{len(common)} positions differ {diff[:10]}")
            continue
        errs = np.array([rel(db[p], da[p]) for p in common])
        i = int(errs.argmax())
        nex = sum(1 for p in common if not torch.equal(da[p], db[p]))
        first = [common[k] for k in range(len(common)) if errs[k] > 1e-2][:10]
        print(f"  L{L} {nm}: max {errs.max():.2e} @pos {common[i]}, median {np.median(errs):.2e}, "
              f"{nex}/{len(common)} not bit-equal, first>1e-2 {first}")
