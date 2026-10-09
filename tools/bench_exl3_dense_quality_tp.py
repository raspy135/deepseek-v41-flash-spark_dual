"""Teacher-forced decode logits, FP8 vs EXL3 dense attention + shared expert, on the same tokens.

Load with DSV41_EXL3_DENSE=1 (both weight sets resident; tools/bench_accept_ab_tp.py's x3_off /
x3_on switch at runtime). Per prompt: a greedy reference continuation with x3_off, then each arm
re-prefills the prompt (FP8 in both: prefill does not use the pack) and walks the reference in
4-row verify blocks through FastDecoder.step -- the graph the arm serves with -- keeping its own
KV. Reports per-position KL(FP8 || EXL3) in nats, top-1 agreement, and how often EXL3's argmax
is the reference's next token. Use the disposable two-node gate, not serving.
"""
import argparse
import json
import os
import sys

sys.path[:0] = ['/app', '/app/tools']
import bench_accept_ab_tp as AB  # noqa: E402  (sets the frozen-map environment first)
import torch  # noqa: E402

V = AB.V


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--prompts', type=int, default=12)
    ap.add_argument('--tokens', type=int, default=96)
    args = ap.parse_args()
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=32768, arena_gb=float(os.environ.get('DSV41_BENCH_ARENA_GB', '88')),
                    trace_stats='/app/results/trace-union/stats/coverage.json',
                    spec=True, prune_keep=.61, transient_slots=8, keep_free_gb=6, expert_format='exl3')
    assert e.exl3_dense is not None, 'load with DSV41_EXL3_DENSE=1'
    e.confidence_depth_policy = None
    e.depth_policy.pinned = 3
    tok, enc = AB.Tok(root), AB.load_encoding_module(root)
    eos = tok.token_to_id(enc.eos_token)
    m, fd = e.model, e.fast
    T = 4

    def walk(ids, ref):
        for _ in e.generate(ids, max_tokens=1, temperature=0, seed=42):
            pass
        pos = m.c.len
        out = []
        for k in range(len(ref) // T):
            block = torch.tensor(ref[k * T:(k + 1) * T], device=e.device)
            hashes = m.hash_state(block[None], pos)[0]
            rows = {L: e.tables[L].rows(hashes[:, li, :]) for li, L in enumerate(e.args.engram_layer_ids)}
            logits, _ = fd.step(block, pos, rows)
            out.append(logits.float().clone())
            pos += T
        return torch.cat(out)

    os.makedirs(args.out, exist_ok=True)
    rows_out, kl_all, top1_all, hit_all = [], [], [], []
    for i, p in enumerate(AB.PROMPTS[:args.prompts]):
        ids = AB.build_chat_prompt({'messages': [{'role': 'user', 'content': p}]}, enc, tok, False, 75, e)[1]
        AB.set_config(e, AB.CONFIGS['x3_off'])
        ref = []
        for burst in e.generate(ids, max_tokens=args.tokens + 1, temperature=0, seed=42, stop_token_ids={eos}):
            ref.extend(burst)
        n = (len(ref) - 1) // T * T
        if n < T:
            continue
        ref = ref[:n + 1]
        lo = walk(ids, ref[:n])
        AB.set_config(e, AB.CONFIGS['x3_on'])
        lx = walk(ids, ref[:n])
        lp_o, lp_x = torch.log_softmax(lo, -1), torch.log_softmax(lx, -1)
        kl = (lp_o.exp() * (lp_o - lp_x)).sum(-1)
        top1 = (lo.argmax(-1) == lx.argmax(-1)).float()
        nxt = torch.tensor(ref[1:n + 1], device=lo.device)
        hit_o = (lo.argmax(-1) == nxt).float()
        hit_x = (lx.argmax(-1) == nxt).float()
        kl_all.append(kl); top1_all.append(top1); hit_all.append((hit_o, hit_x))
        row = dict(prompt=i, positions=n, kl_mean=float(kl.mean()), kl_p99=float(kl.quantile(.99)),
                   kl_max=float(kl.max()), top1=float(top1.mean()), ref_hit_fp8=float(hit_o.mean()),
                   ref_hit_x3=float(hit_x.mean()))
        rows_out.append(row)
        print('X3_QUALITY ' + json.dumps(dict(rank=e.ep.rank, **row)), flush=True)
    kl = torch.cat(kl_all)
    top1 = torch.cat(top1_all)
    summary = dict(prompts=len(rows_out), positions=int(kl.numel()), kl_mean=float(kl.mean()),
                   kl_median=float(kl.median()), kl_p99=float(kl.quantile(.99)), kl_max=float(kl.max()),
                   top1=float(top1.mean()),
                   ref_hit_fp8=float(torch.cat([h[0] for h in hit_all]).mean()),
                   ref_hit_x3=float(torch.cat([h[1] for h in hit_all]).mean()))
    print('X3_QUALITY_SUMMARY ' + json.dumps(summary), flush=True)
    path = f'{args.out}/quality-rank{e.ep.rank}.json'
    with open(path, 'w') as f:
        json.dump(dict(summary=summary, prompts=rows_out), f, indent=1)
    owner = os.stat('/app/results')
    os.chown(path, owner.st_uid, owner.st_gid)
    assert all(e.ep.gather_objects(True))
    os._exit(0)


if __name__ == '__main__':
    main()
