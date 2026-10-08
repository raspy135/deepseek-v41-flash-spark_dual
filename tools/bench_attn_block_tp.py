"""Attention-block kernel profile per decode configuration, in one loaded TP2 process.

Same setup as bench_round_breakdown_tp (frozen map, fixed depth 3, code prompt, CUPTI
kernel trace of a middle decode window), repeated for runtime-switchable arithmetic
paths: staged attention (DSV41_ATTN_STAGED), BF16-valued router gate with FP32
accumulation (DSV41_ROUTER_BF16), the fused router tail (DSV41_ROUTER_FUSED_TAIL) and
the split-K Hyper-Connection projection (DSV41_HC_KERNEL).
Graphs are released between configurations; both ranks switch identically. The
baseline is profiled first and last. Different arithmetic may change tokens, so the
trace is analyzed per layer and per operation, not as end-to-end throughput.
Use the disposable two-node gate, not serving.
"""
import argparse
import hashlib
import json
import os
import sys
sys.path[:0] = ['/app', '/app/tools']
for key in ('DSV41_PRUNE_SWAP', 'DSV41_PRUNE_SWAP_PREFILL', 'DSV41_PREFIX_CACHE',
            'DSV41_PREFIX_DISK', 'DSV41_PREFIX_RESPONSE', 'DSV41_GPU_TIMING', 'DSV41_STEP_TIMING'):
    os.environ[key] = '0'
os.environ['DSV41_PRUNE_ADAPT'] = '1'
import torch
import engine.v41_engine as V
import v41_ref
from engine.expert_profiles import mask_digest
from torch.profiler import profile, ProfilerActivity
from server.app import Tok, load_encoding_module, build_chat_prompt
from bench_decode_timeline_hooks import DecodeTimeline

CONFIGS = {
    'base': dict(staged=0, router_bf16=False, fused_tail=False, hc_kernel=False),
    'staged2': dict(staged=2, router_bf16=False, fused_tail=False, hc_kernel=False),
    'router': dict(staged=0, router_bf16=True, fused_tail=True, hc_kernel=False),
    'hc': dict(staged=0, router_bf16=False, fused_tail=False, hc_kernel=True),
    'all': dict(staged=2, router_bf16=True, fused_tail=True, hc_kernel=True),
    'base_repeat': dict(staged=0, router_bf16=False, fused_tail=False, hc_kernel=False),
}
# Draft-only variants (Markov candidate shortlist); target arithmetic is 'all'.
for _k in (128, 512):
    CONFIGS[f'all_mk{_k}'] = dict(CONFIGS['all'], markov_topk=_k)
# L2 prefetch variants (loads only; tokens must not change). attn=4 is the serving default.
for _name, _qkv, _sh, _attn in (('pf_off', 0, 0, 4), ('pf_qkv12', 12, 0, 4), ('pf_qkv24', 24, 0, 4),
                                ('pf_sh16', 0, 16, 4), ('pf_qkv24_sh16', 24, 16, 4),
                                ('pf_qkv24_sh16_attn12', 24, 16, 12)):
    CONFIGS[_name] = dict(CONFIGS['all'], qkv_mb=_qkv, sh_mb=_sh, attn_mb=_attn)
CONFIGS['pf_off2'] = dict(CONFIGS['pf_off'])
for _mode in ('bulk', 'touchn'):
    for _name, _qkv, _sh in (('q12', 12, 0), ('q24', 24, 0), ('q24s16', 24, 16)):
        CONFIGS[f'pf_{_mode}_{_name}'] = dict(CONFIGS['all'], qkv_mb=_qkv, sh_mb=_sh, attn_mb=4, pf_mode=_mode)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--configs', default=','.join(CONFIGS))
    ap.add_argument('--bursts', type=int, default=8)
    ap.add_argument('--max-tokens', type=int, default=128)
    ap.add_argument('--warmup-tokens', type=int, default=48)
    args = ap.parse_args()
    names = args.configs.split(',')
    assert all(n in CONFIGS for n in names) and 1 <= args.bursts <= 16
    assert 1 <= args.warmup_tokens < args.max_tokens <= 1024
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=32768, arena_gb=90.2,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=8, keep_free_gb=6)
    e.confidence_depth_policy = None
    e.depth_policy.pinned = 3
    fd, lean = e.fast, e.fast.lean
    assert fd is not None and lean is not None, 'lean fast decode path required'
    os.makedirs(args.out, exist_ok=True)
    tok, enc = Tok(root), load_encoding_module(root)
    from bench.bench import WORKLOADS
    _, ids, _, _ = build_chat_prompt({'messages': [{'role': 'user', 'content': '[req 70101] ' + WORKLOADS['code']}]},
                                     enc, tok, False, 75, e)
    signature = hashlib.sha256(json.dumps([ids, names, args.bursts, args.max_tokens]).encode()).hexdigest()
    assert len(set(e.ep.gather_objects(signature))) == 1, 'workload differs between ranks'
    maphash = mask_digest(e.model_prune_mask)
    kwargs = dict(max_tokens=args.max_tokens, temperature=0, seed=42, ignore_eos=True)
    owner = os.stat('/app/results')
    summary = {}
    for name in names:
        cfg = CONFIGS[name]
        torch.cuda.synchronize()
        fd.release_graphs()
        fd.staged_attention = cfg['staged']
        lean.router_bf16 = cfg['router_bf16']
        lean.fused_router_tail = cfg['fused_tail']
        v41_ref.HC_KERNEL = cfg['hc_kernel']  # read at call time by hc_kernel_ok
        fd.draft_markov_topk = cfg.get('markov_topk', 0)
        if 'qkv_mb' in cfg:
            assert fd.l2pf_side is not None, 'L2 prefetch disabled (DSV41_L2PF_MB=0)'
            fd.qkv_prefetch_mb, fd.sh_prefetch_mb, fd.attn_prefetch_mb = cfg['qkv_mb'], cfg['sh_mb'], cfg['attn_mb']
            import engine.l2pf as _l2pf
            _l2pf.MODE = cfg.get('pf_mode', 'touch')  # read by l2pf.touch at capture
        if lean.router_bf16:
            lean.prepare_router_weights([w.gate_w for w in e.W.layers])
        assert len(set(e.ep.gather_objects(json.dumps([name, cfg], sort_keys=True)))) == 1, 'configuration differs between ranks'
        warm = []
        for burst in e.generate(ids, **kwargs):
            warm.extend(burst)
        warm_stats = dict(e.last_stats)
        generator = e.generate(ids, **kwargs)
        output = []
        while len(output) < args.warmup_tokens:
            output.extend(next(generator))
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       record_shapes=False, with_stack=False, profile_memory=False)
        bursts = 0
        with DecodeTimeline(e) as timeline:
            prof.start()
            timeline.active = True
            with timeline.span('decode/window'):
                for _ in range(args.bursts):
                    try:
                        with timeline.span('decode/burst'):
                            output.extend(next(generator))
                        bursts += 1
                    except StopIteration:
                        break
                torch.cuda.synchronize()
            timeline.active = False
            prof.stop()
            for burst in generator:
                output.extend(burst)
            digest = hashlib.sha256(json.dumps(output).encode()).hexdigest()
            assert len(set(e.ep.gather_objects(digest))) == 1, 'rank disagreement'
            assert mask_digest(e.model_prune_mask) == maphash, 'expert map changed'
            row = dict(config=name, settings=cfg, bursts=bursts, hash=digest,
                       warm_matches=output == warm, warm_stats=warm_stats, stats=dict(e.last_stats))
            path = f'{args.out}/rank{e.ep.rank}-{name}.json'
            timeline.export(prof, path, dict(row, rank=e.ep.rank, config_full=e.config()))
            os.chown(path, owner.st_uid, owner.st_gid)
        summary[name] = {k: row[k] for k in ('hash', 'warm_matches', 'bursts')}
        summary[name].update(steps=warm_stats['steps'], decode_s=warm_stats['decode_s'],
                             accept_len_mean=warm_stats['accept_len_mean'])
        print('ATTN_BLOCK_CONFIG ' + json.dumps(summary[name]), flush=True)
    torch.cuda.synchronize()
    fd.release_graphs()
    path = f'{args.out}/rank{e.ep.rank}-summary.json'
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    os.chown(path, owner.st_uid, owner.st_gid)
    assert all(e.ep.gather_objects(True))
    os._exit(0)


if __name__ == '__main__':
    main()
