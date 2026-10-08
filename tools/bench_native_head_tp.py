"""Bounded native/packed ABBA gate with one frozen TP2 model and expert map.

Run with run_two_node_gate.sh after stopping serving. No adaptation/history
writes. Keep native and packed control heads only in this diagnostic, then swap
the same model's head and release its graphs at each arm. Assert both rank token
streams, every sampled eager head logit, acceptance and greedy output parity.
"""
import argparse
import hashlib
import json
import os
import sys

sys.path[:0] = ['/app', '/app/tools']
# The timeline helper freezes adaptation and prefix caching before importing V.
from bench_decode_timeline_tp import V, Tok, load_encoding_module, build_chat_prompt
import torch
import v41_ref as R
from bench.bench import WORKLOADS
from engine.expert_profiles import mask_digest
from engine.native_head import make_packed_head
from engine.tensor_parallel import VocabParallelHead, draft_head_bytes


def fatal(kind, error, tb):
    import traceback
    traceback.print_exception(kind, error, tb)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(1)


sys.excepthook = fatal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-tokens', type=int, default=160)
    args = parser.parse_args()
    assert 64 <= args.max_tokens <= 384
    assert os.environ.get('DSV41_HEAD_KERNEL', 'off') == 'off'
    V.save_prune_db = lambda *a, **k: None
    os.makedirs(args.out, exist_ok=True)
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=65536, arena_gb=90.2,
                    trace_stats='/app/results/trace-union/stats/coverage.json',
                    spec=True, prune_keep=.61, transient_slots=8, keep_free_gb=6)
    assert e.dynamic_experts and not V.ADAPT.swap and not V.ADAPT.decode_tokens
    assert e.W.draft_head is None and e.confidence_depth_policy is None
    native = e.W.head
    assert isinstance(native, VocabParallelHead)
    assert isinstance(native.local, torch.Tensor) and native.local.dtype == torch.bfloat16
    packed = VocabParallelHead(make_packed_head(native.local), native.world)
    # Compare every stored bit without a full-size BF16 temporary.
    for first in range(0, native.local.shape[0], 4096):
        last = min(first + 4096, native.local.shape[0])
        assert torch.equal(packed.local.dequant_rows(first, last).view(torch.int16),
                           native.local[first:last].view(torch.int16))
    map_hash = mask_digest(e.model_prune_mask)
    assert len(set(e.ep.gather_objects(map_hash))) == 1
    tok, enc = Tok(root), load_encoding_module(root)
    prompts = {}
    for name in ('code', 'prose'):
        _, prompts[name], _, _ = build_chat_prompt(
            {'messages': [{'role': 'user', 'content': '[req 41420] ' + WORKLOADS[name]}]},
            enc, tok, False, 75, e)
    report = dict(config=e.config(), map=map_hash, runs=[], checks=[],
                  native_bytes=draft_head_bytes(native), packed_bytes=draft_head_bytes(packed))
    references = {}
    raw_head = R.head_logits
    checking = False

    def checked_head(x, head):
        out = raw_head(x, head)
        if (checking and head is packed.local and not torch.cuda.is_current_stream_capturing()
                and len(report['checks']) < 48):
            ref = raw_head(x, native.local)
            same = torch.equal(out, ref)
            report['checks'].append(dict(rows=x.numel() // x.shape[-1], exact=same,
                max_abs=float((out - ref).abs().max()), argmax_exact=torch.equal(out.argmax(-1), ref.argmax(-1))))
            assert same, 'actual head activation logits changed'
        return out

    R.head_logits = checked_head

    def save():
        path = f'{args.out}/gate-rank{e.ep.rank}.json'
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        owner = os.stat('/app/results')
        os.chown(path, owner.st_uid, owner.st_gid)

    def run(name, label, count, measured):
        out = []
        for burst in e.generate(prompts[name], max_tokens=count, temperature=0,
                                seed=42, ignore_eos=True):
            out.extend(burst)
        digest = hashlib.sha256(json.dumps(out).encode()).hexdigest()
        assert len(set(e.ep.gather_objects(digest))) == 1, 'rank token streams differ'
        assert mask_digest(e.model_prune_mask) == map_hash, 'expert map changed'
        key = (name, count)
        exact = references.setdefault(key, digest) == digest
        st = dict(e.last_stats)
        row = dict(workload=name, label=label, measured=measured, tokens=len(out),
                   token_ids=out, text=tok.decode(out), hash=digest, exact=exact,
                   decode_s=st['decode_s'], decode_tok_s=st['decode_tok_s'],
                   prefill_s=st['prefill_s'], steps=st['steps'],
                   accept_len_mean=st['accept_len_mean'], spec_depth=st['spec_depth'])
        report['runs'].append(row); save()
        print('NATIVE_HEAD_RUN ' + json.dumps({k: v for k, v in row.items()
              if k not in ('text', 'token_ids')}), flush=True)
        assert exact, 'greedy output changed'

    for index, label in enumerate(('native', 'packed', 'packed', 'native')):
        e.fast.release_graphs()
        e.W.head = e.fast.head = e.fast.draft_head = {'native': native, 'packed': packed}[label]
        os.environ['DSV41_HEAD_KERNEL'] = 'packed' if label == 'packed' else 'off'
        e.depth_policy.pinned = 3
        assert len(set(e.ep.gather_objects((index, label)))) == 1
        checking = label == 'packed'
        run('code', label, 64, False)
        checking = False
        for name in ('code', 'prose'):
            run(name, label, args.max_tokens, True)
    assert report['checks'], 'no actual-activation head checks ran'
    assert all(e.ep.gather_objects(True))
    report['passed'] = True; save()
    print('NATIVE_HEAD_GATE_PASS', flush=True)
    sys.stdout.flush(); os._exit(0)


if __name__ == '__main__':
    main()
