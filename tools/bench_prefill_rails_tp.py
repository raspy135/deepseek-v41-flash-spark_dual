"""Full-engine TP A/B: one-rail prefill vs dual-rail prefill, decode unchanged.

Use tools/run_two_node_gate.sh with --out pointing into /app/results.
Freezes adaptation/prefix reuse through the existing timeline benchmark setup.
Compares actual prefill logits and generated tokens, not just a collapse gate.
"""
import argparse
import json
import os
import sys
from unittest.mock import patch
sys.path[:0] = ['/app', '/app/tools']
from bench_decode_timeline_tp import V, torch, Tok, load_encoding_module, build_chat_prompt
from engine import collective_rails as rails


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-tokens', type=int, default=64)
    args = ap.parse_args()
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=16384, arena_gb=90,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=16, keep_free_gb=6)
    assert e.ep.tensor_parallel and e.ep.prefill_dual_rail
    assert e.fast is not None and e.fast.use_graphs
    tok, enc = Tok(root), load_encoding_module(root)
    prompt = ('Read these records, then write a Python function that sums their values.\n' +
              '\n'.join(f'Record {i}: category {i%7}, value {i%13}.' for i in range(180)) +
              '\nOutput a function accepting records as dictionaries, with a short example.')
    _, ids, _, _ = build_chat_prompt({'messages': [{'role': 'user', 'content': prompt}]},
                                    enc, tok, False, 75, e)
    assert len(set(e.ep.gather_objects(tuple(ids)))) == 1
    original_forward = e.model.forward
    logits_seen = []
    def observe(*a, **kw):
        out = original_forward(*a, **kw)
        if kw.get('prefill', a[2] if len(a)>2 else False) and out[0] is not None:
            logits_seen.append(out[0][-1].detach().cpu().clone())
        return out
    e.model.forward = observe
    # CED prefill defers final logits to the bounded decoder replay over the prompt tail.
    original_replay = e.model.decoder_replay
    def observe_replay(*a, **kw):
        out = original_replay(*a, **kw)
        if out[0] is not None:
            logits_seen.append(out[0][-1].detach().cpu().clone())
        return out
    e.model.decoder_replay = observe_replay
    pg = rails._prefill_group
    reference = None
    report = dict(prompt_tokens=len(ids), config=e.config(), runs=[])
    for trial, enabled in enumerate((False, True, True, False)):
        assert all(v == enabled for v in e.ep.gather_objects(enabled))
        logits_seen.clear()
        with patch.object(rails, '_prefill_group', pg if enabled else None):
            output = []
            for burst in e.generate(ids, max_tokens=args.max_tokens, temperature=0,
                                    seed=42, stop_token_ids={tok.token_to_id(enc.eos_token)}):
                output.extend(burst)
        torch.cuda.synchronize()
        assert logits_seen, 'must compare actual prefill logits'
        if reference is None:
            reference = (output, list(logits_seen))
        exact = output == reference[0] and len(logits_seen) == len(reference[1]) and all(
            torch.equal(a,b) for a,b in zip(logits_seen, reference[1]))
        assert all(e.ep.gather_objects(exact)), 'rail switch changed tokens or prefill logits'
        assert all(v == output for v in e.ep.gather_objects(output)), 'ranks disagree on tokens'
        item = dict(trial=trial, dual_prefill=enabled, exact=True, stats=dict(e.last_stats))
        report['runs'].append(item)
        print('PREFILL_RAILS '+json.dumps(dict(rank=e.ep.rank, **item)), flush=True)
    path = f'{args.out}/prefill-rails-rank{e.ep.rank}.json'
    with open(path, 'x') as f:
        json.dump(report, f, indent=2)
    owner = os.stat('/app/results')
    os.chown(path, owner.st_uid, owner.st_gid)
    assert all(e.ep.gather_objects(True))
    print('PREFILL_RAILS_PASS', flush=True)
    # As in other full-engine gates, exit together without tearing down live CUDA graphs.
    os._exit(0)


if __name__ == '__main__':
    main()
