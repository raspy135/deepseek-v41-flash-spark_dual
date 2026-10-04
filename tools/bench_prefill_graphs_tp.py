"""Full TP2 CUDA-prefill qualification: changed prompts, ABBA, exact logits and counters.

Disposable gate only. Requires DSV41_PREFILL_GRAPHS=1; freezes ranking and prefix
reuse. Changes the runtime graph enable switch identically on both ranks.
"""
import argparse
import json
import os
import sys
sys.path[:0] = ['/app', '/app/tools']
from bench_decode_timeline_tp import V, torch, Tok, load_encoding_module, build_chat_prompt
from engine import model as M


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=524288, arena_gb=90,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=16, keep_free_gb=6)
    graphs = e.model.prefill_graphs
    assert graphs is not None and e.ep.tensor_parallel
    tok, enc = Tok(root), load_encoding_module(root)
    ids = {}
    for name, count in (('records180', 180), ('records500', 500)):
        prompt = ('Read these records, then write a Python function that sums their values.\n' +
                  '\n'.join(f'Record {i}: category {i%7}, value {(i*3+count)%13}.' for i in range(count)) +
                  '\nOutput a function accepting records as dictionaries, with a short example.')
        _, ids[name], _, _ = build_chat_prompt({'messages': [{'role': 'user', 'content': prompt}]},
                                             enc, tok, False, 75, e)
    assert all(v == ids for v in e.ep.gather_objects(ids))
    seen = []
    original = e.model.decoder_replay
    def observe(*a, **kw):
        result = original(*a, **kw)
        seen.append((result[0][-1].detach().cpu().clone(),
                     e.model._rec_counts.detach().cpu().clone(),
                     e.model._miss_tot.detach().cpu().clone()))
        return result
    e.model.decoder_replay = observe
    report = dict(config=e.config(), runs=[], qualification=[], mismatches=[])
    references = {}

    def save():
        os.makedirs(args.out, exist_ok=True)
        path = f'{args.out}/prefill-graphs-rank{e.ep.rank}.json'
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        owner = os.stat('/app/results')
        os.chown(path, owner.st_uid, owner.st_gid)

    def run(name, enabled, group='qualification', reference_key=None):
        assert all(x == enabled for x in e.ep.gather_objects(enabled))
        graphs.enabled = enabled
        seen.clear()
        before = (torch.zeros_like(e.model._rec_counts, device='cpu') if M.PRUNE_UNIT_REQUEST
                  else e.model._rec_counts.detach().cpu().clone(),
                  e.model._miss_tot.detach().cpu().clone())
        captures = graphs.captures
        replays = graphs.replays
        out = []
        for burst in e.generate(ids[name], max_tokens=32, temperature=0, seed=42,
                                stop_token_ids={tok.token_to_id(enc.eos_token)}):
            out.extend(burst)
        assert len(seen) == 1
        actual = (out, seen[0][0], seen[0][1]-before[0], seen[0][2]-before[1])
        key = reference_key or name
        reference = references.setdefault(key, actual)
        exact = out == reference[0] and all(torch.equal(a,b) for a,b in zip(actual[1:],reference[1:]))
        parity = all(v == out for v in e.ep.gather_objects(out))
        verdict = all(e.ep.gather_objects(exact and parity))
        item = dict(workload=name, enabled=enabled, prompt_tokens=len(ids[name]),
                    prefill_s=e.last_stats['prefill_s'], prefill_tok_s=e.last_stats['prefill_tok_s'],
                    new_captures=graphs.captures-captures, graph_replays=graphs.replays-replays,
                    exact_logits_tokens_counts=verdict, peak_gb=torch.cuda.max_memory_allocated()/1e9,
                    graphs=graphs.report())
        report[group].append(item)
        if not verdict:
            report['mismatches'].append(item)
        print('PREFILL_GRAPHS '+json.dumps(dict(rank=e.ep.rank, **item)), flush=True)
        save()
        assert verdict, 'graph path changed logits, output or demand counts'

    # Qualify capture/replay on different inputs and multiple context positions.
    for name in ids:
        run(name, False)
        run(name, True)
    for name in ids:
        for flag in (False, True, True, False):
            run(name, flag, 'runs')
    # Change a real routing mask and arena slot in place after capture, then compare
    # eager/graph again. This catches graphs retaining stale compact routing maps.
    mask = e.model_prune_mask[0].cpu()
    outgoing = int(torch.nonzero(mask).flatten()[0])
    incoming = int(torch.nonzero(~mask).flatten()[0])
    e.apply_swaps([(0, outgoing, incoming, 1.)])
    run('records180', False, reference_key='swapped')
    run('records180', True, reference_key='swapped')
    e.apply_swaps([(0, incoming, outgoing, 1.)])
    report['routing_swap_checked'] = True
    save()
    assert all(e.ep.gather_objects(True))
    print('PREFILL_GRAPHS_PASS', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
