"""Collect DSpark confidence-head logits against real acceptance, per decode step. Disposable gate
only; never serving. The question it answers: does the confidence head predict acceptance well
enough for a per-step verify width to beat per-request adaptive depth? Analyse the output with
tools/analyze_spec_conf.py.

Verification is pinned to depth 5 (the drafter's full block), so every step observes the outcome
of all five drafts; at depth 3, drafts 4-5 would be censored. Run it twice, with the flag on and
off: greedy token hashes must match, because the confidence head only adds kernels to the draft
graph and nothing reads its output.

    GATE_ENV="DSV41_BLOCK=5 DSV41_BLOCK_DYNAMIC= DSV41_SPEC_CONF=1" GATE_IMAGE=<id> \\
    GATE_LOG_DIR=results/<dir> GATE_SOURCE_ROOT=<snapshot> \\
    bash tools/run_two_node_gate.sh bench_spec_conf_tp.py --out /app/results/<dir>
"""
import argparse
import hashlib
import json
import os
import sys
sys.path[:0] = ['/app', '/app/tools']
from bench_decode_timeline_tp import V, torch, Tok, load_encoding_module, build_chat_prompt
from bench_decode_block_tp import WORKLOADS
import engine.fastdecode as FD

# Prose with embedded code: the case per-request depth cannot serve well (it picks one depth for
# both kinds of text). Not in bench_decode_block_tp, so it has no historical baseline.
MIXED = ('mixed', 'Explain, in a few prose paragraphs, what a Python context manager is and why '
                  'it is useful. After each paragraph, give a short code example that illustrates it.', 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-tokens', type=int, default=512)
    args = ap.parse_args()
    assert not FD.DYNAMIC_DEPTHS and FD.T_DRAFT == 5, 'pin DSV41_BLOCK=5 and unset DSV41_BLOCK_DYNAMIC'
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=524288, arena_gb=90,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=16, keep_free_gb=6)
    tok, enc = Tok(root), load_encoding_module(root)
    eos = tok.token_to_id(enc.eos_token)
    report = dict(config=e.config(), spec_conf=FD.SPEC_CONF, tree_probe=FD.TREE_PROBE,
                  draft_tokens=FD.T_DRAFT, max_tokens=args.max_tokens, runs=[])

    def emit(kind, item):
        print('SPEC_CONF_' + kind + ' ' + json.dumps(dict(rank=e.ep.rank, **item)), flush=True)

    for warm in (True, False):   # the first pass captures graphs and warms experts; keep the second
        for name, prompt, temperature in WORKLOADS + (MIXED,):
            _, ids, _, _ = build_chat_prompt({'messages': [{'role': 'user', 'content': prompt}]},
                                             enc, tok, False, 75, e)
            out = []
            for burst in e.generate(ids, max_tokens=args.max_tokens, temperature=temperature, seed=42,
                                    stop_token_ids={eos}):
                out.extend(burst)
            st = dict(e.last_stats)
            item = dict(workload=name, temperature=temperature, tokens=st.get('completion_tokens'),
                        decode_tok_s=st.get('decode_tok_s'), steps=st.get('steps'),
                        accept_len_mean=st.get('accept_len_mean'),
                        out_sha256=hashlib.sha256(json.dumps(out).encode()).hexdigest())
            emit('WARMUP' if warm else 'RUN', item)
            if not warm:
                # steps: [verified depth, leading accepts, [confidence logit per draft]]
                report['runs'].append(dict(item, steps_log=st.get('spec_conf'),
                                          tree_log=st.get('tree_probe')))
    # The gate launcher creates its log directory on the head only; the worker's results mount
    # does not have it, and a rank-1 write failure here would fail the whole gate at the end.
    owner = os.stat('/app/results')
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
        os.chown(args.out, owner.st_uid, owner.st_gid)
    path = f'{args.out}/spec-conf-rank{e.ep.rank}.json'
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
        json.dump(report, f)
    os.chown(path, owner.st_uid, owner.st_gid)
    assert all(e.ep.gather_objects(True))
    emit('PASS', {})
    os._exit(0)


if __name__ == '__main__':
    main()
