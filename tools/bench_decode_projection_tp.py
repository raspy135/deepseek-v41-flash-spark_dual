"""Full-engine TP2 A/B for the decode projection candidates. Disposable gate only; never serving.

Arms (switched inside one process, on both ranks together, with separate graph pools):
  baseline   DSV41_FP8_DECODE_BLOCK_N=128, merged projections off, torch prune-miss accounting
  candidate  whatever --experiment turns on:
               prune-miss  fused DSV41_PRUNE_MISS accounting
               block-n     --block-n (default auto) for decode fp8 projections
               merged      one launch for wq_a||wkv and shared w1||w3
               act-qdq     activation quantization inside the fp8 GEMM
               all         all four

Every candidate is claimed to leave logits, hidden states and tokens bit-identical; this checks
that on real weights and routes before timing, then times order-balanced fixed verify steps and
512-token generations, then compares depth-8 nesting output. prune-miss is additionally checked
on the decode demand it records. Same frozen-ranking / prefix-off setup as the timeline bench.

    GATE_IMAGE=<id> GATE_LOG_DIR=results/<new-folder> \\
    bash tools/run_two_node_gate.sh bench_decode_projection_tp.py --experiment all \\
         --out /app/results/<new-folder>
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
sys.path[:0] = ['/app', '/app/tools']
from bench_decode_timeline_tp import V, torch, Tok, load_encoding_module, build_chat_prompt, HTML_PROMPT
from nesting_arm import NEST_PROMPT, grade_nest
import fp8_linear as F8
import engine.fastdecode as FD
import engine.model as M_
import v41_ref as R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--experiment', choices=('prune-miss', 'block-n', 'merged', 'act-qdq', 'all'),
                    default='all')
    ap.add_argument('--block-n', choices=('16', '32', '64', 'auto'), default='auto')
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--quick', action='store_true', help='One measured generation per arm, after warmup')
    args = ap.parse_args()
    assert 64 <= args.max_tokens <= 512
    for key in ('DSV41_FP8_DECODE_BLOCK_N', 'DSV41_DECODE_MERGED_PROJ', 'DSV41_PRUNE_MISS_FUSED',
                'DSV41_FP8_ACT_QDQ_FUSED'):
        # The arms are set below; an inherited value would make "baseline" something else.
        assert os.environ.get(key) in (None, '0', '128'), f'unset {key} for the A/B'
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=524288, arena_gb=90,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=16, keep_free_gb=6)
    tok, enc = Tok(root), load_encoding_module(root)
    eos = tok.token_to_id(enc.eos_token)
    fd, m = e.fast, e.model
    assert fd is not None and fd.use_graphs
    want = {'prune': args.experiment in ('prune-miss', 'all'),
            'block_n': args.experiment in ('block-n', 'all'),
            'merged': args.experiment in ('merged', 'all'),
            'act_qdq': args.experiment in ('act-qdq', 'all')}
    if want['prune'] and not M_.PRUNE_MISS:
        raise SystemExit('prune-miss arm needs DSV41_PRUNE_MISS=1 (as served)')
    # Baseline uses the views, candidate the stacked weight: the same bytes either way.
    merged = FD.merge_decode_projections(list(fd.W.layers) + list(fd.W.mtp)) if want['merged'] else 0
    if want['merged']:
        assert merged > 0, 'no projection pair was mergeable'

    arms = {False: dict(graphs={}, drafts=None, pool=None), True: dict(graphs={}, drafts=None, pool=None)}
    fd.graphs, fd.draft_graphs, fd.pool = {}, None, None   # anything captured at load is neither arm
    selected = None

    def select(enabled):
        nonlocal selected
        assert all(v == enabled for v in e.ep.gather_objects(enabled)), 'A/B rank mismatch'
        torch.cuda.synchronize()
        if selected is not None:
            arms[selected].update(graphs=fd.graphs, drafts=fd.draft_graphs, pool=fd.pool)
        arm = arms[enabled]
        fd.graphs, fd.draft_graphs, fd.pool = arm['graphs'], arm['drafts'], arm['pool']
        M_.PRUNE_MISS_FUSED = enabled and want['prune']
        F8.DECODE_BLOCK_N = args.block_n if enabled and want['block_n'] else '128'
        FD.MERGED_PROJ = enabled and want['merged']
        R.ACT_QDQ_FUSED = enabled and want['act_qdq']
        selected = enabled

    report = dict(config=e.config(), experiment=args.experiment, block_n=args.block_n,
                  merged_pairs=merged, runs=[], fixed_steps=[], logits=[], demand=[], nesting=[])

    def emit(kind, item):
        print('DECODE_PROJ_' + kind + ' ' + json.dumps(dict(rank=e.ep.rank, experiment=args.experiment, **item)),
              flush=True)

    def prompt_ids(prompt):
        _, ids, _, _ = build_chat_prompt({'messages': [{'role': 'user', 'content': prompt}]},
                                        enc, tok, False, 75, e)
        digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        assert len(set(e.ep.gather_objects(digest))) == 1
        return ids

    def generate(ids, limit):
        output = []
        for burst in e.generate(ids, max_tokens=limit, temperature=0, seed=42, stop_token_ids={eos}):
            output.extend(burst)
        return output

    mismatches = []
    ids = prompt_ids(HTML_PROMPT)
    report.update(prompt_tokens=len(ids), max_tokens=args.max_tokens)
    order = (False, True, False, True) if args.quick else (False, True, False, True, True, False)
    reference = None
    for trial, enabled in enumerate(order):
        select(enabled)
        output = generate(ids, args.max_tokens)
        reference = output if reference is None else reference
        exact = all(e.ep.gather_objects(output == reference))
        if not exact:
            mismatches.append(f'generation trial {trial}')
        item = dict(trial=trial, enabled=enabled, warmup=trial < 2, exact_tokens=exact,
                    stats=dict(e.last_stats))
        report['runs'].append(item)
        emit('GENERATION', item)

    pos = m.c.len
    token = reference[-2] if len(reference) > 1 else reference[-1]
    select(False)
    drafts, _ = fd.draft(token, pos - 1, 0.)
    block = torch.cat((torch.tensor([token], device=e.device), drafts.clone()))
    hashes = m.hash_state(block[None], pos)[0]
    rows = {layer: e.tables[layer].rows(hashes[:, li, :])
            for li, layer in enumerate(e.args.engram_layer_ids)}
    pending = {layer: None if value is None else tuple(t.clone() for t in value)
               for layer, value in m.c.pending.items()}

    def step():
        m.c.len = pos
        m.c.pending.update(pending)
        logits, hidden = fd.step(block, pos, rows)
        m.c.rollback(pos)
        return logits, hidden

    def demand():
        if m._want_phase is None:
            return None
        return [t.clone() for t in (m._want_phase[1], m._want_mass, m._miss_phase)]

    step()
    reference_logits, reference_hidden = (t.clone() for t in step())
    deltas = {}
    for enabled in (False, True):
        select(enabled)
        step()
        before = demand()
        logits, hidden = step()
        torch.cuda.synchronize()
        after = demand()
        if before is not None:
            deltas[enabled] = [a - b for a, b in zip(after, before)]
        exact = torch.equal(logits, reference_logits) and torch.equal(hidden, reference_hidden)
        item = dict(enabled=enabled, exact=exact,
                    max_logit_delta=float((logits - reference_logits).abs().max()),
                    max_hidden_delta=float((hidden - reference_hidden).abs().max()),
                    argmax_equal=bool(torch.equal(logits.argmax(-1), reference_logits.argmax(-1))))
        report['logits'].append(item)
        emit('LOGITS', item)
        if not all(e.ep.gather_objects(exact)):
            mismatches.append(f'fixed-step logits enabled={enabled}')
    if len(deltas) == 2:
        (c0, w0, p0), (c1, w1, p1) = deltas[False], deltas[True]
        # A non-finite entry already in the database (e.g. loaded from a saved ranking) makes
        # after - before NaN in BOTH arms; compare the finite entries and report the others.
        finite = torch.isfinite(w0) & torch.isfinite(w1)
        rel = ((w1 - w0).abs() / w0.abs().clamp_min(1e-12))[finite]
        item = dict(decode_counts_equal=bool(torch.equal(c0, c1)),
                    miss_phase_equal=bool(torch.equal(p0, p1)),
                    mass_max_rel=float(rel.max()) if rel.numel() else 0.0,
                    mass_nonfinite_same=bool(torch.equal(torch.isfinite(w0), torch.isfinite(w1))),
                    db_mass_nonfinite=int((~torch.isfinite(m._want_mass)).sum()),
                    db_counts_nonfinite=int((~torch.isfinite(m._want_counts)).sum()),
                    decode_slots=float(p1[1, 1]))
        report['demand'].append(item)
        emit('DEMAND', item)
        if not (item['decode_counts_equal'] and item['miss_phase_equal'] and item['mass_max_rel'] < 1e-9
                and item['mass_nonfinite_same']):
            mismatches.append('decode demand accounting')

    for repeat in range(4):
        for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
            select(enabled)
            step()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                step()
            torch.cuda.synchronize()
            item = dict(repeat=repeat, enabled=enabled,
                        rank_ms=e.ep.gather_objects((time.perf_counter() - start) * 100))
            report['fixed_steps'].append(item)
            emit('STEP', item)
    by_arm = {k: [max(i['rank_ms']) for i in report['fixed_steps'] if i['enabled'] == k] for k in (False, True)}
    report['fixed_step_median_ms'] = {str(k): statistics.median(v) for k, v in by_arm.items()}
    measured = [r for r in report['runs'] if not r['warmup']]
    report['generation_tok_s'] = {str(k): [r['stats'].get('decode_tok_s') for r in measured if r['enabled'] == k]
                                  for k in (False, True)}

    ids = prompt_ids(NEST_PROMPT.format(d=8, leaf=48))
    nested = []
    for enabled in (False, True):
        select(enabled)
        output = generate(ids, 128)
        nested.append(output)
        item = dict(enabled=enabled, tokens=len(output), grade=grade_nest(tok.decode(output), 8, 48))
        report['nesting'].append(item)
        emit('NESTING', item)
    if not all(e.ep.gather_objects(nested[0] == nested[1])):
        mismatches.append('nesting tokens')
    report['mismatches'] = mismatches
    report['peak_allocated_gb'] = torch.cuda.max_memory_allocated() / 1e9
    path = f'{args.out}/{args.experiment}-rank{e.ep.rank}.json'
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
        json.dump(report, f, indent=2)
    owner = os.stat('/app/results')
    os.chown(path, owner.st_uid, owner.st_gid)
    assert all(e.ep.gather_objects(True))
    emit('SUMMARY', dict(fixed_step_median_ms=report['fixed_step_median_ms'],
                         generation_tok_s=report['generation_tok_s'], mismatches=mismatches))
    # Timings are still reported on a mismatch, but the run does not pass.
    emit('PASS' if not mismatches else 'FAIL', dict(peak_allocated_gb=report['peak_allocated_gb']))
    os._exit(0 if not mismatches else 1)


if __name__ == '__main__':
    main()
