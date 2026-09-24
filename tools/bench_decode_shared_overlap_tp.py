"""Bounded shared-expert fork/join A/B. Disposable TP gate only; never serving."""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
sys.path[:0] = ['/app', '/app/tools']
# Reuse the same frozen-ranking/prefix-off diagnostic setup and HTML prompt.
from bench_decode_timeline_tp import V, torch, Tok, load_encoding_module, build_chat_prompt, HTML_PROMPT
from nesting_arm import NEST_PROMPT, grade_nest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--capture', help='Trusted local text-only server capture; pickle can execute code')
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--quick', action='store_true', help='One measured generation per arm, after warmup')
    ap.add_argument('--experiment', choices=('shared-overlap', 'native-gather'), default='shared-overlap')
    ap.add_argument('--extra-capture', help='Optional second trusted local prompt, capped at 128 tokens')
    ap.add_argument('--profile-host', action='store_true', help='Separate main-thread CPU profile, excluded from timings')
    args = ap.parse_args()
    assert 64 <= args.max_tokens <= 512
    cap = None
    if args.capture:
        # Fail before spending ~90 seconds loading weights on a missing capture.
        cap = torch.load(args.capture, map_location='cpu', weights_only=False)
        if cap.get('vl') is not None or cap.get('kwargs', {}).get('grammar') is not None:
            raise ValueError('only trusted text-only, unconstrained captures are supported')
    extra = None
    if args.extra_capture:
        extra = torch.load(args.extra_capture, map_location='cpu', weights_only=False)
        if extra.get('vl') is not None or extra.get('kwargs', {}).get('grammar') is not None:
            raise ValueError('extra capture must be text-only and unconstrained')
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=524288, arena_gb=90,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=16, keep_free_gb=6)
    tok, enc = Tok(root), load_encoding_module(root)
    eos = tok.token_to_id(enc.eos_token)
    assert eos is not None
    fd, m = e.fast, e.model
    assert fd is not None and fd.use_graphs
    # Separate graph pools avoid memory aliasing between A/B captures. Replay is
    # serial; both variants still use the exact same weights, routes and KV state.
    arms = {False: dict(graphs={}, drafts=None, pool=None, stream=None),
            True: dict(graphs={}, drafts=None, pool=None,
                       stream=torch.cuda.Stream(device=e.device))}
    selected = None
    report = dict(config=e.config(), runs=[], fixed_steps=[], logits=[], nesting=[])
    report['experiment'] = args.experiment
    native_tables = {}
    if args.experiment == 'native-gather':
        from engine.engram_native import NativeGather
        native_tables = {layer: t.native_gather or NativeGather(t.w_mm, t.s_mm, workers=64)
                         for layer, t in e.tables.items()}
    gather_samples = {}
    if native_tables:
        for layer, table in e.tables.items():
            original = table._gather_rows
            def measured(ids, original=original, layer=layer):
                start = time.perf_counter_ns()
                result = original(ids)
                if len(ids) <= 384:
                    gather_samples.setdefault(layer, []).append((time.perf_counter_ns()-start)/1e6)
                return result
            table._gather_rows = measured

    def select(enabled):
        nonlocal selected
        assert all(v == enabled for v in e.ep.gather_objects(enabled)), 'A/B rank mismatch'
        torch.cuda.synchronize()
        if native_tables:
            for layer, table in e.tables.items():
                table.native_gather = native_tables[layer] if enabled else None
            selected = enabled
            return
        if selected is not None:
            arms[selected].update(graphs=fd.graphs, drafts=fd.draft_graphs, pool=fd.pool)
        arm = arms[enabled]
        fd.graphs, fd.draft_graphs, fd.pool = arm['graphs'], arm['drafts'], arm['pool']
        fd.shared_stream = arm['stream']
        selected = enabled

    def prompt_ids(prompt):
        _, ids, _, _ = build_chat_prompt({'messages': [{'role': 'user', 'content': prompt}]},
                                        enc, tok, False, 75, e)
        digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        assert len(set(e.ep.gather_objects(digest))) == 1
        return ids

    def generate(ids, limit):
        output = []
        for burst in e.generate(ids, max_tokens=limit, temperature=0, seed=42,
                                stop_token_ids={eos}):
            output.extend(burst)
        return output

    def emit(kind, item):
        print('DECODE_AB_' + kind + ' ' + json.dumps(dict(rank=e.ep.rank, experiment=args.experiment, **item)), flush=True)

    if args.capture:
        ids = list(cap['prompt_ids'])
        digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        assert len(set(e.ep.gather_objects(digest))) == 1
    else:
        ids = prompt_ids(HTML_PROMPT)
    report.update(prompt_tokens=len(ids), source='local_capture' if args.capture else 'synthetic',
                  max_tokens=args.max_tokens)
    outputs = {}
    # Warm each graph set, then order-balanced unprofiled generations (A B B A).
    order = (False, True, False, True) if args.quick else (False, True, False, True, True, False)
    for trial, enabled in enumerate(order):
        select(enabled)
        gather_samples.clear()
        output = generate(ids, args.max_tokens)
        if not outputs:
            reference = output
        exact = output == reference
        assert all(e.ep.gather_objects(exact)), 'Overlap changed generated tokens'
        outputs[enabled] = output
        item = dict(trial=trial, enabled=enabled, warmup=trial < 2,
                    exact_tokens=exact, stats=dict(e.last_stats))
        item['small_gather_ms'] = {layer: dict(calls=len(values), median=statistics.median(values),
                                              max=max(values), total=sum(values))
                                   for layer, values in gather_samples.items()}
        report['runs'].append(item)
        emit('GENERATION', item)

    pos = m.c.len
    token = reference[-2] if len(reference) > 1 else reference[-1]
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

    select(False)
    step()
    reference_logits, reference_hidden = (t.clone() for t in step())
    for enabled in (False, True):
        select(enabled)
        step()
        logits, hidden = step()
        exact = torch.equal(logits, reference_logits) and torch.equal(hidden, reference_hidden)
        item = dict(enabled=enabled, exact=exact,
                    max_logit_delta=float((logits-reference_logits).abs().max()),
                    max_hidden_delta=float((hidden-reference_hidden).abs().max()))
        emit('LOGITS', item)
        report['logits'].append(item)
        assert all(e.ep.gather_objects(exact)), 'Overlap changed fixed-step outputs'
    # Prefetched fixed-step rows bypass gather; do not claim this is a gather timing.
    for repeat in range(0 if native_tables else 4):
        for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
            select(enabled)
            step()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                step()
            torch.cuda.synchronize()
            item = dict(repeat=repeat, enabled=enabled,
                        rank_ms=e.ep.gather_objects((time.perf_counter()-start)*100))
            report['fixed_steps'].append(item)
            emit('STEP', item)

    ids = prompt_ids(NEST_PROMPT.format(d=8, leaf=48))
    nested = []
    for enabled in (False, True):
        select(enabled)
        output = generate(ids, 128)
        nested.append(output)
        item = dict(enabled=enabled, tokens=len(output), grade=grade_nest(tok.decode(output), 8, 48))
        report['nesting'].append(item)
        emit('NESTING', item)
    assert all(e.ep.gather_objects(nested[0] == nested[1])), 'Overlap changed nesting tokens'
    report['nesting_exact_tokens'] = True
    if extra is not None:
        extra_ids = list(extra['prompt_ids'])
        digest = hashlib.sha256(json.dumps(extra_ids).encode()).hexdigest()
        assert len(set(e.ep.gather_objects(digest))) == 1
        report['extra_runs'] = []
        reference_extra = None
        for trial, enabled in enumerate((False, True, False, True)):
            select(enabled)
            gather_samples.clear()
            output = generate(extra_ids, 128)
            if reference_extra is None: reference_extra = output
            exact = output == reference_extra
            assert all(e.ep.gather_objects(exact)), 'Candidate changed long-context tokens'
            item = dict(trial=trial, enabled=enabled, warmup=trial < 2, exact_tokens=exact,
                        stats=dict(e.last_stats), small_gather_ms={
                            layer: dict(calls=len(v), median=statistics.median(v), max=max(v))
                            for layer, v in gather_samples.items()})
            report['extra_runs'].append(item)
            emit('EXTRA', item)
    if args.profile_host:
        select(True)
        profile_ids = prompt_ids(HTML_PROMPT)
        reference_profile = generate(profile_ids, 512) if args.capture or args.max_tokens != 512 else reference
        generator = e.generate(profile_ids, max_tokens=512, temperature=0, seed=42, stop_token_ids={eos})
        output = []
        while len(output) < 128:
            output.extend(next(generator))
        from bench_decode_host_timers import HostTimers
        bursts = 0
        with HostTimers(e) as timers:
            for _ in range(32):
                try: output.extend(next(generator))
                except StopIteration: break
                bursts += 1
        for burst in generator: output.extend(burst)
        assert all(e.ep.gather_objects(output == reference_profile)), 'Host profiling changed HTML output'
        report['host_profile'] = dict(bursts=bursts, timer='per_call_thread_cpu_and_wall',
                                      functions=timers.report())
        emit('HOST_PROFILE', report['host_profile'])
    report['peak_allocated_gb'] = torch.cuda.max_memory_allocated()/1e9
    path = f'{args.out}/{args.experiment}-rank{e.ep.rank}.json'
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as f:
        json.dump(report, f, indent=2)
    owner = os.stat('/app/results')
    os.chown(path, owner.st_uid, owner.st_gid)
    assert all(e.ep.gather_objects(True))
    emit('PASS', dict(peak_allocated_gb=report['peak_allocated_gb']))
    os._exit(0)


if __name__ == '__main__':
    main()
