"""Bounded odd-width qualification: compressor parity, target prefixes and graph costs.

Run --cpu-sources on a host without torch, --compressor-only inside the image,
or qualify(engine, out) from another disposable TP2 driver to share its model.
The full-model run freezes residency and never enables forced draft bypass.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

sys.path[:0] = ['/app', '/app/tools', str(Path(__file__).resolve().parents[1]),
                str(Path(__file__).resolve().parent)]


def fatal(kind, error, traceback):
    import traceback as tb
    message = ''.join(tb.format_exception(kind, error, traceback))
    os.write(2, message.encode('utf-8', 'backslashreplace'))
    directory = globals().get('_output_directory')
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
        Path(directory, 'odd-failure-rank' + os.environ.get('RANK', '?') + '.txt').write_text(message)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(1)


def source_checks():
    """Exercise actual pure config/grouping source without importing CUDA modules."""
    from unittest.mock import patch
    from engine.spec_depth import ConfidenceDepthPolicy, DepthPolicy, confidence_depths
    source = ast.parse((Path(__file__).resolve().parents[1] / 'engine/fastdecode.py').read_text())
    definitions = [n for n in source.body if isinstance(n, ast.FunctionDef)
                   and n.name in ('_compressor_group_cut', '_dynamic_depths', '_draft_block')]
    count = 0
    for odd, depths in ((False, '3,5'), (True, '0,5'), (True, '2,4'), (True, '0,2')):
        ns = {'os': os, 'VERIFY_ODD': odd}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<verify-odd-source>', 'exec'), ns)
        with patch.dict(os.environ, {'DSV41_VERIFY_ODD': str(int(odd)),
                                     'DSV41_BLOCK_DYNAMIC': depths, 'DSV41_BLOCK': '',
                                     'DSV41_BLOCK_CONFIDENCE': '0'}, clear=False):
            assert ns['_dynamic_depths']() == tuple(map(int, depths.split(',')))
        count += 1
    for tokens in range(1, 17):
        for parity in (0, 1):
            cut = ns['_compressor_group_cut'](tokens, parity)
            positions = list(range(-parity, tokens))
            groups = [positions[i:i + 2] for i in range(0, cut, 2)]
            assert all(len(g) == 2 for g in groups)
            assert positions[cut:] == ([tokens - 1] if (tokens + parity) % 2 else [])
            assert len(groups) == (tokens + parity) // 2
            count += 1
    # Exercise step's actual host bookkeeping, rather than reconstructing its
    # pending result as compressor_checks does. Odd widths reverse end parity.
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'FastDecoder')
    step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step')
    bookkeeping = next(n for n in step.body if isinstance(n, ast.For)
                       and ast.unparse(n.iter) == 'self.kvl_buf')
    fn = ast.FunctionDef(name='bookkeeping', args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg=name) for name in ('self', 'S', 'T', 'parity')],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=[bookkeeping], decorator_list=[])
    unit = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    bookkeeping_ns = {}
    exec(compile(unit, '<step-pending-source>', 'exec'), bookkeeping_ns)
    class Row:
        def __init__(self, value): self.value = value
        def clone(self): return self.value
    for parity in (0, 1):
        for T in range(1, 7):
            c = SimpleNamespace(pending={0: ('before-kv', 'before-score')}, _chunk_inputs={})
            fake = SimpleNamespace(c=c, kvl_buf={0: [Row(i) for i in range(T)]},
                                   sc_buf={0: [Row(100 + i) for i in range(T)]})
            bookkeeping_ns['bookkeeping'](fake, 10 + parity, T, parity)
            expected = (T - 1, 100 + T - 1) if (parity + T) % 2 else None
            assert c.pending[0] == expected and c._chunk_inputs[0][-1] == ('before-kv', 'before-score')
            count += 1
    with patch.dict(os.environ, {'DSV41_VERIFY_ODD': '1', 'DSV41_BLOCK_CONFIDENCE': '1',
                                 'DSV41_BLOCK': '', 'DSV41_CONF_COST_REFRESH': '0'}):
        assert confidence_depths((2, 4)) == (1, 2, 3, 4, 5)
        try:
            confidence_depths((0, 5))
        except ValueError:
            pass
        else:
            raise AssertionError('confidence depth 0 can pin the no-draft control branch')
        p = ConfidenceDepthPolicy()
        p.step_s = {1: .088, 2: .097, 3: .106, 4: .115, 5: .124}
        for depth in p.depths:
            p.pinned = depth
            assert p.choose([0.] * 5) == p.choose_sampled([0.] * 5) == depth
        p.pinned = None
        # A sampled decision at depth 1 cannot read confidence after boundary 1.
        p.step_s = {1: .01, 2: .5, 3: .6, 4: .7, 5: .8}
        assert p.choose_sampled([0., 0., float('nan'), 100., -100.]) == 1
        p0 = DepthPolicy((0, 5), start=0, interval=1)
        for _ in range(8):
            p0.observe(0, 0, 1, .04)
        assert p0.steps[0] == 8 and p0.step_s[0] == .04
        count += 8
    return {'passed': True, 'checks': count}


def compressor_checks():
    """Compare the actual fast/eager compressor and every rollback boundary on CPU."""
    import torch
    import v41_ref as R
    from unittest.mock import patch
    from engine import model as M
    from engine import fastdecode as FD
    from engine.fastdecode import FastDecoder
    torch.manual_seed(713)
    a = SimpleNamespace(head_dim=8, dim=16, rope_head_dim=4, norm_eps=1e-20,
                        compress_ratios=(2,), index_topk=4)
    w = SimpleNamespace(ratio=2, is_kv_source=True,
                        comp_wkv=torch.randn(8, 16), comp_wgate=torch.randn(8, 16),
                        comp_norm=torch.randn(8, dtype=torch.bfloat16))
    freq = torch.ones(128, 2, dtype=torch.complex64)
    positions = torch.arange(128)
    flags = M.PACKED_KV, M.KV_CACHE_QDQ
    M.PACKED_KV = M.KV_CACHE_QDQ = False
    count = 0
    products = 0
    try:
        # Exercise the actual broadcast helper and prove the one-row opt-in
        # uses views of both inputs, while default and wider batches keep size.
        for odd in (False, True):
            for T in range(1, 7):
                for equation in ('thd,tnd->thn', 'thn,tnd->thd'):
                    left = torch.randn(T, 3, 8 if equation.endswith('thn') else 7)
                    right = torch.randn(T, 7, 8)
                    batch = 2 if odd and T == 1 else T
                    expected = torch.einsum(equation, left.expand(batch, -1, -1),
                                            right.expand(batch, -1, -1))[:T]
                    with patch.object(FD, 'VERIFY_ODD', odd), patch.object(torch, 'einsum', wraps=torch.einsum) as call:
                        actual = FastDecoder._verify_attention_product(None, equation, left, right)
                    args = call.call_args.args
                    assert torch.equal(actual, expected) and args[1].shape[0] == args[2].shape[0] == batch
                    if odd and T == 1:
                        assert args[1].stride(0) == args[2].stride(0) == 0
                        assert args[1].data_ptr() == left.data_ptr() and args[2].data_ptr() == right.data_ptr()
                    products += 1
        for T in range(1, 7):
            for parity in (0, 1):
                S = 10 + parity
                x = torch.randn(T, 16, dtype=torch.bfloat16)
                before = ((torch.randn(8), torch.randn(8)) if parity else None)
                initial = torch.randn(64, 8, dtype=torch.bfloat16)

                def caches():
                    c = object.__new__(M.Caches)
                    c.args = a; c.len = S + T; c.pending = {0: before}
                    c.ckv = {0: initial.clone()}; c.ik = {0: torch.zeros(64, 8)}; c._chunk_inputs = {}
                    return c

                rc, fc = caches(), caches()
                idx = torch.tensor([[0, 1, 2, 3]]).expand(T, -1).clone()
                rm = SimpleNamespace(args=a, c=rc, W=SimpleNamespace(indexers={}),
                                     freqs_c=freq, _positions=positions, dev='cpu', tap=None,
                                     _tap=lambda *args: None)
                sh = M.Shared(); sh.topk = idx
                reference = M.Model._compressed(rm, x, x, w, 0, S, T, positions[S:S + T], sh)
                fd = SimpleNamespace(a=a, c=fc, W=SimpleNamespace(indexers={}),
                                     kvl_buf={0: torch.empty(T, 8)}, sc_buf={0: torch.empty(T, 8)},
                                     pend_buf={0: torch.zeros(2, 8)}, _ar_t=torch.arange(T),
                                     m=SimpleNamespace(freqs_c=freq), topk=idx, lean=None)
                if before is not None:
                    fd.pend_buf[0][0].copy_(before[0]); fd.pend_buf[0][1].copy_(before[1])
                fd._rmsnorm = R.rmsnorm
                fd._rope = lambda value, fq: FastDecoder._rope(fd, value, fq)
                state = {'parity': parity, 'ckv': None, 'ik': None, 'ratio': 0}
                actual = FastDecoder._compressed(fd, x, x, w, 0, positions[S:S + T], state)
                assert torch.equal(rc.ckv[0], fc.ckv[0]), (T, parity, 'compressed rows')
                assert all(torch.equal(r, f) for r, f in zip(reference, actual)), (T, parity, 'gather')
                pending = None if (S + T) % 2 == 0 else tuple(fd.pend_buf[0])
                assert ((pending is None and rc.pending[0] is None)
                        or all(torch.equal(r, f) for r, f in zip(rc.pending[0], pending)))
                fc.pending[0] = pending
                fc._chunk_inputs[0] = (S, fd.kvl_buf[0], fd.sc_buf[0], before)
                for end in range(S, S + T + 1):
                    rc.len = fc.len = S + T
                    rc.rollback(end); fc.rollback(end)
                    rp, fp = rc.pending[0], fc.pending[0]
                    assert ((rp is None and fp is None)
                            or (rp is not None and fp is not None
                                and all(torch.equal(r, f) for r, f in zip(rp, fp)))), (T, parity, end)
                    count += 1
        return {'passed': True, 'rollback_boundaries': count, 'attention_products': products,
                'widths': list(range(1, 7))}
    finally:
        M.PACKED_KV, M.KV_CACHE_QDQ = flags


def qualify(engine, out, repeats=3, allow_failure=False, prompt_padding=0):
    """Same loaded target, all widths and both parities; save evidence before asserting."""
    import torch
    from engine.expert_profiles import mask_digest
    from server.app import Tok, load_encoding_module, build_chat_prompt
    e, m, fd = engine, engine.model, engine.fast
    assert all(w in fd._width_bufs for w in range(1, 7)), 'use odd confidence mode for all target widths'
    assert fd.graph_limit == 0 or fd.graph_limit >= 12, 'avoid graph rotation in this screen'
    assert not e.stream_layers and fd.lut is not None
    os.makedirs(out, exist_ok=True)
    tok, enc = Tok(os.environ['MODEL_DIR']), load_encoding_module(os.environ['MODEL_DIR'])
    report = {'source': source_checks(), 'compressor': compressor_checks(), 'target': [],
              'config': e.config(), 'passed': False}
    map_hash = mask_digest(e.model_prune_mask)

    def save():
        Path(out, f'odd-rank{e.ep.rank}.json').write_text(json.dumps(report, indent=2))

    def clone_pending():
        return {L: None if value is None else tuple(t.clone() for t in value)
                for L, value in m.c.pending.items()}

    def snapshot():
        return {'win': [t.clone() for t in m.c.win], 'ckv': {L: t.clone() for L, t in m.c.ckv.items()},
                'ik': {L: t.clone() for L, t in m.c.ik.items()},
                'mtp': [t.clone() for t in m.c.mtp_win], 'pending': clone_pending(), 'len': m.c.len}

    def restore(s):
        for destination, original in zip(m.c.win, s['win']): destination.copy_(original)
        for name in ('ckv', 'ik'):
            for L, original in s[name].items(): getattr(m.c, name)[L].copy_(original)
        for destination, original in zip(m.c.mtp_win, s['mtp']): destination.copy_(original)
        m.c.pending = {L: None if value is None else tuple(t.clone() for t in value)
                       for L, value in s['pending'].items()}
        m.c._chunk_inputs.clear(); m.c.len = s['len']

    def eager_trace(width, prefix, block, rows):
        """Diagnostic only: row-prefix taps from the same restored cache/input.

        CPU copies perturb timing, so this never runs in a passing cost screen.
        Hooks are driver-local and restored even on failure; production kernels
        do not acquire debug branches or captured tensor clones.
        """
        trace = {}
        current_layer = [-1]
        saved = {name: getattr(fd, name) for name in ('_layer_a', '_layer_b', '_attention', '_final')}
        saved_tap, saved_graphs, saved_memo = getattr(fd, 'tap', None), fd.use_graphs, fd._memo
        einsum, attn_probs = torch.einsum, fd.lean.attn_probs

        def record(name, L, value):
            if name == 'topk' and not fd.W.layers[L].ratio:
                return  # stale scratch until the first compressed indexer
            trace[(L, name)] = value.detach().cpu().clone()

        def layer_a(L, state):
            current_layer[0] = L
            result = saved['_layer_a'](L, state)
            record('layer_a_h', L, fd.h)
            record('route_w', L, fd.route_w)
            return result

        def layer_b(L):
            result = saved['_layer_b'](L)
            record('layer_b_h', L, fd.h)
            record('pre_mix', L, fd.pre_mix)
            return result

        def attention(*args, **kwargs):
            result = saved['_attention'](*args, **kwargs)
            record('attention_return', args[2], result)
            return result

        def final():
            record('final_h', fd.a.n_layers, fd.h)
            record('final_pre_mix', fd.a.n_layers, fd.pre_mix)
            result = saved['_final']()
            record('logits', fd.a.n_layers, fd.logits)
            record('main_hidden', fd.a.n_layers, fd.main_hidden)
            return result

        def product(equation, *operands, **kwargs):
            result = einsum(equation, *operands, **kwargs)
            names = {'thd,tnd->thn': 'qk', 'thn,tnd->thd': 'pv', 'thd,nd->thn': 'index_scores'}
            if equation in names:
                # T1 exact attention broadcasts two equal batches internally;
                # only the logical first row is part of the target trace.
                record(names[equation], current_layer[0], result[:width])
            return result

        def probabilities(*args, **kwargs):
            result = attn_probs(*args, **kwargs)
            record('attn_probs', current_layer[0], result)
            return result

        try:
            fd.use_graphs, fd._memo, fd.tap = False, None, record
            fd._layer_a, fd._layer_b = layer_a, layer_b
            fd._attention, fd._final = attention, final
            torch.einsum, fd.lean.attn_probs = product, probabilities
            restore(prefix)
            fd.step(block[:width], S, {L: value[:width] for L, value in rows.items()})
            torch.cuda.synchronize()
        finally:
            for name, original in saved.items(): setattr(fd, name, original)
            fd.tap, fd.use_graphs, fd._memo = saved_tap, saved_graphs, saved_memo
            torch.einsum, fd.lean.attn_probs = einsum, attn_probs
            restore(prefix)
        return trace

    def diagnose(prefix, block, rows, failed_widths):
        widths = sorted({1, 3, 5, *failed_widths})
        reference = eager_trace(6, prefix, block, rows)
        diagnostic = {'parity': S % 2, 'position': S, 'widths': []}
        for width in widths:
            actual = eager_trace(width, prefix, block, rows)
            stages = []
            for (L, name), value in actual.items():
                expected = reference[(L, name)][:width]
                assert value.shape == expected.shape, (L, name, width, value.shape, expected.shape)
                equal = torch.equal(value, expected)
                if not equal:
                    stages.append({'layer': L, 'stage': name,
                        'max_delta': float((value.float() - expected.float()).abs().max()),
                        'changed_elements': int(torch.count_nonzero(value != expected)),
                        'elements': value.numel()})
            logits = actual[(fd.a.n_layers, 'logits')]
            digest = hashlib.sha256(logits.numpy().tobytes()).hexdigest()
            row = {'width': width, 'exact_eager_prefix': not stages,
                   'first_difference': stages[0] if stages else None,
                   'differences': stages, 'logit_sha256': digest,
                   'rank_equal': len(set(e.ep.gather_objects(digest))) == 1}
            diagnostic['widths'].append(row)
            print('ODD_VERIFY_EAGER ' + json.dumps({k: v for k, v in row.items() if k != 'differences'}), flush=True)
        report.setdefault('diagnostics', []).append(diagnostic); save()

    completed = set()
    for attempt in range(4):
        _, ids, _, _ = build_chat_prompt({'messages': [{'role': 'user',
            'content': '[odd verify qualification] Explain an LRU cache in Python.' + ' x' * (prompt_padding + attempt)}]},
            enc, tok, False, 75, e)
        output = [t for burst in e.generate(ids, max_tokens=1, temperature=0, seed=42, ignore_eos=True) for t in burst]
        S = m.c.len
        if S % 2 in completed:
            continue
        completed.add(S % 2)
        prefix = snapshot()
        proposals, _ = fd.draft(output[-1], S - 1, 0)
        block = torch.cat([torch.tensor(output[-1:], device=fd.dev), proposals.clone()])
        assert block.numel() == 6
        hashes = m.hash_state(block[None], S)[0]
        rows = {L: e.tables[L].rows(hashes[:, li, :]) for li, L in enumerate(e.args.engram_layer_ids)}
        reference_logits = reference_hidden = None
        failed_widths = []
        for width in (6, 2, 3, 4, 5, 1):
            restore(prefix)
            fd.step(block[:width], S, {L: value[:width] for L, value in rows.items()})  # capture/warm
            restore(prefix)
            logits, hidden = fd.step(block[:width], S, {L: value[:width] for L, value in rows.items()})
            logits = logits.clone(); hidden = hidden.clone(); torch.cuda.synchronize()
            if width == 6:
                reference_logits, reference_hidden = logits, hidden
            exact = torch.equal(logits, reference_logits[:width]) and torch.equal(hidden, reference_hidden[:width])
            delta = float((logits - reference_logits[:width]).abs().max())
            hidden_delta = float((hidden - reference_hidden[:width]).abs().max())
            digest = hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest()
            rank_equal = len(set(e.ep.gather_objects(digest))) == 1
            pending = clone_pending()
            for end in range(S, S + width + 1):
                m.c.len = S + width; m.c.pending = dict(pending); m.c.rollback(end)
                assert (m.c.pending[next(iter(fd.pend_buf))] is None) == (end % 2 == 0)
            samples = []
            key = next(key for key in fd.graphs if key[0] == S % 2 and key[-1] == width)
            graphs = fd.graphs[key]
            assert graphs[-1] is not None, 'timing screen expects resident segment graphs'
            for _ in range(repeats):
                restore(prefix); fd.prepare_pending_buffers()
                # Segment graphs start at layer 0; step's embedding and initial
                # stream mix are host work outside those graphs. Restore both
                # so repeated prices use the same input, not the prior output.
                fd.h.copy_(fd.W.embed[fd.ids].unsqueeze(1).expand(-1, fd.a.hc_mult, -1))
                fd.pre_mix.copy_(fd._premix0)
                begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _, first, second in graphs[-1]:
                    first.replay()
                    if second is not None: second.replay()
                finish.record(); finish.synchronize(); samples.append(begin.elapsed_time(finish))
            row = {'parity': S % 2, 'position': S, 'width': width, 'exact_prefix': exact,
                   'rank_equal': rank_equal, 'max_logit_delta': delta, 'sha256': digest,
                   'max_hidden_delta': hidden_delta,
                   'main_graph_ms': statistics.median(samples), 'samples_ms': samples}
            report['target'].append(row); save()
            print('ODD_VERIFY_TARGET ' + json.dumps(row), flush=True)
            if not exact or not rank_equal:
                failed_widths.append(width)
            assert mask_digest(e.model_prune_mask) == map_hash, 'resident map changed'
        # Both ranks enter diagnostic collectives even if only one saw a drift.
        failed_widths = sorted({w for ws in e.ep.gather_objects(failed_widths) for w in ws})
        if failed_widths:
            diagnose(prefix, block, rows, failed_widths)
        if len(completed) == 2:
            break
    assert len(completed) == 2, 'both start parities required'
    report['passed'] = all(row['exact_prefix'] and row['rank_equal'] for row in report['target'])
    save()
    print('ODD_VERIFY_PASS' if report['passed'] else 'ODD_VERIFY_FAILED', flush=True)
    assert report['passed'] or allow_failure, 'odd/even target rows changed; all discrepancy artifacts retained'
    return report


def main():
    global _output_directory
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cpu-sources', action='store_true')
    ap.add_argument('--compressor-only', action='store_true')
    ap.add_argument('--router-followup', action='store_true')
    ap.add_argument('--depth-followup', action='store_true')
    ap.add_argument('--allow-odd-failure', action='store_true',
                    help='diagnostic only: finish width screens and optional router followup, then exit 1 on drift')
    ap.add_argument('--out')
    ap.add_argument('--prompt-padding', type=int, default=0,
                    help='synthetic repeated tokens; >2048 exercises compressed-key top-k selection')
    args = ap.parse_args()
    _output_directory = args.out
    sys.excepthook = fatal
    if args.cpu_sources:
        print(json.dumps(source_checks(), indent=2)); return
    if args.compressor_only:
        print(json.dumps({'source': source_checks(), 'compressor': compressor_checks()}, indent=2)); return
    if not args.out:
        ap.error('--out is required for the model screen')
    os.environ.update(DSV41_VERIFY_ODD='1', DSV41_BLOCK_CONFIDENCE='1',
                      DSV41_BLOCK_DYNAMIC='3,5', DSV41_BLOCK='', DSV41_GRAPHS_MAX='16',
                      DSV41_ROUTER_BF16='0')
    from bench_decode_timeline_tp import V
    V.save_prune_db = lambda *a, **kw: None
    e = V.V41Engine(os.environ['MODEL_DIR'], max_seq=32768, arena_gb=90.2,
                   trace_stats='/app/results/trace-union/stats/coverage.json', spec=True,
                   prune_keep=.61, transient_slots=8, keep_free_gb=6)
    if not 0 <= args.prompt_padding <= 10000:
        ap.error('--prompt-padding must be in 0..10000')
    report = qualify(e, args.out, allow_failure=args.allow_odd_failure,
                     prompt_padding=args.prompt_padding)
    if args.depth_followup and report['passed']:
        from bench_verify_depths_tp import run_verify_depths
        run_verify_depths(e, args.out)
    if args.router_followup:
        from bench_router_followup_tp import run_router_followup
        run_router_followup(e, args.out)
    passed = all(e.ep.gather_objects(report['passed']))
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0 if passed else 1)


if __name__ == '__main__':
    main()
