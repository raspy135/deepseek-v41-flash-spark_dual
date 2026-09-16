"""Small full-engine TP decode trace; synthetic coding prompt, frozen adaptation."""
import json
import os
import sys
import time
sys.path[:0] = ['/app', '/app/tools']
os.environ['DSV41_HC_MM_TILE'] = '16'  # explicit pre-optimization baseline
for name in ('DSV41_PRUNE_SWAP', 'DSV41_PRUNE_SWAP_PREFILL', 'DSV41_PRUNE_ADAPT',
             'DSV41_PREFIX_CACHE', 'DSV41_PREFIX_DISK', 'DSV41_PREFIX_RESPONSE',
             'DSV41_GPU_TIMING', 'DSV41_STEP_TIMING'):
    os.environ[name] = '0'
import torch
import engine.v41_engine as V
from torch.profiler import profile, ProfilerActivity
from server.app import Tok, load_encoding_module, build_chat_prompt


def main():
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=524288, arena_gb=90,
                    trace_stats='/app/results/trace-union/stats/coverage.json',
                    spec=True, prune_keep=.61, transient_slots=16, keep_free_gb=6)
    tok, enc = Tok(root), load_encoding_module(root)
    code = '\n'.join(f'def item_{i}(cache, key):\n    return cache.get(key, None)\n' for i in range(160))
    text = 'Review this Python cache interface and propose a thread-safe bounded implementation.\n' + code
    _, ids, _, _ = build_chat_prompt({'messages': [{'role': 'user', 'content': text}]},
                                     enc, tok, False, 75, e)
    for _ in range(2):
        emitted = []
        for burst in e.generate(ids, max_tokens=64, temperature=0, seed=42):
            emitted.extend(burst)
        print('DECODE_TRACE_WARM ' + json.dumps({'rank': e.ep.rank, 'stats': e.last_stats}), flush=True)
    fd, m = e.fast, e.model
    pos = m.c.len
    token = emitted[-2] if len(emitted) > 1 else emitted[-1]
    drafts, _ = fd.draft(token, pos-1, 0.0)
    block = torch.cat((torch.tensor([token], device=e.device), drafts.clone()))
    hashes = m.hash_state(block[None], pos)[0]
    rows = {layer: e.tables[layer].rows(hashes[:, li, :])
            for li, layer in enumerate(e.args.engram_layer_ids)}
    # Generation rollback can leave pending compressor values as views of the
    # verify scratch buffers. Replaying overwrites those views: retain immutable
    # copies so repeated steps really have identical history.
    pending = {layer: None if value is None else tuple(t.clone() for t in value)
               for layer, value in m.c.pending.items()}
    def step():
        m.c.len = pos
        m.c.pending.update(pending)
        logits, _ = fd.step(block, pos, rows)
        m.c.rollback(pos)
        return logits
    experiment = os.environ.get('DSV41_BENCH_DECODE_EXPERIMENT', '')
    if experiment in ('fused-routing', 'routing-fp8', 'hc32'):
        import fp4_moe as K
        import v41_ref as R
        from bench_fp4_decode_experiments import fused_routing, narrow_fp8_linear
        original_router, original_fp8 = K.build_routing_small, R.fp8_linear
        original_mm = R.mm
        def hc32_mm(x, weight):
            if (isinstance(weight, torch.Tensor) and weight.dtype == torch.float32
                    and tuple(weight.shape) == (24, 20480) and x.ndim == 2 and x.shape[0] <= 16):
                return torch.nn.functional.linear(
                    torch.nn.functional.pad(x, (0, 0, 0, 32-x.shape[0])), weight)[:x.shape[0]]
            return original_mm(x, weight)
        baseline_graphs = fd.graphs
        baseline_drafts = fd.draft_graphs
        reference = step().clone()
        repeated = step().clone()
        assert all(e.ep.gather_objects(torch.equal(reference, repeated))), 'Baseline replay is not stable'
        if experiment == 'hc32':
            R.mm = hc32_mm
        else:
            K.build_routing_small = fused_routing
        if experiment == 'routing-fp8':
            R.fp8_linear = narrow_fp8_linear
        fd.graphs = {}
        fd.draft_graphs = None
        step()  # cold graph capture is not part of either timing or comparison
        candidate = step().clone()
        candidate_graphs = fd.graphs
        candidate_drafts = fd.draft_graphs
        exact = torch.equal(reference, candidate)
        delta = float((reference.float()-candidate.float()).abs().max())
        matched = all(e.ep.gather_objects(exact))
        if experiment != 'hc32':
            assert matched, f'Router logits differ: {delta}'
        print('DECODE_CANDIDATE_LOGITS ' + json.dumps({'rank': e.ep.rank, 'experiment': experiment,
              'exact': matched, 'delta': delta,
              'argmax_equal': torch.equal(reference.argmax(-1), candidate.argmax(-1))}), flush=True)
        # Retain both captured graph sets and alternate replay. Compilation and
        # graph capture are excluded, and both variants share the same KV state.
        for repeat in range(4):
            arms = [('baseline', baseline_graphs), ('router', candidate_graphs)]
            for label, graphs in (arms if repeat % 2 == 0 else arms[::-1]):
                fd.graphs = graphs
                step()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(10):
                    step()
                torch.cuda.synchronize()
                ms = (time.perf_counter()-start)*100
                ranks = e.ep.gather_objects(ms)
                if e.ep.rank == 0:
                    print('DECODE_ROUTER_AB ' + json.dumps({'repeat': repeat, 'label': label,
                          'rank_ms': ranks, 'exact_logits': exact, 'max_delta': delta}), flush=True)
        def select(label):
            candidate = label == 'router'
            fd.graphs = candidate_graphs if candidate else baseline_graphs
            fd.draft_graphs = candidate_drafts if candidate else baseline_drafts
            K.build_routing_small = fused_routing if candidate and experiment != 'hc32' else original_router
            R.fp8_linear = narrow_fp8_linear if candidate and experiment == 'routing-fp8' else original_fp8
            R.mm = hc32_mm if candidate and experiment == 'hc32' else original_mm

        # Warm both candidate parities, then order-balanced generation timings.
        for trial, label in enumerate(('router', 'baseline', 'router', 'router', 'baseline')):
            select(label)
            output = []
            for burst in e.generate(ids, max_tokens=64, temperature=0, seed=42):
                output.extend(burst)
            same = output == emitted
            matched = all(e.ep.gather_objects(same))
            if experiment != 'hc32':
                assert matched, 'Candidate generation differs'
            print('DECODE_ROUTER_GENERATION ' + json.dumps({'rank': e.ep.rank, 'trial': trial,
                   'label': label, 'warm': trial == 0, 'exact_tokens': same,
                   'stats': e.last_stats}), flush=True)
        from nesting_arm import NEST_PROMPT, grade_nest
        _, nesting_ids, _, _ = build_chat_prompt({'messages': [{'role': 'user',
                    'content': NEST_PROMPT.format(d=8, leaf=48)}]}, enc, tok, False, 75, e)
        nested = []
        for label in ('baseline', 'router'):
            select(label)
            output = []
            for burst in e.generate(nesting_ids, max_tokens=128, temperature=0, seed=42,
                                    stop_token_ids={tok.token_to_id(enc.eos_token)}):
                output.extend(burst)
            nested.append(output)
            print('DECODE_ROUTER_NESTING ' + json.dumps({'rank': e.ep.rank, 'label': label,
                  'text': tok.decode(output), 'grade': grade_nest(tok.decode(output), 8, 48)}), flush=True)
        matched = all(e.ep.gather_objects(nested[0] == nested[1]))
        if experiment != 'hc32':
            assert matched, 'Nesting tokens differ'
        os._exit(0)
    step()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    print('DECODE_TRACE_STEP_MS ' + json.dumps({'rank': e.ep.rank,
          'ms': (time.perf_counter()-start)*100, 'position': pos, 'rows': len(block)}), flush=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            step()
        torch.cuda.synchronize()
    folder = '/app/results/decode-full-trace'
    os.makedirs(folder, exist_ok=True)
    prof.export_chrome_trace(f'{folder}/rank{e.ep.rank}.json')
    table = prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=45)
    print('DECODE_TRACE_KERNELS\n' + table, flush=True)
    with open(f'{folder}/rank{e.ep.rank}.txt', 'w') as f:
        f.write(table)
    assert all(e.ep.gather_objects(True))
    if e.ep.rank == 0:
        print('DECODE_TRACE_PASS', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
