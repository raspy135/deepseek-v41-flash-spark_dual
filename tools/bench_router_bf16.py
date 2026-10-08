"""Cold real-weight router screen. Reports numerical differences, never promotes."""
import json
import os
from pathlib import Path
import statistics
import sys
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent)]
import torch
from safetensors import safe_open
from router_bf16 import project


def timed(fn, pairs):
    for x, w in pairs:
        fn(x, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for x, w in pairs:
            fn(x, w)
    samples = []
    for _ in range(7):
        graph.replay()
        a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        a.record(); graph.replay(); b.record(); b.synchronize()
        samples.append(a.elapsed_time(b) * 1000 / len(pairs))
    return statistics.median(samples)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261007)
    root = Path(os.environ['MODEL_DIR'])
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    def get(key):
        with safe_open(str(root / index[key]), framework='pt', device='cpu') as f:
            return f.get_tensor(key)
    weights, biases = [], []
    for layer in range(40):
        source = get(f'layers.{layer}.ffn.gate.weight')
        assert source.dtype == torch.bfloat16, (layer, source.dtype)
        weights.append(source.cuda())
        biases.append(get(f'layers.{layer}.ffn.gate.bias').cuda().float())
    expanded = [w.float() for w in weights]
    report = {'checkpoint_dtype': 'BF16', 'accumulation_dtype': 'FP32',
              'bit_exact': False, 'rows': [], 'routing': []}
    output = Path(os.environ.get('BENCH_OUT', '/out/router-bf16.json'))
    def save():
        output.write_text(json.dumps(report, indent=2))
    def baseline(x, w):
        pad = torch.zeros((16, x.shape[1]), device=x.device, dtype=torch.float32)
        pad[:x.shape[0]].copy_(x)
        return torch.mm(pad, w.t())[:x.shape[0]]
    # Screening uses layer-zero copies larger than L2, then all 40 different
    # layers for the chosen plan. Both arms include the same input-copy cost.
    copies = [weights[0].clone() for _ in range(16)]
    copies32 = [w.float() for w in copies]
    x = torch.randn(4, 5120, device='cuda', dtype=torch.bfloat16) * .5
    plans = [(s, n, k) for s in (1, 2, 4, 8) for n, k in ((32, 128), (64, 128))]
    plan_times = []
    ref = baseline(x, expanded[0])
    for split, bn, bk in plans:
        fn = lambda x, w: project(x, w, split=split, bn=bn, bk=bk)
        got = fn(x, weights[0])
        us = timed(fn, [(x, w) for w in copies])
        row = dict(split=split, bn=bn, bk=bk, us=us,
                   max_abs=float((got-ref).abs().max()),
                   relative=float((got-ref).norm()/ref.norm()))
        plan_times.append(row)
        print('PLAN', json.dumps(row), flush=True)
    best = min(plan_times, key=lambda r:r['us'])
    report['plans'] = plan_times
    report['selected'] = best
    kwargs = {k:best[k] for k in ('split', 'bn', 'bk')}
    for rows in (1, 4, 6):
        xs = [torch.randn(rows, 5120, device='cuda', dtype=torch.bfloat16)*.5 for _ in weights]
        times = {'baseline':[], 'bf16':[]}
        for arm in ('baseline', 'bf16', 'bf16', 'baseline'):
            fn = baseline if arm=='baseline' else lambda x,w:project(x,w,**kwargs)
            ws = expanded if arm=='baseline' else weights
            times[arm].append(timed(fn, list(zip(xs, ws))))
        row = dict(rows=rows, baseline_us=statistics.median(times['baseline']),
                   bf16_us=statistics.median(times['bf16']))
        report['rows'].append(row)
        print('COLD', json.dumps(row), flush=True)
        save()
    # Several activation magnitudes, pruning densities, and exact ties. This is
    # a router comparison, not a general quality test or actual serving trace.
    for scale in (.001, .1, 1., 10., 100.):
        tokens, changed_sets, changed_order, max_delta, max_weight_delta = 0, 0, 0, 0., 0.
        for layer, w in enumerate(weights):
            x = torch.randn(16, 5120, device='cuda', dtype=torch.bfloat16) * scale
            x[0].zero_()
            ref = baseline(x, expanded[layer])
            got = project(x, w, **kwargs)
            assert torch.equal(got[:1], project(x[:1], w, **kwargs)), 'row variance'
            max_delta = max(max_delta, float((got-ref).abs().max()))
            for fraction in (1., .6, .3):
                keep = torch.rand(384, device='cuda') < fraction
                keep[:6] = True
                a, b = [torch.nn.functional.softplus(g).sqrt() for g in (ref, got)]
                bias = biases[layer]
                ai = (a+bias).masked_fill(~keep, float('-inf')).topk(6,-1).indices
                bi = (b+bias).masked_fill(~keep, float('-inf')).topk(6,-1).indices
                changed_order += int((ai!=bi).any(-1).sum())
                changed_sets += int((ai.sort(-1).values!=bi.sort(-1).values).any(-1).sum())
                aw, bw = a.gather(-1,ai), b.gather(-1,bi)
                aw = aw/aw.sum(-1,keepdim=True)*1.5
                bw = bw/bw.sum(-1,keepdim=True)*1.5
                max_weight_delta = max(max_weight_delta, float((aw-bw).abs().max()))
                tokens += 16
        row = dict(scale=scale, token_cases=tokens, changed_sets=changed_sets,
                   changed_order=changed_order, max_logit_abs=max_delta,
                   max_route_weight_abs=max_weight_delta, row_invariant=True)
        report['routing'].append(row)
        print('ROUTING', json.dumps(row), flush=True)
        save()
    print('ROUTER_BF16_SCREEN_DONE', flush=True)


if __name__ == '__main__':
    main()
