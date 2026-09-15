"""Bounded scheduling sweep for the existing software-FP4 prefill kernel.

Run with the model service stopped. Uses real layer-0 weights and EP-like null routing;
every candidate must be bit-identical to the current FP32 routed output. No arithmetic,
activation format, reduction order, or model routing policy is intentionally changed.
"""
import argparse
import json
import statistics

import torch

import fp4_moe as K
from test_fp4_moe import load_arena, random_routing


def timing(fn, iters):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(iters):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--iters', type=int, default=7)
    ap.add_argument('--confirm', action='store_true', help='alternate baseline/candidate on fixed routing')
    args = ap.parse_args()
    assert not K.DOT_SCALED, 'this sweep preserves the software-decoded arithmetic'
    arena = load_arena(128, 'cuda')
    null = 127
    for name in ('w1', 's1', 'w2', 's2', 'w3', 's3'):
        getattr(arena, name)[null].zero_()
    gen = torch.Generator().manual_seed(130)
    cases = []
    for t in (512, 2048):
        x = torch.randn(t, K.DIM, generator=gen).bfloat16().cuda()
        ids, weights = random_routing(t, 254, gen, 'cuda')
        slots = torch.where(ids % 2 == 0, ids // 2, torch.full_like(ids, null))
        kw = dict(out_dtype=torch.float32, slots_repeat=True, null_slot=null)
        if args.confirm:
            kw.update(routing_ids=slots, routing_slot_map=torch.arange(128, device='cuda', dtype=torch.int32))
        ref = K.moe_forward(x, slots, weights, arena, down_cfg=K._DOWN_CFG[64], **kw)
        cases.append((t, x, slots, weights, kw, ref))
    original_up, original_down = K._UP_CFG[64], K._DOWN_CFG[64]
    candidates = [('baseline', original_up, original_down)]
    candidates += [('up', cfg, original_down) for cfg in
                   ((64, 4, 2), (64, 4, 3), (128, 4, 1), (128, 4, 2))]
    candidates += [('down', original_up, cfg) for cfg in
                   ((64, 4, 3), (128, 4, 2), (128, 4, 3))]
    if args.confirm:
        candidates = [('baseline' if i % 2 == 0 else 'down', original_up,
                       original_down if i % 2 == 0 else (128, 4, 2)) for i in range(6)]
    rows = []
    def run(label, up, down):
        row = dict(label=label, up=up, down=down, cases=[])
        try:
            for t, x, slots, weights, kw, ref in cases:
                fn = lambda: K.moe_forward(x, slots, weights, arena, up_cfg=up,
                                           down_cfg=down, **kw)
                got = fn()
                equal = torch.equal(got, ref)
                item = dict(tokens=t, equal=equal,
                            max_delta=float((got-ref).abs().max()))
                item['ms'] = timing(fn, args.iters) if equal else None
                row['cases'].append(item)
        except Exception as e:
            row['error'] = str(e)
        rows.append(row)
        with open(args.out, 'w') as f:
            json.dump(rows, f, indent=2)
        print(json.dumps(row), flush=True)
        return row
    for candidate in candidates:
        run(*candidate)
    valid = [r for r in rows if 'error' not in r and len(r['cases']) == 2
             and all(c['equal'] for c in r['cases'])]
    best_up = min((r for r in valid if r['label'] in ('up', 'baseline')),
                  key=lambda r: r['cases'][-1]['ms'])['up']
    best_down = min((r for r in valid if r['label'] in ('down', 'baseline')),
                    key=lambda r: r['cases'][-1]['ms'])['down']
    run('combined', best_up, best_down)
    run('baseline-repeat', original_up, original_down)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
