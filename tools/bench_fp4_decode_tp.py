"""Bounded TP2 decode MoE scheduling experiment, real weights / synthetic activations.

No server or demand DB. Preserves software FP4 arithmetic and output-layout TP.
Run via run_two_node_gate.sh with the service stopped. Times CUDA graph replay,
not eager Python dispatch; each accepted candidate must match baseline bitwise.
"""
import json
import os
import statistics
import sys
sys.path[:0] = ['/app', '/app/tools']
os.environ['DSV41_FP4_DOT_SCALED'] = '0'
os.environ['DSV41_TP_EXPERT_LAYOUT'] = 'output'
import torch
from safetensors import safe_open
from engine.dist import EPDistributed
import fp4_moe as K
from bench_fp4_decode_experiments import SeparateUp, fused_routing


def main():
    ep = EPDistributed()
    ep.init('cuda')
    assert ep.world == 2
    arena = K.ExpertArena(64, 'cuda', tp_rank=ep.rank, tp_world=2)
    shard = os.path.join(os.environ['MODEL_DIR'], 'model-00003-of-00048.safetensors')
    with safe_open(shard, framework='pt', device='cpu') as f:
        for expert in range(64):
            prefix = f'layers.0.ffn.experts.{expert}.'
            arena.load_slot(expert, *(f.get_tensor(prefix + name)
                            for name in ('w1.weight', 'w1.scale', 'w2.weight', 'w2.scale',
                                         'w3.weight', 'w3.scale')))
    stream = torch.cuda.Stream()
    rows = []
    gen = torch.Generator().manual_seed(20260916)
    configs = [('baseline', (128, 4, 1), (128, 8, 3)),
               ('up64-s2', (64, 4, 2), (128, 8, 3)),
               ('up128-s2', (128, 4, 2), (128, 8, 3)),
               ('up32-s3', (32, 4, 3), (128, 8, 3)),
               ('down128-w4-s2', (128, 4, 1), (128, 4, 2)),
               ('down64-w4-s2', (128, 4, 1), (64, 4, 2)),
               ('baseline-repeat', (128, 4, 1), (128, 8, 3))]
    original_up = K._moe_up_kernel
    original_router = K.build_routing_small
    if os.environ.get('DSV41_BENCH_DECODE_EXPERIMENT') == 'separate-up':
        configs = [('baseline', (128, 4, 1), (128, 8, 3)),
                   ('separate128-s1', (128, 4, 1), (128, 8, 3)),
                   ('separate128-s2', (128, 4, 2), (128, 8, 3)),
                   ('separate64-s2', (64, 4, 2), (128, 8, 3)),
                   ('baseline-repeat', (128, 4, 1), (128, 8, 3))]
    if os.environ.get('DSV41_BENCH_DECODE_EXPERIMENT') == 'confirm-separate':
        configs = [(('baseline-' if i % 2 == 0 else 'separate-') + str(i),
                    (128, 4, 1) if i % 2 == 0 else (128, 4, 2), (128, 8, 3))
                   for i in range(8)]
    if os.environ.get('DSV41_BENCH_DECODE_EXPERIMENT') == 'fused-routing':
        configs = [(('baseline-' if i % 2 == 0 else 'router-') + str(i),
                    (128, 4, 1), (128, 8, 3)) for i in range(8)]
    for t in (1, 4, 8):  # one token; k=3 verification; two-lane verification
        for sharing in (('shared',) if t == 1 else ('shared', 'mixed', 'disjoint')):
            if sharing == 'shared':
                slots = torch.arange(6).repeat(t, 1)
            elif sharing == 'disjoint':
                slots = torch.arange(t * 6).view(t, 6)
            else:
                slots = torch.stack([torch.randperm(16, generator=gen)[:6] for _ in range(t)])
            slots = slots.int().cuda()
            x = torch.randn(t, K.DIM, generator=gen).bfloat16().cuda()
            weights = torch.rand(t, 6, generator=gen).cuda()
            weights /= weights.sum(1, keepdim=True)
            torch.cuda.synchronize()
            ref = None
            for label, up, down in configs:
                K._moe_up_kernel = SeparateUp() if label.startswith('separate') else original_up
                K.build_routing_small = fused_routing if label.startswith('router') else original_router
                marks = []
                def mark(name):
                    event = torch.cuda.Event(enable_timing=True, external=True)
                    event.record()
                    marks.append((name, event))
                def fn(instrument=False):
                    return K.moe_forward(x, slots, weights, arena, up_cfg=up, down_cfg=down,
                                         out_dtype=torch.float32,
                                         stage_mark=mark if instrument else None)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        got = fn()
                torch.cuda.synchronize()
                if ref is None:
                    ref = got.clone()
                exact = torch.equal(got, ref)
                delta = float((got - ref).abs().max())
                checks = ep.gather_objects((exact, delta))
                row = dict(tokens=t, sharing=sharing, distinct=int(slots.unique().numel()),
                           label=label, up=up, down=down, exact=all(v[0] for v in checks),
                           max_delta=max(v[1] for v in checks))
                if row['exact']:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        result = fn()
                    for _ in range(3):
                        graph.replay()
                    torch.cuda.synchronize()
                    measurements = []
                    for _ in range(5):
                        a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        a.record()
                        for _ in range(20):
                            graph.replay()
                        b.record()
                        b.synchronize()
                        measurements.append(a.elapsed_time(b) / 20)
                    row['rank_ms'] = ep.gather_objects(statistics.median(measurements))
                    row['ms'] = max(row['rank_ms'])
                    assert torch.equal(result, ref)
                    del graph
                    # Separately instrument the baseline, so events cannot favor a candidate.
                    if label in ('baseline', 'baseline-0', 'separate-1', 'router-1'):
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            mark('start')
                            result = fn(True)
                            mark('end')
                        times = {name: [] for name, _ in marks[1:]}
                        for _ in range(10):
                            graph.replay()
                            torch.cuda.synchronize()
                            for (_, a), (name, b) in zip(marks, marks[1:]):
                                times[name].append(a.elapsed_time(b))
                        row['stages_rank_ms'] = ep.gather_objects({
                            name: statistics.median(values) for name, values in times.items()})
                        del graph
                rows.append(row)
                if ep.rank == 0:
                    print('DECODE_MOE ' + json.dumps(row), flush=True)
    if ep.rank == 0:
        print('DECODE_MOE_DONE ' + json.dumps({'cases': len(rows)}), flush=True)
    assert all(ep.gather_objects(True))
    torch.cuda.synchronize()
    os._exit(0)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
