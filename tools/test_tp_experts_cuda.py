"""Two-rank real-weight FP4 TP gate, including graph capture and balanced/skewed timing."""
import json
import os
import sys
import time
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/tools')

import torch
from safetensors import safe_open
import fp4_moe as K
from engine.dist import EPDistributed


def main():
    if '--output' in sys.argv:
        os.environ['DSV41_TP_EXPERT_LAYOUT'] = 'output'
    if '--scatter' in sys.argv:
        os.environ['DSV41_TP_EXPERT_REDUCE'] = 'scatter'
    ep = EPDistributed()
    ep.init('cuda')
    root = os.environ['MODEL_DIR']
    index = json.load(open(root + '/model.safetensors.index.json'))['weight_map']
    full = K.ExpertArena(13, 'cuda')
    tp = K.ExpertArena(13, 'cuda', ep.rank, ep.world)
    for expert in range(12):
        prefix = f'layers.0.ffn.experts.{expert}.'
        values = []
        for name in ('w1.weight', 'w1.scale', 'w2.weight', 'w2.scale', 'w3.weight', 'w3.scale'):
            with safe_open(root + '/' + index[prefix + name], framework='pt', device='cpu') as f:
                values.append(f.get_tensor(prefix + name))
        full.load_slot(expert, *values)
        tp.load_slot(expert, *values)
    for arena in (full, tp):
        for name in ('w1', 's1', 'w2', 's2', 'w3', 's3'):
            getattr(arena, name)[12].zero_()
    def timed(fn, count=5):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(count):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000 / count
    for tokens in (1, 6, 63, 128, 2048):
        torch.manual_seed(42)
        x = torch.randn(tokens, K.DIM, device='cuda', dtype=torch.bfloat16)
        weights = torch.rand(tokens, 6, device='cuda')
        weights /= weights.sum(-1, keepdim=True)
        for skew in (False, True):
            selected = torch.arange(6, device='cuda', dtype=torch.int32)
            if skew:
                selected *= 2
            selected = selected.repeat(tokens, 1)
            remote_null = selected.masked_fill(selected % 2 != ep.rank, 12)
            def ep_run():
                y = K.moe_forward(x, remote_null, weights, full, out_dtype=torch.float32,
                                  slots_repeat=True, null_slot=12)
                ep.combine(y)
                return y
            def tp_run():
                return K.moe_forward(x, selected, weights, tp, out_dtype=torch.float32)
            expected = K.moe_forward(x, selected, weights, full, out_dtype=torch.float32)
            actual = tp_run()
            rel = float((actual - expected).norm() / expected.norm().clamp_min(1e-20))
            finite = bool(torch.isfinite(actual).all())
            assert finite and rel < .005, (tokens, skew, rel)
            if '--output' in sys.argv:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            ep_ms, tp_ms = timed(ep_run), timed(tp_run)
            row = dict(tokens=tokens, skew=skew, relative_l2=rel,
                       max_abs=float((actual - expected).abs().max()), ep_ms=ep_ms, tp_ms=tp_ms)
            if ep.rank == 0:
                print('TP_KERNEL ' + json.dumps(row), flush=True)
            if tokens == 6 and not skew:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        tp_run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = tp_run()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(captured, actual, rtol=0, atol=0)
    assert all(ep.gather_objects(True))
    if ep.rank == 0:
        print('TP_KERNEL_GATE_PASSED', flush=True)
    torch.cuda.synchronize()
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
