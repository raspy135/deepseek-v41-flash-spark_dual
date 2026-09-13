"""GPU parity and timing for skipping EP2's zero expert; no server restart needed."""
import json
import os
import time

import torch
from test_fp4_moe import load_arena
from fp4_moe import DIM, moe_forward


def elapsed(fn, repeats=5):
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return sorted(values)[len(values) // 2]


@torch.no_grad()
def main():
    torch.manual_seed(42)
    arena = load_arena(16, 'cuda')
    null = 15
    for w in (arena.w1, arena.w2, arena.w3):
        w[null].zero_()
    for s in (arena.s1, arena.s2, arena.s3):
        s[null].fill_(127)
    results = []
    for tokens in (17, 128, 513, 2048):
        x = torch.randn(tokens, DIM, device='cuda', dtype=torch.bfloat16) * 0.1
        weights = torch.rand(tokens, 6, device='cuda')
        weights /= weights.sum(-1, keepdim=True)
        original = torch.randint(0, null, (tokens, 6), device='cuda', dtype=torch.int32)
        for remote in (0.0, 0.5, 1.0):
            slots = original.masked_fill(torch.rand(tokens, 6, device='cuda') < remote, null)
            slot_map = torch.randperm(16, device="cuda", dtype=torch.int32)
            inverse = torch.argsort(slot_map).to(torch.int32)
            slot_map = torch.cat((slot_map, torch.full((112,), null, device="cuda", dtype=torch.int32)))
            route_ids = inverse[slots.long()]
            def run(skip, fixed=False):
                return moe_forward(x, slots, weights, arena, out_dtype=torch.float32,
                                   slots_repeat=True, null_slot=null if skip else -1,
                                   routing_ids=route_ids if fixed else None,
                                   routing_slot_map=slot_map if fixed else None)
            baseline, optimized = run(False), run(True)
            assert torch.isfinite(optimized).all()
            assert torch.equal(baseline, optimized), (tokens, remote, (baseline-optimized).abs().max().item())
            if remote == 1.0:
                assert torch.count_nonzero(optimized) == 0
            fixed = run(True, True)
            assert torch.equal(baseline, fixed), (tokens, remote, "fixed routing parity")
            if tokens == 513 and remote == 0.5:
                # Dynamic unique/indexing would fail CUDA graph capture.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run(True, True)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run(True, True)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(baseline, captured)
            old = elapsed(lambda: run(False))
            new = elapsed(lambda: run(True))
            both = elapsed(lambda: run(True, True))
            row = dict(fixed_ms=round(both, 3), combined_speedup=round(old/both, 3), tokens=tokens, remote_fraction=remote, bit_exact=True,
                       baseline_ms=round(old, 3), optimized_ms=round(new, 3), speedup=round(old/new, 3))
            results.append(row)
            print(json.dumps(row), flush=True)
    path = os.environ.get('NULL_BENCH_OUT')
    if path:
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
