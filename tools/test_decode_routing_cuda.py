"""Benchmark router contract, including changing routes during graph replay."""
import sys
sys.path[:0] = ['/app', '/app/tools']
import torch
from bench_fp4_decode_experiments import fused_routing


def check(slots, block_slots, pairs, bm):
    expected = {}
    for i, slot in enumerate(slots.flatten().tolist()):
        expected.setdefault(slot, []).append(i)
    assert max(map(len, expected.values())) <= bm
    actual = {}
    for slot, indices in zip(block_slots.tolist(), pairs.view(-1, bm).tolist()):
        if slot < 0:
            assert all(i == -1 for i in indices)
            continue
        assert slot not in actual
        actual[slot] = [i for i in indices if i >= 0]
    assert actual == expected, (actual, expected)


def main():
    generator = torch.Generator().manual_seed(20260916)
    tested = 0
    for p in (1, 6, 24, 48, 64):
        for bm in (16, 64):
            routes = [torch.arange(p, dtype=torch.int32),
                      torch.arange(p, dtype=torch.int32) % max(1, p // bm + 1)]
            if p <= bm:
                routes.append(torch.full((p,), 12345, dtype=torch.int32))
            for _ in range(20):
                # At most ceil(p / 6) occurrences, with sparse arena indices.
                routes.append((torch.randperm(p, generator=generator) % 6 * 1001).int())
            slots = routes[0].view(1, -1).cuda()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                fused_routing(slots, bm)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                bs, bp, nb = fused_routing(slots, bm)
            assert nb == p
            for route in routes:
                slots.copy_(route.view_as(slots))
                graph.replay()
                check(slots, bs, bp, bm)
                tested += 1
    print(f'DECODE_ROUTING_PASS {tested} graph-replay route mutations')


if __name__ == '__main__':
    main()
