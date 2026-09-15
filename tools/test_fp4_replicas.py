"""Offline real-weight test of relocated EP contributions and unchanged decode slots.

Run with the service stopped. This does not prove full-model quality or speed.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.prefill_replicas import routing_tables
from test_fp4_moe import load_arena, random_routing
from fp4_moe import DIM, moe_forward


def main():
    arena = load_arena(38, 'cuda')
    null = 31
    replica_slots = [[32, 33, 34], [35, 36, 37]]
    moves = [dict(layer=0, expert=0, dst=1, slot=0),
             dict(layer=0, expert=2, dst=1, slot=1),
             dict(layer=0, expert=1, dst=0, slot=0)]
    names = ('w1', 's1', 'w2', 's2', 'w3', 's3')
    for name in names:
        tensor = getattr(arena, name)
        tensor[null].zero_()
        for row in moves:
            tensor[replica_slots[row['dst']][row['slot']]].copy_(tensor[row['expert']])
    tables = []
    originals = []
    for rank in range(2):
        lru = {(0, e): e for e in range(rank, 30, 2)}
        kw = dict(rank=rank, null_slot=null, replica_slots=replica_slots[rank],
                  n_layers=1, n_experts=30, keep=np.ones((1, 30), bool))
        for target, plan in ((tables, moves), (originals, [])):
            lut, routes = routing_tables(lru, plan, **kw)
            target.append((torch.as_tensor(lut[0], device='cuda'),
                           tuple(torch.as_tensor(t, device='cuda') for t in routes[0])))
    gen = torch.Generator().manual_seed(31)
    for n in (1, 6, 63, 512, 2048):
        x = torch.randn(n, DIM, generator=gen).bfloat16().cuda()
        ids, weights = random_routing(n, 30, gen, 'cuda')
        def run(pair):
            halves = []
            for lut, (route_ids, slots) in pair:
                halves.append(moe_forward(x, lut[ids], weights, arena, out_dtype=torch.float32,
                    slots_repeat=True, null_slot=null, routing_ids=route_ids[ids], routing_slot_map=slots))
            return halves[0] + halves[1]
        before, moved, restored = run(originals), run(tables), run(originals)
        assert torch.equal(before, restored), 'replicas modified original weights/routing'
        shared = torch.randn(n, DIM, generator=gen).bfloat16().cuda().float()
        equal = torch.equal((before + shared).bfloat16(), (moved + shared).bfloat16())
        delta = float((before-moved).abs().max())
        print(f'T={n} max_fp32_delta={delta:.9g} final_bf16_equal={equal} restored_exact=True', flush=True)
        assert equal, f'regrouped EP sum changed BF16 output at T={n}'


if __name__ == '__main__':
    with torch.inference_mode():
        main()
