"""Frozen-placement A/B for draft/embedding TP; no private prompt text is logged."""
import argparse
import hashlib
import json
import os
import sys
sys.path[:0] = ['/app', '/app/tools']
ap = argparse.ArgumentParser()
ap.add_argument('--enable', action='store_true')
args = ap.parse_args()
for key in ('DSV41_TP_DRAFT_EXPERTS', 'DSV41_TP_EMBED'):
    os.environ[key] = '1' if args.enable else '0'
for key in ('DSV41_PREFIX_CACHE', 'DSV41_PREFIX_DISK', 'DSV41_PRUNE_ADAPT',
            'DSV41_PRUNE_SWAP', 'DSV41_PRUNE_SWAP_PREFILL'):
    os.environ[key] = '0'
os.environ['DSV41_PRUNE_DB'] = '/app/results/packed-kv-test-demand.npz'

import torch
import engine.v41_engine as V

def main():
    V.save_prune_db = lambda *a, **kw: None
    e = V.V41Engine(os.environ['MODEL_DIR'], max_seq=786432, arena_gb=88,
                    trace_stats='/app/results/trace-union/stats/coverage.json',
                    spec=True, prune_keep=.60, transient_slots=16, keep_free_gb=6)
    cap = (torch.load('/app/results/captured_request.pt', map_location='cpu', weights_only=False)
           if e.ep.rank == 0 else None)
    ids = e.ep.broadcast_obj(cap['prompt_ids'] if cap is not None else None)
    mask = hashlib.sha256()
    for layer in sorted(e.model_prune_mask):
        mask.update(e.model_prune_mask[layer].cpu().numpy().tobytes())
    for repeat in range(3):
        output = []
        for burst in e.generate(ids, max_tokens=128, temperature=0, seed=1234):
            output.extend(burst)
        row = dict(enabled=args.enable, repeat=repeat, mask=mask.hexdigest(),
                   output_hash=hashlib.sha256(json.dumps(output).encode()).hexdigest(),
                   output_tokens=len(output), allocated_bytes=torch.cuda.memory_allocated(),
                   stats={k:e.last_stats.get(k) for k in ('prefill_s', 'decode_s', 'decode_tok_s',
                          'accept_len_mean', 'prefix_cached_tokens')})
        assert row['stats']['prefix_cached_tokens'] == 0
        assert e.expert_generation == 0
        assert len(set(e.ep.gather_objects(row['output_hash']))) == 1
        if e.ep.rank == 0:
            print('TP_MEMORY ' + json.dumps(row), flush=True)
    torch.cuda.synchronize()
    sys.stdout.flush()
    os._exit(0)

if __name__ == '__main__':
    main()
