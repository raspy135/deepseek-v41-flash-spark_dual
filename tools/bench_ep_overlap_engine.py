"""Same-process full-model overlap A/B, run on two nodes with serving stopped.

Frozen experts, no prefix reuse, no demand DB writes. Only this standalone harness
toggles the flag between calls; the mode is broadcast before both ranks generate.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, '/app')

# Must be set before importing the engine's module-level controls.
os.environ['DSV41_PREFILL_EP_OVERLAP'] = '1'  # allocate stream and agree at startup
os.environ['DSV41_PRUNE_SWAP'] = '0'
os.environ['DSV41_PRUNE_SWAP_PREFILL'] = '0'
os.environ['DSV41_PREFIX_CACHE'] = '0'
for flag in ('PREFILL_MOE_TIMING', 'PREFILL_TIMING', 'ATTN_TIMING', 'STEP_TIMING', 'GPU_TIMING'):
    os.environ['DSV41_' + flag] = '0'

import torch
from tokenizers import Tokenizer
import engine.v41_engine as V


def main():
    # Retain demand recording overhead, but do not persist synthetic benchmark votes.
    V.save_prune_db = lambda *args, **kwargs: None
    model_dir = os.environ['MODEL_DIR']
    engine = V.V41Engine(model_dir, max_seq=262144, arena_gb=88,
                        trace_stats='/app/results/trace-union/stats/coverage.json',
                        spec=True, prune_keep=.60, transient_slots=16, keep_free_gb=6)
    text = 'Summarize this documentation concisely.\n\n' + Path('/app/README.md').read_text()
    ids = Tokenizer.from_file(model_dir + '/tokenizer.json').encode(text).ids
    rows = []
    mask_hash = hashlib.sha256(b''.join(m.cpu().numpy().tobytes()
        for _, m in sorted(engine.model_prune_mask.items()))).hexdigest()
    order = (0, 1, 1, 0, 1, 0) if '--reverse' in sys.argv else (0, 1, 0, 1, 0, 1)
    for index, proposed in enumerate(order):
        mode = engine.ep.broadcast_obj(proposed if engine.ep.rank == 0 else None)
        os.environ['DSV41_PREFILL_EP_OVERLAP'] = str(mode)
        output = []
        for burst in engine.generate(ids, max_tokens=16, temperature=0, seed=42, ignore_eos=True):
            output.extend(burst)
        stats = engine.last_stats
        assert stats['prefix_cached_tokens'] == 0
        assert engine.expert_generation == 0
        current_hash = hashlib.sha256(b''.join(m.cpu().numpy().tobytes()
            for _, m in sorted(engine.model_prune_mask.items()))).hexdigest()
        assert current_hash == mask_hash
        row = dict(run=index, warmup=index < 2, overlap=mode, tokens=len(ids),
                   prefill_s=stats['prefill_s'], prefill_tok_s=stats['prefill_tok_s'],
                   output_sha256=hashlib.sha256(json.dumps(output).encode()).hexdigest())
        rows.append(row)
        if engine.ep.rank == 0:
            print('OVERLAP_AB ' + json.dumps(row), flush=True)
    # Compare warmed arms, avoiding startup capture as a numerical confound.
    assert len({r['output_sha256'] for r in rows[2:]}) == 1, 'warmed output changed'
    # Both ranks must have passed their assertions before publishing success.
    engine.ep.broadcast_obj(True if engine.ep.rank == 0 else None)
    if engine.ep.rank == 0:
        print('OVERLAP_AB_OUTPUTS_EQUAL', flush=True)
    # The full-model process-group/graph teardown can hang after all tests pass.
    # This disposable process has no pending writes; let process exit reclaim it.
    torch.cuda.synchronize()
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
