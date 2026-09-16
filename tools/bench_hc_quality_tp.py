"""Small HC16/HC32 quality comparison, with the saved demand ranking frozen."""
import collections
import json
import os
import sys
sys.path[:0] = ['/app', '/app/tools']
for key in ('DSV41_PRUNE_SWAP', 'DSV41_PRUNE_SWAP_PREFILL', 'DSV41_PREFIX_CACHE',
            'DSV41_PREFIX_DISK', 'DSV41_PREFIX_RESPONSE', 'DSV41_GPU_TIMING', 'DSV41_STEP_TIMING'):
    os.environ[key] = '0'
os.environ['DSV41_PRUNE_ADAPT'] = '1'  # read existing demand; do not swap or save it
os.environ['DSV41_HC_MM_TILE'] = '16'
import torch
import engine.v41_engine as V
import v41_ref as R
from server.app import Tok, load_encoding_module, build_chat_prompt
import bench_quality_quant2_fixture as Q


def main():
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=524288, arena_gb=90,
                   trace_stats='/app/results/trace-union/stats/coverage.json',
                   spec=True, prune_keep=.61, transient_slots=16, keep_free_gb=6)
    tok, enc = Tok(root), load_encoding_module(root)
    eos = tok.token_to_id(enc.eos_token)
    assert eos is not None
    fd = e.fast
    graphs = {16: ({}, None), 32: ({}, None)}
    probes = []
    counts = collections.Counter()
    for family, name, prompt, limit, grader in Q.build_probes():
        # Four depths, all three tiny counts, and three constraint cases.
        if family not in ('nesting', 'char_count', 'constraint'):
            continue
        if family == 'constraint' and counts[family] >= 3:
            continue
        counts[family] += 1
        probes.append((family, name, prompt, min(limit, 160), grader))
    probes.append(('japanese', 'brief explanation',
                   '日本語だけで、キャッシュが何かを二文で説明してください。', 100, None))
    scores = collections.defaultdict(list)
    for i, (family, name, prompt, limit, grade) in enumerate(probes):
        _, ids, _, _ = build_chat_prompt({'messages': [{'role': 'user', 'content': prompt}]},
                                        enc, tok, False, 75, e)
        outputs = {}
        for tile in ((16, 32) if i % 2 == 0 else (32, 16)):
            assert all(x == tile for x in e.ep.gather_objects(tile))
            R.HC_MM_TILE = tile
            fd.graphs, fd.draft_graphs = graphs[tile]
            output = []
            for burst in e.generate(ids, max_tokens=limit, temperature=0, seed=42, stop_token_ids={eos}):
                output.extend(burst)
            graphs[tile] = fd.graphs, fd.draft_graphs
            text = tok.decode([t for t in output if t != eos])
            result = grade(text) if grade else None
            outputs[tile] = output
            if result:
                scores[tile, family].append(result[0])
            if e.ep.rank == 0:
                print('HC_QUALITY ' + json.dumps({'tile': tile, 'family': family, 'name': name,
                      'tokens': len(output), 'limit': limit, 'eos': eos in output,
                      'grade': result, 'text': text}, ensure_ascii=False), flush=True)
        if e.ep.rank == 0:
            print('HC_QUALITY_MATCH ' + json.dumps({'name': name, 'exact_tokens': outputs[16] == outputs[32]}), flush=True)
    if e.ep.rank == 0:
        print('HC_QUALITY_SUMMARY ' + json.dumps({f'{tile}/{family}': sum(values)/len(values)
              for (tile, family), values in scores.items()}), flush=True)
    assert all(e.ep.gather_objects(True))
    os._exit(0)


if __name__ == '__main__':
    main()
