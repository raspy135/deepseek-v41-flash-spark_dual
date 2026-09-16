"""Small synthetic-image TP2 gate: RAM/disk reuse and changed-image invalidation."""
import base64
import io
import json
import os
import sys
sys.path[:0] = ['/app', '/app/tools']
os.environ.update(DSV41_PREFIX_RESPONSE='0', DSV41_PREFIX_DISK='1',
                  DSV41_PREFIX_DISK_STRICT='1', DSV41_PREFIX_CACHE='1',
                  DSV41_VISION='1', DSV41_PRUNE_SWAP='0', DSV41_PRUNE_ADAPT='0',
                  DSV41_PRUNE_SWAP_PREFILL='0')
import torch
from PIL import Image
import engine.v41_engine as V
from server.app import Tok, load_encoding_module, build_chat_prompt
import server.app as app


def main():
    V.save_prune_db = lambda *a, **kw: None
    root = os.environ['MODEL_DIR']
    e = V.V41Engine(root, max_seq=786432, arena_gb=92,
                    trace_stats='/app/results/trace-union/stats/coverage.json',
                    spec=True, prune_keep=.63, transient_slots=16, keep_free_gb=6)
    tok, enc = Tok(root), load_encoding_module(root)
    app.VISION_OK = e.vision is not None
    records = []
    def chat(color='red', suffix=''):
        buf = io.BytesIO()
        Image.new('RGB', (64, 64), color).save(buf, format='PNG')
        url = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
        body = {'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Inspect this image.'},
            {'type': 'image_url', 'image_url': {'url': url}},
            {'type': 'text', 'text': ('Context note: describe colors literally. ' * 180)
             + suffix + ' Name the dominant color in one word.'}]}]}
        _, ids, _, vl = build_chat_prompt(body, enc, tok, False, 75, e)
        return ids, vl
    def run(ids, vl, label):
        e.set_vl_inputs(*vl) if vl is not None else e.set_vl_inputs(None, None)
        out = []
        for burst in e.generate(ids, max_tokens=8, temperature=0, seed=42):
            out.extend(burst)
        records.append({'case': label, 'tokens': len(ids), 'prefix': e.last_stats['prefix_cached_tokens'],
                        'prefill_s': e.last_stats['prefill_s'], 'output': tok.decode(out)})
        return out
    ids, vl = chat()
    first = run(ids, vl, 'cold_red')
    warm = run(ids, vl, 'ram_red')
    assert e.last_stats['prefix_cached_tokens'] == len(ids), records
    assert first == warm, records
    e.prefix_disk.join()
    run(tok.encode('Reply OK.'), None, 'displace')
    disk = run(ids, vl, 'disk_red')
    assert e.prefix_disk.stats['source'] == 'disk', e.prefix_disk.stats
    assert e.last_stats['prefix_cached_tokens'] == len(ids) and disk == first, records
    # A changed suffix still reuses image-containing chunk boundaries.
    next_ids, next_vl = chat(suffix='Be concise. ')
    cached = run(next_ids, next_vl, 'suffix_red')
    assert e.last_stats['prefix_cached_tokens'] >= vl[1][0].start + vl[1][0].types.numel(), records
    persistence, e.prefix_disk = e.prefix_disk, None
    e._prefix_cache, e._prefix_snapshots = None, {}
    fresh = run(next_ids, next_vl, 'fresh_suffix_red')
    assert cached == fresh, records
    e.prefix_disk = persistence
    changed_ids, changed_vl = chat(color='blue', suffix='Be concise. ')
    assert changed_ids == next_ids, 'test must collide on placeholder token IDs'
    changed = run(changed_ids, changed_vl, 'changed_blue')
    assert e.last_stats['prefix_cached_tokens'] <= changed_vl[1][0].start, records
    e.prefix_disk.join()
    e.prefix_disk = None
    e._prefix_cache, e._prefix_snapshots = None, {}
    fresh_changed = run(changed_ids, changed_vl, 'fresh_blue')
    assert changed == fresh_changed, records
    assert all(e.ep.gather_objects(True))
    if e.ep.rank == 0:
        print('VISION_PREFIX_GATE_PASS ' + json.dumps(records), flush=True)
    torch.cuda.synchronize()
    os._exit(0)


if __name__ == '__main__':
    main()
