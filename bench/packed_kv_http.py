"""Matched HTTP A/B: exact captured input IDs, greedy output, no prompt content logged."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture')
    ap.add_argument('out')
    ap.add_argument('--runs', type=int, default=3)
    args = ap.parse_args()
    cap = torch.load(args.capture, map_location='cpu', weights_only=False)
    payload = dict(cap['sampling'])
    payload.update(prompt=cap['prompt_ids'], stream=False, temperature=0.0,
                   seed=1234, max_tokens=128)
    rows = []
    for i in range(args.runs):
        req = urllib.request.Request('http://127.0.0.1:8000/v1/completions',
                                     json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=600) as r:
            raw = r.read()
            assert len(raw) == int(r.headers['Content-Length'])
        result = json.loads(raw)
        s = result['x_engine_stats']
        row = dict(run=i, wall_s=time.monotonic()-start,
                   sha256=hashlib.sha256(result['choices'][0]['text'].encode()).hexdigest(),
                   usage=result['usage'], stats={k: s.get(k) for k in
                   ('prefill_s','decode_s','decode_tok_s','accept_len_mean',
                    'prefix_cached_tokens','packed_kv','expert_generation')})
        rows.append(row)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
