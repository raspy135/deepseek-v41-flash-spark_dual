"""Replay a trusted local one-shot capture without printing prompt/completion content.

Run only on captures produced locally by server/app.py: torch's pickle loader can execute code.
Use DSV41_PREFIX_CACHE=0 on the server for matched full-prefill measurements. Keep expert
ranking/settings identical across arms and retain demand logging when measuring its cost.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture')
    ap.add_argument('out')
    ap.add_argument('--base', default='http://127.0.0.1:8000')
    ap.add_argument('--runs', type=int, default=2)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--allow-prefix-cache', action='store_true',
                    help='decode-only follow-up; do not interpret cached prefill as throughput')
    ap.add_argument('--label', required=True)
    args = ap.parse_args()
    cap = torch.load(args.capture, map_location='cpu', weights_only=False)
    if cap.get('vl') is not None or cap.get('kwargs', {}).get('grammar') is not None:
        raise ValueError('raw-token replay does not reproduce vision or constrained grammar')
    ids = cap['prompt_ids']
    payload = dict(cap['sampling'])
    payload.update(model='deepseek', prompt=ids, stream=False,
                   max_tokens=args.max_tokens)
    # Both arms get the same random seed; do not silently switch a sampled workload to greedy.
    if payload.get('seed') is None:
        payload['seed'] = 42
    digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    rows = []
    for i in range(args.runs):
        start = time.perf_counter()
        req = urllib.request.Request(args.base + '/v1/completions', json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as response:
            data = json.load(response)
        stats = data.get('x_engine_stats') or {}
        row = dict(label=args.label, run=i, wall_s=time.perf_counter()-start,
                   prompt_sha256=digest, usage=data.get('usage'), stats=stats,
                   finish=data['choices'][0].get('finish_reason'),
                   completion_sha256=hashlib.sha256(data['choices'][0]['text'].encode()).hexdigest())
        rows.append(row)
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(rows, f, indent=2)
        if stats.get('prefix_cached_tokens', 0) and not args.allow_prefix_cache:
            raise RuntimeError('prefix cache reused input; disable it before benchmarking')
        print(json.dumps(dict(run=i, usage=row['usage'], prefill_s=stats.get('prefill_s'),
                              prefill_tok_s=stats.get('prefill_tok_s'),
                              decode_tok_s=stats.get('decode_tok_s'),
                              accept_len=stats.get('accept_len_mean'))), flush=True)


if __name__ == '__main__':
    main()
