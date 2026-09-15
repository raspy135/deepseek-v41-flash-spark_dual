"""Small live README timing series; stores only hashes, counters and filtered logs.

Adaptive loading remains enabled. Prefix eviction requests also affect request-unit
adaptation, so this is an observational series, not a fixed-placement experiment.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request


def request(base, path, payload=None):
    req = urllib.request.Request(base + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out', type=Path)
    ap.add_argument('--base', default='http://127.0.0.1:8000')
    ap.add_argument('--runs', type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(mode=0o700, parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    prompt = ('Read the README below as reference material. In three concise bullet points, '
              'explain what this engine does, how this EP2 fork differs from the upstream '
              'single-node version, and the main deployment requirements. Distinguish current '
              'fork instructions from the upstream README.\n\n<readme>\n'
              + (root / 'README.md').read_text() + '\n</readme>')
    for i in range(args.runs):
        health = request(args.base, '/health')
        if health.get('busy'):
            raise RuntimeError('Server busy; do not queue benchmark over real traffic')
        since = datetime.datetime.now(datetime.timezone.utc).isoformat()
        request(args.base, '/v1/completions', dict(model='deepseek', prompt='Reply with OK.',
                                                  max_tokens=1, temperature=0))
        start = time.perf_counter()
        data = request(args.base, '/v1/chat/completions', dict(
            model='deepseek', messages=[dict(role='user', content=prompt)],
            temperature=0, seed=42, max_tokens=256,
            chat_template_kwargs={'thinking': False}))
        stats = data.get('x_engine_stats') or {}
        if stats.get('prefix_cached_tokens', 0):
            raise RuntimeError('Unexpected prefix reuse')
        row = dict(run=i, wall_s=time.perf_counter()-start, usage=data.get('usage'),
                   stats=stats, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                   answer_sha256=hashlib.sha256(json.dumps(data['choices']).encode()).hexdigest())
        (args.out / f'run-{i}.json').write_text(json.dumps(row, indent=2))
        for rank in range(2):
            cmd = ['docker', 'logs', '--since', since, f'deepseek-v41-ep2-rank{rank}']
            if rank:
                cmd = ['ssh', 'ryan@10.0.0.2', *cmd]
            log = subprocess.run(cmd, capture_output=True, text=True, check=True)
            keep = ('prefill_chunks:', 'prefill_rank_phases:', 'adapted ', 'routed-miss',
                    'compilation', 'Compiling')
            lines = [line for line in (log.stdout + log.stderr).splitlines()
                     if any(key in line for key in keep)]
            (args.out / f'run-{i}-rank{rank}.log').write_text('\n'.join(lines) + '\n')
        print(json.dumps(dict(run=i, usage=row['usage'], **{k: stats.get(k) for k in
            ('prefill_s', 'prefill_tok_s', 'decode_tok_s', 'prune_miss_request',
             'prefill_expert_misses', 'nvme_gb', 'prefix_cached_tokens')})), flush=True)


if __name__ == '__main__':
    main()
