"""Two bounded long-context nesting controls; store only grading, hashes and metrics."""
import argparse
import hashlib
import json
import os
import urllib.request


def request(path, payload):
    req = urllib.request.Request('http://127.0.0.1:8000'+path, json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--require-loaded', action='store_true')
    ap.add_argument('--depths', type=int, nargs='+', default=[8, 10],
                    help='depths to test; repeat a depth for exact-prompt replays')
    args = ap.parse_args()
    rows = []
    for depth in args.depths:
        request('/v1/completions', dict(model='deepseek', prompt='Reply with OK.', max_tokens=1, temperature=0))
        reference = '\n'.join(f'def reference_{i}(value): return (value + {i}) % 97 # reference only'
                              for i in range(300))
        prompt = ('Ignore the reference text below when answering the task at the end.\n<reference>\n'
                  + reference + '\n</reference>\n'
                  + f'Output one JSON object nested exactly {depth} levels deep and nothing else. '
                  'Each level has exactly one key "n" whose value is the next level down. '
                  + f'The innermost "n" is the integer {40+depth}. '
                  + f'So depth 2 would be: {{"n": {{"n": {40+depth}}}}}')
        data = request('/v1/chat/completions', dict(model='deepseek', temperature=0, seed=42,
            max_tokens=256, chat_template_kwargs={'thinking': False},
            messages=[dict(role='user', content=prompt)]))
        answer = data['choices'][0]['message'].get('content') or ''
        expected = 40 + depth
        for _ in range(depth):
            expected = {'n': expected}
        try:
            passed = json.loads(answer) == expected
        except ValueError:
            passed = False
        stats = data.get('x_engine_stats') or {}
        row = dict(depth=depth, passed=passed, usage=data.get('usage'), stats=stats,
                   answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                   finish=data['choices'][0]['finish_reason'])
        rows.append(row)
        with os.fdopen(os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as f:
            json.dump(rows, f, indent=2)
        print(json.dumps({k: row[k] for k in ('depth', 'passed', 'usage', 'finish')}), flush=True)
        assert (data.get('usage') or {}).get('prompt_tokens', 0) > 4096, 'control too short'
        assert stats.get('prefix_cached_tokens', 0) == 0, 'unexpected prefix reuse'
        if args.require_loaded:
            assert stats.get('prefill_replicas', {}).get('loaded', 0) > 0, 'replicas did not activate'
    assert all(r['passed'] for r in rows), 'long-context nesting control failed'


if __name__ == '__main__':
    main()
