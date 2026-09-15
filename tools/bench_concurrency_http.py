"""Small synthetic serving probe; no private captures or API credentials.

Run against a local server with DSV41_MAX_CONCURRENCY=2. Results describe the
server's current adaptation/cache settings, not a frozen-weight quality A/B.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--tokens', type=int, default=128)
    args = parser.parse_args()
    prompts = [
        'Write a clear explanation of how B-tree insertion and splitting preserve balance.',
        '日本語だけで、二分探索の仕組みと計算量を例を使って説明してください。',
    ]

    def request(index):
        body = {'model': 'deepseek-v4.1-flash', 'messages': [{'role': 'user', 'content': prompts[index]}],
                'temperature': 0, 'max_tokens': args.tokens, 'stream': False,
                'reasoning_effort': 'none'}
        started = time.perf_counter()
        req = Request(args.url + '/v1/chat/completions', json.dumps(body).encode(),
                      {'Content-Type': 'application/json'})
        with urlopen(req, timeout=300) as response:
            result = json.load(response)
        assert result['choices'][0]['message']['content']
        stats = result.get('x_engine_stats', {})
        return {'lane_prompt': index, 'wall_s': time.perf_counter() - started,
                'tokens': result['usage']['completion_tokens'],
                'prefill_s': stats.get('prefill_s'), 'decode_tok_s': stats.get('decode_tok_s'),
                'scheduler': stats.get('scheduler'), 'prefix_cached_tokens': stats.get('prefix_cached_tokens')}

    # Warm both prompts, then time equal request counts serially and concurrently.
    for index in (0, 1):
        print(json.dumps({'warmup': request(index)}), flush=True)
    for parallel in (False, True, True):
        started = time.perf_counter()
        if parallel:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(request, (0, 1)))
        else:
            results = [request(i) for i in (0, 1)]
        elapsed = time.perf_counter() - started
        print(json.dumps({'parallel': parallel, 'wall_s': elapsed,
                          'aggregate_tok_s': sum(r['tokens'] for r in results) / elapsed,
                          'requests': results}), flush=True)


if __name__ == '__main__':
    main()
