"""Matched-prompt prefill A/B probe. Repeated runs intentionally warm Engram caches.

Compare the same prompt/config across engine builds. This is not a cold-workload
benchmark. wall_s includes the short completion; prefill_s is engine accounting.
"""
import argparse
import hashlib
import json
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out')
    parser.add_argument('--base', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='deepseek')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--stream', action='store_true', help='also measure client time to first text')
    args = parser.parse_args()
    prompt = 'Explain the following code and identify patterns.\n' + ''.join(
        f'def function_{i}(value): return value * {i+1} + {i%17}\n' for i in range(300))
    payload = dict(model=args.model, prompt=prompt, max_tokens=16, temperature=0, seed=42)
    if args.stream:
        payload.update(stream=True, stream_options={'include_usage': True})
    body = json.dumps(payload).encode()
    rows = []
    for i in range(args.runs):
        req = urllib.request.Request(args.base + '/v1/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
        start = time.perf_counter()
        ttft = None
        with urllib.request.urlopen(req, timeout=300) as response:
            if args.stream:
                data, text = {}, ''
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    content = line[6:].strip()
                    if content == b'[DONE]':
                        break
                    event = json.loads(content)
                    for choice in event.get('choices', []):
                        delta = choice.get('text', '')
                        if delta and ttft is None:
                            ttft = time.perf_counter() - start
                        text += delta
                    for key in ('usage', 'x_engine_stats'):
                        if event.get(key) is not None:
                            data[key] = event[key]
                data['choices'] = [{'text': text}]
            else:
                data = json.load(response)
        row = dict(run=i, wall_s=time.perf_counter()-start, usage=data.get('usage'),
                   stats=data.get('x_engine_stats'), choices=data.get('choices'),
                   prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        if args.stream:
            row['ttft_s'] = ttft
        rows.append(row)
        with open(args.out, 'w') as f:
            json.dump(rows, f, indent=2)
        stats = row['stats'] or {}
        print(json.dumps(dict(run=i, wall_s=round(row['wall_s'], 3),
                              ttft_s=ttft,
                              prefill_s=stats.get('prefill_s'),
                              prefill_tok_s=stats.get('prefill_tok_s'))), flush=True)


if __name__ == '__main__':
    main()
