"""Low-volume live chat continuation check; synthetic text only."""
import json
import time
import urllib.request

BASE = 'http://127.0.0.1:8000'


def health():
    with urllib.request.urlopen(BASE + '/health', timeout=10) as r:
        return json.load(r)


def idle():
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        h = health()
        if not h['busy']:
            return h
        time.sleep(.2)
    raise RuntimeError('post-response preparation did not finish')


def post(body):
    request = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=180) as r:
        data = r.read()
    return data, time.monotonic() - start


def main():
    assert not health()['busy'], 'server is busy; do not interrupt a user request'
    messages = [{'role': 'user', 'content': 'Explain binary search in three short sentences.'}]
    raw, wall = post(dict(model='deepseek', messages=messages, temperature=0, seed=42,
                         enable_thinking=False, max_tokens=128, stream=False))
    first = json.loads(raw)
    h = idle()
    prepared = h['engine_config']['prefix_response_last']
    assert prepared and prepared['status'] == 'saved', h
    messages += [first['choices'][0]['message'],
                 {'role': 'user', 'content': 'Give its time complexity only.'}]
    raw, second_wall = post(dict(model='deepseek', messages=messages, temperature=0, seed=42,
                                enable_thinking=False, max_tokens=16, stream=True,
                                stream_options={'include_usage': True}))
    assert b'data: [DONE]' in raw
    events = [json.loads(line[6:]) for line in raw.decode().splitlines()
              if line.startswith('data: ') and line != 'data: [DONE]']
    stats = next(e['x_engine_stats'] for e in reversed(events) if 'x_engine_stats' in e)
    assert stats['prefix_cached_tokens'] >= prepared['cached_tokens'], (stats, prepared)
    h = idle()
    assert h['engine_config']['prefix_response_last']['cached_tokens'] > prepared['cached_tokens']
    print(json.dumps({'nonstream_wall_s': wall, 'first_usage': first['usage'],
                      'prepared': prepared, 'stream_wall_s': second_wall,
                      'next_prefix_tokens': stats['prefix_cached_tokens'],
                      'next_prefill_s': stats['prefill_s'], 'healthy': h['status']}))


if __name__ == '__main__':
    main()
