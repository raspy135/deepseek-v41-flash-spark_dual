"""Low-volume live image-history smoke test; no user images or captured prompts."""
import base64
import io
import json
import time
import urllib.request

from PIL import Image


BASE = 'http://127.0.0.1:8000'


def main():
    with urllib.request.urlopen(BASE + '/health', timeout=10) as r:
        assert not json.load(r)['busy'], 'server busy; do not interrupt user traffic'
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), 'red').save(buf, format='PNG')
    url = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    messages = [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Inspect this image.'},
        {'type': 'image_url', 'image_url': {'url': url}},
        {'type': 'text', 'text': 'Context note: describe colors literally. ' * 180
         + 'Name the dominant color in one word.'}]}]
    stats = []
    for turn in range(2):
        request = urllib.request.Request(BASE + '/v1/chat/completions', json.dumps({
            'model': 'deepseek', 'messages': messages, 'enable_thinking': False,
            'max_completion_tokens': 8, 'temperature': 0, 'seed': 42,
            'stream': False}).encode(), headers={'Content-Type': 'application/json'})
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=180) as r:
            result = json.load(r)
        s = result['x_engine_stats']
        stats.append({'turn': turn, 'prompt': s['prompt_tokens'],
                      'cached': s['prefix_cached_tokens'], 'prefill_s': s['prefill_s'],
                      'wall_s': round(time.monotonic() - started, 3)})
        messages += [result['choices'][0]['message'],
                     {'role': 'user', 'content': 'Is it red or blue? Answer one word.'}]
    assert stats[1]['cached'] >= stats[0]['prompt'], stats
    print('VISION_PREFIX_HTTP_PASS ' + json.dumps(stats))


if __name__ == '__main__':
    main()
