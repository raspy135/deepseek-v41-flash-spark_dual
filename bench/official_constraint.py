"""Run only existing synthetic constraint probes against the official API.

Credentials are read locally, never logged or saved; redirects are rejected.
No captured user requests or conversation history are transmitted.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.error


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location('quality_quant2',
        '/home/ryan/git/llm_benchmark/quality_quant2.py')
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    key = Path('/home/ryan/git/data/official_key').read_text().strip()
    if not key or any(c.isspace() for c in key):
        raise SystemExit('Expected a nonempty bare API key in the local key file')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    url = 'https://api.deepseek.com/chat/completions'
    payload = dict(model='deepseek-flash', url=url, label='official-constraint-nothink',
                   when=time.strftime('%F %T'), temperature=0, thinking={'type': 'disabled'}, rows=[])
    for probe, item, prompt, cap, grade in benchmark.build_probes():
        if probe != 'constraint':
            continue
        body = dict(model=payload['model'], messages=[dict(role='user', content=prompt)],
                    temperature=0, max_tokens=cap, thinking=payload['thinking'])
        request = urllib.request.Request(url, json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        start = time.perf_counter()
        try:
            with opener.open(request, timeout=180) as response:
                data = json.load(response)
        except urllib.error.HTTPError as error:
            raise SystemExit(f'Official API HTTP {error.code}; response body suppressed') from None
        except Exception as error:
            raise SystemExit(f'Official API transport failed: {type(error).__name__}') from None
        choice = data['choices'][0]
        message = choice['message']
        content = message.get('content') or ''
        if message.get('reasoning_content'):
            raise SystemExit('Unexpected reasoning content: non-thinking comparison invalid')
        score, checks = grade(content)
        row = dict(probe=probe, item=item, score=round(score, 3), checks=checks,
                   finish=choice['finish_reason'], secs=round(time.perf_counter()-start, 2),
                   content=content, usage=data.get('usage'), response_model=data.get('model'),
                   system_fingerprint=data.get('system_fingerprint'))
        payload['rows'].append(row)
        with os.fdopen(os.open(args.out, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600), 'w') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(json.dumps({k: row[k] for k in ('item', 'score', 'checks', 'finish', 'response_model')}), flush=True)
    print('constraint_mean', round(sum(r['score'] for r in payload['rows']) / len(payload['rows']), 4))


if __name__ == '__main__':
    main()
