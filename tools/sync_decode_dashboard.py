#!/usr/bin/env python3
"""Embed the decode dashboard in /expert-map for hot static-only deployment.

The standalone page is the source of truth. The existing handler reads expert_map.html
on every GET, so /expert-map?view=decode works without restarting an older server.
Run this after editing server/decode_probe.html. --sample-report embeds a measured
report stripped to counters/layouts; prompts and activation values are never included.
"""
import argparse
import base64
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = '<!-- BEGIN GENERATED DECODE DASHBOARD -->'
END = '<!-- END GENERATED DECODE DASHBOARD -->'


def compact_report(raw, title=None, workload=None):
    metadata = {'source': 'Saved measurement', 'title': 'Saved decode measurement',
                **raw.get('_dashboard', {})}
    if title is not None:
        metadata['title'] = title
    if workload is not None:
        metadata['workload'] = workload
    result = {'version': 1, 'ranks': [], '_dashboard': metadata}
    for node in raw['ranks']:
        report = node['report']
        dest = {k: report[k] for k in ('status', 'config', 'profile_config', 'coverage', 'loop_phases') if k in report}
        dest['cases'] = []
        for case in report['cases']:
            c = {k: case[k] for k in ('name', 'key', 'phase', 'live_ms', 'replay_count', 'skip_reason', 'snapshot_bytes') if k in case}
            c['metadata'] = {k: case['metadata'][k] for k in ('kind', 'weight', 'weight_bytes', 'shape', 'scope', 'parent') if k in case['metadata']}
            for kind in ('inputs', 'outputs'):
                c[kind] = [{k: tensor[k] for k in ('shape', 'stride', 'dtype') if k in tensor} for tensor in case.get(kind, [])]
            if 'isolation' in case:
                c['isolation'] = {k: case['isolation'][k] for k in ('exact', 'stale', 'error', 'warm_ms', 'cold_ms', 'warm_samples_ms', 'cold_samples_ms', 'calls', 'repeats', 'warmup', 'flush_bytes') if k in case['isolation']}
            dest['cases'].append(c)
        result['ranks'].append({'rank': node['rank'], 'report': dest})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-report', type=Path)
    parser.add_argument('--title', help='Label for the embedded measured sample')
    parser.add_argument('--workload', help='Measured workload description; never inferred from graph buckets')
    args = parser.parse_args()
    dashboard_path = ROOT / 'server/decode_probe.html'
    page_path = ROOT / 'server/expert_map.html'
    dashboard = dashboard_path.read_text()
    if args.sample_report:
        sample = compact_report(json.loads(args.sample_report.read_text()), args.title, args.workload)
        # A full two-node report has thousands of rows. Store it compressed so the
        # expert map does not carry megabytes of unparsed JSON on each page load.
        packed = gzip.compress(json.dumps(sample, separators=(',', ':')).encode(), mtime=0)
        payload = json.dumps({'encoding': 'gzip+base64', 'data': base64.b64encode(packed).decode()})
        dashboard = re.sub(r'(<script id="decode-dashboard-sample" type="application/json">).*?(</script>)',
                           lambda m: m[1] + payload + m[2], dashboard, count=1, flags=re.S)
        dashboard_path.write_text(dashboard)
    # Encoding '<' keeps embedded HTML script terminators inert in the enclosing page.
    source = json.dumps(dashboard, ensure_ascii=True).replace('<', '\\u003c')
    generated = f'''{START}
<script id="decode-dashboard-bootstrap">
if (new URLSearchParams(location.search).get('view') === 'decode') {{
  document.title = 'Deepseek · Decode performance';
  const source = {source};
  const page = new DOMParser().parseFromString(source, 'text/html');
  document.head.replaceChildren(...page.head.childNodes);
  document.body.replaceChildren(...page.body.childNodes);
  // DOMParser keeps scripts inert. Execute only our two dashboard scripts, in order.
  for (const id of ['decode-dashboard-core', 'decode-dashboard-ui']) {{
    const inert = document.getElementById(id), script = document.createElement('script');
    script.id = id; script.textContent = inert.textContent; inert.replaceWith(script);
  }}
}}
</script>
{END}'''
    page = page_path.read_text()
    if START in page:
        begin = page.index(START)
        end = page.index(END, begin) + len(END)
        page = page[:begin] + generated + page[end:]
    else:
        page = page.replace('</body>', generated + '\n</body>')
    # The expert map must not poll, paint, or register controls in decode view.
    if "if (new URLSearchParams(location.search).get('view') !== 'decode') {" not in page:
        page = page.replace("<script>\n'use strict';", "<script>\nif (new URLSearchParams(location.search).get('view') !== 'decode') {\n'use strict';", 1)
        boundary = page.index(START)
        before = page[:boundary]
        last_close = before.rfind('</script>')
        page = before[:last_close] + '}\n' + before[last_close:] + page[boundary:]
    if 'href="/expert-map?view=decode"' not in page:
        page = page.replace('<main>\n', '<main>\n<nav style="display:flex;gap:20px;margin-bottom:24px;font-size:12px" aria-label="Engine dashboards"><a href="/expert-map" aria-current="page">Expert memory</a><a href="/expert-map?view=decode">Decode performance</a></nav>\n', 1)
    page_path.write_text(page)
    print(f'Synced dashboard ({len(dashboard.encode()):,} bytes) into {page_path.name}.')


if __name__ == '__main__':
    main()
