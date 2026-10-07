#!/usr/bin/env python3
"""Install pinned upstream expert statistics (~13 MB); never fetch model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT = '45a0caffc8f080f8fd32d22f4e3d4e9122e25e5f'
REPO = 'https://github.com/0xBakeer/deepseek-v41-flash-spark'
SOURCE = 'results/keepsets/topics/coverage.json'
SHA256 = 'eb5214a78791a1e8cc0db51353f6ea7931f4f2d18142776f90784e01e67f16d3'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dest', type=Path, default=Path('results/keepsets/upstream-topics/coverage.json'))
    p.add_argument('--source-file', type=Path, help='Reuse a previously obtained identical statistics file')
    a = p.parse_args()
    if a.dest.exists():
        data = a.dest.read_bytes()
    elif a.source_file:
        data = a.source_file.read_bytes()
    else:
        url = f'https://raw.githubusercontent.com/0xBakeer/deepseek-v41-flash-spark/{COMMIT}/{SOURCE}'
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise SystemExit('Profile checksum mismatch; refusing to write or overwrite statistics')
    if not a.dest.exists():
        a.dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = a.dest.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(a.dest)
    provenance = {'repository': REPO, 'commit': COMMIT, 'source': SOURCE,
                  'sha256': digest, 'bytes': len(data), 'license': 'MIT; Copyright (c) 2026 0xBakeer',
                  'contents': '39 topics, counts and saliency for 40 layers x 384 experts; no weights'}
    a.dest.with_name('coverage.provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({'path': str(a.dest), **provenance}))


if __name__ == '__main__':
    main()
