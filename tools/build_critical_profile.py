"""Build a contribution-norm proxy from --output-norms traces, without model weights.

Only traced layers/experts qualify for streaming. This is a calibration profile,
not evidence of downstream quality improvement or zero-impact unobserved experts.
"""
import argparse
import glob
import json
from pathlib import Path
import re

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--layers', type=int, default=40)
    p.add_argument('--experts', type=int, default=384)
    a = p.parse_args()
    sums = np.zeros((a.layers, a.experts), np.float64)
    samples = np.zeros_like(sums)
    for f in sorted(glob.glob(str(Path(a.trace) / 'trace/layer*.npz'))):
        layer = int(re.search(r'layer(\d+)\.npz$', f).group(1))
        with np.load(f, allow_pickle=False) as d:
            ids, norms = d['indices'].astype(np.int64), d['output_norms'].astype(np.float64)
            if (ids.shape != norms.shape or not np.isfinite(norms).all() or
                    (norms < 0).any() or (ids < 0).any() or (ids >= a.experts).any()):
                raise ValueError('invalid output-norm trace: ' + f)
            sums[layer] = np.bincount(ids.ravel(), weights=norms.ravel(), minlength=a.experts)
            samples[layer] = np.bincount(ids.ravel(), minlength=a.experts)
    norms = sums / np.maximum(samples, 1)
    if not samples.any():
        raise ValueError('no calibration observations')
    if Path(a.out).exists():
        raise ValueError('output exists; preserve the previous calibration')
    np.savez_compressed(a.out, norms=norms, samples=samples, profile_version=[1])
    print(json.dumps(dict(layers_observed=int((samples.sum(axis=1) > 0).sum()),
        experts_with_three_observations=int((samples >= 3).sum()),
        routed_observations=int(samples.sum()),
        limitation='Unweighted norm is a residual-contribution proxy, not causal answer importance.')))


if __name__ == '__main__':
    main()
