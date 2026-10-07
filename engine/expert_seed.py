"""Packaged cold-start expert history; never replaces a local demand database."""
from pathlib import Path
import hashlib
import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parents[1] / 'profiles' / 'learned-experts-v1.npz'
VERSION = 1


def load_seed(db_path, *, enabled, request_unit, metric, layers, experts, path=None):
    # Even an unreadable/incompatible local DB belongs to the user. Its normal
    # recovery path must not silently replace it with someone else's history.
    if not enabled or not request_unit or metric != 'score' or Path(db_path).exists():
        return None
    path = DEFAULT_PATH if path is None else Path(path)
    raw = path.read_bytes()
    import io
    with np.load(io.BytesIO(raw), allow_pickle=False) as d:
        if int(d['version'].item()) != VERSION:
            raise ValueError('unsupported expert seed version')
        counts, mass, keep = (d[k].copy() for k in ('counts', 'mass', 'keep'))
    shape = (layers, experts)
    if any(a.shape != shape for a in (counts, mass, keep)):
        raise ValueError('expert seed dimensions do not match model')
    if any(not np.isfinite(a).all() or (a < 0).any() for a in (counts, mass)):
        raise ValueError('invalid expert seed demand')
    if not ((keep == 0) | (keep == 1)).all() or not keep.any(axis=1).all():
        raise ValueError('invalid expert seed resident map')
    if not (counts.sum(axis=1) > 0).all() or not (mass.sum(axis=1) > 0).all():
        raise ValueError('empty expert seed demand')
    return dict(counts=counts, mass=mass, keep=keep.astype(bool),
                sha256=hashlib.sha256(raw).hexdigest())


def seed_selection(seed, budget, floor):
    """Restore the exact captured map only at its original budget and valid floor."""
    if seed is None:
        return None
    keep = seed['keep']
    if int(keep.sum()) != budget or int(keep.sum(axis=1).min()) < floor:
        return None
    return {L: np.flatnonzero(row).tolist() for L, row in enumerate(keep)}
