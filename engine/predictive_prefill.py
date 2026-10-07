"""Bounded prompt-to-routing predictor; no extra forward pass or expert streaming.

Rank 0 owns a bank of hashed text features and measured prefill distributions.
Predictions are provisional ONE-request votes passed to the existing planner.
Shadow evaluation happens before inserting the current sample (no self-training).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np

VERSION = 1
BINS = 1024
FEATURES = 3 * BINS


def normalize(rows):
    rows = np.asarray(rows, dtype=np.float64)
    totals = rows.sum(axis=1, keepdims=True)
    return np.divide(rows, totals, out=np.zeros_like(rows), where=totals > 0)


def prompt_features(tokens, prefix):
    ids = np.asarray(tokens, dtype=np.uint64)
    suffix = ids[prefix:]
    parts = []
    # Context, new-token vocabulary and ordered new-token bigrams. Square-root
    # frequency limits repeated boilerplate; every input token is considered.
    for values, weight in ((ids, .25), (suffix, .5),
                           ((suffix[:-1] * np.uint64(1000003)) ^ suffix[1:], .25)):
        values, frequencies = np.unique(values, return_counts=True)
        # Signed feature hashing avoids an almost-uniform positive vector for
        # long prompts, which otherwise makes unrelated 200k contexts look alike.
        hashed = values ^ (values >> np.uint64(30))
        hashed *= np.uint64(0xbf58476d1ce4e5b9)
        hashed ^= hashed >> np.uint64(27)
        hashed *= np.uint64(0x94d049bb133111eb)
        hashed ^= hashed >> np.uint64(31)
        signs = np.where((hashed >> np.uint64(63)) != 0, 1., -1.)
        bag = np.bincount((hashed % BINS).astype(np.int64),
                          weights=np.sqrt(frequencies)*signs, minlength=BINS).astype(np.float32)
        length = np.linalg.norm(bag)
        parts.append(bag * (np.sqrt(weight) / length) if length else bag)
    feature = np.concatenate(parts).astype(np.float32)
    feature /= max(float(np.linalg.norm(feature)), 1e-20)
    key = hashlib.sha256(ids.tobytes() + str(prefix).encode()).hexdigest()
    return feature, key


class PredictivePrefill:
    def __init__(self, layers, experts, env=None):
        env = os.environ if env is None else env
        self.mode = env.get('DSV41_PREDICTIVE_PREFILL', 'off')
        if self.mode not in ('off', 'shadow', 'apply'):
            raise ValueError('DSV41_PREDICTIVE_PREFILL must be off, shadow or apply')
        self.path = env.get('DSV41_PREDICTIVE_DB', 'results/predictive-prefill.npz')
        self.capacity = int(env.get('DSV41_PREDICTIVE_CAPACITY', '128'))
        self.minimum = int(env.get('DSV41_PREDICTIVE_MIN_SAMPLES', '8'))
        self.similarity = float(env.get('DSV41_PREDICTIVE_MIN_SIMILARITY', '.85'))
        if not 1 <= self.minimum <= self.capacity <= 256 or not 0 < self.similarity <= 1:
            raise ValueError('invalid predictive prefill capacity/minimum/similarity')
        self.shape = (layers, experts)
        self.features = np.empty((0, FEATURES), dtype=np.float32)
        self.counts = np.empty((0, layers, experts), dtype=np.float32)
        self.mass = np.empty_like(self.counts)
        self.keys = []
        self.identity = ''
        self.io_error = None
        self.writable = True
        self.pending = None
        self.last = {'mode': self.mode, 'status': 'not_started'}

    def boot_fields(self):
        return dict(predictive_version=VERSION, predictive_mode=self.mode,
                    predictive_capacity=self.capacity, predictive_minimum=self.minimum,
                    predictive_similarity=self.similarity, predictive_features=FEATURES)

    def configure(self, cfg, model_config, tokenizer):
        # Reuse across ranking/history changes, but never across model, tokenizer,
        # numerical/serving-policy changes. Database is rank-0 authoritative.
        stable = {k:v for k,v in cfg.items() if not k.startswith('predictive_')
                  and k not in ('initial_keep_sha256', 'expert_seed_sha256')}
        h = hashlib.sha256(json.dumps(stable, sort_keys=True).encode())
        for path in (model_config, tokenizer):
            with open(path, 'rb') as f:
                for block in iter(lambda:f.read(1 << 20), b''):
                    h.update(block)
        self.identity = h.hexdigest()
        if not Path(self.path).exists():
            return
        try:
            # Bound uncompressed content before NumPy allocates arrays.
            with zipfile.ZipFile(self.path) as z:
                if sum(i.file_size for i in z.infolist()) > 40 * 1024**2:
                    raise ValueError('predictor database exceeds size bound')
            with np.load(self.path, allow_pickle=False) as d:
                if int(d['version']) != VERSION or str(d['identity']) != self.identity:
                    raise ValueError('predictor database belongs to a different configuration')
                keys = d['keys'].tolist()
                features, counts, mass = d['features'], d['counts'], d['mass']
                n = len(keys)
                if (not 0 <= n <= 256 or features.shape != (n, FEATURES)
                        or counts.shape != (n, *self.shape) or mass.shape != counts.shape
                        or len(set(keys)) != n or any(not isinstance(k,str) or len(k)!=64 for k in keys)):
                    raise ValueError('invalid predictor database shapes/keys')
                for a in (features, counts, mass):
                    if not np.isfinite(a).all() or (a is not features and (a < 0).any()):
                        raise ValueError('invalid predictor database values')
                if n and (not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5)
                          or not np.allclose(counts.sum(axis=2), 1, atol=1e-5)
                          or not np.allclose(mass.sum(axis=2), 1, atol=1e-5)):
                    raise ValueError('unnormalized predictor database')
                self.keys = keys[-self.capacity:]
                self.features = features[-self.capacity:].astype(np.float32)
                self.counts = counts[-self.capacity:].astype(np.float32)
                self.mass = mass[-self.capacity:].astype(np.float32)
        except Exception as exc:
            self.io_error = str(exc)
            self.writable = False  # preserve an incompatible/corrupt file for inspection

    def predict(self, feature):
        if len(self.keys) < self.minimum:
            return None, {'status': 'collecting', 'samples': len(self.keys)}
        sims = self.features @ feature
        nearest = float(sims.max())
        ids = np.flatnonzero(sims >= self.similarity)
        if not len(ids):
            return None, {'status': 'no_similar_prompt', 'similarity': nearest, 'samples': len(self.keys)}
        ids = ids[np.argsort(-sims[ids], kind='stable')[:4]]
        weights = sims[ids].astype(np.float64) ** 8
        weights /= weights.sum()
        demand = tuple(np.tensordot(weights, a[ids], axes=1) for a in (self.counts, self.mass))
        return demand, dict(status='predicted', similarity=nearest, neighbors=len(ids), samples=len(self.keys))

    def observe(self, feature, key, counts, mass):
        arrays = [np.asarray(a, dtype=np.float64) for a in (counts, mass)]
        if any(a.shape != self.shape or not np.isfinite(a).all() or (a < 0).any()
               or (a.sum(axis=1) <= 0).any() for a in arrays):
            raise ValueError('prefill training requires valid observed demand on every layer')
        arrays = [normalize(a).astype(np.float32) for a in arrays]
        # Repeated prompts replace their old entry instead of filling the bank or
        # pretending repeated measurements are distinct training contexts.
        retain = [i for i,k in enumerate(self.keys) if k != key][-(self.capacity-1):] if self.capacity > 1 else []
        self.features = np.concatenate((self.features[retain], feature[None]), axis=0)
        self.counts = np.concatenate((self.counts[retain], arrays[0][None]), axis=0)
        self.mass = np.concatenate((self.mass[retain], arrays[1][None]), axis=0)
        self.keys = [self.keys[i] for i in retain] + [key]

    def save(self):
        if not self.writable:
            return
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + '.tmp')
        try:
            with open(tmp, 'wb') as f:
                np.savez(f, version=VERSION, identity=self.identity, keys=np.asarray(self.keys, dtype='U64'),
                         features=self.features, counts=self.counts, mass=self.mass)
            os.replace(tmp, p)
        except Exception as exc:
            self.io_error = str(exc)
            if tmp.exists():
                tmp.unlink()

    def begin(self, engine, tokens, prefix):
        if self.mode == 'off':
            return
        self.pending = None
        started = time.perf_counter()
        packet = None
        if engine.ep.rank == 0:
            report = dict(mode=self.mode, status='skipped', samples=len(self.keys), swaps=0,
                          new_tokens=len(tokens)-prefix)
            packet = dict(report=report, swaps=[])
            try:
                from engine.v41_engine import ADAPT
                if getattr(engine, '_images', None) is not None:
                    report['status'] = 'skip_vision'
                elif len(tokens)-prefix < ADAPT.swap_prefill_min:
                    report['status'] = 'skip_short_or_cached'
                else:
                    feature, key = prompt_features(tokens, prefix)
                    demand, info = self.predict(feature)
                    report.update(info)
                    planned = []
                    if demand is not None:
                        masks = np.stack([engine.model_prune_mask[L].cpu().numpy() for L in range(self.shape[0])])
                        predicted_miss = float((demand[0] * ~masks).sum()/self.shape[0])
                        report['predicted_miss_rate'] = predicted_miss
                        if predicted_miss >= ADAPT.swap_prefill_min_miss:
                            planned = engine.plan_swaps(max_swaps=ADAPT.swap_max, request_demand=demand)
                        else:
                            report['status'] = 'below_miss_gate'
                    self.pending = dict(feature=feature, key=key, demand=demand, plan=planned)
                    report['planned_swaps'] = len(planned)
                    if self.mode == 'apply':
                        packet['swaps'] = planned
            except Exception as exc:
                self.pending = None
                report.update(status='prediction_error', error=type(exc).__name__)
            report['prediction_ms'] = round(1000*(time.perf_counter()-started), 3)
        # Always reached by both ranks for every request in shadow/apply mode,
        # including empty plans, cache hits, vision and predictor errors.
        packet = engine.ep.broadcast_obj(packet)
        self.last = packet['report']
        load_start = time.perf_counter()
        if packet['swaps']:
            try:
                self.last['swaps'] = engine.apply_swaps(packet['swaps'])
            except Exception:
                # Once a distributed plan starts, partial writes are fatal.
                import traceback
                traceback.print_exc()
                os._exit(1)
        self.last['load_ms'] = round(1000*(time.perf_counter()-load_start), 3)

    def finish(self, engine):
        if self.mode == 'off' or engine.ep.rank != 0 or self.pending is None:
            return
        started = time.perf_counter()
        pending, self.pending = self.pending, None
        try:
            from engine.v41_engine import ADAPT
            counts = engine.model._req_counts.cpu().numpy().copy()
            mass = engine.model._req_mass.cpu().numpy().copy()
            if pending['demand'] is not None:
                actual = normalize(mass if ADAPT.metric == 'score' else counts)
                estimated = normalize(pending['demand'][1 if ADAPT.metric == 'score' else 0])
                self.last['mean_total_variation'] = float(np.abs(actual-estimated).sum(axis=1).mean()/2)
                if self.mode == 'shadow':
                    # Same initial residents/history/cap/gain threshold; only
                    # predicted vs measured prefill demand differs.
                    measured_miss = engine.model.miss_snapshot()
                    before = engine._miss_at_request_start
                    denom = measured_miss[1]-before[1]
                    rate = (measured_miss[0]-before[0])/denom if denom > 0 else 0.
                    target = (engine.plan_swaps(max_swaps=ADAPT.swap_max, request_demand=(counts,mass))
                              if rate >= ADAPT.swap_prefill_min_miss else [])
                    expected = {tuple(s[:4]) for s in target}
                    predicted = {tuple(s[:4]) for s in pending['plan']}
                    # Promotion overlap is more meaningful than exact donor pairing.
                    expected_in = {s[2:4] for s in expected}
                    predicted_in = {s[2:4] for s in predicted}
                    self.last.update(actual_planned_swaps=len(target),
                        promotion_precision=(len(expected_in & predicted_in)/len(predicted_in) if predicted_in else None),
                        promotion_recall=(len(expected_in & predicted_in)/len(expected_in) if expected_in else None),
                        exact_swap_overlap=len(expected & predicted))
            self.observe(pending['feature'], pending['key'], counts, mass)
            self.save()
            self.last['samples_after'] = len(self.keys)
        except Exception as exc:
            self.last.update(observation_error=type(exc).__name__)
        self.last['observation_ms'] = round(1000*(time.perf_counter()-started), 3)
        if self.io_error:
            self.last['database_error'] = self.io_error
        from engine.v41_engine import log
        log('predictive prefill: ' + json.dumps(self.last))
