"""Opt-in immediate full routing in selected layers, using only transient slots."""
import json
from pathlib import Path


def parse_layers(raw, n_layers):
    if not raw.strip():
        return frozenset()
    try:
        layers = frozenset(int(v.strip()) for v in raw.split(','))
    except ValueError as exc:
        raise ValueError('DSV41_STREAM_LAYERS must contain integer layer IDs') from exc
    if any(v < 0 or v >= n_layers for v in layers):
        raise ValueError(f'stream layer IDs must be in [0, {n_layers})')
    return layers


def validate_capacity(layers, keep_masks, transient_slots, experts):
    # Every possible nonresident expert of a selected layer must fit at once.
    # Prefill can touch all of them; using decode's k*T bound would fail later.
    for layer in layers:
        cold = experts - int(keep_masks[layer].sum())
        if cold > transient_slots:
            raise ValueError(f'L{layer} needs {cold} transient slots, got {transient_slots}')


def graph_bounds(n_layers, engram_layers, stream_layers):
    return sorted({0, n_layers} | {l for l in engram_layers if 0 < l < n_layers}
                  | set(stream_layers) | {l + 1 for l in stream_layers})


class LayerAblation:
    """Diagnostic-only mask changes; fixed graph topology and persistent addresses.

    Rank 0 reads the control file, but both ranks always reach the same broadcast.
    Only mask data change; every layer still resolves through the transient ring.
    """
    def __init__(self, path, n_layers):
        self.path, self.n_layers = path.strip(), n_layers
        self.pruned = ()

    def sync(self, model, ep):
        packet = None
        if ep.rank == 0:
            try:
                data = json.loads(Path(self.path).read_text())
                ids = data['pruned_layers']
                if (not isinstance(ids, list) or any(type(l) is not int or
                        not 0 <= l < self.n_layers for l in ids)):
                    raise ValueError('pruned_layers must be a list of valid integer layer IDs')
                packet = {'layers': sorted(set(ids)), 'error': None}
            except Exception as exc:
                packet = {'layers': [], 'error': type(exc).__name__ + ': ' + str(exc)}
        packet = ep.broadcast_obj(packet)  # unconditional, including an invalid file
        if packet['error']:
            raise RuntimeError('layer ablation control rejected: ' + packet['error'])
        self.pruned = tuple(packet['layers'])
        for layer, keep in model.stream_keep.items():
            if layer in self.pruned:
                keep.copy_(model.prune_mask[layer])
            else:
                keep.fill_(True)

    def report(self):
        return {'enabled': bool(self.path), 'pruned_layers': list(self.pruned)}
