"""Image identity for prefix reuse: hash model inputs, not URLs or placeholders."""
import hashlib
import json

import torch


def image_fingerprints(images, token_types=None):
    records = []
    expected = None if token_types is None else torch.full_like(token_types, -1, device='cpu')
    previous_end = 0
    for im in sorted(images or (), key=lambda im: im.start):
        start, end = int(im.start), int(im.start + im.types.numel())
        if start < previous_end or end <= start:
            raise ValueError('overlapping or invalid image spans')
        previous_end = end
        h = hashlib.sha256(b'prefix-image-v1')
        h.update(json.dumps([int(im.n_vit_h), int(im.n_vit_w)]).encode())
        for tensor in (im.types, im.patches):
            value = tensor.detach().cpu().contiguous()
            h.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
            h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        records.append((start, end, h.hexdigest()))
        if expected is not None:
            if end > expected.numel():
                raise ValueError('image extends beyond token types')
            expected[start:end] = im.types.cpu()
    if expected is not None and not torch.equal(expected, token_types.cpu()):
        raise ValueError('image spans do not match token types')
    return tuple(records)


def media_prefix(records, n):
    """Identity up to a boundary; None forbids resuming inside an image."""
    if any(start < n < end for start, end, _ in records):
        return None
    return tuple(tuple(record) for record in records if record[0] < n)
