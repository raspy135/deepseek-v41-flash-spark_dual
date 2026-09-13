"""The vision tower loads correctly and runs on a real image.

The forward math is the checkpoint's own module, so what needs proving is the LOADING -- that
every parameter got the checkpoint tensor it names, not a transposed or same-shaped neighbour --
and that a real image goes through end to end with a sane token grid.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from engine.vision import load_vision, embed_image

MD = os.environ.get("MODEL_DIR", "/models/DeepSeek-V4.1-Flash")
DEV = os.environ.get("DEV", "cpu")

vit, aligner, cfg = load_vision(MD, device=DEV, dtype=torch.bfloat16)
n_p = sum(p.numel() for p in vit.parameters()) + sum(p.numel() for p in aligner.parameters())
print(f"loaded: ViT {cfg['vision_n_layers']} layers dim {cfg['vision_dim']}, "
      f"{n_p/1e6:.1f}M params on {DEV}")

# 1. every parameter must equal its checkpoint tensor, bit for bit
from safetensors import safe_open
index = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
bad, checked = [], 0
for prefix, mod in (("vision.", vit), ("aligner.", aligner)):
    for name, p in mod.state_dict().items():
        key = prefix + name
        with safe_open(os.path.join(MD, index[key]), framework="pt") as f:
            ref = f.get_tensor(key)
        checked += 1
        if not torch.equal(p.cpu().to(ref.dtype), ref):
            bad.append(key)
print(f"1. parameters matching the checkpoint bitwise: {checked - len(bad)}/{checked}"
      + (f"   MISMATCHED: {bad[:5]}" if bad else ""))
assert not bad

# 2. a real image, through the real preprocessor
sys.path.insert(0, os.path.join(MD, "inference"))
import image_processor as IP
img = os.path.join(MD, "assets", "dsv41_kv_cache.png")
fn = next((getattr(IP, n) for n in ("process_image", "load_image", "preprocess") if hasattr(IP, n)), None)
print(f"2. image_processor entry points: {[n for n in dir(IP) if not n.startswith('_') and callable(getattr(IP,n))][:8]}")
print(f"   using: {fn.__name__ if fn else 'NONE FOUND'}")
if fn is None:
    raise SystemExit("no obvious entry point; inspect image_processor.py")
out = fn(img) if "path" in fn.__code__.co_varnames[:2] or fn.__code__.co_argcount == 1 else None
print(f"   -> {type(out).__name__}: {out}" if not hasattr(out, 'patches') else
      f"   patches {tuple(out.patches.shape)} grid {out.n_vit_h}x{out.n_vit_w} types {tuple(out.types.shape)}")
if hasattr(out, "patches"):
    emb = embed_image(vit, aligner, out.patches.to(DEV).to(torch.bfloat16), out.n_vit_h, out.n_vit_w)
    print(f"3. aligner output {tuple(emb.shape)} dtype {emb.dtype} finite={bool(torch.isfinite(emb.float()).all())}")
    print(f"   expected dim {cfg['dim']}: {'ok' if emb.shape[-1] == cfg['dim'] else 'WRONG'}")
    n_img = int((out.types == IP.IMAGE).sum())
    print(f"   IMAGE slots in the token stream: {n_img}, aligner rows: {emb.shape[0]}"
          f"  {'match' if n_img == emb.shape[0] else '<-- MISMATCH: rows must fill IMAGE slots exactly'}")
