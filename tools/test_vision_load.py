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

class A: pass
for k, v in cfg.items():
    setattr(A, k, v)
A.vision_enabled = A.vision_n_layers > 0

img = os.path.join(MD, "assets", "dsv41_kv_cache.png")
patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = IP.load_image({"url": img}, A)
types = IP.image_token_types(n_llm_h, n_llm_w)
print(f"2. image patches {tuple(patches.shape)}, ViT grid {n_vit_h}x{n_vit_w}, "
      f"LLM grid {n_llm_h}x{n_llm_w}, token types {tuple(types.shape)}")
emb = embed_image(vit, aligner, patches.to(DEV).to(torch.bfloat16), n_vit_h, n_vit_w)
print(f"3. aligner output {tuple(emb.shape)} dtype {emb.dtype} finite={bool(torch.isfinite(emb.float()).all())}")
print(f"   expected dim {cfg['dim']}: {'ok' if emb.shape[-1] == cfg['dim'] else 'WRONG'}")
n_img = int((types == IP.IMAGE).sum())
print(f"   IMAGE slots in the token stream: {n_img}, aligner rows: {emb.shape[0]}"
      f"  {'match' if n_img == emb.shape[0] else '<-- MISMATCH: rows must fill IMAGE slots exactly'}")
assert emb.shape[-1] == cfg["dim"] and n_img == emb.shape[0]
assert bool(torch.isfinite(emb.float()).all())
