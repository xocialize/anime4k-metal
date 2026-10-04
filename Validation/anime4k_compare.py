"""anime4k_compare.py — Metal shaders (anime4k-metal --raw) vs the PyTorch interpreter (anime4k.py), same frame.

usage: python anime4k_compare.py <in.png> <metal.f32> <glsl files…>
Prints max |Δ|, PSNR over [0, 1], and the 8-bit picture (share of samples that round differently, max levels).
"""
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from anime4k import Anime4K  # noqa: E402

inp, raw, files = sys.argv[1], sys.argv[2], sys.argv[3:]
x = np.asarray(Image.open(inp).convert("RGB"), np.float32) / 255
h, w = x.shape[:2]
with torch.inference_mode():
    t = Anime4K(files, device="cpu")(torch.from_numpy(x).permute(2, 0, 1)[None])   # fp32, CPU, unclamped? see below
torch_out = t[0].permute(1, 2, 0).numpy()               # Anime4K.__call__ clamps to [0, 1]
metal = np.fromfile(raw, np.float32).reshape(2 * h, 2 * w, 4)[..., :3]
metal = np.clip(metal, 0, 1)
d = np.abs(metal - torch_out)
psnr = 10 * np.log10(1 / max(float((d ** 2).mean()), 1e-20))
q = np.abs(np.round(metal * 255) - np.round(torch_out * 255))
print(f"metal vs anime4k.py: max|Δ| {d.max():.2e}  PSNR {psnr:.1f} dB  8-bit: {(q > 0).mean() * 100:.4f}% samples differ, "
      f"max {int(q.max())} level(s)  ({w}x{h} → {2 * w}x{2 * h})")
