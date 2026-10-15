"""make_fixtures.py — the test fixtures and their references from anime4k.py (the independent PyTorch reading).

  python make_fixtures.py <drawn-sr root> <glsl dir> <fixtures dir>

Inputs: 128×96 crops of two licence-clean images from the G-4 drawn set (mlxengine-image/corpus/drawn-sr), taken
from its ×2 bicubic rung: `c_Angry_anime_girl` (Wikimedia Commons, CC0, by snoopingasusual) and `own_poster` (our
own render). References: every mode through anime4k.py at fp32 on the CPU, clamped to [0, 1] and written as 16-bit
RGB PNGs (round(x·65535)), so the tests can compare at 8 or 16 bits without the reference's rounding mattering.
"""
import os, struct, sys, zlib
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anime4k import Anime4K

MODES = {
    "mode_b": ["Anime4K_Restore_CNN_Soft_VL", "Anime4K_Upscale_CNN_x2_VL"],
    "mode_a": ["Anime4K_Restore_CNN_VL", "Anime4K_Upscale_CNN_x2_VL"],
    "upscale_vl": ["Anime4K_Upscale_CNN_x2_VL"],
    "upscale_m": ["Anime4K_Upscale_CNN_x2_M"],
}
CROPS = {"anime": ("c_Angry_anime_girl", (120, 140)), "poster": ("own_poster", (60, 40))}
SIDE = (128, 96)


def png16(path, rgb16):
    """A 16-bit RGB PNG (Pillow writes 16-bit only for greyscale)."""
    h, w, _ = rgb16.shape
    raw = b"".join(b"\x00" + rgb16[y].astype(">u2").tobytes() for y in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main():
    root, glsl, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    for name, (iid, (x, y)) in CROPS.items():
        im = Image.open(f"{root}/lr/x2_bicubic/{iid}.png").convert("RGB").crop((x, y, x + SIDE[0], y + SIDE[1]))
        im.save(f"{out}/{name}.png")
        t = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None]
        for mode, files in MODES.items():
            with torch.inference_mode():
                r = Anime4K([f"{glsl}/{f}.glsl" for f in files], device="cpu")(t)[0].permute(1, 2, 0).numpy()
            png16(f"{out}/{name}_{mode}.png", np.round(np.clip(r, 0, 1) * 65535).astype(np.uint16))
            print(name, mode, r.shape)


if __name__ == "__main__":
    main()
