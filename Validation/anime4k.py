"""anime4k.py — runs Anime4K's CNN mpv shaders (bloc97/Anime4K, MIT) in PyTorch, for measurement only.

There is no GLSL runtime in this harness (mpv/libplacebo would bring FFmpeg), so each shader pass is
translated into the convolution it computes:

  * `#define go_k(x_off, y_off) (max(±(TEX_texOff(vec2(x_off, y_off))), 0.0))` is input group k: texture
    TEX through ReLU (or ReLU of its negation — Anime4K's CReLU), sampled at a pixel offset. The first pass
    reads MAIN with no activation. `#define g_k (...TEX_tex(TEX_pos)...)` is the same at offset 0 (1x1 passes).
  * `result += mat4(m0..m15) * go_k(dx, dy)` contributes out[i] += m[4*j + i] * in[j] (GLSL mat4 is
    column-major), at kernel tap (row dy+1, col dx+1). texOff clamps at the border -> replicate padding.
  * `Depth-to-Space` picks channel [i.y*2 + i.x] of a 4-channel texture at half resolution = pixel_shuffle(2).
    The M shader writes one residual and copies it to all three channels (c1 = c0); VL writes one per channel.
    The pass adds MAIN sampled at the doubled size — libplacebo's default linear sampler = bilinear,
    align_corners=False.
  * A final pass with `return result + MAIN_tex(MAIN_pos)` (Restore) is a same-size residual.

`flip_y` exists because the texture y direction is the one assumption here not fixed by the shader text;
run.py's `a4kcheck` phase measures both orientations against a full-reference cell and keeps the right one.
Unvalidated against libplacebo pixels — the receipt says so.
"""
import re

import torch
import torch.nn.functional as F

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _floats(s):
    return [float(v) for v in re.findall(_NUM, s)]


class Pass:
    pass


def parse(path):
    text = open(path).read()
    passes = []
    for block in text.split("//!DESC")[1:]:
        p = Pass()
        p.desc = block.split("\n", 1)[0].strip()
        p.binds = re.findall(r"//!BIND (\S+)", block)
        m = re.search(r"//!SAVE (\S+)", block)
        p.save = m.group(1) if m else "MAIN"
        if "Depth-to-Space" in p.desc:
            p.kind = "d2s"
            src = {int(k): t for k, t in re.findall(r"float c(\d) = (\w+?)_tex\(", block)}
            alias = {int(k): int(v) for k, v in re.findall(r"float c(\d) = c(\d);", block)}
            p.channels = []
            for c in range(3):
                k = c
                while k in alias:
                    k = alias[k]
                p.channels.append(src[k])
            passes.append(p)
            continue
        p.kind = "conv"
        groups = {}
        for name, body in re.findall(r"#define (go?_\d+)(?:\(x_off, y_off\))? \((.*)\)\s*$", block, re.M):
            tex = re.search(r"(\w+?)_tex(?:Off)?\(", body).group(1)
            act = "neg" if "max(-(" in body else ("pos" if body.startswith("max(") else "id")
            groups[name] = (tex, act)
        order = sorted(groups, key=lambda n: int(n.split("_")[1]))
        p.groups = [groups[n] for n in order]
        index = {n: i for i, n in enumerate(order)}
        ksize = 3 if any(n.startswith("go") for n in order) else 1
        W = torch.zeros(4, 4 * len(order), ksize, ksize, dtype=torch.float64)
        terms = re.findall(r"mat4\(([^)]*)\) \* (go?_\d+)(?:\(([^,]+), ([^)]+)\))?", block)
        for mat, gname, dx, dy in terms:
            m = _floats(mat)
            assert len(m) == 16, p.desc
            gi = index[gname]
            r = c = 0
            if ksize == 3:
                c, r = int(float(dx)) + 1, int(float(dy)) + 1
            for j in range(4):
                for i in range(4):
                    W[i, 4 * gi + j, r, c] += m[4 * j + i]
        b = re.search(r"result \+= vec4\(([^)]*)\);", block)
        p.bias = torch.tensor(_floats(b.group(1)) if b else [0.0] * 4, dtype=torch.float64)
        p.weight = W
        p.ksize = ksize
        p.residual = "return result + MAIN_tex(MAIN_pos)" in block
        p.nterms = len(terms)
        passes.append(p)
    return passes


class Anime4K:
    """A chain of shader files, applied in order (e.g. Restore_CNN_VL then Upscale_CNN_x2_VL)."""

    def __init__(self, files, device="cpu", flip_y=False):
        self.chain = [parse(f) for f in files]
        self.device = device
        self.flip_y = flip_y
        for shader in self.chain:
            for p in shader:
                if p.kind == "conv":
                    w = p.weight.flip(2) if (flip_y and p.ksize == 3) else p.weight
                    p.w = w.to(torch.float32).to(device)
                    p.b = p.bias.to(torch.float32).to(device)

    @torch.inference_mode()
    def __call__(self, rgb):  # rgb: (1,3,H,W) float in [0,1]
        main = torch.cat([rgb, torch.ones_like(rgb[:, :1])], 1)
        for shader in self.chain:
            tex = {"MAIN": main}
            for p in shader:
                if p.kind == "conv":
                    xs = []
                    for t, act in p.groups:
                        v = tex[t]
                        xs.append(v if act == "id" else F.relu(v) if act == "pos" else F.relu(-v))
                    x = torch.cat(xs, 1)
                    if p.ksize == 3:
                        x = F.pad(x, (1, 1, 1, 1), mode="replicate")
                    out = F.conv2d(x, p.w, p.b)
                    if p.residual:
                        out = out + tex["MAIN"]
                    tex[p.save] = out
                else:  # depth-to-space + bilinear MAIN
                    res = torch.cat([F.pixel_shuffle(tex[t], 2) for t in p.channels], 1)
                    base = F.interpolate(tex["MAIN"][:, :3], scale_factor=2, mode="bilinear", align_corners=False)
                    out = torch.cat([res + base, torch.ones_like(res[:, :1])], 1)
                    tex[p.save] = out
            main = tex["MAIN"]
        return main[:, :3].clamp(0, 1)

    def describe(self):
        lines = []
        for shader in self.chain:
            for p in shader:
                if p.kind == "conv":
                    lines.append(f"{p.desc}: {len(p.groups)} groups -> {p.save} ({p.ksize}x{p.ksize}, {p.nterms} terms"
                                 f"{', +MAIN' if p.residual else ''})")
                else:
                    lines.append(f"{p.desc}: d2s {p.channels} + bilinear MAIN")
        return "\n".join(lines)
