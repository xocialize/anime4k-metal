"""anime4k_msl.py — transpile Anime4K's mpv GLSL hooks into a Metal compute library, one kernel per pass.

A near-literal port of each `hook()`, with the shader's own sampling semantics: every intermediate is a texture,
`TEX_texOff(vec2(x, y))` samples at the texel centre + an integer texel offset through a clamp-to-edge sampler (so
the border behaviour is libplacebo's, not a re-implementation of it), `mat4(16 scalars)` becomes
`float4x4(float4 ×4)` with the same COLUMN-major grouping (GLSL and MSL both fill columns first), and the
Depth-to-Space pass adds `MAIN` sampled BILINEARLY at the output position — libplacebo's default hook sampler.
This is a second, independent implementation of the same shaders: `anime4k.py` reads them as convolutions, this
runs them as shaders. Their agreement is the validation `anime4k.py` lacked.

`emit(files) -> (msl_source, plan)`; `plan` lists each pass's kernel name, input texture names (in binding order),
output texture name and output scale (1 or 2) for the Swift driver (`Anime4KMetal/`).
"""
import re
import sys

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

HEADER = """#include <metal_stdlib>
using namespace metal;
constexpr sampler S(coord::normalized, address::clamp_to_edge, filter::linear);
static inline float4 fetch(texture2d<float, access::sample> t, float2 pos, float2 pt, float2 off) {
    return t.sample(S, pos + off * pt);
}
"""


def _f(s):
    v = float(s)
    return repr(v) if "e" not in repr(v) else f"{v:.9e}"


def _mat(nums):
    n = [_f(x) for x in nums]
    cols = [f"float4({n[4 * c]}, {n[4 * c + 1]}, {n[4 * c + 2]}, {n[4 * c + 3]})" for c in range(4)]
    return f"float4x4({', '.join(cols)})"


def emit(files):
    out = [HEADER]
    plan = []
    kid = 0
    for f in files:
        text = open(f).read()
        for block in text.split("//!DESC")[1:]:
            desc = block.split("\n", 1)[0].strip()
            binds = re.findall(r"//!BIND (\S+)", block)
            m = re.search(r"//!SAVE (\S+)", block)
            save = m.group(1) if m else "MAIN"
            name = f"k{kid:02d}"
            kid += 1
            args = [f"texture2d<float, access::sample> {b} [[texture({i})]]" for i, b in enumerate(binds)]
            args.append(f"texture2d<float, access::write> OUT [[texture({len(binds)})]]")
            args.append("uint2 gid [[thread_position_in_grid]]")
            body = []
            body.append("    if (gid.x >= OUT.get_width() || gid.y >= OUT.get_height()) return;")
            body.append("    float2 opos = (float2(gid) + 0.5) / float2(OUT.get_width(), OUT.get_height());")
            for b in binds:
                body.append(f"    float2 {b}_size = float2({b}.get_width(), {b}.get_height());")
                body.append(f"    float2 {b}_pt = 1.0 / {b}_size;")
                body.append(f"    float2 {b}_pos = opos;")
            if "Depth-to-Space" in desc:
                src = {int(k): t for k, t in re.findall(r"float c(\d) = (\w+?)_tex\(", block)}
                alias = {int(k): int(v) for k, v in re.findall(r"float c(\d) = c(\d);", block)}
                for c in range(4):
                    if c in src:
                        t = src[c]
                        body.append(f"    float2 f{c} = fract({t}_pos * {t}_size);")
                        body.append(f"    int2 i{c} = int2(f{c} * 2.0);")
                        body.append(f"    float c{c} = {t}.sample(S, (float2(0.5) - f{c}) * {t}_pt + {t}_pos)[i{c}.y * 2 + i{c}.x];")
                    else:
                        body.append(f"    float c{c} = c{alias[c]};")
                body.append("    OUT.write(float4(c0, c1, c2, c3) + MAIN.sample(S, MAIN_pos), gid);")
                scale = 2
            else:
                defs = re.findall(r"#define (go?_\d+)(?:\(x_off, y_off\))? \((.*)\)\s*$", block, re.M)
                groups = {}
                for dname, dbody in defs:
                    tex = re.search(r"(\w+?)_tex(?:Off)?\(", dbody).group(1)
                    act = "neg" if "max(-(" in dbody else ("pos" if dbody.startswith("max(") else "id")
                    groups[dname] = (tex, act)

                def tap(g, dx, dy):   # MSL has no lambdas: each tap is an inline expression
                    tex, act = groups[g]
                    off = f"float2({_f(dx)}, {_f(dy)})" if g.startswith("go") else "float2(0.0)"
                    v = f"fetch({tex}, {tex}_pos, {tex}_pt, {off})"
                    return {"id": v, "pos": f"max({v}, 0.0)", "neg": f"max(-{v}, 0.0)"}[act]

                terms = re.findall(r"(vec4 result =|result \+=) mat4\(([^)]*)\) \* (go?_\d+)(?:\(([^,]+), ([^)]+)\))?", block)
                for lead, mat, g, dx, dy in terms:
                    op = "float4 result =" if lead.startswith("vec4") else "result +="
                    body.append(f"    {op} {_mat(re.findall(_NUM, mat))} * {tap(g, dx or 0, dy or 0)};")
                b = re.search(r"result \+= vec4\(([^)]*)\);", block)
                if b:
                    body.append(f"    result += float4({', '.join(_f(x) for x in re.findall(_NUM, b.group(1)))});")
                if "return result + MAIN_tex(MAIN_pos)" in block:
                    body.append("    result += MAIN.sample(S, MAIN_pos);")
                body.append("    OUT.write(result, gid);")
                scale = 1
            out.append(f"// {desc}\nkernel void {name}({', '.join(args)}) {{\n" + "\n".join(body) + "\n}\n")
            plan.append(dict(kernel=name, binds=binds, save=save, scale=scale, desc=desc))
    return "".join(out), plan


if __name__ == "__main__":
    import json
    src, plan = emit(sys.argv[2:])
    open(sys.argv[1], "w").write(src)
    json.dump(plan, open(sys.argv[1] + ".plan.json", "w"), indent=1)
    print(f"{len(plan)} kernels, {len(src) / 1e3:.0f} KB of MSL -> {sys.argv[1]}")
