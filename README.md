# anime4k-metal

[Anime4K](https://github.com/bloc97/Anime4K) (bloc97, MIT) — a set of tiny real-time CNN upscalers distributed as
mpv GLSL shaders — **running as Metal compute shaders**, for testing **mode B** as a playback-time upscaler (the
media-player role) and a candidate for the web build (WGSL). Status: **testing**, not a product package. Background
and the measurements that led here: `mlxengine-forge/Docs/VIDEO2X-EVAL.md` §3.2, §5.

## What's here

| path | what |
|---|---|
| `Generator/anime4k_msl.py` | GLSL → MSL transpiler: one compute kernel per mpv hook pass, with the hook's own sampling (texel-centre offsets through a clamp-to-edge linear sampler; MAIN sampled bilinearly in depth-to-space — libplacebo's defaults). Writes `<mode>.msl` + `<mode>.plan.json` (pass order, bound textures, SAVE target, scale). |
| `Generator/fetch_and_generate.sh` | fetches the GLSL at the pinned revision and regenerates every mode, compile-checking each with `xcrun metal` |
| `Sources/Anime4KMetal/` | runtime: compiles a mode's MSL once, replays the plan (MAIN is replaced when a pass saves it, as in mpv), pools textures, zero-copy `CVPixelBuffer` ↔ `MTLTexture`, and a split-screen compose (MPS Lanczos ×2 vs Anime4K) |
| `Sources/anime4k/` | CLI: `image`, `video [--compare]`, `bench` |
| `Validation/` | `anime4k.py` (an independent PyTorch reading of the same shaders — passes as convolutions) and `anime4k_compare.py` |

Modes (each ×2): **`b`** Restore_CNN_Soft_VL → Upscale_CNN_x2_VL · `a` Restore_CNN_VL → Upscale_CNN_x2_VL ·
`upscale_vl` · `upscale_m`. Modes A/B are Anime4K's own modes minus Clamp_Highlights and the second upscale (this
package does exactly ×2).

## Test it

```bash
swift build -c release
.build/release/anime4k video in.mp4 compare.mp4 --mode b --compare     # left: Lanczos ×2 | right: Anime4K
.build/release/anime4k image in.png out.png --mode b
.build/release/anime4k bench in.png --mode b                           # GPU ms/frame
```

## What has been measured (2026-10-04, M5 Max, macOS 27.2)

- **Correctness:** the Metal shaders agree with the PyTorch reading to max |Δ| ≤ 9.5e-7 (144–151 dB, never more than
  one 8-bit level apart) on Trigun 640×480 and Wistoria 960×540 frames, for mode B and Upscale_VL. Two independent
  implementations agree. Their shared assumption is that the hook sampler is linear + clamp, as in libplacebo. This
  package's fp32 output matches the prototype's to ≤ 1 level on 0.0009 % of samples.
- **Speed** (GPU time per frame, fp16 intermediates, median):

  | mode | 960×540 → 1920×1080 | 1920×1080 → 3840×2160 |
  |---|---:|---:|
  | B | 2.7 ms | 9.0 ms |
  | upscale_vl | 1.4 ms | 5.2 ms |
  | upscale_m | 0.4 ms | 1.2 ms |

  fp16 intermediates are 66.6 dB against fp32 (≤ 1 level). With `--compare` and HEVC encode, a 49-frame
  1080p → 4K clip renders in 0.6 s of wall time.
- **Quality** (from VIDEO2X-EVAL §5, PyTorch reading, 960×540 → 1080p against 1080p Wistoria frames): mode B beats
  the shipped Real-ESRGAN `.anime` on SSIMULACRA2 in all four full-reference cells (67.8 / 35.2 / 80.9 / 37.9 vs
  58.4 / 32.8 / 63.9 / 34.9), deblocking 3–14×. Those cells reward faithfulness; the playback-scale eyeball is the
  open test.

## Licence

MIT (`LICENSE`) for the generator, runtime, CLI and validation code. The generated shaders in
`Sources/Anime4KMetal/Shaders/` derive from bloc97/Anime4K's GLSL, MIT, reproduced in `NOTICE`.

## Open

1. The operator's playback-scale eyeball (`video --compare`) on real anime, especially low-bitrate SD and 720p
   sources.
2. If it holds: a real-time path in the player (an `AVPlayerItemVideoOutput` → this pipeline → `CAMetalLayer`),
   and a WGSL port for the web build (the generator's structure carries over pass for pass).
3. Anime4K's GAN shaders (v4.1) are not generated yet — they use a different graph shape.
