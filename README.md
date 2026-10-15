# anime4k-metal

[Anime4K](https://github.com/bloc97/Anime4K) (bloc97, MIT), a set of tiny real-time CNN upscalers distributed as mpv
GLSL shaders, **running as Metal compute shaders on Apple silicon**. Two ways in:

- **`Anime4KUpscaler`**: a `CGImage` in, a `CGImage` at exactly twice the size out. It keeps 8- or 16-bit depth,
  alpha and the image's RGB colour space, and tiles large images so the GPU working set stays bounded.
- **`Anime4K`**: the shader chain itself, on an `MTLTexture` or a `CVPixelBuffer`, for real-time video and live
  previews.

No MLX, no model download, no vendored binaries: the network weights are constants in the shader source. Metal only,
with no CPU fallback; `Anime4KUpscaler.isAvailable` says whether this Mac can run it. macOS 14 and later.

## Use

```swift
import Anime4KMetal

guard Anime4KUpscaler.isAvailable else { return }          // unavailableReason says why not
let upscaler = try Anime4KUpscaler()                        // mode .upscaleVL, the measured default for still images
let doubled = try upscaler.upscale(image)                   // ×2; upscale(image, doublings: 2) for ×4
```

`Configuration` sets the mode, the intermediate precision (fp16 for 8-bit images and fp32 for deeper ones by
default), the tile size and margin, and whether GPU textures are kept between calls. `memoryEstimate(width:height:)`
gives the GPU memory a call will allocate. The upscaler is thread-safe; calls are serialised. Cancelling the calling
task stops the run between tiles.

### Which mode

| mode | chain | use it for |
|---|---|---|
| `.upscaleVL` | Upscale_CNN_x2_VL | **still images, the image API's default.** Beats Lanczos on every drawn image measured, keeps text as legible as Lanczos, and is no worse than Lanczos on photos. |
| `.b` | Restore_CNN_Soft_VL → Upscale_CNN_x2_VL | heavily compressed sources such as low-bitrate anime video. Its restore pass costs ~12 SSIMULACRA2 on clean images and corrupts 3.6× the words, but wins at JPEG q30. |
| `.a` | Restore_CNN_VL → Upscale_CNN_x2_VL | as `.b`, with the stronger restore. |
| `.upscaleM` | Upscale_CNN_x2_M | the cheapest; close to `.upscaleVL` on quality. |

Anime4K is the instant, model-free tier. On drawn content the learned upscalers (HEART, NERVE, RealPLKSR) score higher
on every class; see the measurements below.

## What's here

| path | what |
|---|---|
| `Sources/Anime4KMetal/Anime4KUpscaler.swift` | the image API: CGImage ↔ GPU, alpha (un-premultiply, colour bleed, Lanczos-3 alpha ×2), exact tiling |
| `Sources/Anime4KMetal/Anime4K.swift` | the runtime: compiles a mode's MSL once, replays its plan (MAIN is replaced when a pass saves it, as in mpv), pools textures, zero-copy `CVPixelBuffer` ↔ `MTLTexture`, a split-screen compare (MPS Lanczos ×2 vs Anime4K) |
| `Sources/Anime4KMetal/Shaders/` | the generated MSL + plan per mode |
| `Sources/anime4k/` | CLI: `image`, `batch`, `benchimage`, `video [--compare]`, `bench` |
| `Generator/anime4k_msl.py`, `fetch_and_generate.sh` | GLSL → MSL transpiler (one kernel per mpv hook pass, the hook's own sampling: texel-centre offsets through a clamp-to-edge linear sampler, MAIN bilinear in depth-to-space, libplacebo's defaults); fetches the GLSL at the pinned revision and regenerates every mode |
| `Validation/` | `anime4k.py` (an independent PyTorch reading of the same shaders, as convolutions), `anime4k_compare.py`, `make_fixtures.py` (the test references) |
| `Tests/` | 11 tests: parity, tiling, alpha, depth, colour space (`swift test`) |

```bash
swift build -c release
.build/release/anime4k image in.png out.png                       # ×2, mode upscale_vl; --mode b, --doublings 2
.build/release/anime4k benchimage in.png --sizes 1024,2048         # end-to-end wall time per ×2
.build/release/anime4k video in.mp4 compare.mp4 --mode b --compare  # left: Lanczos ×2 | right: Anime4K
```

## Measured (v0.1.0, 2026-10-15, M5 Max, macOS 27.2)

**Correctness.**
- Every mode agrees with the PyTorch reading within one 8-bit level on both test fixtures. At fp32 intermediates
  most modes agree exactly; at fp16 1–6 % of samples differ by one level.
- A 16-bit image comes back within one 16-bit level of the reference.
- Tiled output equals untiled output byte for byte, with and without alpha. Tiles stay exact down to a 16-pixel
  margin; the default is 48.
- The image path's Lanczos-3 alpha is our own kernel because MPS's Lanczos is not translation-invariant: it broke
  exact tiling by one level.

**Speed** (end to end, CGImage to CGImage, median of 10, fp16 intermediates):

| mode | 512² → 1024² | 1024² → 2048² | 2048² → 4096² | 4096² → 8192² | GPU working set |
|---|---:|---:|---:|---:|---:|
| `.upscaleVL` | 6.0 ms | 12.4 ms | 41.0 ms | 125 ms | 197 MB |
| `.b` | 6.3 ms | 17.7 ms | 61.7 ms | 178 ms | 340 MB |
| `.upscaleM` | 2.6 ms | 6.9 ms | 21.6 ms | 69 ms | 122 MB |

The working set is per tile and stops growing at the 1024² tile. The process also holds the input and output bitmaps.
fp32 intermediates, the default for 16-bit images, take `.upscaleVL` to 14.2 ms at 1024².

**Quality at ×2** on the G-4 drawn set: 75 licence-clean images in four classes (line art, text, UI, CG), four
damage rungs, scored with SSIMULACRA2, DISTS and an OCR word count. The other arms are AB-R-0486's own numbers.

| arm | SSIMULACRA2 ↑ line / text / ui / cg | DISTS ↓ line / text / ui / cg | words corrupted (text + ui, 4 rungs) |
|---|---|---|---:|
| Anime4K `.upscaleVL` | 57.6 / 44.8 / 38.8 / 60.8 | 0.074 / 0.081 / 0.128 / 0.073 | 100 |
| Anime4K `.b` | 51.8 / 40.0 / 31.8 / 54.7 | 0.086 / 0.088 / 0.154 / 0.068 | 363 |
| HEART `.fidelity` | 67.8 / 59.9 / 44.5 / 69.6 | 0.031 / 0.037 / 0.061 / 0.038 | 245 |
| NERVE `.fidelity` | 59.0 / 49.0 / 44.6 / 57.8 | 0.044 / 0.061 / 0.074 / 0.062 | 298 |
| Lanczos | 43.8 / 31.4 / 28.5 / 55.3 | 0.165 / 0.148 / 0.182 / 0.130 | 108 |

On 100 photo tiles at ×2, `.upscaleVL` scores SSIMULACRA2 77.3 against Lanczos 75.4 and HEART 70.2; `.b` scores 57.0.
Method, scripts and every per-image number: `mlxengine-forge/Tools/anime4k-image/20261015/`.

## Licence

MIT (`LICENSE`) for the generator, runtime, CLI and validation code. The generated shaders in
`Sources/Anime4KMetal/Shaders/` derive from bloc97/Anime4K's GLSL, MIT, reproduced in `NOTICE`. The test fixtures are
crops of a CC0 illustration (Wikimedia Commons, by snoopingasusual) and of our own render.

## Open

1. The playback-scale eyeball (`video --compare`) on real anime, especially low-bitrate SD and 720p sources.
2. A real-time path in a player (`AVPlayerItemVideoOutput` → `Anime4K` → `CAMetalLayer`), and a WGSL port for the
   web (the generator's structure carries over pass for pass).
3. Anime4K's GAN shaders (v4.1) are not generated yet; they use a different graph shape.
