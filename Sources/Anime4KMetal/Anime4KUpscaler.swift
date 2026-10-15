// Anime4KUpscaler — Anime4K as an image upscaler: a CGImage in, a CGImage at exactly twice the size out.
//
// What it adds over `Anime4K` (the shader chain on one texture):
//
// - **Depth.** An 8-bit image comes back 8-bit, a 16-bit one 16-bit. The chain's intermediates are fp16 for 8-bit
//   images (≤ 1 level against fp32, AB-R-0392) and fp32 for 16-bit ones, where fp16's 11-bit mantissa would show.
// - **Alpha.** The shaders compute colour only. A transparent image is un-premultiplied, its colour is bled outward
//   into the transparent area (so the network never sees the black that premultiplied transparency hides, which
//   would ring as a dark fringe), and the chain runs on that colour. Alpha is enlarged ×2 with Lanczos-3 (a kernel
//   here, at fp32; MPS's Lanczos is not translation-invariant, so tiles would not be exact) and multiplied back. An
//   image whose alpha is all opaque takes the plain path.
// - **Tiles.** Every pass keeps its own texture (the last ones read every earlier layer), so the chain's working
//   set is ~35 textures the size of the input. Inputs larger than `maxTileSide` run as equal-size overlapping tiles
//   whose margin covers the chain's receptive field and the bleed, so a tiled result equals the untiled one exactly
//   (the tests check it); `memoryEstimate` says what a call will allocate.
// - **Colour.** The values are processed as encoded, in the image's own RGB colour space when it has one (what mpv
//   does with MAIN: display-referred, not linearised). Other colour models are drawn into sRGB first. Values above
//   1.0 (extended-range or HDR floats) are clamped on entry; HDR is outside this package's remit.
//
// The default mode is `.upscaleVL`, not the video player's mode B. Measured on the G-4 drawn set at ×2
// (mlxengine-forge/Tools/anime4k-image/20261015, AB-T-0291): upscale-only VL beats Lanczos on every image of every
// class and keeps text as legible as Lanczos does, and on photos it is no worse than Lanczos. Mode B's restore pass
// costs ~12 SSIMULACRA2 on clean sources, corrupts 3.6× the words and makes photos look drawn; it wins only on
// heavily compressed input (JPEG q30), which is what it was made for. Neither mode reaches the learned upscalers
// (HEART, NERVE, RealPLKSR) on drawn content: this is the instant, model-free tier, not the best one.

import CoreGraphics
import Foundation
import Metal

public final class Anime4KUpscaler: @unchecked Sendable {

    /// Intermediate precision of the shader chain.
    public enum Precision: Sendable, Hashable {
        /// fp16 for 8-bit images, fp32 for deeper ones.
        case automatic
        case half
        case full
    }

    public struct Configuration: Sendable, Hashable {
        public var mode: Anime4KMode
        public var precision: Precision
        /// The largest tile side, in input pixels. Larger inputs are tiled. 1024 keeps mode B's working set near
        /// 380 MB at fp16; `memoryEstimate(width:height:)` gives the exact figure.
        public var maxTileSide: Int
        /// Input pixels of context a tile carries beyond the part it contributes, on each interior side. It must
        /// cover the chain's receptive field plus `bleedRadius`.
        public var tileMargin: Int
        /// How far, in input pixels, colour is bled into transparent areas before the chain runs.
        public var bleedRadius: Int
        /// Keep the GPU textures between calls (faster repeated calls of one size); otherwise they are released at
        /// the end of every call and only the compiled pipelines stay.
        public var retainsTextures: Bool

        public init(mode: Anime4KMode = .upscaleVL, precision: Precision = .automatic, maxTileSide: Int = 1024,
                    tileMargin: Int = 48, bleedRadius: Int = 16, retainsTextures: Bool = false) {
            self.mode = mode
            self.precision = precision
            self.maxTileSide = maxTileSide
            self.tileMargin = tileMargin
            self.bleedRadius = bleedRadius
            self.retainsTextures = retainsTextures
        }
    }

    private static let systemDevice: MTLDevice? = MTLCreateSystemDefaultDevice()

    /// Whether this Mac can run the upscaler (it needs a Metal GPU; there is no CPU fallback).
    public static var isAvailable: Bool { systemDevice != nil }
    /// Why it can't, in the user's terms; nil when it can.
    public static var unavailableReason: String? {
        isAvailable ? nil : "Anime4K runs on the GPU, and this Mac reports no Metal device."
    }

    public let configuration: Configuration
    public let device: MTLDevice
    private let queue: MTLCommandQueue
    private let lock = NSLock()
    private var chains: [Bool: Anime4K] = [:]          // keyed by halfPrecision
    private let unpremultiplyPSO: MTLComputePipelineState
    private let bleedPSO: MTLComputePipelineState
    private let composePSO: MTLComputePipelineState
    private let alpha2xPSO: MTLComputePipelineState
    private var textures: TileTextures?

    public init(_ configuration: Configuration = Configuration(), device: MTLDevice? = nil) throws {
        guard let dev = device ?? Self.systemDevice, let q = dev.makeCommandQueue() else { throw Anime4KError.noDevice }
        guard configuration.maxTileSide >= 2 * configuration.tileMargin + 64 else {
            throw Anime4KError.unsupportedImage("maxTileSide \(configuration.maxTileSide) leaves no core inside a \(configuration.tileMargin)-pixel margin")
        }
        self.configuration = configuration
        self.device = dev
        self.queue = q
        let lib: MTLLibrary
        do { lib = try dev.makeLibrary(source: Self.kernelSource, options: nil) } catch {
            throw Anime4KError.compile(String(describing: error))
        }
        func pso(_ name: String) throws -> MTLComputePipelineState {
            guard let f = lib.makeFunction(name: name) else { throw Anime4KError.missingResource(name) }
            return try dev.makeComputePipelineState(function: f)
        }
        unpremultiplyPSO = try pso("a4k_unpremultiply")
        bleedPSO = try pso("a4k_bleed")
        composePSO = try pso("a4k_compose")
        alpha2xPSO = try pso("a4k_alpha2x")
    }

    // MARK: - Public API

    /// `image` at exactly twice its width and height, with its depth, colour space and alpha kept.
    public func upscale(_ image: CGImage) throws -> CGImage {
        lock.lock()
        defer { lock.unlock() }
        defer { if !configuration.retainsTextures { releaseTextures() } }
        return try run(image)
    }

    /// `image` doubled `doublings` times (×2, ×4, ×8, …).
    public func upscale(_ image: CGImage, doublings: Int) throws -> CGImage {
        var out = image
        for _ in 0 ..< max(0, doublings) { out = try upscale(out) }
        return out
    }

    /// Bytes of GPU memory one call on a `width`×`height` image allocates at most (one tile's worth of textures).
    public func memoryEstimate(width: Int, height: Int, deep: Bool = false, alpha: Bool = true) -> Int {
        let (tw, th) = (min(width, configuration.maxTileSide), min(height, configuration.maxTileSide))
        let half = usesHalf(deep: deep)
        let chainBytes = (try? chain(half: half).workingSetBytes(width: tw, height: th)) ?? 0
        let io = deep ? 8 : 4
        var bytes = chainBytes + tw * th * io + 4 * tw * th * io                // input tile + output tile
        if alpha { bytes += 2 * tw * th * (half ? 8 : 16) + 4 * tw * th * 4 }  // bleed ping-pong + Lanczos alpha
        return bytes
    }

    /// Releases every GPU texture this upscaler holds; the compiled pipelines stay.
    public func purge() {
        lock.lock()
        defer { lock.unlock() }
        releaseTextures()
    }

    // MARK: - The run

    private func usesHalf(deep: Bool) -> Bool {
        switch configuration.precision {
        case .automatic: return !deep
        case .half: return true
        case .full: return false
        }
    }

    private func chain(half: Bool) throws -> Anime4K {
        if let c = chains[half] { return c }
        let c = try Anime4K(mode: configuration.mode, halfPrecision: half, device: device)
        chains[half] = c
        return c
    }

    private func releaseTextures() {
        textures = nil
        for c in chains.values { c.purgeTextures(); c.pooledInputSize = nil }
    }

    private func run(_ image: CGImage) throws -> CGImage {
        let (w, h) = (image.width, image.height)
        guard w > 0, h > 0 else { throw Anime4KError.unsupportedImage("empty image") }
        guard w <= 16384, h <= 16384 else {
            throw Anime4KError.unsupportedImage("\(w)×\(h) doubles past 32768 pixels a side")
        }
        let deep = image.bitsPerComponent > 8
        let source = try Bitmap(drawing: image, deep: deep)
        let alphaPath = source.hasAlphaChannel && !source.isOpaque
        let half = usesHalf(deep: deep)
        let a4k = try chain(half: half)

        // Equal-size tiles: cores partition the image; each tile is its core plus the margin, clamped inside.
        let tile = (min(w, configuration.maxTileSide), min(h, configuration.maxTileSide))
        let xs = Self.cores(length: w, tile: tile.0, margin: configuration.tileMargin)
        let ys = Self.cores(length: h, tile: tile.1, margin: configuration.tileMargin)

        if let size = a4k.pooledInputSize, size != tile { a4k.purgeTextures() }
        a4k.pooledInputSize = tile
        let tex = try tileTextures(width: tile.0, height: tile.1, deep: deep, half: half, alpha: alphaPath)

        let out = Bitmap(width: 2 * w, height: 2 * h, deep: deep, colorSpace: source.colorSpace,
                         hasAlphaChannel: source.hasAlphaChannel)

        for y in ys {
            for x in xs {
                if Task.isCancelled { throw CancellationError() }
                let ox = Self.origin(of: x, length: w, tile: tile.0, margin: configuration.tileMargin)
                let oy = Self.origin(of: y, length: h, tile: tile.1, margin: configuration.tileMargin)
                tex.input.replace(region: MTLRegionMake2D(0, 0, tile.0, tile.1), mipmapLevel: 0,
                                  withBytes: source.pointer(x: ox, y: oy), bytesPerRow: source.rowBytes)
                guard let cb = queue.makeCommandBuffer() else { throw Anime4KError.gpu("no command buffer") }
                var main: MTLTexture = tex.input
                if alphaPath, let a = tex.workA, let b = tex.workB, let up = tex.alpha2x {
                    dispatch(cb, unpremultiplyPSO, [tex.input, a], grid: (tile.0, tile.1))
                    var (src, dst) = (a, b)
                    for _ in 0 ..< configuration.bleedRadius {
                        dispatch(cb, bleedPSO, [src, dst], grid: (tile.0, tile.1))
                        swap(&src, &dst)
                    }
                    main = src
                    dispatch(cb, alpha2xPSO, [tex.input, up], grid: (2 * tile.0, 2 * tile.1))
                }
                let colour = try a4k.encode(main, into: cb)
                var useAlpha: UInt32 = alphaPath ? 1 : 0
                dispatch(cb, composePSO, [colour, tex.alpha2x ?? colour, tex.output], grid: (2 * tile.0, 2 * tile.1),
                         bytes: &useAlpha)
                cb.commit()
                cb.waitUntilCompleted()
                if let e = cb.error { throw Anime4KError.gpu(String(describing: e)) }
                tex.output.getBytes(out.pointer(x: 2 * x.lowerBound, y: 2 * y.lowerBound), bytesPerRow: out.rowBytes,
                                    from: MTLRegionMake2D(2 * (x.lowerBound - ox), 2 * (y.lowerBound - oy),
                                                          2 * x.count, 2 * y.count), mipmapLevel: 0)
            }
        }
        guard let result = out.makeImage() else { throw Anime4KError.unsupportedImage("cannot build the result image") }
        return result
    }

    private func dispatch(_ cb: MTLCommandBuffer, _ pso: MTLComputePipelineState, _ textures: [MTLTexture],
                          grid: (Int, Int), bytes: UnsafeMutablePointer<UInt32>? = nil) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        for (i, t) in textures.enumerated() { enc.setTexture(t, index: i) }
        if let bytes { enc.setBytes(bytes, length: 4, index: 0) }
        enc.dispatchThreadgroups(MTLSize(width: (grid.0 + 15) / 16, height: (grid.1 + 15) / 16, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }

    // MARK: - Tiling

    /// The cores along one axis: contiguous ranges partitioning `0..<length`, each at most `tile - 2·margin` long
    /// (one range when the axis fits in a tile).
    static func cores(length: Int, tile: Int, margin: Int) -> [Range<Int>] {
        guard length > tile else { return [0 ..< length] }
        let span = tile - 2 * margin
        let n = (length + span - 1) / span
        return (0 ..< n).map { i in (i * length / n) ..< ((i + 1) * length / n) }
    }

    /// Where the tile holding `core` starts: `margin` before the core, kept inside the image.
    static func origin(of core: Range<Int>, length: Int, tile: Int, margin: Int) -> Int {
        guard length > tile else { return 0 }
        return min(max(core.lowerBound - margin, 0), length - tile)
    }

    // MARK: - GPU textures (one tile's worth, reused across tiles and, when retained, across calls)

    private struct TileTextures {
        let key: String
        let input: MTLTexture
        let output: MTLTexture
        let workA: MTLTexture?
        let workB: MTLTexture?
        let alpha2x: MTLTexture?
    }

    private func tileTextures(width: Int, height: Int, deep: Bool, half: Bool, alpha: Bool) throws -> TileTextures {
        let key = "\(width)x\(height) deep:\(deep) half:\(half) alpha:\(alpha)"
        if let t = textures, t.key == key { return t }
        textures = nil
        func make(_ format: MTLPixelFormat, _ w: Int, _ h: Int, shared: Bool) throws -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: w, height: h, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = shared ? .shared : .private
            guard let t = device.makeTexture(descriptor: d) else {
                throw Anime4KError.gpu("cannot allocate a \(w)×\(h) \(format) texture")
            }
            return t
        }
        let io: MTLPixelFormat = deep ? .rgba16Unorm : .rgba8Unorm
        let work: MTLPixelFormat = half ? .rgba16Float : .rgba32Float
        let t = TileTextures(
            key: key,
            input: try make(io, width, height, shared: true),
            output: try make(io, 2 * width, 2 * height, shared: true),
            workA: alpha ? try make(work, width, height, shared: false) : nil,
            workB: alpha ? try make(work, width, height, shared: false) : nil,
            alpha2x: alpha ? try make(.r32Float, 2 * width, 2 * height, shared: false) : nil)
        textures = t
        return t
    }

    // MARK: - Kernels

    static let kernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    // Premultiplied RGBA → straight colour; .a marks pixels that hold colour (1) or not yet (0).
    kernel void a4k_unpremultiply(texture2d<float, access::read> src [[texture(0)]],
                                  texture2d<float, access::write> dst [[texture(1)]],
                                  uint2 g [[thread_position_in_grid]]) {
        if (g.x >= dst.get_width() || g.y >= dst.get_height()) return;
        float4 p = src.read(g);
        dst.write(p.a > 0.0 ? float4(clamp(p.rgb / p.a, 0.0, 1.0), 1.0) : float4(0.0), g);
    }

    // One step of colour bleeding: a pixel without colour takes the mean of its neighbours that have one.
    kernel void a4k_bleed(texture2d<float, access::read> src [[texture(0)]],
                          texture2d<float, access::write> dst [[texture(1)]],
                          uint2 g [[thread_position_in_grid]]) {
        uint w = dst.get_width(), h = dst.get_height();
        if (g.x >= w || g.y >= h) return;
        float4 c = src.read(g);
        if (c.a > 0.0) { dst.write(c, g); return; }
        float3 sum = float3(0.0);
        float n = 0.0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int2 q = clamp(int2(g) + int2(dx, dy), int2(0), int2(w - 1, h - 1));
                float4 s = src.read(uint2(q));
                if (s.a > 0.0) { sum += s.rgb; n += 1.0; }
            }
        }
        dst.write(n > 0.0 ? float4(sum / n, 1.0) : c, g);
    }

    static inline float lanczos3(float d) {
        if (d == 0.0) return 1.0;
        if (fabs(d) >= 3.0) return 0.0;
        float p = M_PI_F * d;
        return 3.0 * sin(p) * sin(p / 3.0) / (p * p);
    }

    // Alpha ×2 with Lanczos-3, from whole texels at the two fixed phases of an exact doubling (output x samples
    // source x/2 − 0.25), edges clamped. Every output depends only on its 6×6 neighbourhood, so tiles are exact.
    kernel void a4k_alpha2x(texture2d<float, access::read> src [[texture(0)]],
                            texture2d<float, access::write> dst [[texture(1)]],
                            uint2 g [[thread_position_in_grid]]) {
        if (g.x >= dst.get_width() || g.y >= dst.get_height()) return;
        int2 size = int2(src.get_width(), src.get_height());
        int2 k = int2(g) / 2;
        int2 odd = int2(g) & 1;
        float2 s = float2(k) + select(float2(-0.25), float2(0.25), odd != 0);
        int2 first = k - 3 + odd;
        float wx[6], wy[6], sx = 0.0, sy = 0.0;
        for (int i = 0; i < 6; i++) {
            wx[i] = lanczos3(s.x - float(first.x + i)); sx += wx[i];
            wy[i] = lanczos3(s.y - float(first.y + i)); sy += wy[i];
        }
        float acc = 0.0;
        for (int j = 0; j < 6; j++) {
            for (int i = 0; i < 6; i++) {
                int2 q = clamp(first + int2(i, j), int2(0), size - 1);
                acc += wx[i] * wy[j] * src.read(uint2(q)).a;
            }
        }
        dst.write(float4(acc / (sx * sy)), g);
    }

    // The chain's colour, clamped, premultiplied by the Lanczos-enlarged alpha (or opaque).
    kernel void a4k_compose(texture2d<float, access::read> colour [[texture(0)]],
                            texture2d<float, access::read> alpha [[texture(1)]],
                            texture2d<float, access::write> dst [[texture(2)]],
                            constant uint& useAlpha [[buffer(0)]],
                            uint2 g [[thread_position_in_grid]]) {
        if (g.x >= dst.get_width() || g.y >= dst.get_height()) return;
        float3 c = clamp(colour.read(g).rgb, 0.0, 1.0);
        float a = useAlpha != 0 ? clamp(alpha.read(g).r, 0.0, 1.0) : 1.0;
        dst.write(float4(c * a, a), g);
    }
    """
}

// MARK: - CPU bitmaps

/// An RGBA bitmap in the layout the GPU textures share: 8-bit components in memory order, or 16-bit little-endian
/// ones; alpha last, premultiplied.
final class Bitmap {
    let width: Int
    let height: Int
    let deep: Bool
    let colorSpace: CGColorSpace
    let hasAlphaChannel: Bool
    let rowBytes: Int
    let data: UnsafeMutableRawPointer
    private(set) var isOpaque = true

    var bytesPerPixel: Int { deep ? 8 : 4 }

    init(width: Int, height: Int, deep: Bool, colorSpace: CGColorSpace, hasAlphaChannel: Bool) {
        self.width = width
        self.height = height
        self.deep = deep
        self.colorSpace = colorSpace
        self.hasAlphaChannel = hasAlphaChannel
        rowBytes = width * (deep ? 8 : 4)
        data = UnsafeMutableRawPointer.allocate(byteCount: rowBytes * height, alignment: 64)
    }

    /// `image` drawn unchanged into a bitmap of its own RGB colour space (sRGB for other colour models).
    convenience init(drawing image: CGImage, deep: Bool) throws {
        let space: CGColorSpace
        if let own = image.colorSpace, own.model == .rgb { space = own } else { space = CGColorSpace(name: CGColorSpace.sRGB)! }
        let alpha: Bool
        switch image.alphaInfo {
        case .none, .noneSkipFirst, .noneSkipLast: alpha = false
        default: alpha = true
        }
        self.init(width: image.width, height: image.height, deep: deep, colorSpace: space, hasAlphaChannel: alpha)
        memset(data, 0, rowBytes * height)
        guard let ctx = CGContext(data: data, width: width, height: height, bitsPerComponent: deep ? 16 : 8,
                                  bytesPerRow: rowBytes, space: space, bitmapInfo: bitmapInfo) else {
            throw Anime4KError.unsupportedImage("CoreGraphics cannot draw a \(deep ? 16 : 8)-bit RGBA bitmap in \(space.name.map { $0 as String } ?? "this colour space")")
        }
        ctx.setBlendMode(.copy)
        ctx.interpolationQuality = .none
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        if alpha { isOpaque = scanOpaque() }
    }

    deinit { data.deallocate() }

    var bitmapInfo: UInt32 {
        let a = hasAlphaChannel ? CGImageAlphaInfo.premultipliedLast : .noneSkipLast
        return a.rawValue | (deep ? CGBitmapInfo.byteOrder16Little.rawValue : 0)
    }

    func pointer(x: Int, y: Int) -> UnsafeMutableRawPointer { data + y * rowBytes + x * bytesPerPixel }

    private func scanOpaque() -> Bool {
        if deep {
            let p = data.assumingMemoryBound(to: UInt16.self)
            for i in stride(from: 3, to: width * height * 4, by: 4) where UInt16(littleEndian: p[i]) != .max { return false }
        } else {
            let p = data.assumingMemoryBound(to: UInt8.self)
            for i in stride(from: 3, to: width * height * 4, by: 4) where p[i] != .max { return false }
        }
        return true
    }

    /// The bitmap as a CGImage. The image takes over the memory.
    func makeImage() -> CGImage? {
        let bytes = rowBytes * height
        let owner = Unmanaged.passRetained(self)
        guard let provider = CGDataProvider(dataInfo: owner.toOpaque(), data: data, size: bytes, releaseData: { info, _, _ in
            if let info { Unmanaged<Bitmap>.fromOpaque(info).release() }
        }) else {
            owner.release()
            return nil
        }
        return CGImage(width: width, height: height, bitsPerComponent: deep ? 16 : 8, bitsPerPixel: deep ? 64 : 32,
                       bytesPerRow: rowBytes, space: colorSpace, bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}
