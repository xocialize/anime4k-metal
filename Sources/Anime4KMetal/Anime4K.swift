// Anime4K on Metal — runs bloc97/Anime4K's mpv GLSL hook chains (MIT) as Metal compute shaders.
//
// The shaders are GENERATED (Generator/anime4k_msl.py): one kernel per GLSL pass, with the hook's own sampling —
// texel-centre offsets through a clamp-to-edge linear sampler, MAIN sampled bilinearly in the depth-to-space pass
// (libplacebo's defaults). Each mode ships as MSL source + a plan (pass order, bound texture names, SAVE target,
// scale); the runtime compiles one library per mode and replays the plan, replacing MAIN when a pass saves it — as
// mpv does. Validated against an independent PyTorch reading of the same shaders at 144–151 dB (Validation/).

import CoreVideo
import Foundation
import Metal
import MetalPerformanceShaders

public enum Anime4KMode: String, CaseIterable, Sendable {
    /// Restore_CNN_Soft_VL → Upscale_CNN_x2_VL — Anime4K's mode B minus its Clamp_Highlights/second upscale.
    case b = "mode_b"
    /// Restore_CNN_VL → Upscale_CNN_x2_VL — mode A, same trimming.
    case a = "mode_a"
    /// Upscale_CNN_x2_VL alone (no restore).
    case upscaleVL = "upscale_vl"
    /// Upscale_CNN_x2_M alone — the cheapest.
    case upscaleM = "upscale_m"
}

public enum Anime4KError: Error, CustomStringConvertible {
    case noDevice, missingResource(String), compile(String), unbound(String), pixelBuffer(String)
    /// An image the upscaler cannot take (its size, or a bitmap CoreGraphics cannot draw into RGBA).
    case unsupportedImage(String)
    /// The GPU reported an error while running the chain.
    case gpu(String)
    public var description: String {
        switch self {
        case .noDevice: return "no Metal device"
        case .missingResource(let r): return "missing bundled resource \(r)"
        case .compile(let e): return "MSL compile failed: \(e)"
        case .unbound(let t): return "pass reads unbound texture \(t)"
        case .pixelBuffer(let e): return "pixel buffer: \(e)"
        case .unsupportedImage(let e): return "unsupported image: \(e)"
        case .gpu(let e): return "GPU: \(e)"
        }
    }
}

struct Pass: Decodable { let kernel: String; let binds: [String]; let save: String; let scale: Int }

public final class Anime4K {
    public let mode: Anime4KMode
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    /// rgba16Float intermediates (what mpv allocates; 66.6 dB vs fp32, ≤ 1 8-bit level) or rgba32Float.
    public let halfPrecision: Bool
    let passes: [Pass]
    let pipelines: [MTLComputePipelineState]
    let util: MTLLibrary
    let toBGRA: MTLComputePipelineState
    let splitPSO: MTLComputePipelineState
    var pool: [String: MTLTexture] = [:]
    var cache: CVMetalTextureCache?

    public init(mode: Anime4KMode = .b, halfPrecision: Bool = true, device: MTLDevice? = nil) throws {
        guard let dev = device ?? MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else {
            throw Anime4KError.noDevice
        }
        self.mode = mode
        self.device = dev
        self.queue = q
        self.halfPrecision = halfPrecision
        guard let src = Bundle.module.url(forResource: mode.rawValue, withExtension: "msl", subdirectory: "Shaders"),
              let planURL = Bundle.module.url(forResource: mode.rawValue, withExtension: "plan.json",
                                              subdirectory: "Shaders") else {
            throw Anime4KError.missingResource(mode.rawValue)
        }
        let lib: MTLLibrary
        do {
            lib = try dev.makeLibrary(source: try String(contentsOf: src, encoding: .utf8), options: nil)
            util = try dev.makeLibrary(source: Self.utilitySource, options: nil)
        } catch { throw Anime4KError.compile(String(describing: error)) }
        passes = try JSONDecoder().decode([Pass].self, from: Data(contentsOf: planURL))
        pipelines = try passes.map { p in
            guard let f = lib.makeFunction(name: p.kernel) else { throw Anime4KError.missingResource(p.kernel) }
            return try dev.makeComputePipelineState(function: f)
        }
        toBGRA = try dev.makeComputePipelineState(function: util.makeFunction(name: "to_output")!)
        splitPSO = try dev.makeComputePipelineState(function: util.makeFunction(name: "split")!)
        CVMetalTextureCacheCreate(nil, nil, dev, nil, &cache)
    }

    public var passCount: Int { passes.count }

    /// Bytes of intermediate textures one `encode` of a `width`×`height` input allocates (each pass keeps its own
    /// output, because the last passes read every earlier layer). Exact, from the plan.
    public func workingSetBytes(width: Int, height: Int) -> Int {
        var size: [String: (Int, Int)] = ["MAIN": (width, height)]
        var total = 0
        let texel = halfPrecision ? 8 : 16
        for p in passes {
            let base = size[p.binds.last ?? "MAIN"] ?? (width, height)
            let out = (base.0 * p.scale, base.1 * p.scale)
            total += out.0 * out.1 * texel
            size[p.save] = out
        }
        return total
    }

    /// Releases the pooled intermediate textures; the compiled pipelines stay.
    public func purgeTextures() { pool.removeAll() }

    /// The pool's texture size, so a caller changing size can purge first instead of growing the pool.
    var pooledInputSize: (Int, Int)?

    func pooled(_ key: String, _ w: Int, _ h: Int) -> MTLTexture {
        let k = "\(key)@\(w)x\(h)"
        if let t = pool[k] { return t }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: halfPrecision ? .rgba16Float : .rgba32Float,
                                                         width: w, height: h, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        let t = device.makeTexture(descriptor: d)!
        pool[k] = t
        return t
    }

    /// Encode the whole chain on `input` (any readable RGBA-sampled texture, e.g. a BGRA8 video frame — Metal
    /// returns components in RGBA order). Returns the ×2 output (an intermediate-format texture from the pool,
    /// valid until the next call).
    public func encode(_ input: MTLTexture, into cb: MTLCommandBuffer) throws -> MTLTexture {
        var tex: [String: MTLTexture] = ["MAIN": input]
        for (i, (p, pso)) in zip(passes, pipelines).enumerated() {
            let ins = try p.binds.map { b -> MTLTexture in
                guard let t = tex[b] else { throw Anime4KError.unbound(b) }
                return t
            }
            let base = ins.last!
            let out = pooled("p\(i)", base.width * p.scale, base.height * p.scale)
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pso)
            for (j, t) in ins.enumerated() { enc.setTexture(t, index: j) }
            enc.setTexture(out, index: ins.count)
            enc.dispatchThreadgroups(MTLSize(width: (out.width + 15) / 16, height: (out.height + 15) / 16, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.endEncoding()
            tex[p.save] = out
        }
        return tex["MAIN"]!
    }

    /// Wrap a CVPixelBuffer (32BGRA, IOSurface-backed) as a Metal texture — zero copy. The CVMetalTexture must stay
    /// alive until the GPU is done with the texture, so it is returned too.
    public func texture(_ pb: CVPixelBuffer) throws -> (MTLTexture, CVMetalTexture) {
        var cvTex: CVMetalTexture?
        let (w, h) = (CVPixelBufferGetWidth(pb), CVPixelBufferGetHeight(pb))
        guard let cache, CVMetalTextureCacheCreateTextureFromImage(nil, cache, pb, nil, .bgra8Unorm, w, h, 0, &cvTex)
                == kCVReturnSuccess, let cvTex, let t = CVMetalTextureGetTexture(cvTex) else {
            throw Anime4KError.pixelBuffer("texture from \(w)x\(h)")
        }
        return (t, cvTex)
    }

    /// Upscale one 32BGRA frame into `output` (32BGRA, 2× its size). `compare`: left of the divider is
    /// MPS Lanczos ×2 of the input, right is Anime4K — for side-by-side eyeballing at playback scale.
    public func upscale(_ input: CVPixelBuffer, into output: CVPixelBuffer, compare: Bool = false,
                        divider: Float = 0.5) throws -> Double {
        let (inTex, inHold) = try texture(input), (outTex, outHold) = try texture(output)
        defer { withExtendedLifetime((inHold, outHold)) {} }
        let cb = queue.makeCommandBuffer()!
        let a4k = try encode(inTex, into: cb)
        let grid = MTLSize(width: (outTex.width + 15) / 16, height: (outTex.height + 15) / 16, depth: 1)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        if compare {
            let lz = pooled("lanczos", outTex.width, outTex.height)
            MPSImageLanczosScale(device: device).encode(commandBuffer: cb, sourceTexture: inTex, destinationTexture: lz)
            let enc = cb.makeComputeCommandEncoder()!
            var d = divider
            enc.setComputePipelineState(splitPSO)
            enc.setTexture(lz, index: 0)
            enc.setTexture(a4k, index: 1)
            enc.setTexture(outTex, index: 2)
            enc.setBytes(&d, length: 4, index: 0)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            enc.endEncoding()
        } else {
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(toBGRA)
            enc.setTexture(a4k, index: 0)
            enc.setTexture(outTex, index: 1)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
            enc.endEncoding()
        }
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw Anime4KError.pixelBuffer("GPU: \(e)") }
        return cb.gpuEndTime - cb.gpuStartTime
    }

    static let utilitySource = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void to_output(texture2d<float, access::read> src [[texture(0)]],
                          texture2d<float, access::write> dst [[texture(1)]], uint2 g [[thread_position_in_grid]]) {
        if (g.x >= dst.get_width() || g.y >= dst.get_height()) return;
        dst.write(float4(clamp(src.read(g).rgb, 0.0, 1.0), 1.0), g);
    }
    kernel void split(texture2d<float, access::read> left [[texture(0)]],
                      texture2d<float, access::read> right [[texture(1)]],
                      texture2d<float, access::write> dst [[texture(2)]],
                      constant float& divider [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
        if (g.x >= dst.get_width() || g.y >= dst.get_height()) return;
        uint cut = uint(divider * float(dst.get_width()));
        float3 c = (g.x < cut) ? left.read(g).rgb : right.read(g).rgb;
        if (g.x == cut || g.x + 1 == cut) c = float3(1.0, 0.85, 0.0);
        dst.write(float4(clamp(c, 0.0, 1.0), 1.0), g);
    }
    """
}
