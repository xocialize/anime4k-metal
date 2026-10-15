// anime4k — test driver for Anime4KMetal.
//
//   anime4k image <in> <out.png> [--mode b|a|upscale_vl|upscale_m] [--fp32] [--tile N] [--doublings K]
//         the image API (Anime4KUpscaler): keeps alpha and 16-bit depth, tiles large inputs
//   anime4k batch <in-dir> <out-dir> [--mode …] [--fp32] [--tile N]      every .png in <in-dir>, same names
//   anime4k benchimage <in> [--mode …] [--sizes 512,1024,2048,4096] [--iters 10] [--tile N]
//         end-to-end CGImage → CGImage wall time per ×2 (median), at square crops of <in> resized to each size
//   anime4k video <in.mp4|mov|m4v> <out.mp4> [--mode …] [--fp32] [--compare] [--divider 0.5] [--limit N]
//         --compare: left of the yellow divider is MPS Lanczos ×2, right is Anime4K — the playback-scale eyeball
//   anime4k bench <in.png> [--mode …] [--fp32] [--iters 50]
//
// Video: AVFoundation decode (BGRA) → zero-copy Metal textures → the shader chain → HEVC at ~0.15 bits/px, source
// timestamps preserved. Audio is not carried (eyeballing only).

import AVFoundation
import Anime4KMetal
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

func die(_ s: String) -> Never { FileHandle.standardError.write((s + "\n").data(using: .utf8)!); exit(1) }

var args = Array(CommandLine.arguments.dropFirst())
func opt(_ n: String) -> String? {
    guard let i = args.firstIndex(of: n), i + 1 < args.count else { return nil }
    let v = args[i + 1]; args.removeSubrange(i ... i + 1); return v
}
func flag(_ n: String) -> Bool { if let i = args.firstIndex(of: n) { args.remove(at: i); return true }; return false }

let modeName = opt("--mode") ?? "b"
guard let mode = Anime4KMode(rawValue: modeName.count == 1 ? "mode_\(modeName)" : modeName) else {
    die("unknown mode \(modeName) (b, a, upscale_vl, upscale_m)")
}
let half = !flag("--fp32")

func pixelBuffer(_ w: Int, _ h: Int) -> CVPixelBuffer {
    var pb: CVPixelBuffer?
    let attrs: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                                kCVPixelBufferMetalCompatibilityKey as String: true]
    CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
    return pb!
}

func loadImage(_ path: String) -> CVPixelBuffer {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { die("cannot read \(path)") }
    let pb = pixelBuffer(cg.width, cg.height)
    CVPixelBufferLockBaseAddress(pb, [])
    let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: cg.width, height: cg.height, bitsPerComponent: 8,
                        bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpace(name: CGColorSpace.sRGB)!,
                        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: cg.width, height: cg.height))
    CVPixelBufferUnlockBaseAddress(pb, [])
    return pb
}

func savePNG(_ pb: CVPixelBuffer, _ path: String) {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: CVPixelBufferGetWidth(pb),
                        height: CVPixelBufferGetHeight(pb), bitsPerComponent: 8,
                        bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpace(name: CGColorSpace.sRGB)!,
                        bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
    let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
    CGImageDestinationFinalize(dest)
}

func median(_ v: [Double]) -> Double { let s = v.sorted(); return s.isEmpty ? .nan : s[s.count / 2] }

func readCG(_ path: String) -> CGImage {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { die("cannot read \(path)") }
    return cg
}

func writePNG(_ image: CGImage, _ path: String) {
    guard let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { die("cannot write \(path)") }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else { die("cannot write \(path)") }
}

/// Resident memory as the kernel accounts it (what Activity Monitor and the editor's memory tiers read).
func physFootprint() -> Int {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return kr == KERN_SUCCESS ? Int(info.phys_footprint) : 0
}

let tileSide = Int(opt("--tile") ?? "") ?? 1024
let upConfig = Anime4KUpscaler.Configuration(mode: mode, precision: half ? .automatic : .full, maxTileSide: tileSide)

if ["image", "batch", "benchimage"].contains(args.first ?? "") {
    let up: Anime4KUpscaler
    do { up = try Anime4KUpscaler(upConfig) } catch { die("\(error)") }
    switch args[0] {
    case "image":
        let doublings = Int(opt("--doublings") ?? "1") ?? 1
        guard args.count == 3 else { die("image <in> <out.png>") }
        let cg = readCG(args[1])
        let t0 = Date()
        let out = try up.upscale(cg, doublings: doublings)
        let ms = Date().timeIntervalSince(t0) * 1000
        writePNG(out, args[2])
        print(String(format: "%@ (%@): %dx%d %d-bit → %dx%d %d-bit in %.1f ms wall (first call, includes pipeline compile)",
                     mode.rawValue, half ? "auto precision" : "fp32", cg.width, cg.height, cg.bitsPerComponent,
                     out.width, out.height, out.bitsPerComponent, ms))
    case "batch":
        guard args.count == 3 else { die("batch <in-dir> <out-dir>") }
        let fm = FileManager.default
        try fm.createDirectory(atPath: args[2], withIntermediateDirectories: true)
        let names = try fm.contentsOfDirectory(atPath: args[1]).filter { $0.lowercased().hasSuffix(".png") }.sorted()
        let t0 = Date()
        for n in names { writePNG(try up.upscale(readCG(args[1] + "/" + n)), args[2] + "/" + n) }
        print(String(format: "%@: %d images in %.1f s", mode.rawValue, names.count, Date().timeIntervalSince(t0)))
    default: // benchimage
        let iters = Int(opt("--iters") ?? "10") ?? 10
        let sizes = (opt("--sizes") ?? "512,1024,2048,4096").split(separator: ",").compactMap { Int($0) }
        guard args.count == 2 else { die("benchimage <in>") }
        let src = readCG(args[1])
        let side = min(src.width, src.height)
        guard let square = src.cropping(to: CGRect(x: (src.width - side) / 2, y: (src.height - side) / 2, width: side, height: side))
        else { die("crop failed") }
        _ = try up.upscale(square)                                         // pipeline compile, outside the timing
        for s in sizes {
            let ctx = CGContext(data: nil, width: s, height: s, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            ctx.interpolationQuality = .high
            ctx.draw(square, in: CGRect(x: 0, y: 0, width: s, height: s))
            let img = ctx.makeImage()!
            let before = physFootprint()
            var peak = before
            var ts: [Double] = []
            for _ in 0 ..< iters {
                let t0 = DispatchTime.now().uptimeNanoseconds
                let out = try up.upscale(img)
                ts.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
                peak = max(peak, physFootprint())
                _ = out
            }
            let tiles = Int(ceil(Double(s) / Double(max(1, tileSide - 96)))) * Int(ceil(Double(s) / Double(max(1, tileSide - 96))))
            print(String(format: "%@ %@ %d² → %d²: %.1f ms median (min %.1f, n %d)  tiles %d  estimate %.0f MB  phys +%.0f MB",
                         mode.rawValue, half ? "fp16" : "fp32", s, 2 * s, median(ts), ts.min() ?? 0, iters,
                         s > tileSide ? tiles : 1,
                         Double(up.memoryEstimate(width: s, height: s, deep: false, alpha: false)) / 1e6,
                         Double(peak - before) / 1e6))
        }
    }
    exit(0)
}

let a4k: Anime4K
do { a4k = try Anime4K(mode: mode, halfPrecision: half) } catch { die("\(error)") }

switch args.first ?? "" {

case "bench":
    let iters = Int(opt("--iters") ?? "50") ?? 50
    guard args.count == 2 else { die("bench <in.png>") }
    let inPB = loadImage(args[1])
    let outPB = pixelBuffer(CVPixelBufferGetWidth(inPB) * 2, CVPixelBufferGetHeight(inPB) * 2)
    _ = try a4k.upscale(inPB, into: outPB)
    var ts: [Double] = []
    for _ in 0 ..< iters { ts.append(try a4k.upscale(inPB, into: outPB)) }
    print(String(format: "%@ %@ %dx%d → ×2: GPU %.2f ms/frame (median of %d, min %.2f) ≈ %.0f fps", mode.rawValue,
                 half ? "fp16" : "fp32", CVPixelBufferGetWidth(inPB), CVPixelBufferGetHeight(inPB),
                 median(ts) * 1000, iters, (ts.min() ?? 0) * 1000, 1 / median(ts)))

case "video":
    let compare = flag("--compare")
    let divider = Float(opt("--divider") ?? "0.5") ?? 0.5
    let limit = Int(opt("--limit") ?? "") ?? .max
    guard args.count == 3 else { die("video <in> <out.mp4> [--compare] [--limit N]") }
    let asset = AVURLAsset(url: URL(fileURLWithPath: args[1]))
    guard let track = try await asset.loadTracks(withMediaType: .video).first else { die("no video track") }
    let size = try await track.load(.naturalSize)
    let fps = try await track.load(.nominalFrameRate)
    let (w, h) = (Int(size.width), Int(size.height))
    let reader = try AVAssetReader(asset: asset)
    let ro = AVAssetReaderTrackOutput(track: track, outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferMetalCompatibilityKey as String: true])
    reader.add(ro)
    let outURL = URL(fileURLWithPath: args[2])
    try? FileManager.default.removeItem(at: outURL)
    let writer = try AVAssetWriter(outputURL: outURL, fileType: .mp4)
    let bitrate = Int(0.15 * Double(4 * w * h) * Double(fps))
    let wi = AVAssetWriterInput(mediaType: .video, outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.hevc, AVVideoWidthKey: 2 * w, AVVideoHeightKey: 2 * h,
        AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: bitrate]])
    let ad = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: wi, sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: 2 * w, kCVPixelBufferHeightKey as String: 2 * h,
        kCVPixelBufferMetalCompatibilityKey as String: true, kCVPixelBufferIOSurfacePropertiesKey as String: [:]])
    writer.add(wi)
    guard reader.startReading(), writer.startWriting() else { die("reader/writer start failed") }
    var started = false
    var n = 0
    var gpu: [Double] = []
    let t0 = Date()
    while n < limit, let sb = ro.copyNextSampleBuffer() {
        guard let inPB = CMSampleBufferGetImageBuffer(sb) else { continue }
        let pts = CMSampleBufferGetPresentationTimeStamp(sb)
        if !started { writer.startSession(atSourceTime: pts); started = true }
        var outPB: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, ad.pixelBufferPool!, &outPB)
        gpu.append(try a4k.upscale(inPB, into: outPB!, compare: compare, divider: divider))
        while !wi.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 1_000_000) }
        ad.append(outPB!, withPresentationTime: pts)
        n += 1
    }
    wi.markAsFinished()
    await writer.finishWriting()
    guard writer.status == .completed else { die("writer: \(String(describing: writer.error))") }
    print(String(format: "%@ %@: %d frames %dx%d → %dx%d%@ in %.1f s wall; GPU %.2f ms/frame median → %@",
                 mode.rawValue, half ? "fp16" : "fp32", n, w, h, 2 * w, 2 * h, compare ? " (compare split)" : "",
                 Date().timeIntervalSince(t0), median(gpu) * 1000, outURL.path))

default:
    die("usage: anime4k image|batch|benchimage|video|bench … (see the header of main.swift)")
}
