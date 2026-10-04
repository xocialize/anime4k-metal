// anime4k — test driver for Anime4KMetal.
//
//   anime4k image <in.png> <out.png> [--mode b|a|upscale_vl|upscale_m] [--fp32]
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

let a4k: Anime4K
do { a4k = try Anime4K(mode: mode, halfPrecision: half) } catch { die("\(error)") }

switch args.first ?? "" {
case "image":
    guard args.count == 3 else { die("image <in.png> <out.png>") }
    let inPB = loadImage(args[1])
    let outPB = pixelBuffer(CVPixelBufferGetWidth(inPB) * 2, CVPixelBufferGetHeight(inPB) * 2)
    let t = try a4k.upscale(inPB, into: outPB)
    savePNG(outPB, args[2])
    print(String(format: "%@ (%d passes, %@): %dx%d → %dx%d in %.2f ms GPU (first frame, includes pipeline warm-up)",
                 mode.rawValue, a4k.passCount, half ? "fp16" : "fp32", CVPixelBufferGetWidth(inPB),
                 CVPixelBufferGetHeight(inPB), CVPixelBufferGetWidth(outPB), CVPixelBufferGetHeight(outPB), t * 1000))

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
    die("usage: anime4k image|video|bench … (see the header of main.swift)")
}
