// The image API against the independent PyTorch reading (Validation/anime4k.py, references made by
// Validation/make_fixtures.py), and the properties the API promises: exact tiling, alpha, depth, colour space.

import CoreGraphics
import Foundation
import ImageIO
import XCTest
@testable import Anime4KMetal

final class Anime4KUpscalerTests: XCTestCase {

    override func setUpWithError() throws {
        try XCTSkipUnless(Anime4KUpscaler.isAvailable, "no Metal device")
    }

    // MARK: - Parity with the PyTorch reading

    /// Every mode, both fixtures, at the default precision (fp16 intermediates for 8-bit images) and at fp32:
    /// never more than one 8-bit level from the reference (the AB-R-0392 bar).
    func testParityWithPyTorchReading() throws {
        for fixture in ["anime", "poster"] {
            let input = try load(fixture)
            for mode in Anime4KMode.allCases {
                let reference = try rgba(load("\(fixture)_\(mode.rawValue)"))
                for precision in [Anime4KUpscaler.Precision.automatic, .full] {
                    let out = try Anime4KUpscaler(.init(mode: mode, precision: precision)).upscale(input)
                    XCTAssertEqual(out.width, 2 * input.width)
                    XCTAssertEqual(out.height, 2 * input.height)
                    XCTAssertEqual(out.bitsPerComponent, 8)
                    let got = try rgba(out)
                    var worst = 0, differing = 0, samples = 0
                    for i in 0 ..< got.count where i % 4 != 3 {
                        let d = abs(level8(got[i]) - level8(reference[i]))
                        worst = max(worst, d)
                        differing += d > 0 ? 1 : 0
                        samples += 1
                    }
                    print("parity \(fixture) \(mode.rawValue) \(precision): max \(worst) level(s), "
                          + String(format: "%.4f%% of samples differ", 100 * Double(differing) / Double(samples)))
                    XCTAssertLessThanOrEqual(worst, 1, "\(fixture) \(mode.rawValue) \(precision)")
                }
            }
        }
    }

    // MARK: - Tiling

    func testTiledEqualsUntiled() throws {
        let opaque = try mosaic(width: 410, height: 290, alpha: false)
        let transparent = try mosaic(width: 410, height: 290, alpha: true)
        for image in [opaque, transparent] {
            let whole = try Anime4KUpscaler(.init(maxTileSide: 1024)).upscale(image)
            let tiled = try Anime4KUpscaler(.init(maxTileSide: 160)).upscale(image)
            XCTAssertEqual(try bytes(whole), try bytes(tiled), "alpha: \(image.alphaInfo)")
        }
    }

    /// The smallest margin at which tiles are still exact, for the record; the default must be at or above it.
    func testDefaultMarginCoversReceptiveField() throws {
        let image = try mosaic(width: 410, height: 290, alpha: true)
        let whole = try bytes(Anime4KUpscaler(.init(maxTileSide: 1024)).upscale(image))
        var smallest: Int?
        for margin in stride(from: 48, through: 0, by: -4) {
            let tiled = try bytes(Anime4KUpscaler(.init(maxTileSide: 200, tileMargin: margin)).upscale(image))
            if tiled == whole { smallest = margin } else { break }
        }
        print("smallest exact margin (alpha path, bleed 16): \(smallest.map(String.init) ?? "none ≤ 48")")
        XCTAssertNotNil(smallest)
        XCTAssertLessThanOrEqual(smallest ?? .max, Anime4KUpscaler.Configuration().tileMargin)
    }

    func testCoresPartitionTheAxis() {
        for (length, tile, margin) in [(1000, 160, 48), (4097, 1024, 48), (160, 160, 48), (161, 160, 48), (90, 160, 48)] {
            let tileSide = min(length, tile)
            let cores = Anime4KUpscaler.cores(length: length, tile: tileSide, margin: margin)
            XCTAssertEqual(cores.first?.lowerBound, 0)
            XCTAssertEqual(cores.last?.upperBound, length)
            for (a, b) in zip(cores, cores.dropFirst()) { XCTAssertEqual(a.upperBound, b.lowerBound) }
            for core in cores {
                let o = Anime4KUpscaler.origin(of: core, length: length, tile: tileSide, margin: margin)
                XCTAssertGreaterThanOrEqual(o, 0)
                XCTAssertLessThanOrEqual(o + tileSide, length)
                XCTAssert(core.lowerBound == 0 || core.lowerBound - o >= margin, "\(length) \(core)")
                XCTAssert(core.upperBound == length || o + tileSide - core.upperBound >= margin, "\(length) \(core)")
            }
        }
    }

    // MARK: - Alpha

    /// A red disc on transparency keeps its colour to the edge: no dark fringe from the black that premultiplied
    /// transparency stores.
    func testTransparentEdgesKeepTheirColour() throws {
        let disc = try drawn(width: 96, height: 96, alpha: true) { ctx in
            ctx.setFillColor(CGColor(srgbRed: 0.9, green: 0.1, blue: 0.1, alpha: 1))
            ctx.fillEllipse(in: CGRect(x: 18, y: 18, width: 60, height: 60))
        }
        let out = try Anime4KUpscaler().upscale(disc)
        XCTAssertEqual(out.alphaInfo, .premultipliedLast)
        let px = try rgba(out)
        var fringe = 0, worst: Float = 0
        for i in stride(from: 0, to: px.count, by: 4) where px[i + 3] >= 0.25 {
            let a = px[i + 3]
            let (r, g, b) = (px[i] / a, px[i + 1] / a, px[i + 2] / a)
            let err = max(abs(r - 0.9), abs(g - 0.1), abs(b - 0.1))
            worst = max(worst, err)
            fringe += err > 0.08 ? 1 : 0
        }
        print("disc: worst colour error under alpha ≥ 0.25: \(worst), pixels off by > 0.08: \(fringe)")
        XCTAssertEqual(fringe, 0)
        XCTAssertEqual(px[(96 * 192 + 96) * 4 + 3], 1, accuracy: 1e-3)   // centre opaque
        XCTAssertEqual(px[3], 0, accuracy: 1e-3)                          // corner transparent
    }

    /// An alpha channel that is opaque everywhere takes the plain path: the colour equals the no-alpha result.
    func testOpaqueAlphaChannelTakesThePlainPath() throws {
        let rgb = try mosaic(width: 130, height: 97, alpha: false)
        let withAlpha = try redraw(rgb, alpha: true)
        let a = try rgba(Anime4KUpscaler().upscale(rgb))
        let b = try rgba(Anime4KUpscaler().upscale(withAlpha))
        XCTAssertEqual(a, b)
    }

    // MARK: - Depth

    /// A 16-bit image comes back 16-bit, through fp32 intermediates, within a fraction of an 8-bit level of the
    /// reference (the fixture's values are exact at 16 bits).
    func testSixteenBitRoundTrip() throws {
        let input = try redraw(try load("anime"), alpha: false, bits: 16)
        let out = try Anime4KUpscaler(.init(mode: .b)).upscale(input)
        XCTAssertEqual(out.bitsPerComponent, 16)
        let got = try rgba(out), ref = try rgba(load("anime_mode_b"))
        var worst: Float = 0
        for i in 0 ..< got.count where i % 4 != 3 { worst = max(worst, abs(got[i] - ref[i])) }
        print("16-bit vs reference: max |Δ| \(worst) (\(worst * 65535) 16-bit levels)")
        XCTAssertLessThan(worst, 0.5 / 255)
    }

    func testSixteenBitKeepsMoreThan256Levels() throws {
        let ramp = try drawn(width: 512, height: 32, alpha: false, bits: 16) { ctx in
            let g = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB),
                               colors: [CGColor(srgbRed: 0.2, green: 0.2, blue: 0.2, alpha: 1),
                                        CGColor(srgbRed: 0.35, green: 0.35, blue: 0.35, alpha: 1)] as CFArray,
                               locations: [0, 1])!
            ctx.drawLinearGradient(g, start: .zero, end: CGPoint(x: 512, y: 0), options: [])
        }
        let out = try Anime4KUpscaler().upscale(ramp)
        let px = try rgba(out)
        let row = (0 ..< out.width).map { px[(32 * out.width + $0) * 4] }
        let distinct = Set(row.map { Int(($0 * 65535).rounded()) }).count
        print("16-bit ramp: \(distinct) distinct levels across \(out.width) pixels (an 8-bit path would give ≤ 39)")
        XCTAssertGreaterThan(distinct, 256)
    }

    // MARK: - Colour space, size, memory

    func testKeepsItsRGBColourSpace() throws {
        let p3 = CGColorSpace(name: CGColorSpace.displayP3)!
        let image = try drawn(width: 40, height: 30, alpha: false, space: p3) { ctx in
            ctx.setFillColor(CGColor(colorSpace: p3, components: [0.1, 0.8, 0.3, 1])!)
            ctx.fill(CGRect(x: 10, y: 5, width: 20, height: 20))
        }
        let out = try Anime4KUpscaler().upscale(image)
        XCTAssertEqual(out.colorSpace?.name, CGColorSpace.displayP3)
    }

    func testDoublings() throws {
        let out = try Anime4KUpscaler().upscale(try load("poster"), doublings: 2)
        XCTAssertEqual(out.width, 512)
        XCTAssertEqual(out.height, 384)
    }

    func testMemoryEstimateIsCappedByTheTile() throws {
        let up = try Anime4KUpscaler()
        let small = up.memoryEstimate(width: 512, height: 512)
        let tileSized = up.memoryEstimate(width: 1024, height: 1024)
        let huge = up.memoryEstimate(width: 8192, height: 8192)
        XCTAssertGreaterThan(tileSized, small)
        XCTAssertEqual(huge, tileSized)
        print("memory estimate, mode B, alpha path: 512² \(small / 1_000_000) MB, 1024² and up \(tileSized / 1_000_000) MB")
    }

    // MARK: - Helpers

    private func load(_ name: String) throws -> CGImage {
        guard let url = Bundle.module.url(forResource: name, withExtension: "png", subdirectory: "Fixtures"),
              let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { throw Anime4KError.missingResource(name) }
        return img
    }

    /// RGBA floats, premultiplied, read without colour conversion (drawn in the image's own colour space).
    private func rgba(_ image: CGImage) throws -> [Float] {
        let bitmap = try Bitmap(drawing: image, deep: true)
        let p = bitmap.data.assumingMemoryBound(to: UInt16.self)
        let n = image.width * image.height * 4
        var out = [Float](repeating: 1, count: n)
        for i in 0 ..< n { out[i] = Float(UInt16(littleEndian: p[i])) / 65535 }
        if !bitmap.hasAlphaChannel { for i in stride(from: 3, to: n, by: 4) { out[i] = 1 } }
        return out
    }

    private func bytes(_ image: CGImage) throws -> Data {
        let bitmap = try Bitmap(drawing: image, deep: image.bitsPerComponent > 8)
        return Data(bytes: bitmap.data, count: bitmap.rowBytes * bitmap.height)
    }

    private func level8(_ v: Float) -> Int { Int((v * 255).rounded()) }

    private func drawn(width: Int, height: Int, alpha: Bool, bits: Int = 8, space: CGColorSpace? = nil,
                       _ draw: (CGContext) -> Void) throws -> CGImage {
        let info = (alpha ? CGImageAlphaInfo.premultipliedLast : .noneSkipLast).rawValue
            | (bits == 16 ? CGBitmapInfo.byteOrder16Little.rawValue : 0)
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: bits, bytesPerRow: 0,
                                  space: space ?? CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: info) else {
            throw Anime4KError.unsupportedImage("test context")
        }
        ctx.clear(CGRect(x: 0, y: 0, width: width, height: height))
        draw(ctx)
        guard let img = ctx.makeImage() else { throw Anime4KError.unsupportedImage("test image") }
        return img
    }

    private func redraw(_ image: CGImage, alpha: Bool, bits: Int = 8) throws -> CGImage {
        try drawn(width: image.width, height: image.height, alpha: alpha, bits: bits, space: image.colorSpace) { ctx in
            ctx.setBlendMode(.copy)
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        }
    }

    /// Both fixtures, enlarged and repeated over a canvas bigger than a test tile; with `alpha`, a ring of the
    /// canvas is transparent and an ellipse is half-transparent, so the bleed and the alpha enlargement cross tile
    /// borders too.
    private func mosaic(width: Int, height: Int, alpha: Bool) throws -> CGImage {
        let a = try load("anime"), p = try load("poster")
        return try drawn(width: width, height: height, alpha: alpha) { ctx in
            ctx.interpolationQuality = .high
            var x = 0, k = 0
            while x < width {
                var y = 0
                while y < height {
                    ctx.draw(k % 2 == 0 ? a : p, in: CGRect(x: x, y: y, width: 150, height: 113))
                    y += 113; k += 1
                }
                x += 150; k += 1
            }
            if alpha {
                ctx.setBlendMode(.clear)
                ctx.fill(CGRect(x: 0, y: 0, width: width, height: 30))
                ctx.fill(CGRect(x: 0, y: 0, width: 25, height: height))
                ctx.setBlendMode(.destinationIn)
                ctx.setFillColor(CGColor(gray: 0, alpha: 0.5))
                ctx.fillEllipse(in: CGRect(x: 120, y: 80, width: 200, height: 150))
            }
        }
    }
}
