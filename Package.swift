// swift-tools-version: 6.0
import PackageDescription

// anime4k-metal — bloc97/Anime4K (MIT) as Metal compute shaders, for testing Anime4K mode B as a playback-time /
// web upscaler (VIDEO2X-EVAL §3.2). The shaders are generated from the GLSL by Generator/ and validated against an
// independent PyTorch reading (Validation/). Status: testing — not a product package yet.
let package = Package(
    name: "anime4k-metal",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "Anime4KMetal", targets: ["Anime4KMetal"]),
        .executable(name: "anime4k", targets: ["anime4k"]),
    ],
    targets: [
        .target(
            name: "Anime4KMetal",
            resources: [.copy("Shaders")],
            linkerSettings: [.linkedFramework("Metal"), .linkedFramework("MetalPerformanceShaders"),
                             .linkedFramework("CoreVideo")]
        ),
        .executableTarget(
            name: "anime4k",
            dependencies: ["Anime4KMetal"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
