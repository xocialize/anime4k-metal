// swift-tools-version: 6.0
import PackageDescription

// anime4k-metal — bloc97/Anime4K (MIT) as Metal compute shaders: an image upscaler (`Anime4KUpscaler`, CGImage in,
// CGImage at twice the size out) and the shader chain on textures and pixel buffers (`Anime4K`), for drawings, line
// art and graphics. The shaders are generated from the GLSL by Generator/ and validated against an independent
// PyTorch reading (Validation/). Metal only: no CPU fallback, no vendored binaries.
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
        .testTarget(
            name: "Anime4KMetalTests",
            dependencies: ["Anime4KMetal"],
            resources: [.copy("Fixtures")]
        ),
        .executableTarget(
            name: "anime4k",
            dependencies: ["Anime4KMetal"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
