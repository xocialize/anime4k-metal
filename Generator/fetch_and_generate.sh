#!/bin/zsh
# Fetch Anime4K's GLSL (bloc97/Anime4K @ 7684e958, MIT) and regenerate the Metal shader libraries the package bundles.
# Each mode = an ordered chain of mpv hook files, transpiled into one .metal + .plan.json by anime4k_msl.py.
set -euo pipefail
here=${0:A:h}
rev=7684e9586f8dcc738af08a1cdceb024cc184f426
glsl=$here/glsl
out=$here/../Sources/Anime4KMetal/Shaders
mkdir -p $glsl $out
for f in Upscale/Anime4K_Upscale_CNN_x2_M Upscale/Anime4K_Upscale_CNN_x2_VL \
         Restore/Anime4K_Restore_CNN_VL Restore/Anime4K_Restore_CNN_Soft_VL; do
  curl -fsSL -o $glsl/${f:t}.glsl "https://raw.githubusercontent.com/bloc97/Anime4K/$rev/glsl/$f.glsl"
done
shasum -a 256 $glsl/*.glsl
py=${PYTHON:-python3}
$py $here/anime4k_msl.py $out/mode_b.metal      $glsl/Anime4K_Restore_CNN_Soft_VL.glsl $glsl/Anime4K_Upscale_CNN_x2_VL.glsl
$py $here/anime4k_msl.py $out/mode_a.metal      $glsl/Anime4K_Restore_CNN_VL.glsl      $glsl/Anime4K_Upscale_CNN_x2_VL.glsl
$py $here/anime4k_msl.py $out/upscale_vl.metal  $glsl/Anime4K_Upscale_CNN_x2_VL.glsl
$py $here/anime4k_msl.py $out/upscale_m.metal   $glsl/Anime4K_Upscale_CNN_x2_M.glsl
for f in $out/*.metal; do xcrun -sdk macosx metal -c $f -o /dev/null 2>&1 | grep -v "unused variable" | grep -E "error" && exit 1; done
# SwiftPM must not compile these (four libraries share kernel names): ship them as source, the runtime compiles one per mode
for f in $out/*.metal; do mv $f ${f%.metal}.msl; mv $f.plan.json ${f%.metal}.plan.json; done
echo "generated + compile-checked: $(ls $out)"
