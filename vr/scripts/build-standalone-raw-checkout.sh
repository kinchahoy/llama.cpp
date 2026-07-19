#!/usr/bin/env bash
#
# build-standalone-raw-checkout.sh
#
# WHAT THIS IS FOR:
#   A copy-pasteable build script for a *separate, plain* llama.cpp checkout
#   (no vr/ tree, no gfx906 patches) used purely as the "vanilla upstream"
#   comparison baseline when benchmarking this repo's gfx906 optimizations.
#
# HOW TO USE IT:
#   1. Get a clean llama.cpp checkout somewhere else, e.g.:
#        git clone https://github.com/ggml-org/llama.cpp ~/infer/some-baseline-checkout
#      (or reuse an existing throwaway clone).
#   2. Copy this script into the root of that checkout:
#        cp ~/infer/llama.cpp/vr/scripts/build-standalone-raw-checkout.sh \
#           ~/infer/some-baseline-checkout/build-gfx906.sh
#   3. From that checkout's root, run it:
#        cd ~/infer/some-baseline-checkout
#        ./build-gfx906.sh
#      This builds llama-cli, llama-server, llama-bench into ./build/gfx906
#      using the same CMake flags as this repo's build-gfx906-vanilla.sh, so
#      the two trees are comparable.
#   4. Benchmark it against this repo's optimized build with bench-head.sh /
#      _bench_table.py, pointing at:
#        ~/infer/some-baseline-checkout/build/gfx906/bin/llama-bench
#
# NOTE: This script is standalone (no _gfx906_build.sh / _llama_root.sh
# dependency) because it's meant to live outside this repo, in the raw
# checkout it builds.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/gfx906}"
AMDGPU_ARCH="${AMDGPU_ARCH:-gfx906}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
TARGETS_STRING="${TARGETS:-llama-cli llama-server llama-bench}"
read -r -a TARGETS <<< "$TARGETS_STRING"

configure_rocm_environment() {
    if [[ -n "${CC:-}" && -n "${CXX:-}" ]]; then
        return
    fi

    if command -v rocm-sdk >/dev/null 2>&1; then
        local rocm_path
        rocm_path="$(rocm-sdk path --root)"

        export ROCM_PATH="$rocm_path"
        export HIP_PATH="$rocm_path"
        export HIP_PLATFORM=amd
        export CMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
        export PATH="$(rocm-sdk path --bin):$PATH"

        if command -v amdclang >/dev/null 2>&1 && command -v amdclang++ >/dev/null 2>&1; then
            export CC="$(command -v amdclang)"
            export CXX="$(command -v amdclang++)"
        elif [[ -x "$rocm_path/llvm/bin/clang" && -x "$rocm_path/llvm/bin/clang++" ]]; then
            export CC="$rocm_path/llvm/bin/clang"
            export CXX="$rocm_path/llvm/bin/clang++"
        fi

        if [[ -d "$rocm_path/llvm/bin" ]]; then
            export HIP_CLANG_PATH="$rocm_path/llvm/bin"
        fi
    elif [[ -x /opt/rocm/llvm/bin/clang && -x /opt/rocm/llvm/bin/clang++ ]]; then
        export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
        export HIP_PATH="$ROCM_PATH"
        export HIP_PLATFORM=amd
        export HIP_CLANG_PATH="$ROCM_PATH/llvm/bin"
        export CC="$ROCM_PATH/llvm/bin/clang"
        export CXX="$ROCM_PATH/llvm/bin/clang++"
        export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"
        export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

    if [[ -z "${CC:-}" || -z "${CXX:-}" ]]; then
        echo "Error: could not find ROCm clang compilers." >&2
        echo "Activate the ROCm/TheRock venv (e.g. ~/amd-clean/therock-venv) or set CC and CXX." >&2
        exit 2
    fi
}

main() {
    configure_rocm_environment

    echo "Vanilla llama.cpp gfx906 build (comparison baseline):"
    echo "  source: $ROOT_DIR"
    echo "  build:  $BUILD_DIR"
    echo "  arch:   $AMDGPU_ARCH"
    echo "  CC:     $CC"
    echo "  CXX:    $CXX"

    cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_HIP_ARCHITECTURES="$AMDGPU_ARCH" \
        -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native -DNDEBUG" \
        -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -DNDEBUG" \
        -DCMAKE_HIP_FLAGS="-Wno-ignored-attributes -Wno-cuda-compat -Wno-unused-result" \
        -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        "-DCMAKE_INSTALL_RPATH=\$ORIGIN" \
        -DGGML_HIP=ON \
        -DGGML_HIP_GRAPHS=ON \
        -DGGML_HIP_NO_VMM=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DGGML_VULKAN=OFF \
        -DBUILD_SHARED_LIBS=ON

    cmake --build "$BUILD_DIR" -j"$BUILD_JOBS" --target "${TARGETS[@]}"

    echo
    echo "Build complete:"
    for target in "${TARGETS[@]}"; do
        if [[ -e "$BUILD_DIR/bin/$target" ]]; then
            echo "  $BUILD_DIR/bin/$target"
        fi
    done
}

main "$@"
