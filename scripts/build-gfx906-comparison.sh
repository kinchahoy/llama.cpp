#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAINLINE_REF="${MAINLINE_REF:-origin/master}"
MAINLINE_SOURCE="${MAINLINE_SOURCE:-$ROOT_DIR/build/.mainline-src}"
MAINLINE_BUILD="${MAINLINE_BUILD:-$ROOT_DIR/build/mainline}"
CANDIDATE_BUILD="${CANDIDATE_BUILD:-$ROOT_DIR/build/gfx906-2026-06}"
AMDGPU_ARCH="${AMDGPU_ARCH:-gfx906}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
TARGETS=(llama-bench test-backend-ops llama-cli llama-server)

setup_rocm() {
    if ! command -v rocm-sdk >/dev/null 2>&1; then
        echo "Error: rocm-sdk not found. Activate the TheRock environment first." >&2
        exit 2
    fi

    ROCM_PATH="$(rocm-sdk path --root)"
    export ROCM_PATH
    export HIP_PATH="$ROCM_PATH"
    export HIP_PLATFORM=amd
    export CMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    export PATH="$(rocm-sdk path --bin):$PATH"

    if command -v amdclang >/dev/null 2>&1 && command -v amdclang++ >/dev/null 2>&1; then
        CC="$(command -v amdclang)"
        CXX="$(command -v amdclang++)"
    elif [[ -x "$ROCM_PATH/llvm/bin/clang" && -x "$ROCM_PATH/llvm/bin/clang++" ]]; then
        CC="$ROCM_PATH/llvm/bin/clang"
        CXX="$ROCM_PATH/llvm/bin/clang++"
    else
        echo "Error: ROCm clang and clang++ were not found." >&2
        exit 2
    fi

    export CC CXX
    if [[ -d "$ROCM_PATH/llvm/bin" ]]; then
        export HIP_CLANG_PATH="$ROCM_PATH/llvm/bin"
    fi
}

prepare_mainline_source() {
    local top_level

    git -C "$ROOT_DIR" rev-parse --verify "$MAINLINE_REF^{commit}" >/dev/null
    if [[ ! -e "$MAINLINE_SOURCE" ]]; then
        mkdir -p "$(dirname "$MAINLINE_SOURCE")"
        git -C "$ROOT_DIR" worktree add --detach "$MAINLINE_SOURCE" "$MAINLINE_REF"
        return
    fi

    top_level="$(git -C "$MAINLINE_SOURCE" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ "$top_level" != "$MAINLINE_SOURCE" ]]; then
        echo "Error: $MAINLINE_SOURCE exists but is not the expected git worktree." >&2
        exit 2
    fi
    if [[ -n "$(git -C "$MAINLINE_SOURCE" status --short)" ]]; then
        echo "Error: mainline worktree is dirty: $MAINLINE_SOURCE" >&2
        exit 2
    fi

    git -C "$MAINLINE_SOURCE" checkout --detach "$MAINLINE_REF"
}

configure() {
    local source_dir="$1"
    local build_dir="$2"

    cmake -S "$source_dir" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_HIP_ARCHITECTURES="$AMDGPU_ARCH" \
        -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native -DNDEBUG" \
        -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -DNDEBUG" \
        -DCMAKE_HIP_FLAGS="-Wno-ignored-attributes -Wno-cuda-compat -Wno-unused-result" \
        -DGGML_HIP=ON \
        -DGGML_HIP_GRAPHS=ON \
        -DGGML_HIP_NO_VMM=ON \
        -DLLAMA_BUILD_TESTS=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DGGML_VULKAN=ON \
        -DBUILD_SHARED_LIBS=ON
}

build_tree() {
    local label="$1"
    local source_dir="$2"
    local build_dir="$3"

    echo "Configuring $label from $source_dir"
    configure "$source_dir" "$build_dir"
    echo "Building $label: ${TARGETS[*]}"
    cmake --build "$build_dir" -j"$BUILD_JOBS" --target "${TARGETS[@]}"
}

main() {
    setup_rocm
    prepare_mainline_source

    echo "Mainline ref: $(git -C "$MAINLINE_SOURCE" rev-parse --short=12 HEAD)"
    echo "Candidate ref: $(git -C "$ROOT_DIR" rev-parse --short=12 HEAD)"
    echo "ROCm path: $ROCM_PATH"
    echo "GPU architecture: $AMDGPU_ARCH"

    build_tree mainline "$MAINLINE_SOURCE" "$MAINLINE_BUILD"
    build_tree gfx906-2026-06 "$ROOT_DIR" "$CANDIDATE_BUILD"

    echo
    echo "Comparison binaries:"
    for target in "${TARGETS[@]}"; do
        echo "  $MAINLINE_BUILD/bin/$target"
        echo "  $CANDIDATE_BUILD/bin/$target"
    done
}

main "$@"
