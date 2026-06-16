#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAINLINE_REF="${MAINLINE_REF:-origin/master}"
MAINLINE_SOURCE="${MAINLINE_SOURCE:-$ROOT_DIR/build/.mainline-src}"
MAINLINE_BUILD="${MAINLINE_BUILD:-$ROOT_DIR/build/mainline}"
CONTROL_PATCH="${CONTROL_PATCH:-$ROOT_DIR/gfx906-q8-rocblas-dispatch.patch}"
CANDIDATE_BUILD="${CANDIDATE_BUILD:-$ROOT_DIR/build/gfx906-2026-06}"
AMDGPU_ARCH="${AMDGPU_ARCH:-gfx906}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
CONFIGURE_ONLY="${CONFIGURE_ONLY:-0}"
TARGETS=(llama-bench test-backend-ops)
if [[ "${BUILD_FULL:-0}" == "1" ]]; then
    TARGETS+=(llama-cli llama-server)
fi

check_environment() {
    if [[ -z "${CC:-}" || -z "${CXX:-}" ]]; then
        echo "Error: CC and CXX are not set." >&2
        echo "For TheRock, run: source scripts/setup-therock-env.sh" >&2
        exit 2
    fi
    if ! command -v "$CC" >/dev/null 2>&1 || ! command -v "$CXX" >/dev/null 2>&1; then
        echo "Error: configured CC or CXX is not executable." >&2
        exit 2
    fi
}

prepare_mainline_source() {
    local current_commit
    local target_commit
    local top_level

    target_commit="$(git -C "$ROOT_DIR" rev-parse --verify "$MAINLINE_REF^{commit}")"
    if [[ ! -e "$MAINLINE_SOURCE" ]]; then
        mkdir -p "$(dirname "$MAINLINE_SOURCE")"
        git -C "$ROOT_DIR" worktree add --detach "$MAINLINE_SOURCE" "$MAINLINE_REF"
    fi

    top_level="$(git -C "$MAINLINE_SOURCE" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ "$top_level" != "$MAINLINE_SOURCE" ]]; then
        echo "Error: $MAINLINE_SOURCE exists but is not the expected git worktree." >&2
        exit 2
    fi
    if [[ -n "$(git -C "$MAINLINE_SOURCE" status --short)" ]]; then
        if [[ -n "$CONTROL_PATCH" ]] && git -C "$MAINLINE_SOURCE" apply -R --check "$CONTROL_PATCH" >/dev/null 2>&1; then
            git -C "$MAINLINE_SOURCE" apply -R "$CONTROL_PATCH"
        else
            echo "Error: mainline worktree has unexpected changes: $MAINLINE_SOURCE" >&2
            exit 2
        fi
    fi

    current_commit="$(git -C "$MAINLINE_SOURCE" rev-parse HEAD)"
    if [[ "$current_commit" != "$target_commit" ]]; then
        git -C "$MAINLINE_SOURCE" checkout --detach "$target_commit"
    fi

    if [[ -n "$CONTROL_PATCH" ]]; then
        git -C "$MAINLINE_SOURCE" apply --check "$CONTROL_PATCH"
        git -C "$MAINLINE_SOURCE" apply "$CONTROL_PATCH"
    fi
}

configure() {
    local source_dir="$1"
    local build_dir="$2"
    local cached_cc
    local selected_cc

    selected_cc="$(command -v "$CC")"
    cached_cc="$(sed -n 's/^CMAKE_C_COMPILER:[^=]*=//p' "$build_dir/CMakeCache.txt" 2>/dev/null || true)"
    if [[ -n "$cached_cc" && "$cached_cc" != "$selected_cc" ]]; then
        echo "Resetting stale CMake configuration in $build_dir"
        cmake -E remove -f "$build_dir/CMakeCache.txt"
        cmake -E remove_directory "$build_dir/CMakeFiles"
    fi

    cmake -S "$source_dir" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_HIP_ARCHITECTURES="$AMDGPU_ARCH" \
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
    if [[ "$CONFIGURE_ONLY" == "1" ]]; then
        return
    fi
    echo "Building $label: ${TARGETS[*]}"
    cmake --build "$build_dir" -j"$BUILD_JOBS" --target "${TARGETS[@]}"
}

main() {
    check_environment
    prepare_mainline_source

    echo "Mainline ref: $(git -C "$MAINLINE_SOURCE" rev-parse --short=12 HEAD)"
    echo "Control patch: ${CONTROL_PATCH:-none}"
    echo "Candidate ref: $(git -C "$ROOT_DIR" rev-parse --short=12 HEAD)"
    echo "C compiler: $CC"
    echo "C++ compiler: $CXX"
    echo "GPU architecture: $AMDGPU_ARCH"

    build_tree mainline "$MAINLINE_SOURCE" "$MAINLINE_BUILD"
    build_tree gfx906-2026-06 "$ROOT_DIR" "$CANDIDATE_BUILD"

    if [[ "$CONFIGURE_ONLY" != "1" ]]; then
        echo
        echo "Comparison binaries:"
        for target in "${TARGETS[@]}"; do
            echo "  $MAINLINE_BUILD/bin/$target"
            echo "  $CANDIDATE_BUILD/bin/$target"
        done
    fi
}

main "$@"
