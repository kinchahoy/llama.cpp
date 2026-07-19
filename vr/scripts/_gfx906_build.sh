#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_llama_root.sh"
ROOT_DIR="$(resolve_llama_root)"

AMDGPU_ARCH="${AMDGPU_ARCH:-gfx906}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
CONFIGURE_ONLY="${CONFIGURE_ONLY:-0}"
INSTALL_DIR="${INSTALL_DIR:-}"
GGML_VULKAN="${GGML_VULKAN:-OFF}"

gfx906_configure_rocm_environment() {
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
        echo "Source vr/scripts/setup-therock-env.sh, install ROCm, or set CC and CXX." >&2
        exit 2
    fi
}

gfx906_check_environment() {
    if [[ "${REQUIRE_GFX906:-1}" == "1" && "$AMDGPU_ARCH" != "gfx906" ]]; then
        echo "Error: this MI50/MI60 build requires AMDGPU_ARCH=gfx906." >&2
        echo "Set REQUIRE_GFX906=0 only for an intentional non-gfx906 build." >&2
        exit 2
    fi

    if ! command -v "$CC" >/dev/null 2>&1 || ! command -v "$CXX" >/dev/null 2>&1; then
        echo "Error: configured CC or CXX is not executable." >&2
        exit 2
    fi
}

gfx906_reset_stale_cmake_cache() {
    local build_dir="$1"
    local cached_arch
    local cached_cc
    local selected_cc

    selected_cc="$(command -v "$CC")"
    cached_cc="$(sed -n 's/^CMAKE_C_COMPILER:[^=]*=//p' "$build_dir/CMakeCache.txt" 2>/dev/null || true)"
    cached_arch="$(sed -n 's/^CMAKE_HIP_ARCHITECTURES:[^=]*=//p' "$build_dir/CMakeCache.txt" 2>/dev/null || true)"

    if [[ -n "$cached_cc" && "$cached_cc" != "$selected_cc" ]] ||
       [[ -n "$cached_arch" && "$cached_arch" != "$AMDGPU_ARCH" ]]; then
        echo "Resetting stale CMake configuration in $build_dir"
        cmake -E remove -f "$build_dir/CMakeCache.txt"
        cmake -E remove_directory "$build_dir/CMakeFiles"
    fi
}

gfx906_configure_tree() {
    local source_dir="$1"
    local build_dir="$2"
    local extra_args=()

    gfx906_reset_stale_cmake_cache "$build_dir"

    if [[ -n "${CMAKE_EXTRA_ARGS:-}" ]]; then
        read -r -a extra_args <<< "$CMAKE_EXTRA_ARGS"
    fi

    cmake -S "$source_dir" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_HIP_ARCHITECTURES="$AMDGPU_ARCH" \
        -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native -DNDEBUG" \
        -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -DNDEBUG" \
        -DCMAKE_HIP_FLAGS="-Wno-ignored-attributes -Wno-cuda-compat -Wno-unused-result" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        "-DCMAKE_INSTALL_RPATH=\$ORIGIN" \
        -DGGML_HIP=ON \
        -DGGML_HIP_GRAPHS=ON \
        -DGGML_HIP_NO_VMM=ON \
        -DGGML_CCACHE=ON \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=ON \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DGGML_VULKAN="$GGML_VULKAN" \
        -DBUILD_SHARED_LIBS=ON \
        "${extra_args[@]}"
}

gfx906_build_targets() {
    local build_dir="$1"
    shift

    if [[ "$CONFIGURE_ONLY" == "1" ]]; then
        echo "Configured only; build skipped."
        return
    fi

    cmake --build "$build_dir" -j"$BUILD_JOBS" --target "$@"
}

gfx906_install_tree() {
    local build_dir="$1"

    if [[ -n "$INSTALL_DIR" ]]; then
        cmake --install "$build_dir" --prefix "$INSTALL_DIR"
        echo "Installed relocatable tree: $INSTALL_DIR"
    fi
}

gfx906_print_binaries() {
    local build_dir="$1"
    shift

    echo
    echo "Build complete:"
    for target in "$@"; do
        if [[ -e "$build_dir/bin/$target" ]]; then
            echo "  $build_dir/bin/$target"
        fi
    done
    printf '%s\n' 'The build-tree binaries use CMAKE_INSTALL_RPATH=$ORIGIN for local shared libraries.'
}
