#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_gfx906_build.sh"

BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/gfx906-optimal}"
TARGETS_STRING="${TARGETS:-llama-cli llama-server llama-bench test-backend-ops}"
read -r -a TARGETS <<< "$TARGETS_STRING"

OPTIMAL_PATCH_SERIES="$ROOT_DIR/vr/patches/optimal-gfx906.series"
OPTIMAL_DEFINES=(
    GGML_CUDA_MMVQ_Q4K_GFX906_BRANCHLESS_SCALES
    GGML_CUDA_MMVQ_Q4K_GFX906_PAIRED_FUSION
    GGML_CUDA_MMVQ_Q8_0_GFX906_WIDE_VDR
    GGML_CUDA_MMVQ_Q8_0_GFX906_PAIRED_FUSION
    GGML_CUDA_MMQ_Q4K_GFX906_PRECOMPUTE
    GGML_CUDA_MMQ_Q4K_GFX906_MIN_BLOCKS_3
    GGML_CUDA_MMQ_Q6K_GFX906_MIN_BLOCKS_1
    GGML_CUDA_Q8_1_GFX906_DPP_REDUCE
    GGML_CUDA_FATTN_VEC_GFX906_DPP_Q8_1
    GGML_CUDA_FATTN_VEC_GFX906_KQ8
)

print_optimal_state() {
    cat <<EOF
Current gfx906 bigbang build includes:
  - Q8_0 selective rocBLAS dispatch for wide dense prefill.
  - Q8_0 MMVQ wide-VDR candidate for token generation.
  - Q8_0 MMVQ paired value+gate fused activation load.
  - Q4_K MMQ metadata precompute plus stride-9 LDS layout for prefill.
  - Q4_K MMQ min_blocks=3 occupancy experiment.
  - Q6_K MMQ min_blocks=1 launch bound for the accepted Q6_K prefill path.
  - Q4_K MMVQ branch-free scale decoding plus Q8_1 sum reuse for depth-0 TG.
  - Q4_K MMVQ paired value+gate fused activation load.
  - Q8_1 MMQ quantization DPP reductions.
  - FlashAttention vector Q8_1 quantization DPP reductions.
  - FlashAttention vector quantized KQ uses 8 lanes per dot.

Build:
  source: $ROOT_DIR
  build:  $BUILD_DIR
  arch:   $AMDGPU_ARCH
  CC:     $CC
  CXX:    $CXX
EOF

    if [[ -r "$OPTIMAL_PATCH_SERIES" ]]; then
        echo "Patch series:"
        echo "  $OPTIMAL_PATCH_SERIES"
    else
        echo "Patch series: not found; assuming the source tree is already patched."
    fi
}

verify_compile_gate() {
    local compile_commands="$BUILD_DIR/compile_commands.json"

    if [[ ! -r "$compile_commands" ]]; then
        echo "Error: missing $compile_commands; cannot verify gfx906 compile gates." >&2
        exit 2
    fi

    local define
    for define in "${OPTIMAL_DEFINES[@]}"; do
        if ! grep -q "$define" "$compile_commands"; then
            echo "Error: missing $define in compile_commands.json." >&2
            echo "The build would not include all optimal gfx906 specializations." >&2
            exit 2
        fi
    done
}

main() {
    gfx906_configure_rocm_environment
    gfx906_check_environment
    print_optimal_state
    gfx906_configure_tree "$ROOT_DIR" "$BUILD_DIR"
    verify_compile_gate
    gfx906_build_targets "$BUILD_DIR" "${TARGETS[@]}"
    gfx906_install_tree "$BUILD_DIR"
    gfx906_print_binaries "$BUILD_DIR" "${TARGETS[@]}"
}

main "$@"
