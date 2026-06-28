#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_gfx906_build.sh"

BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/gfx906-vanilla}"
TARGETS_STRING="${TARGETS:-llama-cli llama-server llama-bench}"
read -r -a TARGETS <<< "$TARGETS_STRING"

main() {
    gfx906_configure_rocm_environment
    gfx906_check_environment

    echo "Vanilla gfx906 build:"
    echo "  source: $ROOT_DIR"
    echo "  build:  $BUILD_DIR"
    echo "  arch:   $AMDGPU_ARCH"
    echo "  CC:     $CC"
    echo "  CXX:    $CXX"

    gfx906_configure_tree "$ROOT_DIR" "$BUILD_DIR"
    gfx906_build_targets "$BUILD_DIR" "${TARGETS[@]}"
    gfx906_install_tree "$BUILD_DIR"
    gfx906_print_binaries "$BUILD_DIR" "${TARGETS[@]}"
}

main "$@"
