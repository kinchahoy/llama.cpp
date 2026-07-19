#!/usr/bin/env bash
# Build one exact private gfx906 profile with only the requested targets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_gfx906_build.sh"

ACTION=run
case "${1:-}" in
    "")
        ;;
    --dry-run)
        ACTION=dry-run
        ;;
    -h|--help)
        cat <<'EOF'
Usage:
  PROFILE=common vr/scripts/build-gfx906-variant.sh --dry-run
  PROFILE=common vr/scripts/build-gfx906-variant.sh

Profiles: none, common, q4, q6, combined
Defaults: BUILD_DIR=build/gfx906-ablate-<profile>
          TARGETS=test-backend-ops
EOF
        exit 0
        ;;
    *)
        echo "Error: unknown argument: $1" >&2
        exit 2
        ;;
esac

PROFILE="${PROFILE:-common}"
case "$PROFILE" in
    none|common|q4|q6|combined)
        ;;
    *)
        echo "Error: PROFILE must be none, common, q4, q6, or combined." >&2
        exit 2
        ;;
esac

BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/gfx906-ablate-$PROFILE}"
TARGETS_STRING="${TARGETS:-test-backend-ops}"
read -r -a TARGETS <<< "$TARGETS_STRING"
(( ${#TARGETS[@]} > 0 )) || {
    echo "Error: TARGETS must not be empty." >&2
    exit 2
}

if [[ -n "${CMAKE_EXTRA_ARGS:-}" ]]; then
    CMAKE_EXTRA_ARGS="$CMAKE_EXTRA_ARGS -DGGML_HIP_GFX906_PROFILE=$PROFILE"
else
    CMAKE_EXTRA_ARGS="-DGGML_HIP_GFX906_PROFILE=$PROFILE"
fi

if [[ "$ACTION" == "dry-run" ]]; then
    echo "profile=$PROFILE"
    echo "build=$BUILD_DIR"
    echo "targets=${TARGETS[*]}"
    echo "cmake_extra_args=$CMAKE_EXTRA_ARGS"
    exit 0
fi

main() {
    gfx906_configure_rocm_environment
    gfx906_check_environment

    echo "gfx906 variant:"
    echo "  profile: $PROFILE"
    echo "  source:  $ROOT_DIR"
    echo "  build:   $BUILD_DIR"
    echo "  targets: ${TARGETS[*]}"

    gfx906_configure_tree "$ROOT_DIR" "$BUILD_DIR"
    python3 "$SCRIPT_DIR/check-gfx906-defines.py" "$BUILD_DIR" "$PROFILE"
    gfx906_build_targets "$BUILD_DIR" "${TARGETS[@]}"
    gfx906_print_binaries "$BUILD_DIR" "${TARGETS[@]}"
}

main "$@"
