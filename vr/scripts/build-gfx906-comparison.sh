#!/usr/bin/env bash

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
  vr/scripts/build-gfx906-comparison.sh --dry-run
  vr/scripts/build-gfx906-comparison.sh

Defaults:
  MAINLINE_REF=571d0d540
  CANDIDATE_PROFILE=combined
  TARGETS=test-backend-ops
EOF
        exit 0
        ;;
    *)
        echo "Error: unknown argument: $1" >&2
        exit 2
        ;;
esac

MAINLINE_REF="${MAINLINE_REF:-571d0d540}"
MAINLINE_SOURCE="${MAINLINE_SOURCE:-$ROOT_DIR/build/.mainline-src}"
MAINLINE_BUILD="${MAINLINE_BUILD:-$ROOT_DIR/build/head-control}"
CONTROL_PATCH="${CONTROL_PATCH:-}"
CANDIDATE_BUILD="${CANDIDATE_BUILD:-$ROOT_DIR/build/head-candidate}"
CANDIDATE_PROFILE="${CANDIDATE_PROFILE:-combined}"
TARGETS_STRING="${TARGETS:-test-backend-ops}"
read -r -a TARGETS <<< "$TARGETS_STRING"
(( ${#TARGETS[@]} > 0 )) || {
    echo "Error: TARGETS must not be empty." >&2
    exit 2
}
case "$CANDIDATE_PROFILE" in
    none|common|q4|q6|combined)
        ;;
    *)
        echo "Error: invalid CANDIDATE_PROFILE=$CANDIDATE_PROFILE" >&2
        exit 2
        ;;
esac

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

build_tree() {
    local label="$1"
    local source_dir="$2"
    local build_dir="$3"
    local profile="$4"
    local saved_extra_args="${CMAKE_EXTRA_ARGS:-}"

    echo "Configuring $label from $source_dir"
    if [[ "$label" == "candidate" ]]; then
        CMAKE_EXTRA_ARGS="${saved_extra_args:+$saved_extra_args }-DGGML_HIP_GFX906_PROFILE=$profile"
    fi
    gfx906_configure_tree "$source_dir" "$build_dir"
    CMAKE_EXTRA_ARGS="$saved_extra_args"
    python3 "$SCRIPT_DIR/check-gfx906-defines.py" "$build_dir" "$profile"
    if [[ "$CONFIGURE_ONLY" != "1" ]]; then
        echo "Building $label: ${TARGETS[*]}"
    fi
    gfx906_build_targets "$build_dir" "${TARGETS[@]}"
}

main() {
    if [[ "$ACTION" == "dry-run" ]]; then
        echo "mainline_ref=$MAINLINE_REF"
        echo "mainline_source=$MAINLINE_SOURCE"
        echo "mainline_build=$MAINLINE_BUILD"
        echo "candidate_source=$ROOT_DIR"
        echo "candidate_build=$CANDIDATE_BUILD"
        echo "candidate_profile=$CANDIDATE_PROFILE"
        echo "targets=${TARGETS[*]}"
        exit 0
    fi

    gfx906_configure_rocm_environment
    gfx906_check_environment
    prepare_mainline_source

    echo "Mainline ref: $(git -C "$MAINLINE_SOURCE" rev-parse --short=12 HEAD)"
    echo "Control patch: ${CONTROL_PATCH:-none}"
    echo "Candidate ref: $(git -C "$ROOT_DIR" rev-parse --short=12 HEAD)"
    echo "Candidate profile: $CANDIDATE_PROFILE"
    echo "C compiler: $CC"
    echo "C++ compiler: $CXX"
    echo "GPU architecture: $AMDGPU_ARCH"

    build_tree control "$MAINLINE_SOURCE" "$MAINLINE_BUILD" none
    build_tree candidate "$ROOT_DIR" "$CANDIDATE_BUILD" "$CANDIDATE_PROFILE"

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
