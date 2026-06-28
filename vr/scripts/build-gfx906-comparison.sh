#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_gfx906_build.sh"
MAINLINE_REF="${MAINLINE_REF:-origin/master}"
MAINLINE_SOURCE="${MAINLINE_SOURCE:-$ROOT_DIR/build/.mainline-src}"
MAINLINE_BUILD="${MAINLINE_BUILD:-$ROOT_DIR/build/mainline}"
CONTROL_PATCH="${CONTROL_PATCH:-}"
CANDIDATE_BUILD="${CANDIDATE_BUILD:-$ROOT_DIR/build/gfx906-2026-06}"
TARGETS_STRING="${TARGETS:-llama-bench test-backend-ops}"
if [[ "${BUILD_FULL:-0}" == "1" && -z "${TARGETS:-}" ]]; then
    TARGETS_STRING="llama-bench test-backend-ops llama-cli llama-server"
fi
read -r -a TARGETS <<< "$TARGETS_STRING"

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

    echo "Configuring $label from $source_dir"
    gfx906_configure_tree "$source_dir" "$build_dir"
    if [[ "$CONFIGURE_ONLY" != "1" ]]; then
        echo "Building $label: ${TARGETS[*]}"
    fi
    gfx906_build_targets "$build_dir" "${TARGETS[@]}"
}

main() {
    gfx906_configure_rocm_environment
    gfx906_check_environment
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
