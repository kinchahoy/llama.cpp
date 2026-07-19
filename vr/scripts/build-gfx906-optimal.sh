#!/usr/bin/env bash
# Historical compatibility name for the retained/common profile.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PROFILE=common
export BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/gfx906-ablate-common}"
export TARGETS="${TARGETS:-test-backend-ops}"

echo "Note: build-gfx906-optimal.sh is a compatibility alias for PROFILE=common."
exec bash "$SCRIPT_DIR/build-gfx906-variant.sh" "$@"
