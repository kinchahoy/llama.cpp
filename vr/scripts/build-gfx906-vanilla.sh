#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
Error: the current source tree contains gfx906 dispatch changes, so it cannot
produce a vanilla control. Use build-gfx906-comparison.sh for the pinned
571d0d540 control, or build-standalone-raw-checkout.sh in a clean checkout.
EOF
exit 2
