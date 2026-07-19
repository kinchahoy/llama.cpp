#!/usr/bin/env bash
set -euo pipefail

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$PATCH_DIR/../.." && pwd)"

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <series-file>" >&2
    exit 2
fi

SERIES="$1"
if [[ "$SERIES" != /* ]]; then
    SERIES="$PATCH_DIR/$SERIES"
fi
if [[ ! -f "$SERIES" ]]; then
    echo "series file not found: $SERIES" >&2
    exit 2
fi

patches=()
while IFS= read -r entry || [[ -n "$entry" ]]; do
    case "$entry" in
        ""|\#*) continue ;;
    esac
    patch="$PATCH_DIR/$entry"
    if [[ ! -f "$patch" ]]; then
        echo "patch listed by series does not exist: $patch" >&2
        exit 2
    fi
    patches+=("$patch")
done < "$SERIES"

if [[ ${#patches[@]} -eq 0 ]]; then
    echo "series contains no patches: $SERIES" >&2
    exit 2
fi

git -C "$ROOT_DIR" apply "${patches[@]}"
echo "Applied $(basename "$SERIES")"
