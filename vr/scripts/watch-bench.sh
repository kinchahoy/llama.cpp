#!/usr/bin/env bash
# Watch a bench-head.sh output directory without guessing process state.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_llama_root.sh"
ROOT="$(resolve_llama_root)"

OUT="${1:-${OUT:-}}"
REFRESH="${REFRESH:-3}"
ONCE="${ONCE:-0}"

if [[ -z "$OUT" ]]; then
    OUT="$(ls -dt "$ROOT"/vr/bench-results/gfx906-model-* 2>/dev/null | head -1)"
fi
[[ -n "$OUT" && -d "$OUT" ]] || {
    echo "Error: pass a bench output directory." >&2
    exit 2
}

status_value() {
    local key="$1"
    sed -n "s/^${key}=//p" "$OUT/status" 2>/dev/null | head -1
}

render() {
    local state current completed total newest_error
    state="$(status_value state)"
    current="$(status_value current)"
    completed="$(status_value completed)"
    total="$(status_value total)"
    newest_error="$(ls -t "$OUT"/*/*.jsonl.err 2>/dev/null | head -1)"

    [[ -t 1 ]] && clear 2>/dev/null || true
    printf 'bench=%s state=%s progress=%s/%s\n' \
        "$OUT" "${state:-unknown}" "${completed:-0}" "${total:-?}"
    printf 'current=%s\n\n' "${current:-unknown}"
    if [[ -n "$newest_error" ]]; then
        printf 'latest=%s\n' "$newest_error"
        tail -n 5 "$newest_error" 2>/dev/null | sed 's/^/  /'
        printf '\n'
    fi
    python3 "$SCRIPT_DIR/_bench_table.py" "$OUT"
}

if [[ "$ONCE" == "1" ]]; then
    render
    exit 0
fi

trap 'echo; exit 0' INT
while true; do
    render
    if [[ ! -r "$OUT/status" ]]; then
        exit 0
    fi
    state="$(status_value state)"
    if [[ "$state" == "complete" || "$state" == "failed" ]]; then
        exit 0
    fi
    sleep "$REFRESH"
done
