#!/usr/bin/env bash
# Live dashboard for the head-vs-head+patches benchmark (or any llama-bench
# A/B run that writes <OUT>/<build>/<config>.jsonl with -o jsonl).
#
# Auto-follows the newest .err (current benchmark heartbeat) and reprints the
# parsed control-vs-candidate results table. Refreshes until the driver writes
# "BENCH DONE" to its main log, then prints the final table and exits.
#
# Usage:
#   vr/scripts/watch-bench.sh [OUT_DIR]
# Env overrides:
#   OUT       results dir   (default: newest vr/bench-results/gfx906-head-*)
#   MAIN_LOG  driver log    (default: /tmp/vr-bench-main.log)
#   REFRESH   seconds       (default: 5)
#   ONCE=1    render once and exit (no loop)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TABLE_PY="$ROOT/vr/scripts/_bench_table.py"

OUT="${1:-${OUT:-}}"
if [[ -z "$OUT" ]]; then
    OUT="$(ls -dt "$ROOT"/vr/bench-results/gfx906-head-* 2>/dev/null | head -1)"
fi
MAIN_LOG="${MAIN_LOG:-/tmp/vr-bench-main.log}"
REFRESH="${REFRESH:-5}"

if [[ -z "$OUT" ]]; then
    echo "No results dir found. Pass one: watch-bench.sh <OUT_DIR>" >&2
    exit 2
fi

render() {
    local newest_err
    newest_err="$(ls -t "$OUT"/*/*.jsonl.err 2>/dev/null | head -1)"
    clear 2>/dev/null
    echo "=== bench watch  ($(date +%T))  OUT=$OUT"
    if pgrep -f 'vr-bench\.sh|bench-head\.sh' >/dev/null 2>&1; then
        echo "driver: RUNNING (pid $(pgrep -f 'vr-bench\.sh|bench-head\.sh' | head -1))"
    elif grep -q "BENCH DONE" "$MAIN_LOG" 2>/dev/null; then
        echo "driver: FINISHED"
    else
        echo "driver: NOT RUNNING (no DONE marker - may have died; check $MAIN_LOG)"
    fi
    echo
    echo "--- driver log (last 2) ---"
    tail -n 2 "$MAIN_LOG" 2>/dev/null | sed 's/^/  /'
    echo
    echo "--- current benchmark heartbeat: $(basename "${newest_err:-none}") ---"
    [[ -n "$newest_err" ]] && tail -n 5 "$newest_err" 2>/dev/null | sed 's/^/  /'
    echo
    echo "--- results so far ---"
    python3 "$TABLE_PY" "$OUT"
}

if [[ "${ONCE:-0}" == "1" ]]; then
    render
    exit 0
fi

trap 'echo; echo "(stopped watching - bench keeps running in the background)"; exit 0' INT
while true; do
    render
    if grep -q "BENCH DONE" "$MAIN_LOG" 2>/dev/null && ! pgrep -f 'vr-bench\.sh|bench-head\.sh' >/dev/null 2>&1; then
        echo
        echo "*** BENCH DONE - final table above ***"
        exit 0
    fi
    sleep "$REFRESH"
done
