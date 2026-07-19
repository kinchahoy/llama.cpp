#!/usr/bin/env bash
# One focused correctness gate and one exact-shape operator A/B.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_llama_root.sh"
source "$SCRIPT_DIR/_bench_common.sh"
ROOT="$(resolve_llama_root)"
cd "$ROOT"

usage() {
    cat <<'EOF'
Usage:
  OPS=/tmp/model-ops.txt TYPES=q4_K N=512 \
  CONTROL_BUILD=build/gfx906-ablate-common \
  CANDIDATE_BUILD=build/gfx906-ablate-q4 \
  vr/scripts/bench-ops.sh --dry-run

  OPS=... TYPES=... N=... CONTROL_BUILD=... CANDIDATE_BUILD=... \
  vr/scripts/bench-ops.sh

Environment:
  DEVICE=ROCm0
  OP=MUL_MAT
  CHECK_CORRECTNESS=0       set to 1 only when correctness is not already proven
  ORDER="control candidate"
  CONTROL_ENV=""            optional NAME=VALUE assignments for control
  CANDIDATE_ENV=""          optional NAME=VALUE assignments for candidate
  MIN_SPEEDUP_PCT=0         stop when any candidate shape is slower
  COOL_TEMP=55              gate once before the performance pair
  OUT=...                   must not already exist
EOF
}

ACTION=run
case "${1:-}" in
    "")
        ;;
    --dry-run)
        ACTION=dry-run
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Error: unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
esac

OPS="${OPS:-}"
TYPES="${TYPES:-${TYPE:-}}"
N="${N:-}"
CONTROL_BUILD="${CONTROL_BUILD:-}"
CANDIDATE_BUILD="${CANDIDATE_BUILD:-}"
DEVICE="${DEVICE:-ROCm0}"
OP="${OP:-MUL_MAT}"
CHECK_CORRECTNESS="${CHECK_CORRECTNESS:-0}"
ORDER="${ORDER:-control candidate}"
CONTROL_ENV="${CONTROL_ENV:-}"
CANDIDATE_ENV="${CANDIDATE_ENV:-}"
MIN_SPEEDUP_PCT="${MIN_SPEEDUP_PCT:-0}"
COOL_TEMP="${COOL_TEMP:-55}"
COOL_SENSOR="${COOL_SENSOR:-edge}"
COOL_TIMEOUT="${COOL_TIMEOUT:-120}"
COOL_POLL="${COOL_POLL:-3}"
TYPE_TAG="${TYPES// /_}"
OUT="${OUT:-$ROOT/vr/bench-results/gfx906-op-${TYPE_TAG:-unset}-n${N:-unset}-$(date +%Y-%m-%d-%H%M%S)}"

[[ -n "$OPS" && -n "$TYPES" && -n "$N" && -n "$CONTROL_BUILD" && -n "$CANDIDATE_BUILD" ]] || {
    echo "Error: OPS, TYPES, N, CONTROL_BUILD, and CANDIDATE_BUILD are required." >&2
    usage >&2
    exit 2
}
read -r -a TYPE_LIST <<< "$TYPES"
for type in "${TYPE_LIST[@]}"; do
    [[ "$type" =~ ^[A-Za-z0-9_]+$ ]] || {
        echo "Error: invalid type: $type" >&2
        exit 2
    }
done
[[ "$(wc -w <<< "$TYPES")" -eq "$(tr ' ' '\n' <<< "$TYPES" | sed '/^$/d' | sort -u | wc -l)" ]] || {
    echo "Error: TYPES contains a duplicate." >&2
    exit 2
}
[[ "$N" =~ ^[1-9][0-9]*$ ]] || {
    echo "Error: N must be a positive integer." >&2
    exit 2
}
[[ "$DEVICE" =~ ^ROCm[0-9]+$ ]] || {
    echo "Error: DEVICE must name one ROCm backend device, for example ROCm0." >&2
    exit 2
}
[[ "$CHECK_CORRECTNESS" == "0" || "$CHECK_CORRECTNESS" == "1" ]] || {
    echo "Error: CHECK_CORRECTNESS must be 0 or 1." >&2
    exit 2
}

read -r -a CONTROL_ENV_ARGV <<< "$CONTROL_ENV"
read -r -a CANDIDATE_ENV_ARGV <<< "$CANDIDATE_ENV"
for assignment in "${CONTROL_ENV_ARGV[@]}" "${CANDIDATE_ENV_ARGV[@]}"; do
    [[ -z "$assignment" || "$assignment" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || {
        echo "Error: invalid environment assignment: $assignment" >&2
        exit 2
    }
done

for build in $ORDER; do
    [[ "$build" == "control" || "$build" == "candidate" ]] || {
        echo "Error: ORDER accepts only control and candidate." >&2
        exit 2
    }
done
[[ "$(wc -w <<< "$ORDER")" -eq 2 &&
   "$(tr ' ' '\n' <<< "$ORDER" | sed '/^$/d' | sort -u | wc -l)" -eq 2 ]] || {
    echo "Error: ORDER must contain control and candidate exactly once." >&2
    exit 2
}
[[ "$MIN_SPEEDUP_PCT" =~ ^[+-]?[0-9]+([.][0-9]+)?$ ]] || {
    echo "Error: MIN_SPEEDUP_PCT must be numeric." >&2
    exit 2
}
[[ -z "$COOL_TEMP" || "$COOL_TEMP" =~ ^[0-9]+$ ]] || {
    echo "Error: COOL_TEMP must be an integer or empty." >&2
    exit 2
}
[[ "$COOL_TIMEOUT" =~ ^[0-9]+$ && "$COOL_POLL" =~ ^[1-9][0-9]*$ ]] || {
    echo "Error: COOL_TIMEOUT must be non-negative and COOL_POLL must be positive." >&2
    exit 2
}
[[ "$COOL_SENSOR" == "edge" || "$COOL_SENSOR" == "junction" || "$COOL_SENSOR" == "memory" ]] || {
    echo "Error: COOL_SENSOR must be edge, junction, or memory." >&2
    exit 2
}

CONTROL_BUILD="$(realpath -m "$CONTROL_BUILD")"
CANDIDATE_BUILD="$(realpath -m "$CANDIDATE_BUILD")"
TYPE_PATTERN="$(IFS='|'; echo "${TYPE_LIST[*]}")"
FILTER="ne=\\[[0-9]+,${N},1,1\\],op_params=\\[\\],sources=(${TYPE_PATTERN})\\["

binary_for_build() {
    case "$1" in
        control)
            printf '%s/bin/test-backend-ops\n' "$CONTROL_BUILD"
            ;;
        candidate)
            printf '%s/bin/test-backend-ops\n' "$CANDIDATE_BUILD"
            ;;
    esac
}

environment_for_build() {
    if [[ "$1" == "control" ]]; then
        BUILD_ENV=("${CONTROL_ENV_ARGV[@]}")
    else
        BUILD_ENV=("${CANDIDATE_ENV_ARGV[@]}")
    fi
}

print_plan() {
    local build binary
    printf 'filter=%s\n' "$FILTER"
    if [[ "$CHECK_CORRECTNESS" == "1" ]]; then
        printf 'correctness:'
        printf ' %q' env "${CANDIDATE_ENV_ARGV[@]}" "$(binary_for_build candidate)" test -b "$DEVICE" \
            --test-file "$OPS" -o "$OP" -p "$FILTER" -j 1
        printf '\n'
    fi
    for build in $ORDER; do
        binary="$(binary_for_build "$build")"
        environment_for_build "$build"
        printf '%s:' "$build"
        printf ' %q' env "${BUILD_ENV[@]}" "$binary" perf -b "$DEVICE" \
            --test-file "$OPS" -o "$OP" -p "$FILTER"
        printf '\n'
    done
}

if [[ "$ACTION" == "dry-run" ]]; then
    print_plan
    exit 0
fi

[[ ! -e "$OUT" ]] || {
    echo "Error: OUT already exists; refusing to append or repeat: $OUT" >&2
    exit 2
}
[[ -r "$OPS" ]] || {
    echo "Error: unreadable OPS file: $OPS" >&2
    exit 2
}
for build in control candidate; do
    binary="$(binary_for_build "$build")"
    [[ -x "$binary" ]] || {
        echo "Error: missing executable: $binary" >&2
        exit 2
    }
done

source "$ROOT/vr/scripts/setup-therock-env.sh" >/dev/null
mkdir -p "$OUT"
print_plan > "$OUT/plan.txt"
{
    printf 'started=%s\n' "$(date --iso-8601=seconds)"
    printf 'branch_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'git_status=%q\n' "$(git status --short)"
    printf 'ops=%s\ntypes=%s\nn=%s\nfilter=%s\n' "$OPS" "$TYPES" "$N" "$FILTER"
    printf 'device=%s\nop=%s\ncheck_correctness=%s\norder=%s\n' \
        "$DEVICE" "$OP" "$CHECK_CORRECTNESS" "$ORDER"
    printf 'control_env=%q\ncandidate_env=%q\n' "$CONTROL_ENV" "$CANDIDATE_ENV"
    printf 'min_speedup_pct=%s\n' "$MIN_SPEEDUP_PCT"
    printf 'control_build=%s\ncandidate_build=%s\n' "$CONTROL_BUILD" "$CANDIDATE_BUILD"
    for build in control candidate; do
        binary="$(binary_for_build "$build")"
        sha256sum "$binary"
        for library in libllama.so libggml.so libggml-hip.so; do
            [[ -r "$(dirname "$binary")/$library" ]] && sha256sum "$(dirname "$binary")/$library"
        done
    done
    stat -c 'ops_stat=%n,%s,%Y' "$OPS"
} > "$OUT/manifest.txt"

total=$((2 + CHECK_CORRECTNESS))
completed=0
vr_write_status "$OUT" running "starting" "$completed" "$total"
vr_capture_hardware "$OUT/hardware.log"
vr_capture_gpu_state "$OUT/telemetry.log" "start"

finish() {
    local rc=$?
    local state=complete
    trap - EXIT
    (( rc == 0 )) || state=failed
    vr_capture_gpu_state "$OUT/telemetry.log" "$state"
    vr_write_status "$OUT" "$state" "none" "$completed" "$total"
    exit "$rc"
}
trap finish EXIT

if [[ "$CHECK_CORRECTNESS" == "1" ]]; then
    current="correctness $TYPES n=$N"
    vr_write_status "$OUT" running "$current" "$completed" "$total"
    echo "$current"
    env "${CANDIDATE_ENV_ARGV[@]}" "$(binary_for_build candidate)" test -b "$DEVICE" \
        --test-file "$OPS" -o "$OP" -p "$FILTER" -j 1 \
        > "$OUT/correctness.log" 2>&1
    rg -q "Backend [0-9]+/[0-9]+: ${DEVICE}$" "$OUT/correctness.log" || {
        echo "Error: correctness did not initialize $DEVICE." >&2
        exit 2
    }
    rg -q '  [1-9][0-9]*/[1-9][0-9]* tests passed' "$OUT/correctness.log" || {
        echo "Error: correctness ran no matching cases." >&2
        exit 2
    }
    completed=$((completed + 1))
fi

vr_wait_for_cool "$COOL_TEMP" "$COOL_SENSOR" "$COOL_TIMEOUT" "$COOL_POLL" "$DEVICE"
for build in $ORDER; do
    current="$build $TYPES n=$N"
    vr_write_status "$OUT" running "$current" "$completed" "$total"
    vr_capture_gpu_state "$OUT/telemetry.log" "before:$current"
    echo "$current"
    environment_for_build "$build"
    env "${BUILD_ENV[@]}" "$(binary_for_build "$build")" perf -b "$DEVICE" \
        --test-file "$OPS" -o "$OP" -p "$FILTER" \
        > "$OUT/$build.log" 2> "$OUT/$build.err"
    vr_capture_gpu_state "$OUT/telemetry.log" "after:$current"
    completed=$((completed + 1))
done

python3 "$SCRIPT_DIR/_op_table.py" \
    "$OUT/control.log" "$OUT/candidate.log" \
    --fail-below "$MIN_SPEEDUP_PCT" | tee "$OUT/comparison.txt"
echo "Results: $OUT"
