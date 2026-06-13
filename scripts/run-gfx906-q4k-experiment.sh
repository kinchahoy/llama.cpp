#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTROL_BIN="${CONTROL_BIN:-$ROOT_DIR/build/mainline/bin/llama-bench}"
CANDIDATE_BUILD="${CANDIDATE_BUILD:-$ROOT_DIR/build/gfx906-2026-06}"
CANDIDATE_BIN="${CANDIDATE_BIN:-$CANDIDATE_BUILD/bin/llama-bench}"
DEVICE="${DEVICE:-ROCm0}"
MIN_GAIN="${MIN_GAIN:-5}"
CORE_PROMPT="${CORE_PROMPT:-8192}"
LONG_PROMPT="${LONG_PROMPT:-20000}"
DELTA="${DELTA:-Q4_K gfx906: keep mmq_y=128/mmq_x=64/four waves, but change launch-bounds minimum resident blocks 2->1 to trade occupancy for zero spill}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-q4k-min-blocks-1}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/bench-results/gfx906-q4k-experiments}"
RUN_DIR="$RESULTS_ROOT/$RUN_ID"
REPO="unsloth/Qwen3.6-27B-GGUF"
FILE_NAME="Qwen3.6-27B-Q4_K_M.gguf"

usage() {
    echo "Usage: $0 build|core|quick|long|all|profile"
    echo "core and quick both run pp$CORE_PROMPT; long runs pp$LONG_PROMPT after the core gate passes."
    echo "Set RUN_ID to reuse a run directory across separate commands."
}

write_metadata() {
    mkdir -p "$RUN_DIR"
    printf '%s\n' "Delta: $DELTA" > "$RUN_DIR/delta.txt"
    {
        echo "Delta: $DELTA"
        echo "Run ID: $RUN_ID"
        echo "Generated: $(date --iso-8601=seconds)"
        echo "Control: $CONTROL_BIN"
        echo "Candidate: $CANDIDATE_BIN"
        echo "Device: $DEVICE"
        echo "Model: $REPO/$FILE_NAME"
        echo "Minimum gain: $MIN_GAIN%"
        echo "Core test: pp$CORE_PROMPT"
        echo "Long test: pp$LONG_PROMPT"
        echo "Source: $(git -C "$ROOT_DIR" rev-parse --short=12 HEAD)"
    } > "$RUN_DIR/metadata.txt"
}

require_binary() {
    local binary="$1"
    [[ -x "$binary" ]] || { echo "Not executable: $binary" >&2; exit 2; }
}

build_candidate() {
    [[ -n "${CC:-}" && -n "${CXX:-}" ]] || {
        echo "CC and CXX are not set. Source scripts/setup-therock-env.sh first." >&2
        exit 2
    }
    export CCACHE_DIR="${CCACHE_DIR:-$ROOT_DIR/build/.ccache}"
    mkdir -p "$CCACHE_DIR"
    echo "Delta: $DELTA"
    cmake --build "$CANDIDATE_BUILD" --target llama-bench -j"${BUILD_JOBS:-$(nproc)}" 2>&1 | tee "$RUN_DIR/build.log"
}

bench() {
    local binary="$1"
    local output="$2"
    local prompt="$3"

    "$binary" \
        -r 1 \
        --no-warmup \
        -o jsonl \
        -ngl 99 \
        -fa on \
        -b 2048 \
        -ub 2048 \
        -sm none \
        -dev "$DEVICE" \
        -hf "$REPO" \
        -hff "$FILE_NAME" \
        -p "$prompt" \
        -n 0 > "$output"
}

run_comparison() {
    local stage="$1"
    local prompt="$2"
    local stage_dir="$RUN_DIR/$stage"
    local comparison="$stage_dir/comparison.txt"
    local report="$stage_dir/report.md"
    local status

    mkdir -p "$stage_dir"
    printf '%s\n' "Delta: $DELTA" > "$stage_dir/delta.txt"
    echo "[$stage][control] pp$prompt"
    bench "$CONTROL_BIN" "$stage_dir/control.jsonl" "$prompt"
    echo "[$stage][candidate] pp$prompt"
    bench "$CANDIDATE_BIN" "$stage_dir/candidate.jsonl" "$prompt"

    set +e
    "$ROOT_DIR/scripts/compare-gfx906-q4k-experiment.py" \
        --prompt "$prompt" \
        --minimum-gain "$MIN_GAIN" \
        "$stage_dir/control.jsonl" \
        "$stage_dir/candidate.jsonl" > "$comparison"
    status=$?
    set -e

    {
        echo "# gfx906 Q4_K experiment: $stage"
        echo
        echo "Delta: $DELTA"
        echo
        printf '%s\n' "- Run ID: \`$RUN_ID\`"
        printf '%s\n' "- Device: \`$DEVICE\`"
        printf '%s\n' "- Test: \`pp$prompt\`"
        echo "- Repetitions: 1"
        echo "- Warmup: disabled"
        printf '%s\n' "- Gate: at least \`+$MIN_GAIN%\`"
        echo
        echo '```text'
        cat "$comparison"
        echo '```'
    } > "$report"

    cat "$comparison"
    echo "Report: $report"
    return "$status"
}

run_profile() {
    local profile_dir="$RUN_DIR/profile-pp$CORE_PROMPT"

    command -v rocprofv3 >/dev/null 2>&1 || { echo "rocprofv3 not found" >&2; exit 2; }
    mkdir -p "$profile_dir"
    printf '%s\n' "Delta: $DELTA" > "$profile_dir/delta.txt"
    echo "Delta: $DELTA"
    rocprofv3 --kernel-trace --scratch-memory-trace --stats -f csv \
        -d "$profile_dir" -o "q4k-experiment-pp$CORE_PROMPT" -- \
        "$CANDIDATE_BIN" -r 1 --no-warmup -o jsonl -ngl 99 -fa on \
        -b 2048 -ub 2048 -sm none -dev "$DEVICE" \
        -hf "$REPO" -hff "$FILE_NAME" -p "$CORE_PROMPT" -n 0 \
        > "$profile_dir/candidate.jsonl"
    echo "Profile: $profile_dir"
}

main() {
    [[ $# -eq 1 ]] || { usage; exit 2; }
    write_metadata
    case "$1" in
        build)
            build_candidate
            ;;
        core|quick)
            require_binary "$CONTROL_BIN"
            require_binary "$CANDIDATE_BIN"
            run_comparison core "$CORE_PROMPT"
            ;;
        long)
            require_binary "$CONTROL_BIN"
            require_binary "$CANDIDATE_BIN"
            if [[ ! -f "$RUN_DIR/core/report.md" ]]; then
                echo "Missing core pp$CORE_PROMPT result for RUN_ID=$RUN_ID" >&2
                exit 2
            fi
            "$ROOT_DIR/scripts/compare-gfx906-q4k-experiment.py" \
                --prompt "$CORE_PROMPT" --minimum-gain "$MIN_GAIN" \
                "$RUN_DIR/core/control.jsonl" "$RUN_DIR/core/candidate.jsonl" >/dev/null
            run_comparison long "$LONG_PROMPT"
            ;;
        profile)
            require_binary "$CANDIDATE_BIN"
            run_profile
            ;;
        all)
            require_binary "$CONTROL_BIN"
            build_candidate
            require_binary "$CANDIDATE_BIN"
            run_comparison core "$CORE_PROMPT"
            run_comparison long "$LONG_PROMPT"
            ;;
        *)
            usage
            exit 2
            ;;
    esac
}

main "$@"
