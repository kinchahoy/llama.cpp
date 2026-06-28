#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_llama_root.sh"
ROOT_DIR="$(resolve_llama_root)"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/vr/bench-results/gfx906-qwen36}"
CONTROL_BIN="${CONTROL_BIN:-$ROOT_DIR/build/mainline/bin/llama-bench}"
CANDIDATE_BIN="${CANDIDATE_BIN:-$ROOT_DIR/build/gfx906-2026-06/bin/llama-bench}"
DEVICE="${DEVICE:-ROCm1/ROCm0}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-2048}"
FORCE_LONG="${FORCE_LONG:-0}"

MODELS=(
    "mtp|unsloth/Qwen3.6-27B-MTP-GGUF|q4_0|Qwen3.6-27B-Q4_0.gguf"
    "mtp|unsloth/Qwen3.6-27B-MTP-GGUF|q4_k_m|Qwen3.6-27B-Q4_K_M.gguf"
    "mtp|unsloth/Qwen3.6-27B-MTP-GGUF|q8_0|Qwen3.6-27B-Q8_0.gguf"
    "base|unsloth/Qwen3.6-27B-GGUF|q4_0|Qwen3.6-27B-Q4_0.gguf"
    "base|unsloth/Qwen3.6-27B-GGUF|q4_k_m|Qwen3.6-27B-Q4_K_M.gguf"
    "base|unsloth/Qwen3.6-27B-GGUF|q8_0|Qwen3.6-27B-Q8_0.gguf"
)

usage() {
    echo "Usage: $0 quick|long|all [--force]"
    echo "Defaults:"
    echo "  CONTROL_BIN=$CONTROL_BIN"
    echo "  CANDIDATE_BIN=$CANDIDATE_BIN"
}

require_binaries() {
    [[ -x "$CONTROL_BIN" ]] || { echo "Not executable: $CONTROL_BIN" >&2; exit 2; }
    [[ -x "$CANDIDATE_BIN" ]] || { echo "Not executable: $CANDIDATE_BIN" >&2; exit 2; }
}

bench_args() {
    printf '%s\n' \
        -r 1 \
        --no-warmup \
        --progress \
        -o jsonl \
        -ngl 99 \
        -fa on \
        -b "$BATCH_SIZE" \
        -ub "$UBATCH_SIZE" \
        -sm layer \
        -dev "$DEVICE"
}

run_one() {
    local stage="$1"
    local binary="$2"
    local build_name="$3"
    local model_name="$4"
    local repo="$5"
    local quant="$6"
    local file_name="$7"
    local output_dir="$RESULTS_DIR/$stage/$build_name"
    local output="$output_dir/${model_name}__${quant}.jsonl"
    local quick_args=(-p 32,512,2048 -n 128)
    local long_args=(-p 32 -n 0 -pg 20000,5000)
    local stage_args=()

    mkdir -p "$output_dir"
    if [[ "$stage" == "quick" ]]; then
        stage_args=("${quick_args[@]}")
    else
        stage_args=("${long_args[@]}")
    fi

    mapfile -t common_args < <(bench_args)
    echo "[$stage][$build_name] $repo $file_name"
    "$binary" \
        "${common_args[@]}" \
        -hf "$repo" \
        -hff "$file_name" \
        "${stage_args[@]}" > "$output"
}

run_stage() {
    local stage="$1"

    for entry in "${MODELS[@]}"; do
        IFS='|' read -r model_name repo quant file_name <<< "$entry"
        run_one "$stage" "$CONTROL_BIN" control "$model_name" "$repo" "$quant" "$file_name"
        run_one "$stage" "$CANDIDATE_BIN" candidate "$model_name" "$repo" "$quant" "$file_name"
    done
}

compare_quick() {
    compare_and_report quick \
        "$RESULTS_DIR/quick/control" \
        "$RESULTS_DIR/quick/candidate"
}

compare_and_report() {
    local stage="$1"
    local control_dir="$2"
    local candidate_dir="$3"
    local report="$RESULTS_DIR/$stage/report.txt"
    local comparison
    local status

    comparison="$(mktemp)"
    set +e
    "$ROOT_DIR/vr/scripts/compare-gfx906-qwen36.py" \
        "$control_dir" \
        "$candidate_dir" > "$comparison"
    status=$?
    set -e

    {
        echo "# gfx906 Qwen3.6 $stage comparison"
        echo
        echo "Generated: $(date --iso-8601=seconds)"
        echo
        echo "## Test setup"
        echo
        echo "- Control: \`$CONTROL_BIN\`"
        echo "- Candidate: \`$CANDIDATE_BIN\`"
        echo "- Devices: \`$DEVICE\`"
        echo "- GPU layers: 99"
        echo "- Flash attention: on"
        echo "- Batch / microbatch: \`$BATCH_SIZE / $UBATCH_SIZE\`"
        echo "- Split mode: layer"
        echo "- Repetitions: 1"
        echo "- Built-in warmup: disabled"
        if [[ "$stage" == "quick" ]]; then
            echo "- Tests: \`pp32\`, \`pp512\`, \`pp2048\`, \`tg128\`"
        else
            echo "- Tests: \`pp32\`, \`pp20000+tg5000\`"
            echo "- Quick gate override: $([[ "$FORCE_LONG" == "1" ]] && echo yes || echo no)"
        fi
        echo "- Models: Qwen3.6-27B and Qwen3.6-27B-MTP"
        echo "- Quant files: Q4_0, Q4_K_M, Q8_0"
        echo "- Gate thresholds: Q8_0 +${MIN_Q8_GAIN:-10}%, Q4_K_M +${MIN_Q4K_GAIN:-5}%, max regression ${MAX_REGRESSION:-3}%"
        echo
        echo "## Comparison"
        echo
        echo '```text'
        cat "$comparison"
        echo '```'
    } > "$report"

    cat "$comparison"
    echo "Report: $report"
    rm -f "$comparison"
    return "$status"
}

run_quick() {
    run_stage quick
    compare_quick
}

run_long() {
    if ! compare_quick; then
        if [[ "$FORCE_LONG" != "1" ]]; then
            return 1
        fi
        echo "Quick gate failed; continuing because the long run was forced."
    fi
    run_stage long
    compare_and_report long \
        "$RESULTS_DIR/long/control" \
        "$RESULTS_DIR/long/candidate" || true
}

main() {
    [[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
    if [[ "${2:-}" == "--force" ]]; then
        FORCE_LONG=1
    elif [[ $# -eq 2 ]]; then
        usage
        exit 2
    fi
    require_binaries
    case "$1" in
        quick) run_quick ;;
        long)  run_long ;;
        all)
            if [[ "$FORCE_LONG" == "1" ]]; then
                run_quick || true
                run_long
            else
                run_quick && run_long
            fi
            ;;
        *)     usage; exit 2 ;;
    esac
}

main "$@"
