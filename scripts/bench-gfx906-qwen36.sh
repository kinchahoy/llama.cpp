#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/bench-results/gfx906-qwen36}"
CONTROL_BIN="${CONTROL_BIN:-$ROOT_DIR/build/mainline/bin/llama-bench}"
CANDIDATE_BIN="${CANDIDATE_BIN:-$ROOT_DIR/build/gfx906-2026-06/bin/llama-bench}"
DEVICE="${DEVICE:-ROCm1/ROCm0}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-2048}"

MODELS=(
    "mtp|unsloth/Qwen3.6-27B-MTP-GGUF|q4_0|Qwen3.6-27B-Q4_0.gguf"
    "mtp|unsloth/Qwen3.6-27B-MTP-GGUF|q4_k_m|Qwen3.6-27B-Q4_K_M.gguf"
    "mtp|unsloth/Qwen3.6-27B-MTP-GGUF|q8_0|Qwen3.6-27B-Q8_0.gguf"
    "base|unsloth/Qwen3.6-27B-GGUF|q4_0|Qwen3.6-27B-Q4_0.gguf"
    "base|unsloth/Qwen3.6-27B-GGUF|q4_k_m|Qwen3.6-27B-Q4_K_M.gguf"
    "base|unsloth/Qwen3.6-27B-GGUF|q8_0|Qwen3.6-27B-Q8_0.gguf"
)

usage() {
    echo "Usage: $0 quick|long|all"
    echo "Defaults:"
    echo "  CONTROL_BIN=$CONTROL_BIN"
    echo "  CANDIDATE_BIN=$CANDIDATE_BIN"
}

setup_rocm_runtime() {
    local sdk_root
    local package_root
    local sdk_libs

    if ! command -v rocm-sdk >/dev/null 2>&1; then
        return
    fi

    sdk_root="$(rocm-sdk path --root)"
    package_root="$(dirname "$sdk_root")"
    sdk_libs="$package_root/_rocm_sdk_devel/lib:$package_root/_rocm_sdk_libraries/lib:$package_root/_rocm_sdk_core/lib"
    export LD_LIBRARY_PATH="$sdk_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
    "$ROOT_DIR/scripts/compare-gfx906-qwen36.py" \
        "$RESULTS_DIR/quick/control" \
        "$RESULTS_DIR/quick/candidate"
}

run_quick() {
    run_stage quick
    compare_quick
}

run_long() {
    compare_quick
    run_stage long
    "$ROOT_DIR/scripts/compare-gfx906-qwen36.py" \
        "$RESULTS_DIR/long/control" \
        "$RESULTS_DIR/long/candidate" || true
}

main() {
    [[ $# -eq 1 ]] || { usage; exit 2; }
    setup_rocm_runtime
    require_binaries
    case "$1" in
        quick) run_quick ;;
        long)  run_long ;;
        all)   run_quick && run_long ;;
        *)     usage; exit 2 ;;
    esac
}

main "$@"
