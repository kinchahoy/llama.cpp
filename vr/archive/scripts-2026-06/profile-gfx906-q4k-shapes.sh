#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_llama_root.sh"
ROOT_DIR="$(resolve_llama_root)"
PROFILE_BIN="${PROFILE_BIN:-$ROOT_DIR/build/mainline/bin/llama-bench}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/vr/bench-results/gfx906-q4k-profile}"
REPORT="${REPORT:-$ROOT_DIR/vr/bench-results/gfx906-q4k-profile-diff.txt}"
DEVICE="${DEVICE:-ROCm0}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-2048}"
PROMPTS="${PROMPTS:-512 2048}"
QUANTS="${QUANTS:-q4_0 q4_k_m}"
Q4K_PRECOMPUTE="${Q4K_PRECOMPUTE:-0}"
COUNTERS=(
    SQ_INSTS_VALU
    SQ_INSTS_LDS
    SQ_INSTS_VMEM_RD
    SQ_INSTS_VMEM_WR
    SQ_WAIT_INST_LDS
    SQ_LDS_BANK_CONFLICT
    TCC_HIT_sum
    TCC_MISS_sum
    TCC_EA_RDREQ_32B_sum
    TCC_EA_WRREQ_64B_sum
    SQ_WAVES_sum
)
CASES=(
    "q4_0|Qwen3.6-27B-Q4_0.gguf"
    "q4_k_m|Qwen3.6-27B-Q4_K_M.gguf"
)

usage() {
    echo "Usage: $0 trace|counters|all|report"
}

require_tools() {
    [[ -x "$PROFILE_BIN" ]] || { echo "Not executable: $PROFILE_BIN" >&2; exit 2; }
    command -v rocprofv3 >/dev/null 2>&1 || { echo "rocprofv3 not found" >&2; exit 2; }
    command -v rocprofv3-avail >/dev/null 2>&1 || { echo "rocprofv3-avail not found" >&2; exit 2; }
}

bench_command() {
    local file_name="$1"
    local prompt_tokens="$2"

    printf '%s\n' \
        "$PROFILE_BIN" \
        -r 1 \
        --no-warmup \
        -o jsonl \
        -ngl 99 \
        -fa on \
        -b "$BATCH_SIZE" \
        -ub "$UBATCH_SIZE" \
        -sm none \
        -dev "$DEVICE" \
        -hf unsloth/Qwen3.6-27B-GGUF \
        -hff "$file_name" \
        -p "$prompt_tokens" \
        -n 0
}

run_trace_case() {
    local quant="$1"
    local file_name="$2"
    local prompt_tokens="$3"
    local label="${quant}_pp${prompt_tokens}"
    local case_dir="$OUTPUT_DIR/trace/$label"

    mkdir -p "$case_dir"
    mapfile -t command < <(bench_command "$file_name" "$prompt_tokens")
    printf '%q ' "${command[@]}" > "$case_dir/command.txt"
    printf '\n' >> "$case_dir/command.txt"
    echo "[trace] $label"
    rocprofv3 \
        --kernel-trace \
        --scratch-memory-trace \
        --stats \
        -f csv \
        -d "$case_dir" \
        -o "$label" \
        -- "${command[@]}" > "$case_dir/llama-bench.jsonl"
}

run_counter_case() {
    local quant="$1"
    local file_name="$2"
    local prompt_tokens="$3"
    local label="${quant}_pp${prompt_tokens}"
    local case_dir="$OUTPUT_DIR/counters/$label"

    mkdir -p "$case_dir"
    mapfile -t command < <(bench_command "$file_name" "$prompt_tokens")
    printf '%q ' "${command[@]}" > "$case_dir/command.txt"
    printf '\n' >> "$case_dir/command.txt"
    echo "[counters] $label"
    rocprofv3 \
        --pmc "${COUNTERS[@]}" \
        -f csv \
        -d "$case_dir" \
        -o "$label" \
        -- "${command[@]}" > "$case_dir/llama-bench.jsonl"
}

run_cases() {
    local mode="$1"

    for entry in "${CASES[@]}"; do
        IFS='|' read -r quant file_name <<< "$entry"
        if [[ " $QUANTS " != *" $quant "* ]]; then
            continue
        fi
        for prompt_tokens in $PROMPTS; do
            if [[ "$mode" == "trace" ]]; then
                run_trace_case "$quant" "$file_name" "$prompt_tokens"
            else
                run_counter_case "$quant" "$file_name" "$prompt_tokens"
            fi
        done
    done
}

write_manifest() {
    mkdir -p "$OUTPUT_DIR"
    {
        echo "binary=$PROFILE_BIN"
        echo "device=$DEVICE"
        echo "batch_size=$BATCH_SIZE"
        echo "ubatch_size=$UBATCH_SIZE"
        echo "repo=unsloth/Qwen3.6-27B-GGUF"
        echo "contexts=${PROMPTS// /,}"
        echo "quants=${QUANTS// /,}"
        echo "q4k_precompute=$Q4K_PRECOMPUTE"
        echo "repetitions=1"
        echo "warmup=disabled"
        echo "counters=${COUNTERS[*]}"
    } > "$OUTPUT_DIR/manifest.txt"
}

write_report() {
    "$ROOT_DIR/vr/scripts/analyze-gfx906-q4k-profile.py" \
        --input "$OUTPUT_DIR" \
        --output "$REPORT"
    echo "Report: $REPORT"
}

main() {
    [[ $# -eq 1 ]] || { usage; exit 2; }
    if [[ "$1" == "report" ]]; then
        write_report
        exit
    fi

    require_tools
    write_manifest
    case "$1" in
        trace)
            run_cases trace
            write_report
            ;;
        counters)
            rocprofv3-avail pmc-check "${COUNTERS[@]}"
            run_cases counters
            write_report
            ;;
        all)
            run_cases trace
            rocprofv3-avail pmc-check "${COUNTERS[@]}"
            run_cases counters
            write_report
            ;;
        *)
            usage
            exit 2
            ;;
    esac
}

main "$@"
