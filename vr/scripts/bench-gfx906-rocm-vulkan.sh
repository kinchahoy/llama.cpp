#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_BIN="${BENCH_BIN:-$ROOT_DIR/build/gfx906-2026-06/bin/llama-bench}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/vr/bench-results/gfx906-rocm-vulkan/quick}"
ROCM_DEVICE="${ROCM_DEVICE:-ROCm1/ROCm0}"
VULKAN_DEVICE="${VULKAN_DEVICE:-Vulkan2/Vulkan1}"
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

require_devices() {
    local devices

    [[ -x "$BENCH_BIN" ]] || { echo "Not executable: $BENCH_BIN" >&2; exit 2; }
    devices="$("$BENCH_BIN" --list-devices 2>&1)"
    for device in ROCm1 ROCm0 Vulkan2 Vulkan1; do
        if ! grep -q "$device" <<< "$devices"; then
            echo "Error: $device is not available from $BENCH_BIN" >&2
            printf '%s\n' "$devices" >&2
            exit 2
        fi
    done
}

run_one() {
    local backend="$1"
    local device="$2"
    local model_name="$3"
    local repo="$4"
    local quant="$5"
    local file_name="$6"
    local output_dir="$RESULTS_DIR/$backend"

    mkdir -p "$output_dir"
    echo "[$backend] $repo $file_name"
    "$BENCH_BIN" \
        -r 1 \
        --no-warmup \
        --progress \
        -o jsonl \
        -ngl 99 \
        -fa on \
        -b "$BATCH_SIZE" \
        -ub "$UBATCH_SIZE" \
        -sm layer \
        -dev "$device" \
        -hf "$repo" \
        -hff "$file_name" \
        -p 32,512 \
        -n 128 > "$output_dir/${model_name}__${quant}.jsonl"
}

main() {
    local comparison
    local report="$RESULTS_DIR/report.txt"

    require_devices
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r model_name repo quant file_name <<< "$entry"
        run_one rocm "$ROCM_DEVICE" "$model_name" "$repo" "$quant" "$file_name"
        run_one vulkan "$VULKAN_DEVICE" "$model_name" "$repo" "$quant" "$file_name"
    done

    comparison="$(mktemp)"
    "$ROOT_DIR/vr/scripts/compare-gfx906-qwen36.py" \
        --informational \
        --control-label ROCm \
        --candidate-label Vulkan \
        "$RESULTS_DIR/rocm" \
        "$RESULTS_DIR/vulkan" > "$comparison"

    {
        echo "# gfx906 ROCm and Vulkan quick comparison"
        echo
        echo "Generated: $(date --iso-8601=seconds)"
        echo
        echo "## Test setup"
        echo
        echo "- Binary: \`$BENCH_BIN\`"
        echo "- ROCm devices: \`$ROCM_DEVICE\`"
        echo "- Vulkan devices: \`$VULKAN_DEVICE\`"
        echo "- GPU layers: 99"
        echo "- Flash attention: on"
        echo "- Batch / microbatch: \`$BATCH_SIZE / $UBATCH_SIZE\`"
        echo "- Split mode: layer"
        echo "- Repetitions: 1"
        echo "- Built-in warmup: disabled"
        echo "- Tests: \`pp32\`, \`pp512\`, \`tg128\`"
        echo "- Models: Qwen3.6-27B and Qwen3.6-27B-MTP"
        echo "- Quant files: Q4_0, Q4_K_M, Q8_0"
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
}

main "$@"
