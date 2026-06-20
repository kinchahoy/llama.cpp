#!/usr/bin/env bash
# Isolated Q6_K min_blocks=1 alternating bench:
#   control   = build/mainline           (origin/master + Q8_0 dispatch patch)
#   candidate = build/q6k-only-candidate (control + GGML_CUDA_MMQ_Q6K_GFX906_MIN_BLOCKS_1 on q6_k TU)
# Model: Qwen3.6-27B-Q6_K.gguf (Q6_K is the dominant per-tensor type).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTROL_BIN="${CONTROL_BIN:-$ROOT_DIR/build/mainline/bin/llama-bench}"
CANDIDATE_BIN="${CANDIDATE_BIN:-$ROOT_DIR/build/q6k-only-candidate/bin/llama-bench}"
DEVICE="${DEVICE:-ROCm0}"
RUNS="${RUNS:-3}"
PROMPT="${PROMPT:-8192}"
REPO="${REPO:-unsloth/Qwen3.6-27B-GGUF}"
FILE_NAME="${FILE_NAME:-Qwen3.6-27B-Q6_K.gguf}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-q6k-min-blocks-1}"
RUN_DIR="$ROOT_DIR/vr/bench-results/gfx906-q6k-experiments/$RUN_ID"

mkdir -p "$RUN_DIR"
: > "$RUN_DIR/control.jsonl"
: > "$RUN_DIR/candidate.jsonl"

bench() {
    local binary="$1"
    local output="$2"
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
        -p "$PROMPT" \
        -n 0 > "$output"
}

{
    echo "Run ID: $RUN_ID"
    echo "Control: $CONTROL_BIN"
    echo "Candidate: $CANDIDATE_BIN"
    echo "Model: $REPO/$FILE_NAME"
    echo "Device: $DEVICE"
    echo "Test: pp$PROMPT"
    echo "Alternating runs: $RUNS"
} | tee "$RUN_DIR/metadata.txt"

for ((run = 1; run <= RUNS; ++run)); do
    if ((run % 2 == 1)); then
        echo "[$run/$RUNS] control pp$PROMPT"
        bench "$CONTROL_BIN" "$RUN_DIR/control-$run.jsonl"
        echo "[$run/$RUNS] candidate pp$PROMPT"
        bench "$CANDIDATE_BIN" "$RUN_DIR/candidate-$run.jsonl"
    else
        echo "[$run/$RUNS] candidate pp$PROMPT"
        bench "$CANDIDATE_BIN" "$RUN_DIR/candidate-$run.jsonl"
        echo "[$run/$RUNS] control pp$PROMPT"
        bench "$CONTROL_BIN" "$RUN_DIR/control-$run.jsonl"
    fi
    cat "$RUN_DIR/control-$run.jsonl" >> "$RUN_DIR/control.jsonl"
    cat "$RUN_DIR/candidate-$run.jsonl" >> "$RUN_DIR/candidate.jsonl"
done

"$ROOT_DIR/vr/scripts/compare-gfx906-q4k-experiment.py" \
    --prompt "$PROMPT" \
    --minimum-gain 0 \
    "$RUN_DIR/control.jsonl" "$RUN_DIR/candidate.jsonl" | tee "$RUN_DIR/comparison.txt"
