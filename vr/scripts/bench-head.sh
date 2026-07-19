#!/usr/bin/env bash
# Minimal one-sample model A/B driver for the gfx906 worktree.
#
# The default is one UD-Q4_K_XL raw-PP cell on ROCm0. Select only the affected
# path. Incremental PP and TG at the same depth share one llama-bench process,
# model load, and saved depth state.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_llama_root.sh"
source "$SCRIPT_DIR/_bench_common.sh"
ROOT="$(resolve_llama_root)"
cd "$ROOT"

usage() {
    cat <<'EOF'
Usage:
  vr/scripts/bench-head.sh --list
  vr/scripts/bench-head.sh --dry-run
  vr/scripts/bench-head.sh

Core environment:
  CONFIGS="ud_q4_k_xl_single"    named model/device configurations
  MODES="raw_pp"                 raw_pp, inc_pp, and/or tg
  BUILDS="control candidate"     use "candidate control" for one reversed A/B
  CONTROL_BIN_DIR=...            directory containing control llama-bench
  CANDIDATE_BIN_DIR=...          directory containing candidate llama-bench

Workload:
  RAW_PROMPT=8192
  INC_PROMPT=256
  CONTEXT_DEPTH=8192
  GEN_TOKENS=64
  N_BATCH=8192
  N_UBATCH=512

Policy:
  REPS=1                         any other value is rejected
  WARMUP=auto                    skip duplicate raw PP; keep cheap context init
  COOL_TEMP=55                   gate once before each A/B pair
  COOL_BETWEEN_BUILDS=0          set to 1 only for a deliberate cooled pair
  FAIL_BELOW_PCT=                stop before the next job if any A/B is lower
  OUT=...                        must not already exist
EOF
}

ACTION=run
case "${1:-}" in
    "")
        ;;
    --list)
        ACTION=list
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

REPS="${REPS:-1}"
[[ "$REPS" == "1" ]] || {
    echo "Error: REPS must remain 1. Use one reversed A/B in a new output directory when evidence is borderline." >&2
    exit 2
}

BUILDS="${BUILDS:-control candidate}"
CONFIGS="${CONFIGS:-ud_q4_k_xl_single}"
MODES="${MODES:-raw_pp}"
RAW_PROMPT="${RAW_PROMPT:-8192}"
INC_PROMPT="${INC_PROMPT:-256}"
CONTEXT_DEPTH="${CONTEXT_DEPTH:-8192}"
GEN_TOKENS="${GEN_TOKENS:-64}"
N_BATCH="${N_BATCH:-8192}"
N_UBATCH="${N_UBATCH:-512}"
WARMUP="${WARMUP:-auto}"
COOL_TEMP="${COOL_TEMP:-55}"
COOL_SENSOR="${COOL_SENSOR:-edge}"
COOL_TIMEOUT="${COOL_TIMEOUT:-120}"
COOL_POLL="${COOL_POLL:-3}"
COOL_BETWEEN_BUILDS="${COOL_BETWEEN_BUILDS:-0}"
FAIL_BELOW_PCT="${FAIL_BELOW_PCT:-}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"
CONTROL_BIN_DIR="${CONTROL_BIN_DIR:-$ROOT/build/head-control/bin}"
CANDIDATE_BIN_DIR="${CANDIDATE_BIN_DIR:-$ROOT/build/gfx906-merge-review/bin}"
OUT="${OUT:-$ROOT/vr/bench-results/gfx906-model-$(date +%Y-%m-%d-%H%M%S)}"

SNAP="${SNAP:-$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/ac393bc3d23fd5a929a85e2f33c7c4fd5be02d43}"
LEGACY_SNAP="${LEGACY_SNAP:-$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace}"

SPECS=(
    "ud_q4_k_xl_single|$SNAP/Qwen3.6-27B-UD-Q4_K_XL.gguf|none|ROCm0"
    "ud_q4_k_xl_dual|$SNAP/Qwen3.6-27B-UD-Q4_K_XL.gguf|tensor|ROCm0/ROCm1"
    "q8_0_dual|$SNAP/Qwen3.6-27B-Q8_0.gguf|tensor|ROCm0/ROCm1"
    "q4_0_single|$LEGACY_SNAP/Qwen3.6-27B-Q4_0.gguf|none|ROCm0"
    "q4_1_single|$LEGACY_SNAP/Qwen3.6-27B-Q4_1.gguf|none|ROCm0"
)

if [[ "$ACTION" == "list" ]]; then
    printf '%s\n' "${SPECS[@]%%|*}"
    exit 0
fi

contains_word() {
    local words="$1"
    local wanted="$2"
    [[ " $words " == *" $wanted "* ]]
}

for requested in $CONFIGS; do
    known=0
    for spec in "${SPECS[@]}"; do
        [[ "${spec%%|*}" == "$requested" ]] && known=1
    done
    (( known == 1 )) || {
        echo "Error: unknown configuration: $requested" >&2
        exit 2
    }
done
[[ "$(tr ' ' '\n' <<< "$CONFIGS" | sed '/^$/d' | sort -u | wc -l)" -eq "$(wc -w <<< "$CONFIGS")" ]] || {
    echo "Error: CONFIGS contains a duplicate; repeated cells are not allowed." >&2
    exit 2
}

for build in $BUILDS; do
    [[ "$build" == "control" || "$build" == "candidate" ]] || {
        echo "Error: unknown build label: $build" >&2
        exit 2
    }
done
[[ "$(wc -w <<< "$BUILDS")" -eq 2 &&
   "$(tr ' ' '\n' <<< "$BUILDS" | sed '/^$/d' | sort -u | wc -l)" -eq 2 ]] || {
    echo "Error: BUILDS must contain control and candidate exactly once." >&2
    exit 2
}

for mode in $MODES; do
    [[ "$mode" == "raw_pp" || "$mode" == "inc_pp" || "$mode" == "tg" ]] || {
        echo "Error: unknown mode: $mode" >&2
        exit 2
    }
done
[[ "$(tr ' ' '\n' <<< "$MODES" | sed '/^$/d' | sort -u | wc -l)" -eq "$(wc -w <<< "$MODES")" ]] || {
    echo "Error: MODES contains a duplicate; repeated cells are not allowed." >&2
    exit 2
}

for value in "$RAW_PROMPT" "$INC_PROMPT" "$CONTEXT_DEPTH" "$GEN_TOKENS" "$N_BATCH" "$N_UBATCH"; do
    [[ "$value" =~ ^[0-9]+$ ]] || {
        echo "Error: workload sizes must be non-negative integers." >&2
        exit 2
    }
done
(( N_BATCH > 0 && N_UBATCH > 0 && N_UBATCH <= N_BATCH )) || {
    echo "Error: invalid batch or microbatch size." >&2
    exit 2
}
if contains_word "$MODES" raw_pp; then
    (( RAW_PROMPT > 0 )) || {
        echo "Error: raw_pp requires a positive RAW_PROMPT." >&2
        exit 2
    }
fi
if contains_word "$MODES" inc_pp; then
    (( INC_PROMPT > 0 && CONTEXT_DEPTH > 0 )) || {
        echo "Error: inc_pp requires positive INC_PROMPT and CONTEXT_DEPTH." >&2
        exit 2
    }
fi
if contains_word "$MODES" tg; then
    (( GEN_TOKENS > 0 && CONTEXT_DEPTH > 0 )) || {
        echo "Error: tg requires positive GEN_TOKENS and CONTEXT_DEPTH." >&2
        exit 2
    }
fi

[[ "$WARMUP" == "auto" || "$WARMUP" == "0" || "$WARMUP" == "1" ]] || {
    echo "Error: WARMUP must be auto, 0, or 1." >&2
    exit 2
}
[[ "$COOL_BETWEEN_BUILDS" == "0" || "$COOL_BETWEEN_BUILDS" == "1" ]] || {
    echo "Error: COOL_BETWEEN_BUILDS must be 0 or 1." >&2
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
[[ -z "$FAIL_BELOW_PCT" || "$FAIL_BELOW_PCT" =~ ^[+-]?[0-9]+([.][0-9]+)?$ ]] || {
    echo "Error: FAIL_BELOW_PCT must be numeric or empty." >&2
    exit 2
}

read -r -a BENCH_EXTRA_ARGV <<< "$BENCH_EXTRA_ARGS"
for arg in "${BENCH_EXTRA_ARGV[@]}"; do
    case "$arg" in
        -m|--model|-hf|-hfr|--hf-repo|-hff|--hf-file|-pg|-p|--n-prompt|-n|--n-gen|-d|--n-depth|-b|--batch-size|-ub|--ubatch-size|-r|--repetitions|-o|--output|-sm|--split-mode|-dev|--device|-fa|--flash-attn|-ngl|--n-gpu-layers|--delay|--no-warmup)
            echo "Error: BENCH_EXTRA_ARGS may not override workload, model, output, split, device, FA, or repetition arguments." >&2
            exit 2
            ;;
    esac
done

JOBS=()
selected_configs=0
for spec in "${SPECS[@]}"; do
    IFS='|' read -r label model split_mode devices <<< "$spec"
    contains_word "$CONFIGS" "$label" || continue
    selected_configs=$((selected_configs + 1))
    if contains_word "$MODES" raw_pp; then
        JOBS+=("$label|$model|$split_mode|$devices|raw")
    fi
    if contains_word "$MODES" inc_pp || contains_word "$MODES" tg; then
        JOBS+=("$label|$model|$split_mode|$devices|context")
    fi
done

(( selected_configs > 0 )) || {
    echo "Error: CONFIGS selected no known configurations." >&2
    exit 2
}
(( ${#JOBS[@]} > 0 )) || {
    echo "Error: MODES selected no jobs." >&2
    exit 2
}

binary_for_build() {
    case "$1" in
        control)
            printf '%s/llama-bench\n' "$CONTROL_BIN_DIR"
            ;;
        candidate)
            printf '%s/llama-bench\n' "$CANDIDATE_BIN_DIR"
            ;;
    esac
}

args_for_scenario() {
    local scenario="$1"
    SCENARIO_ARGS=(-b "$N_BATCH" -ub "$N_UBATCH")
    EXPECTED_ROWS=1
    if [[ "$scenario" == "raw" ]]; then
        SCENARIO_ARGS+=(-p "$RAW_PROMPT" -n 0 -d 0)
        return
    fi

    if contains_word "$MODES" inc_pp; then
        SCENARIO_ARGS+=(-p "$INC_PROMPT")
    else
        SCENARIO_ARGS+=(-p 0)
    fi
    if contains_word "$MODES" tg; then
        SCENARIO_ARGS+=(-n "$GEN_TOKENS")
    else
        SCENARIO_ARGS+=(-n 0)
    fi
    SCENARIO_ARGS+=(-d "$CONTEXT_DEPTH")
    EXPECTED_ROWS=0
    if contains_word "$MODES" inc_pp; then
        EXPECTED_ROWS=$((EXPECTED_ROWS + 1))
    fi
    if contains_word "$MODES" tg; then
        EXPECTED_ROWS=$((EXPECTED_ROWS + 1))
    fi
}

warmup_for_scenario() {
    local scenario="$1"
    WARMUP_ENABLED="$WARMUP"
    if [[ "$WARMUP" == "auto" ]]; then
        if [[ "$scenario" == "raw" ]]; then
            WARMUP_ENABLED=0
        else
            WARMUP_ENABLED=1
        fi
    fi
}

print_jobs() {
    local index=0
    local job label model split_mode devices scenario build binary
    for job in "${JOBS[@]}"; do
        IFS='|' read -r label model split_mode devices scenario <<< "$job"
        args_for_scenario "$scenario"
        warmup_for_scenario "$scenario"
        index=$((index + 1))
        printf 'job=%d config=%s scenario=%s model=%s split=%s devices=%s rows=%d\n' \
            "$index" "$label" "$scenario" "$(basename "$model")" "$split_mode" "$devices" "$EXPECTED_ROWS"
        for build in $BUILDS; do
            binary="$(binary_for_build "$build")"
            printf '  %s:' "$build"
            printf ' %q' "$binary" -ngl 99 -fa on -sm "$split_mode" -dev "$devices" \
                -o jsonl -r 1 "${BENCH_EXTRA_ARGV[@]}" -m "$model" "${SCENARIO_ARGS[@]}"
            [[ "$WARMUP_ENABLED" == "0" ]] && printf ' %q' --no-warmup
            printf '\n'
        done
    done
}

if [[ "$ACTION" == "dry-run" ]]; then
    print_jobs
    exit 0
fi

[[ ! -e "$OUT" ]] || {
    echo "Error: OUT already exists; refusing to append or repeat cells: $OUT" >&2
    exit 2
}

source "$ROOT/vr/scripts/setup-therock-env.sh" >/dev/null

for build in $BUILDS; do
    binary="$(binary_for_build "$build")"
    [[ -x "$binary" ]] || {
        echo "Error: missing executable: $binary" >&2
        exit 2
    }
done
for job in "${JOBS[@]}"; do
    IFS='|' read -r _ model _ _ _ <<< "$job"
    [[ -r "$model" ]] || {
        echo "Error: missing model: $model" >&2
        exit 2
    }
done

mkdir -p "$OUT"
for build in $BUILDS; do
    mkdir -p "$OUT/$build"
done

{
    printf 'started=%s\n' "$(date --iso-8601=seconds)"
    printf 'branch_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'git_status=%q\n' "$(git status --short)"
    printf 'builds=%s\n' "$BUILDS"
    printf 'configs=%s\n' "$CONFIGS"
    printf 'modes=%s\n' "$MODES"
    printf 'raw_prompt=%s\ninc_prompt=%s\ncontext_depth=%s\ngen_tokens=%s\n' \
        "$RAW_PROMPT" "$INC_PROMPT" "$CONTEXT_DEPTH" "$GEN_TOKENS"
    printf 'n_batch=%s\nn_ubatch=%s\nreps=1\nwarmup=%s\n' "$N_BATCH" "$N_UBATCH" "$WARMUP"
    printf 'cool_temp=%s\ncool_sensor=%s\ncool_between_builds=%s\n' \
        "$COOL_TEMP" "$COOL_SENSOR" "$COOL_BETWEEN_BUILDS"
    printf 'bench_extra_args=%q\n' "$BENCH_EXTRA_ARGS"
    for build in $BUILDS; do
        binary="$(binary_for_build "$build")"
        printf 'binary_%s=%s\n' "$build" "$binary"
        sha256sum "$binary"
        for library in libllama.so libggml.so libggml-hip.so; do
            [[ -r "$(dirname "$binary")/$library" ]] && sha256sum "$(dirname "$binary")/$library"
        done
    done
    for job in "${JOBS[@]}"; do
        IFS='|' read -r label model split_mode devices scenario <<< "$job"
        printf 'job=%s,%s,%s,%s,%s\n' "$label" "$model" "$split_mode" "$devices" "$scenario"
        stat -c 'model_stat=%n,%s,%Y' "$model"
    done
} > "$OUT/manifest.txt"
print_jobs > "$OUT/plan.txt"
vr_capture_hardware "$OUT/hardware.log"
vr_capture_gpu_state "$OUT/telemetry.log" "start"

total=$(( ${#JOBS[@]} * $(wc -w <<< "$BUILDS") ))
completed=0
run_state=running
vr_write_status "$OUT" "$run_state" "starting" "$completed" "$total"

finish() {
    local rc=$?
    trap - EXIT
    if (( rc == 0 )); then
        run_state=complete
    else
        run_state=failed
    fi
    vr_capture_gpu_state "$OUT/telemetry.log" "$run_state"
    vr_write_status "$OUT" "$run_state" "none" "$completed" "$total"
    exit "$rc"
}
trap finish EXIT

job_index=0
for job in "${JOBS[@]}"; do
    IFS='|' read -r label model split_mode devices scenario <<< "$job"
    args_for_scenario "$scenario"
    warmup_for_scenario "$scenario"
    job_index=$((job_index + 1))

    vr_wait_for_cool "$COOL_TEMP" "$COOL_SENSOR" "$COOL_TIMEOUT" "$COOL_POLL" "$devices"
    build_index=0
    for build in $BUILDS; do
        build_index=$((build_index + 1))
        if (( build_index > 1 )) && [[ "$COOL_BETWEEN_BUILDS" == "1" ]]; then
            vr_wait_for_cool "$COOL_TEMP" "$COOL_SENSOR" "$COOL_TIMEOUT" "$COOL_POLL" "$devices"
        fi

        binary="$(binary_for_build "$build")"
        output_file="$OUT/$build/$label.jsonl"
        error_file="$output_file.err"
        before=0
        [[ -r "$output_file" ]] && before="$(wc -l < "$output_file")"

        common_args=(-ngl 99 -fa on -sm "$split_mode" -dev "$devices" -o jsonl -r 1 --progress)
        common_args+=("${BENCH_EXTRA_ARGV[@]}")
        [[ "$WARMUP_ENABLED" == "0" ]] && common_args+=(--no-warmup)

        current="$build $label $scenario"
        vr_write_status "$OUT" running "$current" "$completed" "$total"
        vr_capture_gpu_state "$OUT/telemetry.log" "before:$current"
        {
            printf '[%s]' "$(date --iso-8601=seconds)"
            printf ' %q' "$binary" "${common_args[@]}" -m "$model" "${SCENARIO_ARGS[@]}"
            printf '\n'
        } >> "$OUT/commands.log"

        echo "[$job_index/${#JOBS[@]}] $current"
        "$binary" "${common_args[@]}" -m "$model" "${SCENARIO_ARGS[@]}" \
            >> "$output_file" 2>> "$error_file"
        vr_capture_gpu_state "$OUT/telemetry.log" "after:$current"

        after="$(wc -l < "$output_file")"
        if (( after - before != EXPECTED_ROWS )); then
            echo "Error: $current emitted $((after - before)) result rows; expected $EXPECTED_ROWS." >&2
            exit 2
        fi
        completed=$((completed + 1))
    done

    if [[ -n "$FAIL_BELOW_PCT" ]]; then
        if ! python3 "$SCRIPT_DIR/_bench_table.py" "$OUT" --require-pairs --fail-below "$FAIL_BELOW_PCT"; then
            echo "Stopped before the next model job because the configured A/B gate failed." >&2
            exit 1
        fi
    fi
done

python3 "$SCRIPT_DIR/_bench_table.py" "$OUT" --require-pairs
echo "Results: $OUT"
