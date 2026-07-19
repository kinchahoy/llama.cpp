#!/usr/bin/env bash
# Reusable head-vs-candidate benchmark for Qwen3.6-27B on gfx906.
#
# The default screen runs UD-Q4_K_XL on one GPU and Q8_0 on two GPUs at pp8192
# and tg128@d8192. Each build pays for one repetition plus warmup.
#
# Usage:
#   vr/scripts/bench-head.sh [FILTER]      # run configs whose label matches FILTER (default: all)
#   vr/scripts/bench-head.sh --list        # list config labels and exit
#   vr/scripts/bench-head.sh --dry-run     # print selected jobs and exit
#
# Examples:
#   vr/scripts/bench-head.sh
#   BUILDS="candidate control" REPS=1 MODES=tg vr/scripts/bench-head.sh
#   CONFIGS="ud_q4_k_xl_single q8_0_dual" vr/scripts/bench-head.sh
#
# Env overrides:
#   REPS=1                  llama-bench repetitions
#   BUILDS="control candidate"
#   CONFIGS="ud_q4_k_xl_single q8_0_dual"
#   MODES="pp tg"           select prompt processing, token generation, or both
#   DEPTHS="8192"           TG depths
#   PROMPTS="8192"          PP sizes
#   GEN_TOKENS=128          tokens per TG cell
#   BENCH_EXTRA_ARGS=""     extra llama-bench arguments, recorded in the manifest
#   COOLDOWN=0              seconds idle between every run (e.g. 30) to relieve heat-soak
#   COOL_TEMP=              if set (e.g. 60), wait before each run until the hottest GPU
#                          reaches this temp. Overrides COOLDOWN. COOL_SENSOR=edge|junction|
#                          memory (default edge), COOL_TIMEOUT=300 caps the wait.
#   RANDOMIZE=0             1 = shuffle test order
# A/B note: control and candidate are adjacent per test to limit drift. Reverse
# the build order only when confirming a winning screen.
#   OUT=vr/bench-results/gfx906-head-<timestamp>
#   CONTROL_BIN_DIR=build/head-control/bin
#   CANDIDATE_BIN_DIR=build/head-candidate/bin
#   SNAP=<snapshot containing UD-Q4_K_XL and Q8_0>
#   LEGACY_SNAP=<snapshot containing Q4_0, Q4_1, and Q4_K_M>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_llama_root.sh"
ROOT="$(resolve_llama_root)"
cd "$ROOT"

REPS="${REPS:-1}"
BUILDS="${BUILDS:-control candidate}"
CONFIGS="${CONFIGS:-ud_q4_k_xl_single q8_0_dual}"
MODES="${MODES:-pp tg}"
DEPTHS="${DEPTHS:-8192}"
PROMPTS="${PROMPTS:-8192}"
GEN_TOKENS="${GEN_TOKENS:-128}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"
read -r -a BENCH_EXTRA_ARGV <<< "$BENCH_EXTRA_ARGS"
SNAP="${SNAP:-$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/ac393bc3d23fd5a929a85e2f33c7c4fd5be02d43}"
LEGACY_SNAP="${LEGACY_SNAP:-$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace}"
OUT="${OUT:-$ROOT/vr/bench-results/gfx906-head-$(date +%Y-%m-%d-%H%M%S)}"
CONTROL_BIN_DIR="${CONTROL_BIN_DIR:-$ROOT/build/head-control/bin}"
CANDIDATE_BIN_DIR="${CANDIDATE_BIN_DIR:-$ROOT/build/head-candidate/bin}"
UDQ4KXL="$SNAP/Qwen3.6-27B-UD-Q4_K_XL.gguf"
Q8="$SNAP/Qwen3.6-27B-Q8_0.gguf"
Q40="$LEGACY_SNAP/Qwen3.6-27B-Q4_0.gguf"
Q41="$LEGACY_SNAP/Qwen3.6-27B-Q4_1.gguf"
Q4KM="$LEGACY_SNAP/Qwen3.6-27B-Q4_K_M.gguf"

# label | model | split-mode | devices  (single-GPU only where the model fits in 32 GiB)
SPECS=(
  "ud_q4_k_xl_single|$UDQ4KXL|none|ROCm0"
  "q8_0_dual|$Q8|tensor|ROCm0/ROCm1"
  "q4_0_single|$Q40|none|ROCm0"
  "q4_1_single|$Q41|none|ROCm0"
  "q4_k_m_dual|$Q4KM|layer|ROCm0/ROCm1"
  "q4_k_m_single|$Q4KM|none|ROCm0"
)

if [[ "${1:-}" == "--list" ]]; then
  printf '%s\n' "${SPECS[@]%%|*}"; exit 0
fi
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  FILTER=""
else
  FILTER="${1:-}"
fi

COOLDOWN="${COOLDOWN:-0}"     # seconds to idle between every llama-bench run (thermal soak relief)
RANDOMIZE="${RANDOMIZE:-0}"   # 1 = shuffle the test order
COOL_TEMP="${COOL_TEMP:-}"          # if set (e.g. 60), wait before each run until max GPU temp <= this. Overrides COOLDOWN.
COOL_SENSOR="${COOL_SENSOR:-edge}"  # which sensor to gate on: edge | junction | memory
COOL_TIMEOUT="${COOL_TIMEOUT:-300}" # give up waiting after this many seconds (don't hang forever)
COOL_POLL="${COOL_POLL:-5}"         # seconds between temp polls

# max temp (rounded C) across all GPUs for the chosen sensor; "0" if unreadable
gpu_max_temp() {
  rocm-smi --showtemp 2>/dev/null | awk -v s="$COOL_SENSOR" '
    $0 ~ ("Sensor " s) { v=$NF; gsub(/[^0-9.]/,"",v); if (v+0>m) m=v+0 }
    END { printf "%.0f", m+0 }'
}
# block until the hottest GPU is at/below target C (or COOL_TIMEOUT elapses)
wait_for_cool() {
  local target="$1" t waited=0
  while :; do
    t="$(gpu_max_temp)"
    [[ -z "$t" || "$t" -le "$target" ]] && break
    if [[ "$waited" -ge "$COOL_TIMEOUT" ]]; then
      echo "  [cool] timeout: ${COOL_SENSOR} ${t}C still > ${target}C after ${waited}s"; break
    fi
    sleep "$COOL_POLL"; waited=$((waited + COOL_POLL))
  done
  [[ -n "$t" ]] && echo "  [cool] ${COOL_SENSOR} ${t}C <= ${target}C (waited ${waited}s)"
}

# Expand selected specs into separate PP and TG jobs. Keeping them separate
# avoids the cross-product that llama-bench creates between -p, -n, and -d.
JOBS=()
for spec in "${SPECS[@]}"; do
  IFS='|' read -r label model sm dev <<< "$spec"
  [[ " $CONFIGS " != *" $label "* ]] && continue
  [[ -n "$FILTER" && "$label" != *"$FILTER"* ]] && continue
  if [[ " $MODES " == *" pp "* ]]; then
    JOBS+=("$label|$model|$sm|$dev|-p $PROMPTS -n 0 -d 0")
  fi
  if [[ " $MODES " == *" tg "* ]]; then
    for d in $DEPTHS; do
      JOBS+=("$label|$model|$sm|$dev|-p 0 -n $GEN_TOKENS -d $d")
    done
  fi
done
[[ ${#JOBS[@]} -gt 0 ]] || { echo "No benchmark configurations selected."; exit 2; }
[[ "$RANDOMIZE" == "1" ]] && mapfile -t JOBS < <(printf '%s\n' "${JOBS[@]}" | shuf)
if [[ "$DRY_RUN" == "1" ]]; then
  printf '%s\n' "${JOBS[@]}"
  exit 0
fi
source vr/scripts/setup-therock-env.sh >/dev/null 2>&1 || { echo "ROCm env failed"; exit 2; }

gap_desc=$([[ -n "$COOL_TEMP" ]] && echo "cool-to-${COOL_TEMP}C(${COOL_SENSOR})" || echo "cooldown=${COOLDOWN}s")
echo "REPS=$REPS BUILDS='$BUILDS' CONFIGS='$CONFIGS' MODES='$MODES' PROMPTS=$PROMPTS DEPTHS='$DEPTHS' GEN_TOKENS=$GEN_TOKENS gap=$gap_desc RANDOMIZE=$RANDOMIZE EXTRA='$BENCH_EXTRA_ARGS'"
echo "OUT=$OUT FILTER='${FILTER:-<all>}'  jobs=${#JOBS[@]} x builds"
echo "Models:"
for s in "${SPECS[@]}"; do
  IFS='|' read -r l m _ _ <<< "$s"
  [[ " $CONFIGS " != *" $l "* ]] && continue
  [[ -n "$FILTER" && "$l" != *"$FILTER"* ]] && continue
  [[ -r "$m" ]] || { echo "missing model: $m"; exit 2; }
  printf '  %-16s %s\n' "$l" "$(basename "$m")"
done
binary_for_build() {
  case "$1" in
    control)   printf '%s/llama-bench\n' "$CONTROL_BIN_DIR" ;;
    candidate) printf '%s/llama-bench\n' "$CANDIDATE_BIN_DIR" ;;
    *) echo "unknown build label: $1" >&2; return 2 ;;
  esac
}

for b in $BUILDS; do
  bin="$(binary_for_build "$b")"
  [[ -x "$bin" ]] || { echo "missing binary: $bin"; exit 2; }
  mkdir -p "$OUT/$b"
done
{
  date --iso-8601=seconds
  printf 'branch_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'builds=%s\nconfigs=%s\nmodes=%s\nprompts=%s\ndepths=%s\ngen_tokens=%s\nreps=%s\nbench_extra_args=%s\n' \
    "$BUILDS" "$CONFIGS" "$MODES" "$PROMPTS" "$DEPTHS" "$GEN_TOKENS" "$REPS" "$BENCH_EXTRA_ARGS"
  for b in $BUILDS; do
    bin="$(binary_for_build "$b")"
    sha256sum "$bin"
    for lib in libllama.so libggml.so libggml-hip.so; do
      [[ -r "$(dirname "$bin")/$lib" ]] && sha256sum "$(dirname "$bin")/$lib"
    done
  done
  rocm-smi -c -P -t 2>/dev/null || true
} > "$OUT/manifest.txt"

# Run each job on control AND candidate back-to-back (interleaved) so both builds
# see similar thermal/clock state. This limits drift but does not remove
# fixed-order bias, which is why only a winning screen is repeated in reverse
# order.
declare -A SEEN
first=1
for job in "${JOBS[@]}"; do
  IFS='|' read -r label model sm dev targs <<< "$job"
  for build in $BUILDS; do
    BIN="$(binary_for_build "$build")"
    of="$OUT/$build/${label}.jsonl"
    [[ -z "${SEEN[$build/$label]:-}" ]] && { : > "$of"; : > "$of.err"; SEEN[$build/$label]=1; }
    if [[ $first -eq 0 ]]; then
      if [[ -n "$COOL_TEMP" ]]; then wait_for_cool "$COOL_TEMP"      # temp-gated wait (preferred)
      elif [[ $COOLDOWN -gt 0 ]]; then sleep "$COOLDOWN"; fi          # fixed cooldown fallback
    fi
    first=0
    common=(-ngl 99 -fa on -o jsonl -r "$REPS" --progress -m "$model" -sm "$sm" -dev "$dev")
    common+=("${BENCH_EXTRA_ARGV[@]}")
    # Keep the warmup run (default): absorbs HIP-graph capture, rocBLAS init,
    # weight page-in, -d KV prefill, clock ramp. Most important at REPS=1.
    [[ "${WARMUP:-1}" == "1" ]] || common+=(--no-warmup)
    echo "[$(date +%T)] [$build] $label  model=$(basename "$model")  :: $targs"
    "$BIN" "${common[@]}" $targs >> "$of" 2>>"$of.err"
  done
done
echo "[$(date +%T)] BENCH DONE -> $OUT"
echo "View: vr/scripts/watch-bench.sh $OUT   (or: python3 vr/scripts/_bench_table.py $OUT)"
