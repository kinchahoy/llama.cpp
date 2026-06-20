#!/usr/bin/env bash
# Reusable head-vs-head+patches benchmark for Qwen3.6-27B MTP on gfx906.
# Fast by default (REPS=1); bump reps to confirm a specific result.
#
# Defaults are tuned for SPEED: one repetition. llama-bench still does a warmup
# run, so r=1 is a usable single-sample. When a delta looks real or noisy,
# re-run just that config with more reps (examples below) instead of paying for
# high reps on every cell.
#
# Usage:
#   vr/scripts/bench-head.sh [FILTER]      # run configs whose label matches FILTER (default: all)
#   vr/scripts/bench-head.sh --list        # list config labels and exit
#
# Examples:
#   vr/scripts/bench-head.sh                       # full matrix, r=1 (fast)
#   REPS=5 vr/scripts/bench-head.sh q4_k_m_dual    # double-check ONE config at r=5
#   BUILDS=candidate vr/scripts/bench-head.sh q8_0 # only the candidate build, q8_0 configs
#
# Env overrides:
#   REPS=1                  llama-bench repetitions
#   BUILDS="control candidate"
#   DEPTHS="8192 16384"     high-context TG depths (short TG @ d0 always included)
#   PROMPTS="512,8192"      prompt sizes for the depth-0 PP sweep
#   COOLDOWN=0              seconds idle between every run (e.g. 30) to relieve heat-soak
#   COOL_TEMP=              if set (e.g. 60), wait before each run until the hottest GPU
#                          reaches this temp. Overrides COOLDOWN. COOL_SENSOR=edge|junction|
#                          memory (default edge), COOL_TIMEOUT=300 caps the wait.
#   RANDOMIZE=0             1 = shuffle test order
# A/B note: control and candidate are interleaved per test, so the delta stays
# unbiased under thermal drift even at COOLDOWN=0.
#   OUT=vr/bench-results/gfx906-head-<date>
#   SNAP=<hf snapshot dir with the gguf files>
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source vr/scripts/setup-therock-env.sh >/dev/null 2>&1 || { echo "ROCm env failed"; exit 2; }

REPS="${REPS:-1}"
BUILDS="${BUILDS:-control candidate}"
# Lean, high-context-weighted defaults. TG is the priority, so we keep two TG
# depths and only a minimal PP set. Expand via env when you want more:
#   high-context PP too:  PROMPTS=512,8192,16384
#   deeper TG:            DEPTHS="8192 16384 32768"   (check VRAM fits)
#   max-lean smoke test:  PROMPTS=8192 DEPTHS=16384
DEPTHS="${DEPTHS:-8192 16384}"     # high-context TG depths (the focus)
PROMPTS="${PROMPTS:-512,8192}"     # short ref (512) + 8K high-context prefill
SNAP="${SNAP:-$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace}"
OUT="${OUT:-$ROOT/vr/bench-results/gfx906-head-$(date +%Y-%m-%d)}"
Q4KM="$SNAP/Qwen3.6-27B-Q4_K_M.gguf"
Q8="$SNAP/Qwen3.6-27B-Q8_0.gguf"

# label | model | split-mode | devices  (single-GPU only where the model fits in 32 GiB)
SPECS=(
  "q4_k_m_dual|$Q4KM|layer|ROCm0/ROCm1"
  "q4_k_m_single|$Q4KM|none|ROCm0"
  "q8_0_dual|$Q8|layer|ROCm0/ROCm1"
)

if [[ "${1:-}" == "--list" ]]; then
  printf '%s\n' "${SPECS[@]%%|*}"; exit 0
fi
FILTER="${1:-}"

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

# Expand specs x invocations into a flat job list: "label|model|sm|dev|testargs".
# Depth runs use -p 0 so we don't pay for a pointless pp512 timing at depth.
JOBS=()
for spec in "${SPECS[@]}"; do
  IFS='|' read -r label model sm dev <<< "$spec"
  [[ -n "$FILTER" && "$label" != *"$FILTER"* ]] && continue
  JOBS+=("$label|$model|$sm|$dev|-p $PROMPTS -n 128")          # PP sweep + short TG @ d0
  for d in $DEPTHS; do
    JOBS+=("$label|$model|$sm|$dev|-p 0 -n 128 -d $d")         # TG @ depth (high-context, the focus)
  done
done
[[ "$RANDOMIZE" == "1" ]] && mapfile -t JOBS < <(printf '%s\n' "${JOBS[@]}" | shuf)

gap_desc=$([[ -n "$COOL_TEMP" ]] && echo "cool-to-${COOL_TEMP}C(${COOL_SENSOR})" || echo "cooldown=${COOLDOWN}s")
echo "REPS=$REPS BUILDS='$BUILDS' PROMPTS=$PROMPTS DEPTHS='$DEPTHS' gap=$gap_desc RANDOMIZE=$RANDOMIZE"
echo "OUT=$OUT FILTER='${FILTER:-<all>}'  jobs=${#JOBS[@]} x builds"
echo "Models:"
for s in "${SPECS[@]}"; do
  IFS='|' read -r l m _ _ <<< "$s"
  [[ -n "$FILTER" && "$l" != *"$FILTER"* ]] && continue
  printf '  %-16s %s\n' "$l" "$(basename "$m")"
done
for b in $BUILDS; do
  [[ -x "$ROOT/build/head-$b/bin/llama-bench" ]] || { echo "missing binary: build/head-$b/bin/llama-bench"; exit 2; }
  mkdir -p "$OUT/$b"
done

# Run each job on control AND candidate back-to-back (interleaved) so both builds
# see near-identical thermal/clock state — the A/B delta stays unbiased even if
# absolute throughput drifts as the cards heat-soak over a long run. (The old
# all-control-then-all-candidate ordering biased whichever build ran second.)
declare -A SEEN
first=1
for job in "${JOBS[@]}"; do
  IFS='|' read -r label model sm dev targs <<< "$job"
  for build in $BUILDS; do
    BIN="$ROOT/build/head-$build/bin/llama-bench"
    of="$OUT/$build/${label}.jsonl"
    [[ -z "${SEEN[$build/$label]:-}" ]] && { : > "$of"; : > "$of.err"; SEEN[$build/$label]=1; }
    if [[ $first -eq 0 ]]; then
      if [[ -n "$COOL_TEMP" ]]; then wait_for_cool "$COOL_TEMP"      # temp-gated wait (preferred)
      elif [[ $COOLDOWN -gt 0 ]]; then sleep "$COOLDOWN"; fi          # fixed cooldown fallback
    fi
    first=0
    common=(-ngl 99 -fa on -o jsonl -r "$REPS" --progress -m "$model" -sm "$sm" -dev "$dev")
    # Keep the warmup run (default): absorbs HIP-graph capture, rocBLAS init,
    # weight page-in, -d KV prefill, clock ramp. Most important at REPS=1.
    [[ "${WARMUP:-1}" == "1" ]] || common+=(--no-warmup)
    echo "[$(date +%T)] [$build] $label  model=$(basename "$model")  :: $targs"
    "$BIN" "${common[@]}" $targs >> "$of" 2>>"$of.err"
  done
done
echo "[$(date +%T)] BENCH DONE -> $OUT"
echo "View: vr/scripts/watch-bench.sh $OUT   (or: python3 vr/scripts/_bench_table.py $OUT)"
