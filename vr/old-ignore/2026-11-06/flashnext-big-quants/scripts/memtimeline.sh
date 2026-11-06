#!/usr/bin/env bash
# usage: memtimeline.sh OUT BUILD LABEL args...   run llama-cli with the given args, sample per-GPU VRAM every 0.2 s with elapsed seconds
OUT=$1; B=$2; LABEL=$3; shift 3
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
C0=/sys/class/drm/card1/device/mem_info_vram_used
C1=/sys/class/drm/card2/device/mem_info_vram_used
T0=$(date +%s.%N)
( while [ ! -f "$OUT/$LABEL.done" ]; do
    printf '%s %s %s\n' "$(python3 -I -c "print(f'{$(date +%s.%N)-$T0:.1f}')")" "$(cat $C0)" "$(cat $C1)"
    sleep 0.2
  done > "$OUT/$LABEL.timeline" ) &
timeout --kill-after=5s 600s "$B/bin/llama-cli" "$@" > "$OUT/$LABEL.txt" 2> "$OUT/$LABEL.err" < /dev/null
echo "rc=$?"
touch "$OUT/$LABEL.done"; sleep 0.5
grep -iE 'out of memory|failed to (allocate|load)|load_tensors|sched_reserve|common_speculative_init|llama_context:' "$OUT/$LABEL.err" | sed -E 's/^([0-9.]+) [A-Z] /\1 /' | cut -c1-130 | head -20
python3 -I - "$OUT/$LABEL.timeline" <<'PY'
import sys
rows = [l.split() for l in open(sys.argv[1]) if len(l.split()) == 3]
G = 2**30
last = None
for t, a, b in rows:
    key = (round(int(a) / G, 1), round(int(b) / G, 1))
    if key != last:
        print(f"  t={float(t):6.1f}s  ROCm0 {int(a)/G:6.2f}  ROCm1 {int(b)/G:6.2f} GiB")
        last = key
PY
