#!/usr/bin/env bash
# usage: memprobe.sh OUT BUILD LABEL [extra args...]   UD-Q3_K_XL (Q8_0 table) plain at ctx 100352, verbose log + per-GPU VRAM peak from sysfs
OUT=$1; B=$2; LABEL=$3; shift 3
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
M=/home/raistlin/models/flashnext/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf
C0=/sys/class/drm/card1/device/mem_info_vram_used
C1=/sys/class/drm/card2/device/mem_info_vram_used
T0=$(cat /sys/class/drm/card1/device/mem_info_vram_total)
T1=$(cat /sys/class/drm/card2/device/mem_info_vram_total)
B0=$(cat $C0); B1=$(cat $C1)
( m0=0; m1=0; while [ ! -f "$OUT/$LABEL.done" ]; do u0=$(cat $C0); u1=$(cat $C1); [ $u0 -gt $m0 ] && m0=$u0; [ $u1 -gt $m1 ] && m1=$u1; echo "$m0 $m1" > "$OUT/$LABEL.peak"; sleep 0.5; done ) &
timeout --kill-after=5s 600s "$B/bin/llama-completion" -m "$M" -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on \
  -ctk q8_0 -ctv q8_0 -b 1024 -ub 512 -c 100352 -t 20 --temp 0 --top-k 1 -s 42 -n 32 -no-cnv --simple-io \
  -p 'The history of the printing press begins in' -v "$@" < /dev/null > "$OUT/$LABEL.txt" 2> "$OUT/$LABEL.log"
rc=$?
touch "$OUT/$LABEL.done"; sleep 1
read p0 p1 < "$OUT/$LABEL.peak"
gib() { python3 -I -c "print(f'{$1/2**30:.2f}')"; }
echo "$LABEL rc=$rc  ROCm0 peak $(gib $p0)/$(gib $T0) GiB (idle $(gib $B0)), free $(gib $((T0-p0)))  |  ROCm1 peak $(gib $p1)/$(gib $T1) GiB (idle $(gib $B1)), free $(gib $((T1-p1)))"
grep -E 'model buffer size|KV buffer size|RS buffer size|compute buffer size|layer input buffer|repack|q8_1 cache' "$OUT/$LABEL.log" | sed -E 's/^[0-9.]+ [A-Z] //' | sort | uniq -c | head -25
