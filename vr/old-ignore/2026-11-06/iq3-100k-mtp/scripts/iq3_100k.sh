#!/usr/bin/env bash
# usage: iq3_100k.sh OUT BUILD   UD-IQ3_XXS (Q8_0 table), ctx 100352, F16 KV: plain and MTP2 with three drafts on ROCm1; -ub 512, falling back to 256 on failure
OUT=$1; B=$2
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
M=/home/raistlin/models/flashnext/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf
HUB=/home/raistlin/.cache/huggingface/hub
DQ4=$HUB/models--bartowski--Qwen3.8-Flash-Next-GGUF/snapshots/db3a185731e162be9ffa4d52d20d7509939e60e8/mtp-Qwen3.8-Flash-Next-Q4_0.gguf
DQ4KM=$HUB/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/d96201194f103c906994f283f504209dc8d2a10c/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf
DQ8=$HUB/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/d96201194f103c906994f283f504209dc8d2a10c/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
PROMPT=/home/raistlin/infer/vrversion-llama.cpp/flake.nix
SHA=/home/raistlin/infer/vrversion-llama.cpp/vr/old-ignore/2026-11-05/catchup-4b77dee05/scripts/reply_sha.py
run() {  # label ts b ub [spec...]
  local label=$1 ts=$2 b=$3 ub=$4; shift 4
  local f=$OUT/$label
  timeout --kill-after=5s 600s "$B/bin/llama-cli" --offline -m "$M" -ngl 999 -dev ROCm0,ROCm1 -sm tensor -ts $ts -fa on \
    -b $b -ub $ub -c 100352 -t 20 --temp 0 --top-k 1 --seed 42 \
    -f "$PROMPT" -n 512 --no-display-prompt --simple-io "$@" > "$f.txt" 2> "$f.err" < /dev/null
  local rc=$?
  echo "$label rc=$rc $(grep -oE 'Prompt: [0-9.]+ t/s|Generation: [0-9.]+ t/s' "$f.txt" | tr '\n' ' ') $(python3 -I "$SHA" "$f.txt" 2>/dev/null | cut -d' ' -f1-3)"
  grep -iE 'out of memory|failed to (allocate|load)|abort' "$f.err" | head -1
  return $rc
}
mtp() { echo -md "$1" -devd ROCm1 --spec-draft-n-ubatch 64 --spec-type draft-mtp --spec-draft-n-max "$2" --spec-draft-backend-sampling; }
run plain-ub512 1,1 1024 512 || run plain-ub256 1,1 512 256
for d in q8:$DQ8; do
  name=${d%%:*}; path=${d#*:}
  ts=1,1; [ $name = q8 ] && ts=0.52,0.48
  run mtp2-$name-ub512 $ts 1024 512 $(mtp "$path" 2) || run mtp2-$name-ub256 $ts 512 256 $(mtp "$path" 2)
done
