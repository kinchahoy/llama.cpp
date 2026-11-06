#!/usr/bin/env bash
# usage: moe_default.sh BUILD OUT   rebuild, repack harness (grouped MoE now default), 100K MTP3 with no env set
B=$1; OUT=$2
mkdir -p "$OUT"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
cmake --build "$B" --target llama-cli llama-bench vr-test-flashnext-moe -j 8 > "$OUT/build.log" 2>&1 || { echo BUILD_FAILED; grep error: "$OUT/build.log" | head; exit 1; }
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE GGML_CUDA_CPY_2D
timeout 1200s "$B/bin/vr-test-flashnext-moe" > "$OUT/repack-moe.log" 2>&1
echo "repack-moe rc=$? cases=$(grep -c nmse= "$OUT/repack-moe.log") worst=$(grep -oE 'nmse=[0-9.e+-]+' "$OUT/repack-moe.log" | cut -d= -f2 | sort -g | tail -1)"
M=/home/raistlin/models/flashnext/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf
DQ8=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/d96201194f103c906994f283f504209dc8d2a10c/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
SHA=/home/raistlin/infer/vrversion-llama.cpp/vr/old-ignore/2026-11-05/catchup-4b77dee05/scripts/reply_sha.py
for r in 1 2; do
  f=$OUT/mtp3-default-r$r
  timeout --kill-after=5s 600s "$B/bin/llama-cli" --offline -m "$M" -ngl 999 -dev ROCm0,ROCm1 -sm tensor -ts 0.52,0.48 -fa on \
    -b 1024 -ub 512 -c 100352 -t 20 --temp 0 --top-k 1 --seed 42 -f /home/raistlin/infer/vrversion-llama.cpp/flake.nix -n 512 --no-display-prompt --simple-io \
    -md "$DQ8" -devd ROCm1 --spec-draft-n-ubatch 64 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling \
    > "$f.txt" 2> "$f.err" < /dev/null
  echo "mtp3 default r$r rc=$? $(grep -oE 'Prompt: [0-9.]+ t/s|Generation: [0-9.]+ t/s' "$f.txt" | tr '\n' ' ') $(python3 -I "$SHA" "$f.txt" 2>/dev/null | cut -d' ' -f1-3)"
done
