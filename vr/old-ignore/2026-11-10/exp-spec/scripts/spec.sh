#!/usr/bin/env bash
# usage: spec.sh NAME BUILD OUT mtp2|d5 [ENV=VAL ...]   (vr/scripts/spec_accept.sh workload, 27B)
set -u
NAME=$1; BUILD=$2; OUT=$3; MODE=$4; shift 4
REPO=/home/raistlin/infer/vrversion-llama.cpp/.claude/worktrees/exp-spec
mkdir -p "$OUT/prompts"
head -c 6000 "$REPO/flake.nix" > "$OUT/prompts/p1.txt"
head -c 6000 "$REPO/src/llama-context.cpp" > "$OUT/prompts/p2.txt"
head -c 4000 "$REPO/tools/server/README.md" > "$OUT/prompts/p3.txt"
printf 'Write a detailed history of the printing press, covering Gutenberg, the spread across Europe, and its social effects.' > "$OUT/prompts/p4.txt"
printf 'Write a Python function that parses an ISO 8601 duration string like P3DT4H5M into seconds, with tests.' > "$OUT/prompts/p5.txt"
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
for kv in "$@"; do export "$kv"; done
HF=/home/raistlin/.cache/huggingface/hub
M27=$HF/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
DFL=$HF/models--z-lab--Qwen3.8-27B-DFlash2-GGUF/snapshots/2d9571f8ce46e151f61c6499c99dee6079e1d610/Qwen3.8-27B-DFlash2-Q8_0.gguf
A="-m $M27 -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -b 2048 -ub 2048 -c 16384 -t 20"
if [ "$MODE" = mtp2 ]; then
  A="$A --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling"
else
  A="$A --spec-type draft-dflash --spec-draft-n-max 5 -md $DFL"
fi
PORT=${PORT:-18093}
timeout 560 "$BUILD/bin/llama-server" --offline $A ${EXTRA:-} -np 1 --host 127.0.0.1 --port $PORT > "$OUT/$NAME.server.log" 2>&1 &
S=$!
for i in $(seq 1 150); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 2; done
echo "== $NAME"
timeout 400 python3 -I /tmp/exp-spec-x/spec.py $PORT "$OUT/prompts" 256 "$OUT/$NAME" | tee "$OUT/$NAME.txt"
kill $S; wait $S 2> /dev/null
grep -iE "warn|fall|dev rows|device|dflash.*dev" "$OUT/$NAME.server.log" | grep -viE "ggml_cuda_init|Device [0-9]|load_tensors|llama_model_load|using device" | head -8
