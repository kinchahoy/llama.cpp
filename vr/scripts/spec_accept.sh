#!/usr/bin/env bash
# usage: spec_accept.sh NAME BUILD_DIR OUT_DIR q27|fn [ENV=VAL ...]
# Speculative A/B with acceptance: llama-server, 5 prompts (code and prose), 256 greedy tokens each.
# q27 = 27B Q8_0 MTP2; fn = Flash Next UD-IQ3_XXS MTP3, Q8_0 sidecar, 100K ctx (TS and DEVD env override the split and draft device).
set -u
NAME=$1; BUILD=$2; OUT=$3; MODEL=$4; shift 4
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
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
FN=/home/raistlin/models/flashnext/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf
DQ8=$HF/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/d96201194f103c906994f283f504209dc8d2a10c/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
if [ "$MODEL" = q27 ]; then
  A="-m $M27 -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -b 2048 -ub 2048 -c 16384 -t 20 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling"
else
  A="-m $FN -ngl 999 -dev ROCm0,ROCm1 -sm tensor -ts ${TS:-0.52,0.48} -fa on -b 1024 -ub 512 -c 100352 -t 20 -md $DQ8 -devd ${DEVD:-ROCm1} --spec-draft-n-ubatch 64 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-backend-sampling"
fi
PORT=${PORT:-18091}
"$BUILD/bin/llama-server" --offline $A -np 1 --host 127.0.0.1 --port $PORT > "$OUT/$NAME.server.log" 2>&1 &
S=$!
for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 2; done
echo "== $NAME"
python3 -I "$HERE/spec_accept.py" $PORT "$OUT/prompts" 256 | tee "$OUT/$NAME.txt"
kill $S; wait $S 2> /dev/null
