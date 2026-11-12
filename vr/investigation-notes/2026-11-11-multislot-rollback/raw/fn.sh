#!/usr/bin/env bash
# usage: fn.sh BUILD_DIR TAG NP SPEC(0|1) [CTX]   Flash Next: token check + 1-user speed + 2 concurrent
D=$(cd "$(dirname "$0")" && pwd)
REPO=/home/raistlin/infer/vrversion-llama.cpp
P=$REPO/vr/investigation-notes/2026-11-11-unified-kv-unverified/verification/prompts
B=$1; TAG=$2; NP=$3; SPEC=$4; CTX=${5:-32768}
ARGS="-ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -t 20 --fit off -ctk f16 -ctv f16 -m /home/raistlin/models/flashnext/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf -ts 0.52,0.48 -b 1024 -ub 512 -np $NP --kv-unified -c $CTX"
if [ "$SPEC" = 1 ]; then
  ARGS="$ARGS -md /home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/d96201194f103c906994f283f504209dc8d2a10c/mtp-Qwen3.8-Flash-Next-Q8_0.gguf -devd ROCm1 --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 64 --spec-draft-backend-sampling"
fi
CMD="python3 -I $D/tokcheck.py \$PORT $P 256 $D/out/$TAG.json $NP; ${NOSPEED:+true ||} python3 -I $REPO/vr/scripts/spec_accept.py \$PORT $P 256; ${NOSPEED:+true ||} python3 -I $D/../../2026-11-11-multiuser-serving/raw/conc.py \$PORT $P 256 $NP"
bash $D/srv.sh $B $TAG "$CMD" -- $ARGS
