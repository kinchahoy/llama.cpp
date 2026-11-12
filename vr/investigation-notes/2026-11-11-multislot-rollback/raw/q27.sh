#!/usr/bin/env bash
# usage: q27.sh BUILD_DIR TAG NP SPEC(0|1) [CTX]   Qwen 27B Q8_0 MTP2: token check + 1-user speed + NP concurrent
D=$(cd "$(dirname "$0")" && pwd)
REPO=/home/raistlin/infer/vrversion-llama.cpp
P=$REPO/vr/investigation-notes/2026-11-11-unified-kv-unverified/verification/prompts
B=$1; TAG=$2; NP=$3; SPEC=$4; CTX=${5:-32768}
ARGS="-ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -t 20 --fit off -ctk f16 -ctv f16 -m /home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf -b 2048 -ub 2048 -np $NP --kv-unified -c $CTX"
if [ "$SPEC" = 1 ]; then
  ARGS="$ARGS --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling"
fi
CMD="python3 -I $D/tokcheck.py \$PORT $P 256 $D/out/$TAG.json $NP; ${NOSPEED:+true ||} python3 -I $REPO/vr/scripts/spec_accept.py \$PORT $P 256; ${NOSPEED:+true ||} python3 -I $D/../../2026-11-11-multiuser-serving/raw/conc.py \$PORT $P 256 $NP"
bash $D/srv.sh $B $TAG "$CMD" -- $ARGS
