#!/usr/bin/env bash
# usage: vr/server/serve.sh [extra llama-server args]
# Router llama-server with the presets in models.ini (one model loaded at a time).
# Env: BUILD (default: build-play-b1), HOST (127.0.0.1), PORT (8080).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
BUILD=${BUILD:-/home/raistlin/infer/vrversion-llama.cpp/build-play-b1}
source /home/raistlin/amd-clean/env/therock.sh
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
exec "$BUILD/bin/llama-server" --offline \
  --models-preset "$HERE/models.ini" --models-max 1 \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" "$@"
