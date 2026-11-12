#!/usr/bin/env bash
# usage: vr/server/serve.sh [extra llama-server args]
# Router llama-server with the presets in models.ini (one model loaded at a time).
# Env: BUILD (default: build-current), HOST (0.0.0.0), PORT (8080).
# build-current and build-previous are symlinks to real build dirs (never rename a build dir: its RUNPATH is absolute).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
BUILD=${BUILD:-/home/raistlin/infer/vrversion-llama.cpp/build-current}
source /home/raistlin/amd-clean/env/therock.sh
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP GGML_CUDA_FLASHNEXT_MOE
exec "$BUILD/bin/llama-server" --offline \
  --models-preset "$HERE/models.ini" --models-max 1 \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8080}" "$@"
