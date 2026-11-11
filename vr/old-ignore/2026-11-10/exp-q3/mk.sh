#!/bin/bash
# usage: mk.sh <src.hip> <out>
source /home/raistlin/amd-clean/env/therock.sh
W=/home/raistlin/infer/vrversion-llama.cpp/.claude/worktrees/exp-q3
CXX=$(rocm-sdk path --root)/lib/llvm/bin/clang++
[ -x "$CXX" ] || CXX=/home/raistlin/amd-clean/therock-venv/.venv/lib/python3.14/site-packages/_rocm_sdk_devel/llvm/bin/clang++
ROOT=$(rocm-sdk path --root)
$CXX -x hip -O3 -std=gnu++17 --offload-arch=gfx906 -mllvm -amdgpu-sched-strategy=max-ilp \
  -DGGML_USE_HIP -DNDEBUG -D__HIP_PLATFORM_AMD__=1 \
  -I$W/ggml/src -I$W/ggml/include -I$W/ggml/src/ggml-cuda \
  "$1" -o "$2" -L/home/raistlin/infer/vrversion-llama.cpp/build-catchup-20261110-151625/bin -lggml-base \
  -Wl,-rpath,/home/raistlin/infer/vrversion-llama.cpp/build-catchup-20261110-151625/bin \
  --rocm-path=$ROOT -lamdhip64 -Wno-unused-command-line-argument 2>&1 | grep -v "warning:" | head -40
