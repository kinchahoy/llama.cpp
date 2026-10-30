#!/usr/bin/env bash
# usage: configure_stage.sh SRC_DIR  (configures + builds SRC_DIR/build-rb with the build-ilp flags of stage B)
set -eu
SRC=$1
R=/home/raistlin/amd-clean/therock-venv/.venv/lib/python3.14/site-packages/_rocm_sdk_devel
rm -f "$SRC/build-rb/CMakeCache.txt"
cmake -S "$SRC" -B "$SRC/build-rb" -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_RCCL=OFF -DGGML_HIP_GRAPHS=ON -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_PREFIX_PATH=$R/lib/cmake -DHIP_COMPILER=clang \
  -DCMAKE_C_COMPILER=$R/llvm/bin/clang -DCMAKE_CXX_COMPILER=$R/llvm/bin/clang++ -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++ \
  "-DCMAKE_CXX_FLAGS=-O3 -Wno-unused-command-line-argument" "-DCMAKE_HIP_FLAGS=-mllvm -amdgpu-sched-strategy=max-ilp" > /dev/null
cmake --build "$SRC/build-rb" -j 20 > "$SRC/build-rb/build.log" 2>&1 || { grep -E 'error:' "$SRC/build-rb/build.log" | head -20; exit 1; }
echo "build ok: $SRC"
