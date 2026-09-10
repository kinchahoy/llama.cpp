# Build and Quickstart Guide: ROCm Multi-GPU Inference

This branch combines:
1. **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp):** Upstream base engine (`b10884`)
2. **[mixa3607/mx-llama.cpp](https://github.com/mixa3607/mx-llama.cpp):** Pipeline tensor parallelism
3. **[Stastez/llama.cpp](https://github.com/ggml-org/llama.cpp/pull/27825):** ROCm PCIe AllReduce (PR #27825)
4. **[iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906):** gfx906 Q8 repacking

**New in Patch 03:** Adds `gfx906` Q8_0/MXFP4 weight repacking, chunked gated-delta-net prefill kernels, and multi-GPU MoE shared expert tensor parallelism.



---

## 1. Prerequisites

- Linux (Ubuntu 22.04/24.04 or compatible)
- AMD ROCm 6.x or 7.x with HIP development libraries
- Clang / hipcc
- CMake 3.22+

Ensure ROCm environment variables are set (adjust path if using a custom ROCm installation):

```sh
export ROCM_PATH=/opt/rocm
export HIP_PATH=$ROCM_PATH
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

---

## 2. Build Instructions

Configure with CMake targeting your AMD GPU architecture (e.g., `gfx906` for Vega 20 / Radeon VII / MI50 / MI60, `gfx908` for MI100, `gfx90a` for MI210/MI250):

```sh
cmake -S . -B build \
  -DAMDGPU_TARGETS=gfx906 \
  -DGGML_HIP=ON \
  -DGGML_HIP_RCCL=OFF \
  -DGGML_CCACHE=OFF \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j $(nproc) --target llama-cli llama-bench llama-server
```

---

## 3. Quickstart: Canonical Benchmark (Qwen3.8-27B-Q8_0)

Screening benchmark across 2x ROCm GPUs over PCIe at 16K context depth. This automatically resolves and downloads or uses the cached Hugging Face model:

```sh
HIP_VISIBLE_DEVICES=0,1 \
./build/bin/llama-bench \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 \
  -ngl 999 -fa on -sm tensor -dev ROCm0/ROCm1 \
  -b 4096 -ub 4096 -p 0 -n 0 \
  -pg 2048,0 -pg 0,256 -d 16384 \
  -r 1 -o jsonl --progress --no-warmup
```

Expected performance on 2x AMD MI60 (`gfx906`):
- **PP2048:** ~380-420 t/s
- **TG256:** ~25-27 t/s

---

## 4. CLI Inference with Prompt File

Run interactive or prompt-file inference with `llama-cli`:

```sh
HIP_VISIBLE_DEVICES=0,1 \
./build/bin/llama-cli \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 \
  -f flake.nix \
  -ngl 999 -fa on -sm tensor -dev ROCm0/ROCm1 \
  -b 4096 -ub 4096 -c 16384 \
  -n 256 --no-warmup
```

---

## 5. Local Server

Start the multi-GPU OpenAI-compatible server:

```sh
HIP_VISIBLE_DEVICES=0,1 \
./build/bin/llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 \
  -ngl 999 -fa on -sm tensor -dev ROCm0/ROCm1 \
  -b 4096 -ub 4096 -c 16384 \
  --host 127.0.0.1 --port 8080
```

