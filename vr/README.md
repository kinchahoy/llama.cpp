# gfx906 Q8 tensor-parallel overlap experiment

This directory isolates a private-fork experiment for Qwen3.6 and Qwen3.8 27B Q8_0 tensor-parallel down projections on two MI60 GPUs. It also accelerates eligible Q8_0 down tensors inside UD-Q8_K_XL.

The optimized path is deliberately narrow. It requires HIP on gfx906, Q8_0 weights, the exact local matrix shape `8704 x 5120` by `8704 x 2048`, two devices, stream 0, and internal all-reduce. Other shapes use the normal paths.

## Build

Current merged base: `46b95d97e` (build 10279). The original performance campaign used `b8efb9510` (build 10275).

```sh
cmake -S . -B build \
  -DAMDGPU_TARGETS=gfx906 -DGGML_HIP=ON -DGGML_HIP_RCCL=ON \
  -DGGML_CCACHE=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_VR_EXPERIMENTS=ON
cmake --build build -j 16 --target \
  llama-server llama-cli llama-bench vr-test-gfx906-tp
```

## GOOO: Qwen3.8 Q8_0

These commands use the tested local model, tensor split, 2048 batch and ubatch, the exact PP overlap, and MTP depth 2. The server binds only to localhost; use an SSH tunnel or add authentication before exposing it.

Server:

```sh
cd ~/infer/mx-llama.cpp && env HIP_VISIBLE_DEVICES=0,1 GGML_CUDA_ALLREDUCE=internal GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_CUDA_TP_OVERLAP=1 GGML_CUDA_TP_OVERLAP_BF16=1 ./build/bin/llama-server -m /home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/fe1e2a23d973adb629709749dc4f6756df66ef10/Qwen3.8-27B-Q8_0.gguf -ngl 99 -fa on -sm tensor -dev ROCm0,ROCm1 -lm dio -b 2048 -ub 2048 -c 16384 --spec-type draft-mtp --spec-draft-n-max 2 --host 127.0.0.1 --port 8080
```

CLI:

```sh
cd ~/infer/mx-llama.cpp && env HIP_VISIBLE_DEVICES=0,1 GGML_CUDA_ALLREDUCE=internal GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_CUDA_TP_OVERLAP=1 GGML_CUDA_TP_OVERLAP_BF16=1 ./build/bin/llama-cli -m /home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/fe1e2a23d973adb629709749dc4f6756df66ef10/Qwen3.8-27B-Q8_0.gguf -ngl 99 -fa on -sm tensor -dev ROCm0,ROCm1 -lm dio -b 2048 -ub 2048 -c 16384 --spec-type draft-mtp --spec-draft-n-max 2 -cnv
```

Paired PP2048/TG1024 benchmark at depth 16384:

```sh
cd ~/infer/mx-llama.cpp && env HIP_VISIBLE_DEVICES=0,1 GGML_CUDA_ALLREDUCE=internal GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_CUDA_TP_OVERLAP=1 GGML_CUDA_TP_OVERLAP_BF16=1 ./build/bin/llama-bench -m /home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/fe1e2a23d973adb629709749dc4f6756df66ef10/Qwen3.8-27B-Q8_0.gguf -ngl 99 -fa on -sm tensor -dev ROCm0/ROCm1 -lm dio -b 2048 -ub 2048 -p 0 -n 0 -pg 2048,0 -pg 0,1024 -d 16384 -r 1 -o jsonl --progress --no-warmup
```

`GGML_CUDA_TP_OVERLAP_BF16=1` changes reduction precision. Remove both `GGML_CUDA_TP_OVERLAP` variables to use the normal F32 path.

## Controls

- `GGML_CUDA_ALLREDUCE=internal`: use the HIP internal two-GPU transport.
- `GGML_CUDA_TP_OVERLAP=1`: split the exact Q8_0 down projection into two 1024-column slabs and overlap MMQ with transport.
- `GGML_CUDA_TP_OVERLAP_BF16=1`: use BF16 instead of F32 for the overlap wire buffer. This only has an effect when overlap is enabled.

All controls are opt-in. Keep `GGML_ENABLE_CUSTOM_AR=1` and `HSA_FORCE_FINE_GRAIN_PCIE=1` set in full-model tests so small-message behavior matches the established mx configuration.

The measured fast configuration enables both overlap variables. BF16 wire rounds the two partial sums before they are made identical on both GPUs. It is not numerically equivalent to the F32 reduction, so treat it as a speed/precision tradeoff and validate model quality for the workload.

## Test order

1. Build and run `vr-test-gfx906-tp --n 1 --check` with overlap disabled.
2. Run `--n 2048 --check` with internal transport only, F32 overlap, and BF16 overlap.
3. Compare one-run, no-warmup Q8_0 PP2048 and TG1024 at context 16384. Wait at least 3 seconds between arms.
4. Keep the change only if the weighted score `TG + PP/10` improves without a correctness failure.

Commands and measured results are in [RESULTS.md](RESULTS.md). The compact machine-readable record is [results/2026-08-13-gfx906-q8.csv](results/2026-08-13-gfx906-q8.csv), with provenance in [results/README.md](results/README.md).

The standalone implementation diff is [patches/q8-tp-overlap.patch](patches/q8-tp-overlap.patch). It excludes the `/vr` harness and documentation.

## Current recommendation

For Qwen3.6 or Qwen3.8 27B Q8_0, or eligible UD-Q8_K_XL tensors, at PP2048 on this two-MI60 host:

```sh
export GGML_CUDA_ALLREDUCE=internal
export GGML_ENABLE_CUSTOM_AR=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GGML_CUDA_TP_OVERLAP=1
export GGML_CUDA_TP_OVERLAP_BF16=1
```

This is a private, exact-shape optimization. Leave the two overlap variables unset for other models, shapes, GPUs, or when F32 reduction precision is required.
