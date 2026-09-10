# gfx906 Q8 tensor-parallel overlap experiment

This directory isolates a private-fork experiment for Qwen3.6 and Qwen3.8 27B Q8_0 tensor-parallel down projections on two MI60 GPUs. It also accelerates eligible Q8_0 down tensors inside UD-Q8_K_XL.

The optimized path is deliberately narrow. It requires HIP on gfx906, Q8_0 weights, the exact local matrix shape `8704 x 5120` by `8704 x 2048`, two devices, stream 0, and internal all-reduce. Other shapes use the normal paths.

## 2026-09-10 pure-head campaign

The current branch merges mx head `0c81bd502` and llama.cpp head `434ddbbc0`, then evaluates the ROCm host-staged AllReduce from PR 27825. RCCL is disabled because it does not work on this host's PCIe topology.

Current winner: 394.1701 PP and 25.8895 TG at 16K, one repetition, after removing the obsolete HIP overlap-only guard in `ggml_backend_cuda_comm_allreduce_internal()`. The earlier arms labeled PR 27825 still used the meta-backend fallback when overlap was off. A focused kernel trace now confirms execution of `ggml_cuda_ar_kernel<float, __hip_bfloat16>`. Keep repacking and parallel dispatch enabled, with the old overlap variables unset. The generic pipeline defaults to BF16 transport; focused execution checks passed, but model-quality equivalence has not been measured.

Use this as the screening benchmark. It loads the model once, performs no warmup, measures PP2048 and TG256, and places both measurements behind 16384 tokens of real cached depth. The 150000-token context allocation in the interactive `flake.nix` command is useful for application validation, but the 7246-byte file does not itself create a long-context workload.

```sh
cd /home/raistlin/infer/mx-llama.cpp && \
HIP_VISIBLE_DEVICES=0,1 \
./build/bin/llama-bench \
  -m /home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/fe1e2a23d973adb629709749dc4f6756df66ef10/Qwen3.8-27B-Q8_0.gguf \
  -ngl 999 -fa on -sm tensor -dev ROCm0/ROCm1 \
  -b 4096 -ub 4096 -p 0 -n 0 \
  -pg 2048,0 -pg 0,256 -d 16384 \
  -r 1 -o jsonl --progress --no-warmup
```

The active target is 300+ PP and 22-25 TG at depth 16384. The user canceled further 64K testing on 2026-09-10. Preserve the existing 64K result as historical evidence; do not run another 64K acceptance gate.

Keep batch and ubatch at 4096 for the canonical record. The 2048/2048 private-overlap arm reached 325.09 PP and 18.57 TG, so it did not beat the 4096/4096 PR 27825 arm at 325.71 PP and 18.76 TG. A smaller batch remains useful only when testing the exact PP2048 overlap path.

### Optimization backlog, 2026-09-10

These are ranked for this Qwen3.8 Q8_0 workload. Lift estimates are hypotheses, not measurements. PP rank and TG rank use 1 for the largest expected lift. Test rank uses 1 for the easiest useful falsification test.

| Overall | Idea | Expected PP lift | PP rank | Expected TG lift | TG rank | Test difficulty | Test rank | First minimal test |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 1 | Tune gfx906 Flash Attention tiles, wave count, and split-KV reduction specifically for Qwen3.8 attention layers at 64K | 5% to 15% | 1 | 4% to 10% | 1 | Medium-hard | 3 | Time existing FA variants on the model's exact head geometry at 64K, then run only the best full arm |
| 2 | Dispatch TG1 to a canonical-layout gfx906 GEMV while retaining repacked MMQ for PP | 0% to 1% | 8 | 5% to 10% | 3 | Medium | 2 | Compare one TG256 arm with only TG1 dispatch changed |
| 3 | Fuse Q8_0 dequantization, dot products, scale application, and output accumulation into a persistent gfx906 GEMV | 0% to 2% | 7 | 6% to 12% | 2 | Hard | 6 | Benchmark the dominant Q8_0 matrix shape in isolation, then TG256 once |
| 4 | Select PR 27825 copy-engine or chunked transport per tensor size and retain exact BF16 PP overlap only where it wins | 2% to 6% | 5 | 2% to 5% | 6 | Medium | 4 | Log tensor sizes and time one fixed-threshold arm against the PR default |
| 5 | Pipeline the next layer's TG weight reads with the current layer's host-staged reduction on separate streams | 1% to 4% | 6 | 4% to 8% | 4 | Hard | 7 | Add event timing around two adjacent layers before changing scheduling |
| 6 | Keep row-parallel output sharded through residual add and RMSNorm, reducing only the norm statistics instead of every hidden vector | 4% to 10% | 2 | 4% to 8% | 5 | Very hard | 8 | Prototype one block with a correctness comparison and focused timing |
| 7 | Fuse each Qwen3.8 recurrent block's gate, state update, and output elementwise work to remove launches and intermediate traffic | 3% to 8% | 4 | 2% to 5% | 7 | Hard | 5 | Profile launch and memory time for one recurrent block, then fuse its hottest chain |
| 8 | Sweep only 2048, 3072, and 4096 batch/ubatch and derive per-op MMQ thresholds instead of one global batch choice | 2% to 8% | 3 | 0% | 8 | Easy | 1 | Use PP2048 at zero depth for the three settings, then validate only the winner at 16K |

The backlog above predates the corrected AllReduce integration. The active 16K throughput target is now met at 394.17 PP and 25.89 TG. No further 64K tests are authorized. See RESULTS.md for the fallback comparison and the numerical-validation limitation.

## Build

Current HEAD: `c0df68229`, with the resolved `434ddbbc0` merge still uncommitted (build 10921). The original performance campaign used `b8efb9510` (build 10275).

```sh
cmake -S . -B build \
  -DAMDGPU_TARGETS=gfx906 -DGGML_HIP=ON -DGGML_HIP_RCCL=OFF \
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

## Historical exact-overlap recommendation

For Qwen3.6 or Qwen3.8 27B Q8_0, or eligible UD-Q8_K_XL tensors, at PP2048 on this two-MI60 host:

```sh
export GGML_CUDA_ALLREDUCE=internal
export GGML_ENABLE_CUSTOM_AR=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GGML_CUDA_TP_OVERLAP=1
export GGML_CUDA_TP_OVERLAP_BF16=1
```

This is a private, exact-shape optimization. Leave the two overlap variables unset for other models, shapes, GPUs, or when F32 reduction precision is required.
