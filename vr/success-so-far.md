# Success so far: deltas over llama.cpp head 434ddbbc0

- **Result versus upstream (2026-09-10):** clean llama.cpp `434ddbbc0` measured **196.97 PP / 20.13 TG**; the final private fork measured **394.17 PP / 25.89 TG**: **+100.1% PP and +28.6% TG**. Both use Qwen3.8 27B Q8_0, 4096/4096 batch limits, and actual 16K cached history. These are single-run measurements taken at different times, not confidence intervals.
- **Attribution:** the private fork before the final fix measured **325.49 PP / 18.68 TG** in the same session. Enabling the intended HIP AllReduce raised that to 394.17 / 25.89 (**+21.1% / +38.6%**). The total gain over upstream includes inherited private-fork optimizations; the small AllReduce port alone has not been benchmarked on otherwise-clean upstream.
- **Hardware/software:** two AMD MI60 GPUs, 32 GiB each, connected over PCIe; `gfx906` is their GPU architecture identifier. ROCm is AMD's GPU software stack; HIP is its CUDA-like C++ programming interface. `llama.cpp` runs inference; its `ggml` backend implements tensor operations and GPU communication. Q8_0 denotes blockwise 8-bit weight quantization; GGUF is the model file format.
- **Workload:** tensor parallelism splits each layer's matrix work across GPUs. PP measures processing a batch of prompt tokens; TG measures ordinary autoregressive generation, without speculative decoding. Flash Attention avoids materializing the full attention matrix. Here PP2048 and TG256 both follow a real 16K cache fill, not merely a 16K context allocation.

## Delta 1: inherited private-fork compute and scheduling

- **AllReduce** sums partial results across GPUs and makes the result available on each GPU. **RCCL** is AMD's collective-communication library, analogous to NVIDIA NCCL; it is disabled because it does not work on this host's PCIe topology. **Butterfly** is ggml's generic staged exchange-and-add fallback for the same collective, not a different mathematical operation.
- **Weight repacking:** `ggml/src/ggml-cuda/q8_repack/` separates Q8_0 integer weights and scales at upload. Its matrix-matrix kernels serve PP; matrix-vector kernels serve TG, including fused gate/up projections. Loader buffer selection, dispatch, and tensor-parallel buffer handling must agree on this layout. This is inherited code, not a kernel rewrite from this session; its implementation map is in [q8_repack/README.md](ggml/src/ggml-cuda/q8_repack/README.md).
- **Scheduling:** inherited `ggml/src/ggml-backend-meta.cpp` changes dispatch GPU work concurrently through a host worker pool. Preserve these and the other private mx changes to reproduce the final number. Disabling repacking measured 130.44 PP / 16.91 TG; serial dispatch measured 317.03 / 18.76. Neither beat the control. These ablations do not isolate every private-fork contribution.

## Delta 2: reimplement the HIP AllReduce port from upstream

- **Reuse upstream's existing algorithm.** At `434ddbbc0`, `ggml/src/ggml-cuda/allreduce.cu` already implements host-staged AllReduce for CUDA but compiles a null-pipeline stub for HIP. Host staging means exchanging through pinned, GPU-accessible system RAM. Small messages use GPU kernels; larger messages use copy engines. No new reduction algorithm, graph scheduler, or weight format is needed for this delta.
- **Enable its existing body for HIP:** change the opening condition from `!defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)` to `!defined(GGML_USE_MUSA)`. Keep the final stub for MUSA (another GPU backend). Do not copy the private fork's now-unreachable legacy `#elif defined(GGML_USE_HIP)` implementation into an upstream port.
- **Provide four aliases** in `ggml/src/ggml-cuda/vendors/hip.h`, alongside the existing CUDA-to-HIP API mappings:

```cpp
#define cudaHostAlloc hipHostMalloc
#define cudaHostAllocPortable hipHostMallocPortable
#define cudaHostAllocMapped hipHostMallocMapped
#define cudaHostGetDevicePointer hipHostGetDevicePointer
```

- **Replace only the wait primitive** inside `ggml_cuda_ar_kernel()`'s existing arrival-token polling loop:

```cpp
#ifdef GGML_USE_HIP
    __builtin_amdgcn_s_sleep(4);
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    __nanosleep(100);
#else
    NO_DEVICE_CODE;
#endif
```

- Preserve upstream's memory-ordering protocol: each GPU writes its partial result to mapped host memory, fences those writes, publishes a per-block arrival token, waits for its peer's token, then sums. Preserve the event-protected buffer reuse, shape/type checks, copy threshold, and BF16 behavior. The sleep instruction is only a polling delay; it does not replace synchronization.
- HIP marks cleanup API results as `nodiscard`. Explicitly discard already-ignored results with `(void)` for `cudaFreeHost`, `cudaStreamSynchronize`, `cudaFree`, `cudaEventDestroy`, and `cudaStreamDestroy` in the existing cleanup/error paths. Keep checked calls checked.
- **Private-fork-only integration fix:** remove the `#ifdef GGML_USE_HIP` block in `ggml_backend_cuda_comm_allreduce_internal()` that scans `cuda_ctx->tp_overlap.active` and returns `false` if none is active. That 11-line guard forced fallback even after the HIP port compiled. **Upstream `434ddbbc0` has no such guard, so a clean upstream port needs no corresponding deletion.** Keep the old overlap variables unset. Earlier runs labeled PR 27825 were misleading: pipeline initialization did not establish execution.

## Reproduce

- Use the **complete current worktree**, not HEAD alone: `/home/raistlin/infer/mx-llama.cpp`, branch `mx-upstream-2026-09-09`, HEAD `c0df68229`, with the resolved but uncommitted upstream merge `434ddbbc0`. Inputs include private mx head `0c81bd502` and PR 27825 head `ee5b990d1`. Preserve the staged merge and unstaged changes; no commit was made.
- In the existing ROCm development shell, with access to both GPUs, configure and build:

```sh
cd /home/raistlin/infer/mx-llama.cpp
cmake -S . -B build \
  -DAMDGPU_TARGETS=gfx906 -DGGML_HIP=ON -DGGML_HIP_RCCL=OFF \
  -DGGML_CCACHE=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_VR_EXPERIMENTS=ON
cmake --build build -j 16 --target llama-cli llama-bench vr-test-gfx906-tp
```

- Start from a shell without prior `GGML_*`, `LLAMA_*`, or GPU-tuning overrides. Leave old `GGML_CUDA_TP_OVERLAP` and `GGML_CUDA_TP_OVERLAP_BF16` controls unset. Run the exact winning benchmark:

```sh
HIP_VISIBLE_DEVICES=0,1 ./build/bin/llama-bench \
  -m /home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/fe1e2a23d973adb629709749dc4f6756df66ef10/Qwen3.8-27B-Q8_0.gguf \
  -ngl 999 -fa on -sm tensor -dev ROCm0/ROCm1 \
  -b 4096 -ub 4096 -p 0 -n 0 \
  -pg 2048,0 -pg 0,256 -d 16384 \
  -r 1 -o jsonl --progress --no-warmup
```

- `-b` / `-ub` set the logical batch and GPU microbatch limits; `-ngl 999` requests full GPU offload. For a fallback comparison on the fixed binary, set `GGML_CUDA_ALLREDUCE=none`: this selects the generic fallback, not omission of the required sum. The reported control was measured before removing the guard.
- Run the existing focused checks:

```sh
HIP_VISIBLE_DEVICES=0,1 GGML_CUDA_ALLREDUCE=internal \
  ./build/bin/vr-test-gfx906-tp --n 1 --check
HIP_VISIBLE_DEVICES=0,1 GGML_CUDA_ALLREDUCE=internal \
  ./build/bin/vr-test-gfx906-tp --n 2048 --check
```

## Evidence and limits

- Both checks passed. A focused `rocprofv3 --kernel-trace --stats` trace explicitly recorded `ggml_cuda_ar_kernel<float, __hip_bfloat16>`, proving execution; initialization messages alone were misleading. Full-model tracing imposed extreme overhead, so profiled throughput was excluded.
- **Precision caveat:** the active pipeline defaults to BF16 transport, a 16-bit floating-point format, for F32 inputs (`GGML_CUDA_AR_BF16_THRESHOLD=1`). This differs numerically from the fallback. Setting the threshold to `0` disables that conversion, but its performance was not measured. All-ones checks validate limited execution correctness; perplexity and model-output equivalence remain untested.
- Detailed results: [vr/RESULTS.md](vr/RESULTS.md). Raw logs/traces: `/tmp/mx-tg-20260910/`, especially `control.jsonl`, `ar-enabled.jsonl`, and `ar-check/n1_kernel_stats.csv`. Temporary artifacts are not durable. No additional 64K tests were run; the current target and evidence are specifically 16K.

## Four next experiments (unmeasured)

- **PP 1 - Tune large-message transfer chunks.** In `allreduce.cu`, the copy-engine path divides a reduction into chunks so transfers and addition can overlap. Smaller chunks can start useful work earlier; larger chunks reduce event/launch overhead. Compare the current automatic heuristic against `GGML_CUDA_AR_COPY_CHUNK_BYTES=524288`, `1048576`, and `2097152` (0.5, 1, and 2 MiB), changing nothing else. Time representative PP reductions first, then validate only the best candidate with PP2048/TG256 at 16K. Keep the BF16 setting fixed so transport precision does not confound the comparison.
- **PP 2 - Tune repacked matrix-matrix tiles.** A tile is the block of output computed together while reusing weights and activations. Inspect `q8_repack/repack-common.cuh` and `repack-kernels.cuh`: the current Q8 settings include `BK=4`, `TN=2`, `BM=64`, and four row lanes. Benchmark valid alternative tile instantiations on the actual per-GPU projection shapes; adjust launch geometry and shared-memory indexing consistently. Larger tiles may improve reuse but consume more registers/shared memory and reduce concurrent work. Keep single-token dispatch unchanged and verify numerical agreement before full-model PP validation.
- **TG 1 - Reduce fused matrix-vector workgroup size.** A workgroup is a set of GPU threads scheduled together; a gfx906 wavefront contains 64 threads. In `q8_repack/mul-mat.cu`, the fused Q8 gate/up kernel currently uses `<16,16,false,64,true>` with 1024 threads. Compare `<4,4,false,64,true>` with 256 threads and a grid of `(ne01+3)/4` blocks, leaving other type branches and the unfused kernel unchanged. This preserves one wavefront per output row but may permit more concurrent workgroups. Measure representative gate/up shapes and compare outputs against the existing kernel using nonuniform inputs; do not assume smaller is faster.
- **TG 2 - Tune small-message AllReduce parallelism.** The active small-message kernel uses eight blocks of 256 threads, each with its own peer-arrival token. For TG's short vectors, fewer blocks may reduce PCIe signaling overhead. Compare two and four blocks against eight using a size-dependent launch policy in `allreduce.cu`; retain the existing maximum-sized arrival buffers and use identical launch geometry on both GPUs. Preserve fences, per-block token indexing, and event-protected slot reuse. Stress repeated reductions with nonuniform inputs before timing TG256 at 16K; keep large PP reductions on their existing path.
