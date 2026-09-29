# Local MX improvements

## Measured gains over clean llama.cpp

The like-for-like comparison on 2026-09-10 used Qwen3.8 27B Q8_0 on two PCIe-connected gfx906 GPUs, tensor split, Flash Attention, 4096 batch and microbatch, PP2048 and TG256 after 16384 cached tokens. Clean llama.cpp `434ddbbc0` measured 196.97 PP and 20.13 TG tokens/s; the merged MX build with working HIP AllReduce measured 394.17 PP and 25.89 TG tokens/s. These were single runs at different times, so the roughly 100% PP and 29% TG gains describe the complete fork, not any one change. [RESULTS.md](RESULTS.md) has the controls and corrections.

| Improvement | Implementation map | Direct evidence on the merged build |
| --- | --- | --- |
| gfx906 repacked Q8_0 weights | [q8_repack/README.md](../ggml/src/ggml-cuda/q8_repack/README.md), `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-backend-meta.cpp` | Disabling repack changed 325.49 PP / 18.68 TG to 130.44 PP / 16.91 TG before the AllReduce fix. |
| HIP host-staged AllReduce | `ggml/src/ggml-cuda/allreduce.cu`, `ggml/src/ggml-cuda/vendors/hip.h`, `ggml/src/ggml-cuda/ggml-cuda.cu` | Enabling the intended path on the same build changed 325.49 PP / 18.68 TG to 394.17 PP / 25.89 TG. |
| Concurrent tensor-parallel lane dispatch | `ggml/src/ggml-backend-meta.cpp` | Serial dispatch measured 317.03 PP / 18.76 TG against a 325.49 PP / 18.68 TG control; the small TG difference does not establish a TG gain. |
| Shared-expert tensor split | `src/llama-model.cpp` | Mirroring shared experts measured 314.04 PP / 17.68 TG against 325.71 PP / 18.76 TG with the split, before the working AllReduce path. |

### How to carry the gains to another head

1. Repack eligible Q8_0 weights during upload into separate int8 and f16 scale planes. In `q8_repack/buffer.cu`, expose a gfx906 extra buffer type and convert weights on `set_tensor`; in the meta backend, compose one repack buffer type per tensor-parallel lane and keep split slices block-aligned. Route repacked `MUL_MAT` and `MUL_MAT_ID` through the `q8_repack` mat-vec kernels for narrow token counts and tiled GEMM for prompt batches. Keep the loader buffer choice, `supports_op`, view handling, and compute predicate consistent so a canonical kernel never reads a repacked weight. The linked implementation map covers the layouts, thresholds, fused single-token gate/up path, and fallback behavior.
2. Keep the host-staged two-GPU AllReduce compiled for HIP by allowing HIP through the opening condition in `allreduce.cu` (`#if !defined(GGML_USE_MUSA)` in this tree). Map CUDA pinned-host APIs to HIP in `vendors/hip.h` and use `__builtin_amdgcn_s_sleep(4)` in the device wait loop. Small reductions use mapped host memory, a system fence, and per-block arrival tokens; large reductions use copy-engine chunks and events. Preserve buffer reuse and ordering, and remove any HIP guard in `ggml_backend_cuda_comm_allreduce_internal()` that requires the separate Q8 overlap mode for normal reductions. Confirm execution, not just pipeline initialization: the earlier missing guard caused a fallback despite a successful init. The default F32-to-BF16 wire conversion (`GGML_CUDA_AR_BF16_THRESHOLD=1`) is a precision tradeoff; set it to `0` for F32 wire transport and measure that configuration separately.
3. Issue each tensor-parallel lane's `ggml_backend_graph_compute_async` concurrently from a persistent host worker pool in `ggml-backend-meta.cpp`, then join before inspecting statuses or advancing the graph. This reduces the stagger before the per-layer AllReduce. Keep a serial fallback and bound the worker wait so idle lanes do not spin indefinitely. `GGML_META_PARALLEL_DISPATCH=0` selects the serial path here.
4. Split shared-expert up/gate weights on `GGML_BACKEND_SPLIT_AXIS_1` and the matching down weight on `GGML_BACKEND_SPLIT_AXIS_0` in `src/llama-model.cpp`. Merge the shared expert's partial output with the routed experts' partial output before the existing per-layer AllReduce; this avoids reading full shared-expert weights on every GPU and adds no separate reduction. `LLAMA_SHEXP_SPLIT=0` restores mirroring here.

The September 29 head merge briefly excluded HIP from the host-staged AllReduce. With otherwise identical PP2048/TG256 settings, the September 10 pre-merge build measured 414.24/30.28 tokens/s and the merged build measured 325.75/18.73. Restoring the HIP condition rebuilt successfully and passed a short PP512/TG32 check, but that short check is not a full repeat of the 16K benchmark. Check this path and compare against the pre-merge commit after each head merge.

### Optional exact-shape overlap

The older Q8_0 down-projection experiment below is separate from the generic AllReduce win above. It splits the exact `8704 x 5120` by `8704 x 2048` prefill case into two 1024-column slabs and overlaps matrix multiplication with peer transport; the BF16 wire option changes precision. At 2048 batch/microbatch it raised PP by 12.63% over its same-build control, with no established TG gain. Keep it opt-in and restricted to the measured shape rather than using it to replace the normal host-staged path.

This branch adds an exact-shape Q8_0 tensor-parallel overlap path beyond public mx-llama.cpp. It targets the Qwen3.6 and Qwen3.8 27B down projection on two gfx906 GPUs at PP2048.

The committed implementation in `d78f9d6f4` overlaps MMQ with internal peer transport. It divides the 2048 prompt columns between two fixed 1024-column ownership halves and optionally transfers partial sums as BF16. The path is opt-in and restricted to HIP gfx906, two devices, Q8_0, stream 0, and the exact local matrix shape `8704 x 5120` by `8704 x 2048`.

An uncommitted extension added `GGML_CUDA_TP_OVERLAP_SLAB_COLS` to test 256, 512, 1024, and 2048-column MMQ compute slabs. Post-merge tests rejected it: 512 columns was slower than 1024, 256 produced incorrect output, and 2048 caused an illegal GPU memory access. The committed implementation therefore keeps the proven fixed 1024-column slabs.

Historical best results from the committed experiment were PP2048 319.48 t/s, TG1024 18.40 t/s, and about 29.7 t/s generation with MTP depth 2. See `README.md` and `RESULTS.md` for the exact controls, precision caveat, commands, and model provenance.

`patches/local-compute-slab-tuning.patch` records the rejected extension as it existed immediately before merging official llama.cpp HEAD.
