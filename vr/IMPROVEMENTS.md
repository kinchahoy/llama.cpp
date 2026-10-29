# gfx906 improvements

The aim is a small delta from llama.cpp that keeps useful gfx906 performance. Some support code is backend-generic because tensor parallelism, graph replay, and speculative state must agree across the runtime.

| Improvement | Why it helps | Main source |
| --- | --- | --- |
| Repacked weights and gfx906 MMQ/MMVQ tuning | Match weight layout, tile sizes, and lane work to GCN hardware; reduce decode memory and instruction cost | `ggml/src/ggml-cuda/q8_repack/`, `mmq-config-gfx906.cuh`, `mmq.cuh`, `mmvq.cu` |
| Graph-safe AllReduce, duplex copies and overlapped prefill slabs | Extend upstream's host-staged transport to reduce communication and scheduling cost | `ggml/src/ggml-cuda/allreduce.cu`, `allreduce.cuh` |
| Concurrent TP lanes and whole-token graph replay | Reduce host launch gaps and keep both GPUs working | `ggml/src/ggml-backend-meta.cpp`, `ggml-backend.cpp` |
| Fused normalization, activation production, and narrow matvec paths | Avoid launches and intermediate traffic, including small speculative verification batches | `ggml/src/ggml-cuda/ggml-cuda.cu`, `norm.cu`, `q8_repack/` |
| Grouped MoE weight reuse | Decode each expert weight block once for several verification tokens, retaining separate dot products and original routing | `ggml/src/ggml-cuda/flashnext-moe.cuh`, `q8_repack/flashnext-moe.cuh` |
| Speculative prompt replay and repacked draft weights | Reduce draft preparation cost; the single-GPU DFlash draft avoids extra TP communication | `common/speculative.cpp`, `src/llama-context.cpp` |
| Split KV, recurrent snapshots, and per-sequence taps | Keep speculative execution and rollback correct with TP and concurrent requests | `src/llama-graph.cpp`, `llama-memory-recurrent.cpp`, `llama-context.cpp` |

The [repack README](../ggml/src/ggml-cuda/q8_repack/README.md) explains its layout and integration. Preserve upload, allocation, dispatch, and consumers together; canonical kernels cannot read repacked weights. For HIP AllReduce, keep the first `#if !defined(GGML_USE_MUSA)` branch in `allreduce.cu` available to HIP. The normal configuration must not depend on `GGML_CUDA_TP_OVERLAP=1`.

Upstream already provides the basic HIP host-staged AllReduce. The local delta adds graph-replay device counters, separate H2D/D2H streams, prefill slab overlap, prepare/launch scheduler hooks, and a separate conditional peer-write/two-shot transport (`tp-allreduce.cu`). The unreachable old HIP overlap-only `#elif` has been removed from `allreduce.cu`; do not restore that duplicate implementation. BF16 transport and its threshold already exist in the upstream base; grouped MoE adds no new precision reduction.

Correctness fixes matter as much as speed: graph reservation can round persistent tap storage above the requested ubatch, prompt state must remain per sequence, and chunked recurrent kernels must preserve every requested rollback snapshot. A successful build or fast PP run does not validate those cases.

## Accepted grouped MoE idea

For narrow MoE verification batches on gfx906, group assignments to the same expert and reuse decoded weights across up to eight entries. Expert selection, quantization and each token's output remain unchanged in intent; there is no intentional precision or quality tradeoff. Floating-point execution differences still need correctness checks.

The [recorded experiment](old-ignore/2026-10-25/flashnext-fast/README.md) found about 6% higher Flash Next TG with grouped-only enabled, identical text/acceptance on the tested prompt, and passing CPU-reference cases. This accepts the weight-reuse idea, not a guarantee for every workload. Keep `GGML_CUDA_FLASHNEXT_MOE=1` opt-in for now. The separate GPU-feedback path is not included. The wider MMQ fallback remains outside this validation.

## Worth revisiting

- RCCL after both GPUs have suitable CPU-connected PCIe lanes. Its current exclusion reflects this host's topology.
- TP split and batch sizing after hardware or workload changes. Equal split and batch/ubatch 2048 are comparison defaults, not hardware-independent optima.
- Narrow verification kernels under concurrent speculative load; more than eight rows can leave the optimized path.
- Q4/K-quant performance, populated deep contexts, and quantized KV. They need their own measurements.
- DFlash vision and mixed token/embedding requests, which remain unvalidated.

Earlier failed or slower experiments describe their tested workload and implementation, not permanent limits. Consult the [dated archive](old-ignore/README.md) before repeating work, then recheck assumptions against the current source. Get explicit OK before experiments or longer tests.

## Built on

- [mixa3607/mx-llama.cpp](https://github.com/mixa3607/mx-llama.cpp) - tensor parallelism and speculative runtime work.
- [iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906) - gfx906 Q8 weight repacking and kernel work.
- [Stastez's host-staged AllReduce work](https://github.com/ggml-org/llama.cpp/pull/27825) - ROCm communication over PCIe.

This fork integrates and extends those ideas on [llama.cpp](https://github.com/ggml-org/llama.cpp); the list is a short lineage note, not a per-line authorship claim.

## Upstream overlap (checked 2026-10-29)

- Host-staged ROCm AllReduce already landed in [PR 27825](https://github.com/ggml-org/llama.cpp/pull/27825), merged Sep 15. It is not gfx906-only. Compare local transport/overlap changes against that implementation before proposing anything new.
- gfx906 MMQ tuning has an existing [discussion, 23881](https://github.com/ggml-org/llama.cpp/discussions/23881). Reported Q8 improvements and Q4 regressions favor small, quant-specific candidates over a blanket architecture default.
- [PR 28198](https://github.com/ggml-org/llama.cpp/pull/28198), merged Sep 3, enables concurrent streams per multi-GPU graph split. It is related groundwork, not the same as this fork's whole-token TP graph implementation.
- The [repack design discussion, 21130](https://github.com/ggml-org/llama.cpp/discussions/21130), identifies testing and RPC/TP integration concerns. Our current GPU repack kernels are gfx906-specific; upstreaming them would need more than a tuning-table change.
- No exact upstream submission for this grouped-MoE implementation was identified in the targeted search. Weight reuse is a general idea, but this implementation and its evidence are gfx906-specific. Absence from search results is not evidence of rejection.

A small per-quant gfx906 tuning change is the clearest first upstream candidate. Larger repack, scheduler and speculative changes need separate discussion and validation. Upstreaming could reduce merge work, but the current fork's measurements do not establish benefits on other GPUs.
