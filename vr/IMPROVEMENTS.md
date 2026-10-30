# gfx906 improvements

The aim is a small delta from llama.cpp that keeps useful gfx906 performance. Some support code is backend-generic because tensor parallelism, graph replay, and speculative state must agree across the runtime.

| Improvement | Why it helps | Main source |
| --- | --- | --- |
| Repacked weights and gfx906 MMQ/MMVQ tuning | Match weight layout, tile sizes, and lane work to GCN hardware; reduce decode memory and instruction cost | `ggml/src/ggml-cuda/q8_repack/`, `mmq.cuh`, `mmvq.cu` |
| Narrow one-token IQ2_S expert MMVQ | Use a gfx906-specific two-row path for even-row, one-column expert operations; odd rows keep the existing path | `ggml/src/ggml-cuda/mmvq.cu` |
| Graph-safe AllReduce, duplex copies and overlapped prefill slabs | Extend upstream's host-staged transport to reduce communication and scheduling cost | `ggml/src/ggml-cuda/allreduce.cu`, `allreduce.cuh` |
| Concurrent TP lanes and whole-token graph replay | Reduce host launch gaps and keep both GPUs working | `ggml/src/ggml-backend-meta.cpp`, `ggml-backend.cpp` |
| Fused normalization, activation production, and narrow matvec paths | Avoid launches and intermediate traffic, including small speculative verification batches | `ggml/src/ggml-cuda/ggml-cuda.cu`, `norm.cu`, `q8_repack/` |
| Grouped MoE weight reuse | Decode each expert weight block once for several verification tokens, retaining separate dot products and original routing | `ggml/src/ggml-cuda/flashnext-moe.cuh`, `q8_repack/flashnext-moe.cuh` |
| Speculative prompt replay and repacked draft weights | Reduce draft preparation cost; the single-GPU DFlash draft avoids extra TP communication | `common/speculative.cpp`, `src/llama-context.cpp` |
| Split KV, recurrent snapshots, and per-sequence taps | Keep speculative execution and rollback correct with TP and concurrent requests | `src/llama-graph.cpp`, `llama-memory-recurrent.cpp`, `llama-context.cpp` |

The [repack README](../ggml/src/ggml-cuda/q8_repack/README.md) explains its layout and integration. Preserve upload, allocation, dispatch, and consumers together; canonical kernels cannot read repacked weights. For HIP AllReduce, keep the first `#if !defined(GGML_USE_MUSA)` branch in `allreduce.cu` available to HIP. The normal configuration must not depend on an environment switch.

Upstream already provides the basic HIP host-staged AllReduce. The local delta adds graph-replay device counters, separate H2D/D2H streams, prefill slab overlap, prepare/launch scheduler hooks, and a separate opt-in peer-write/two-shot transport (`tp-allreduce.cu`, `GGML_CUDA_PEER_AR=1`). The unreachable old HIP overlap-only `#elif` has been removed from `allreduce.cu`; do not restore that duplicate implementation. BF16 transport and its threshold already exist in the upstream base; grouped MoE adds no new precision reduction.

Keep the gfx906 MMQ launch traversal aligned with the gfx906 kernel's row-first indexing. The first port of upstream `418f905a4` changed the block-axis order and showed a large PP drop in one matched Flash Next run. Restoring the gfx906 traversal removed that identified porting regression; that correction is in the active patch set. The initial negative result remains archived and does not measure the corrected tree.

Correctness fixes matter as much as speed: graph reservation can round persistent tap storage above the requested ubatch, prompt state must remain per sequence, and chunked recurrent kernels must preserve every requested rollback snapshot. A successful build or fast PP run does not validate those cases.

## Accepted grouped MoE idea

For narrow MoE verification batches on gfx906, group assignments to the same expert and reuse decoded weights across up to eight entries. Expert selection, quantization and each token's output remain unchanged in intent; there is no intentional precision or quality tradeoff. Floating-point execution differences still need correctness checks.

The [recorded experiment](old-ignore/2026-10-25/flashnext-fast/README.md) found about 6% higher Flash Next TG with grouped-only enabled, identical text/acceptance on the tested prompt, and passing CPU-reference cases. This accepts the weight-reuse idea, not a guarantee for every workload. Keep `GGML_CUDA_FLASHNEXT_MOE=1` opt-in for now. The separate GPU-feedback path is not included. The wider MMQ fallback remains outside this validation.

The active code also accepts the narrow IQ2_S single-token expert path described above. One matched 20-thread pair measured 38.08 vs 37.20 TG (+2.37%) with identical output, but this is not repeatability evidence. The code only selects for IQ2_S expert operations with one output column and even rows; it does not apply to dense Q8_0, and no Q8_0 benefit was measured. Three focused CPU-reference cases passed. See the [dated single-token report](old-ignore/2026-10-29/iq2s-single-token-experiment/README.md).

## 2026-10-29 rebuild: removed and changed

The fork was rebuilt on upstream `36a0e2ffc` as stages B-D. Each stage matched the old master on 27B Q8_0 and Flash Next PP/TG, and greedy output was identical across B, C and D. Details and numbers are in the [rebuild record](rebuild-2026-10-29/README.md).

Removed:
- Multi-stage TP / pipeline stages (`-tps`, `--tensor-parallel-size`, `-tpsd`). They were unused on two GPUs. To restore, start from mx-llama commit `77ffa6093` ([mixa3607/mx-llama.cpp](https://github.com/mixa3607/mx-llama.cpp)).
- Fork-only DeepSeek V4/V4.1 code: MLA head split, vocabulary-parallel output head, narrow shared-expert buffer rule, HC gate fusion, DSV4 KV cache changes, DFlash DSV4-backbone drafter split, V4.1 hc_shift. These models run on upstream code again. None of them is local, so the loss was not measured.
- MiniMax-M2 under `-sm tensor` (upstream rejects it again) and the MiMo-2 SWA pattern fallback for files that count MTP layers.
- Dead code and debug knobs: scheduler timing and traces, realloc trace, repack fusion stats, unused repack variants, the dead `mmq-config-gfx906.cuh`, the uncalled `tp_custom_ar_allreduce`, the unused ext API for layer accumulation, and fork-only Engram additions.

Kept as generic: shared-expert split (`LLAMA_SHEXP_SPLIT`), tied output split, MTP block mirror, PLE shard, dense DFlash drafter replication, and the DFlash SWA pattern fix.

Fixed:
- B12: unsharded IQ4_NL/MXFP4 getrows gather with `ne00 % 256 != 0` wrote zeros.
- B1: gguf-split split a string literal at a newline.

Environment knobs:
- Peer-write AllReduce is now opt-in with `GGML_CUDA_PEER_AR=1`. Before, it turned on by itself when peer access was reported.
- Removed with the stages: `LLAMA_STAGE_LOCAL_MIRROR`, `GGML_META_SPLIT_CARRY_STAGE`, `GGML_META_XFER_RCCL`, `GGML_META_MOE_XFER_RCCL_THRESHOLD`, `GGML_META_NO_DEDICATED_CPY`, `GGML_META_XFER_ROWS`, `GGML_META_LAYER_SEAM_COST`.
- Removed with DeepSeek V4/V4.1: `LLAMA_MLA_TP` and the fork `LLAMA_DSV41_*` knobs (they were inert). Upstream `LLAMA_DSV41_SPARSE` remains.
- Removed in cleanup (default path unchanged): `GGML_CUDA_TP_OVERLAP`, `GGML_SCHED_TIMING`, `GGML_SCHED_TIMING_INTERVAL`, `GGML_SCHED_TRACE_INPUTS`, `GGML_SCHED_REBIND_EVENT_TRACE`, `GGML_SCHED_SYNC_NONGRAPH`, `GGML_ALLOC_REALLOC_TRACE`, `GGML_CUDA_GATE_DEBUG`, `GGML_CUDA_REPACK_MOE_FUSION_STATS`, `GGML_CUDA_OP_SYNC`, `GGML_RP_NC_FUSION`, `GGML_RP_NC_RPL`, `GGML_RP_NC_23`, `GGML_RP_GENERIC_MMV`, `LLAMA_REUSE_NO_DRAIN`, `LLAMA_NO_FUSED_LID`.
- `GGML_TP_AR_NO_GATE` and `GGML_TP_AR_ONESHOT` are now read only in NCCL builds. They were no-ops without NCCL.

Builds must use the README build-ilp flags (`-mllvm -amdgpu-sched-strategy=max-ilp`, CXX `-O3`). A build without them was 2-3% slower on 27B.

Still open:
- HIP gating of default-on shared-code changes: B8 `calc_nwarps`, B14 BF16 rule, B15 graph key, B16 H2D copy, eager cuBLAS handle, AR slabs, `GGML_SCHED_SHARE_VIEW_INPUTS`. B8 needs an MMVQ launch-path change.
- Move `ggml-cuda.cu` glue into separate files.
- Expose `event_query` through the proc-address table instead of one-line backend edits.

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
