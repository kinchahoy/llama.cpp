
## 5. Environment variables named in the reports

Default = behavior when unset. Recommendation: keep / remove / make-cmake / review. "keep (fold)" = keep behavior, consolidate with siblings. "pre-existing" = already in base (listed for completeness only).

| Variable | Default | File | Recommended |
|---|---|---|---|
| GGML_META_SPLIT_CARRY_STAGE | on | ggml-backend-meta.cpp:2520 | keep |
| GGML_META_NO_COMM | off | meta ctor | keep (debug A/B) |
| GGML_META_XFER_RCCL | rule (HWQ>4 or ROCm tps>=3) | meta ctor | keep |
| GPU_MAX_HW_QUEUES | unset (ROCm name) | meta ctor; ggml-cuda.cu:3916 | keep |
| GGML_META_MOE_XFER_RCCL_THRESHOLD | 1048576 | meta ctor | keep |
| GGML_META_NO_DEDICATED_CPY | off | meta ctor | keep |
| GGML_META_PART_DEBUG | off | meta ctor | remove (fold) |
| GGML_META_RUN_DEBUG | off | meta ctor | remove (fold) |
| GGML_META_XFER_DEBUG | off | meta ctor | remove (fold) |
| GGML_META_XFER_ROWS | on | meta ctor | keep |
| GGML_META_CHUNK_TRACE | off | meta ctor | remove |
| GGML_META_LAYER_SEAM_COST | on | meta ctor | keep |
| GGML_META_PROFILE | off | meta ctor | remove or move out |
| GGML_META_PARALLEL_DISPATCH | on | meta ctor | keep |
| GGML_META_TOKEN_GRAPH | on | meta ctor | keep |
| GGML_META_TOPK_BCAST | on | meta static L2555 | keep |
| GGML_META_TOPK_BCAST_ALL | off | meta L2573 | remove |
| GGML_META_TOPK_BCAST_LOGITS | off | meta L2579 | remove |
| GGML_META_INNER_GRAPHS_EAGER_OFF | off | meta L5024 | review (opt-in) |
| GGML_META_TG_LIMIT | unset | meta L5098 | remove (bisect) |
| GGML_META_DEBUG | off | base L448 | pre-existing, keep |
| GGML_SCHED_N_COPIES | max(4, gpus), meta stages, cap 16 | ggml-backend.cpp default_n_copies | keep |
| GGML_SCHED_MAX_COPIES | 16 | CMake | make-cmake (keep) |
| GGML_SCHED_MAX_SPLIT_INPUTS | 64 | CMake; ggml-backend.cpp:1181 | make-cmake; confirm need |
| GGML_SCHED_TIMING | 0 | ggml-backend.cpp new | remove |
| GGML_SCHED_TIMING_INTERVAL | 64 | ggml-backend.cpp new | remove |
| GGML_SCHED_TRACE_INPUTS | 0 | ggml-backend.cpp new | remove |
| GGML_SCHED_INPUT_STAGING_MAX | 16384 B | ggml-backend.cpp | keep (fold) |
| GGML_SCHED_STAGE_SIMPLE_GPU_INPUTS | off | ggml-backend.cpp | keep |
| GGML_SCHED_INPUT_STAGING_I64_MAX | 64 B | ggml-backend.cpp | keep (fold) |
| GGML_SCHED_INPUT_STAGING_F32_MAX | meta 4096 B, else 0 | ggml-backend.cpp | keep (fold) |
| GGML_SCHED_INPUT_STAGING_F16_MAX | meta 512 MiB, else 0 | ggml-backend.cpp | keep (fold) |
| GGML_SCHED_NON_GRAPH_INPUT_STAGING_MAX | meta 32768 B, else 0 | ggml-backend.cpp | keep (fold) |
| GGML_SCHED_SHARE_VIEW_INPUTS | on | ggml-backend.cpp split_graph | review (default) |
| GGML_SCHED_REBIND_EVENT_TRACE | off | ggml-backend.cpp | remove |
| GGML_SCHED_SYNC_NONGRAPH | n_ubatch < 64 | llama-context.cpp:810 | remove env, keep rule |
| GGML_SCHED_DEBUG | off | ggml-backend.cpp | pre-existing |
| GGML_GALLOC_LAYOUT_CACHE | on | ggml-alloc.c | keep |
| GGML_ALLOC_REALLOC_TRACE | off | ggml-alloc.c | remove |
| GGML_TP_AR_NO_GATE | gate on | ggml-cuda.cu:1410 | keep only under NCCL (no effect otherwise) |
| GGML_TP_AR_ONESHOT | off | ggml-cuda.cu:1414 | keep only under NCCL |
| GGML_CUDA_TP_OVERLAP | off | ggml-cuda.cu:1708, 3989 | remove |
| GGML_ENABLE_CUSTOM_AR | on | ggml-cuda.cu:1779 | keep |
| GGML_CUDA_CUBLAS_CHUNK_MIB | 256 | ggml-cuda.cu:2277 | keep |
| GGML_CUDA_Q8_1_CACHE | on | ggml-cuda.cu:2742 | keep (add hazard review) |
| GGML_CUDA_Q8_1_PRODUCER | on | ggml-cuda.cu:5116 | keep |
| GGML_CUDA_AR_SLABS | 4 (0/1 off, max 8) | ggml-cuda.cu:2953 | keep, document |
| GGML_PP_DRAIN_FORCE_SYNC | off | ggml-cuda.cu:3909 | keep (escape hatch) |
| GGML_CUDA_MOE_WEIGHTED_REDUCTION | on | ggml-cuda.cu:5103 | keep |
| GGML_CUDA_FUSE_ADD_RMS_NORM | on | ggml-cuda.cu:5147 | keep |
| GGML_CUDA_REPACK_MOE_FUSION_STATS | off | ggml-cuda.cu:5742 | remove |
| GGML_CUDA_OP_SYNC | off | ggml-cuda.cu:6223 | remove |
| GGML_CUDA_EAGER_BLAS_HANDLE | on | ggml-cuda.cu:7973 | review (default) |
| GGML_CUDA_TOP_K_NROWS_THRESHOLD | not given | top-k.cu (06 F6) | review |
| GGML_CUDA_CONCAT_TRANSPOSED | on | concat.cu | keep |
| GGML_CUDA_GATE_DEBUG | off | softcap.cu:63 | remove |
| GGML_CUDA_GETROWS_CHECK | off | getrows.cu:548 | remove |
| GGML_CUDA_FATTN_MAX_PARALLEL_BLOCKS | 0 | fattn-common.cuh:1219 | remove |
| GGML_CUDA_FLASHNEXT_MOE | 0 (opt-in) | flashnext-moe.cuh:165; q8_repack/flashnext-moe.cuh:55 | keep one reader |
| GGML_CUDA_REPACK_KQUANT_MOE_FUSION | 1 | repack-common.cu:81-90 | keep |
| GGML_RP_MMV_ROWS | 8 | mul-mat.cu:15-26 | keep |
| GGML_RP_NC_PLANAR | 1 | mul-mat.cu:47-53 | keep |
| GGML_RP_NC_L16 | 1 | mul-mat.cu:291-295 | keep |
| GGML_RP_NC_LDS | 1 | mul-mat.cu:305-310 | keep (fix default, low priority) |
| GGML_RP_NC_RPL | off | mul-mat.cu:297-298 | remove |
| GGML_RP_NC_23 | off | mul-mat.cu:300-305 | remove |
| GGML_RP_NC_FUSION | off | repack-common.cu:47-52 | remove |
| GGML_RP_GENERIC_MMV | unset | mul-mat.cu:391; mul-mat-id.cu:146 | remove |
| GGML_RP_* (other, seen in passing) | see 04 | q8_repack | review |
| GGML_RP_NC_* (see 04) | | | |
| HSA_FORCE_FINE_GRAIN_PCIE | unset | tp-allreduce.cu:692 | keep; ask user (peer path) |
| GGML_CUDA_AR_COPY_THRESHOLD | 1 MiB | allreduce.cu:450 | pre-existing, keep |
| GGML_CUDA_AR_COPY_CHUNK_BYTES | 0 (heuristic) | allreduce.cu:453 | pre-existing, keep |
| GGML_CUDA_AR_BF16_THRESHOLD | 1 | allreduce.cu:462 | pre-existing, keep (precision-changing) |
| GGML_CUDA_AR_DUPLEX | 1 | allreduce.cu:481 | keep |
| GGML_CUDA_ALLREDUCE | nccl (falls back to internal) | ggml-cuda.cu:1787 | pre-existing, keep |
| GGML_TP_AR_KTIMING | unset | tp-allreduce.cu:719, 944 | remove or keep (ask) |
| GGML_TP_AR_TWOSHOT | -1 (auto) | tp-allreduce.cu:794 | keep (dead in target) |
| GGML_TP_AR_TWOSHOT_MIN_NE | -1 (auto) | tp-allreduce.cu:815 | keep (dead in target) |
| GGML_TP_AR_BLOCKS | 16 HIP / 36 CUDA | tp-allreduce.cu:932 | keep |
| LLAMA_SPEC_LAZY_PREFILL | 1 | speculative.cpp:1022 | keep; remove read once stable |
| LLAMA_DFLASH_ASYNC_REPLAY | 1 | speculative.cpp:1105 | remove (drop thread) |
| LLAMA_DFLASH_DEFER | 1 | speculative.cpp:1272 | keep |
| LLAMA_DSPARK_DRAFT_WINDOW | 4096 | speculative.cpp:1359 | review (opt-in or document) |
| LLAMA_DFLASH_DEFER_TAIL | 1 | speculative.cpp:1548 | keep or hardcode |
| LLAMA_DSPARK_FLUSH_ROWS | 2048 | speculative.cpp:1591 | remove (hardcode) |
| LLAMA_DSPARK_OPPORTUNISTIC | 1 | speculative.cpp:1595 | remove (hardcode) |
| LLAMA_MTP_ADAPTIVE_DISABLE | 1 | speculative.cpp:2110 | remove env (keep non-finite guard) |
| LLAMA_MTP_DEFERRED_PREFILL | 1 | speculative.cpp:2119 | keep |
| LLAMA_MTP_NO_KV_ONLY | off (KV-only on) | speculative.cpp:2325 | remove |
| LLAMA_DRAFT_REPACK | 1 | speculative.cpp:3433 | remove |
| LLAMA_DFLASH_DRAFT_TP | 0 | speculative.cpp:3445 | keep; document default |
| LLAMA_DFLASH_MIRROR_OUTPUT | 0 | common.cpp:1749-1755 | remove |
| LLAMA_DSPARK_TARGET_REPACK | 0 | common.cpp:1765 | review (default change) |
| LLAMA_NO_MIDPROMPT_CKPT | off | server-context.cpp:4609 | remove |
| LLAMA_DSPARK_PROMPT_RESERVE | 1 | server-context.cpp:4652 | keep |
| LLAMA_ARG_TENSOR_PARALLEL_SIZE | 0 | arg.cpp (CLI env) | keep |
| LLAMA_ARG_SPEC_DRAFT_TENSOR_PARALLEL_SIZE | 0 | arg.cpp (CLI env) | keep |
| LLAMA_MTP_DRAFT_PIPELINE | unset (presence) | llama-context.cpp:572 | keep; use atoi parse |
| LLAMA_DRAFT_PIPELINE | unset (presence) | llama-context.cpp:584 | keep; use atoi parse |
| LLAMA_NO_FUSED_LID | off | llama-context.cpp:744-745 | remove |
| LLAMA_TAP_STATIC | on | llama-context.cpp:876 | review (remove knob) |
| LLAMA_REUSE_NO_DRAIN | off | llama-context.cpp:1923 | remove |
| LLAMA_TAP_ACCUM_MIN | 1 | llama-context.cpp:2872 | keep |
| LLAMA_TAP_EMBD_SKIP | on | llama-context.cpp:2878 | keep |
| LLAMA_TAP_ACCUM_MROPE | on | llama-context.cpp:2892 | keep |
| LLAMA_DFLASH_PRIVATE_HEAD | on | llama-context.cpp:3206 | keep |
| LLAMA_GRAPH_REUSE_DISABLE | off | llama-context.cpp:404 | pre-existing, keep |
| LLAMA_GDN_NO_CHUNK_SNAPSHOTS | off | gated_delta_net.cu | keep (used in logs) |
| LLAMA_PLE_SHARD | 0 | llama-model.cpp:411-421 | keep; consider default on |
| LLAMA_PLE_PREFAULT | 0 | llama-model.cpp:429-462 | remove |
| LLAMA_MLA_TP | 1 | llama-model.cpp:629 | keep |
| LLAMA_DRAFT_SPLIT | unset | llama-model.cpp:730 | keep; document |
| LLAMA_SHEXP_SPLIT | 1 | llama-model.cpp:776 | keep or re-scope (validate) |
| LLAMA_STAGE_LOCAL_MIRROR | 1 | llama-model.cpp:1328 | keep |
| LLAMA_MTP_REPACK | 1 | llama-model.cpp:2154 | keep |
| LLAMA_DSV41_QNORM | 0 | deepseek4.cpp:955-962 | remove (inert) |
| LLAMA_DSV41_NO_COMPRESS | 0 | deepseek4.cpp:996-1003 | remove (inert) |
| LLAMA_DSV41_ENGRAM_CAST_DEVICE | 0 | llama-model-loader.cpp:1241-1247 | remove |
| LLAMA_DSV41_NO_ENGRAM | (documented, absent in code) | FEATURES.md | remove doc line |
| LLAMA_KV_REPLICA | on | llama-kv-cache.cpp:277 | keep |
| LLAMA_DSV4_NO_PREFILL_SNAPSHOT | on | llama-kv-cache-dsv4.cpp:722-750 | keep only with rollback test |
| LLAMA_TRACE | unset | speculative.cpp:2770 | pre-existing |
| LLAMA_BUILD_VR_EXPERIMENTS (CMake) | OFF | CMakeLists.txt:139 | keep in fork only |
