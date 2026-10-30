
## 3. Quick wins (deduplicated)

Locations are HEAD line numbers as given in the reports. "?" = no estimate given. Items marked "behavior" change default behavior and need a decision.

| # | File:line | What | Est. LOC removed | Confidence | Source |
|---|---|---|---|---|---|
| Q1 | ggml-backend-meta.cpp:162-163 | Delete duplicate mmap_support line | 1 | high | 01 |
| Q2 | ggml-backend-meta.cpp:5-8, ~2000-2014, ~3369-3391, 2744, 3033, 3422 | Remove chunk trace and backtrace (GGML_META_CHUNK_TRACE) | ~40 | high | 01 |
| Q3 | ggml-backend-meta.cpp:2375-2390, 2766-2815, prof_* uses | Remove profiler (GGML_META_PROFILE) or move out | ~110 | med | 01 |
| Q4 | ggml-backend-meta.cpp:2737-2739, 3029-3031, 4056, 4352, 4743-4845, 5246-5252 | Fold dbg_part, dbg_run, dbg_xfer into one flag or remove | ~60 | med | 01 |
| Q5 | ggml-backend-meta.cpp:5095-5099 | Remove GGML_META_TG_LIMIT static (default unchanged) | ~5 | high | 01 |
| Q6 | ggml-backend-meta.cpp:4146, 4152 | Remove GGML_META_TOPK_BCAST_ALL and _LOGITS opt-in knobs | ? | med | 01 |
| Q7 | ggml-backend-meta.cpp:1968, ~2080 | Skip idx stash when GGML_META_XFER_ROWS=0 (no default change) | 0 (perf) | med | 01 |
| Q8 | ggml-backend-meta.cpp:1743-1746 | Delete unreachable multi-buffer abort (checked at 1652) | ~4 | med-high | 01 |
| Q9 | ggml-backend-meta.cpp:461-462, 965-970, 996-997 | Cosmetic: blank lines, stale spin-count comment | ~5 | high | 01 |
| Q10 | ggml-backend.cpp ~1208, 1362-1400, 2840-2846, compute_splits | Remove scheduler timing instrumentation (GGML_SCHED_TIMING) | ~190 | high | 02 |
| Q11 | ggml-backend.cpp:2406, 2419-2460 | Remove rebind event trace (keep early return path) | ~30 | high | 02 |
| Q12 | ggml-alloc.c:1152, 1214, 1259, 1301, 1327 | Remove realloc trace logs (keep graph_fits) | ? | high | 02 |
| Q13 | ggml-backend.cpp ~2486 | Restore blank line before "allocate graph" | 0 | high | 02 |
| Q14 | ggml-backend.cpp:962 and 967-968 | Move event_query decl to public header; drop duplicate. NULL default true to false is a behavior change (see B3) | -1 | high | 02 |
| Q15 | ggml-backend.cpp ~2140 | Share-view inputs: default off or restrict to meta (behavior change) | 0 | med | 02 |
| Q16 | ggml/CMakeLists.txt:189; ggml/src/CMakeLists.txt:5 | Delete GGML_SCHED_MAX_SPLIT_INPUTS knob if no need for 64 | 2 | low-med | 02 |
| Q17 | common.cuh:1313-1315 | Delete dead kernel_funcs member | ~3 | high | 03 |
| Q18 | common.cuh:1530-1534 | Delete dead top_k_workspace_rows_cap fields and comment | ~5 | high | 03 |
| Q19 | common.cuh:1486-1493; mmq.cu:352-353 | Delete write-only tp_overlap dst and slab_elements | ~3 | high | 03 |
| Q20 | ggml-cuda.cu:3989-3996 | Remove per-node getenv(GGML_CUDA_TP_OVERLAP) (or hoist to static) | ~8 | high | 03 |
| Q21 | ggml-cuda.cu:6008-6011 | Delete dead SCALE,UNARY,SCALE,DSV4_HC_POST try_fuse site (confirm with graph trace) | ~4 | med-high | 03 |
| Q22 | ggml-cuda.cu:5111 vs 5138 | Move misplaced "try and fuse" comment to ggml_cuda_try_fuse | 0 | high | 03 |
| Q23 | ggml-cuda.cu:13-75 (diff), 5742-5746; 6218-6233 | Remove GGML_CUDA_REPACK_MOE_FUSION_STATS and GGML_CUDA_OP_SYNC | ~67 | high | 03 |
| Q24 | ggml-cuda.cu:1708-1714, 1357, 3073-3081, 829-831; mmq.cu:290-360 | Remove GGML_CUDA_TP_OVERLAP entirely (covers Q19, Q20 overlap parts; vr/README.md:55 requires it unset) | ~100 | med-high | 03, 04 |
| Q25 | ggml-cuda.cu:7972-7982 | Eager cuBLAS handle: default off or HIP-only (behavior change) | 0 | med | 03 |
| Q26 | ggml-cuda.cu HIP graph update (~4062-4105) | Cache hipGraphGetNodes func list at capture (perf, not cleanup) | 0 | med | 03 |
| Q27 | ggml-cuda.cu diff ~986-987, 1435-1436, 1670-1672 | Remove duplicate return line and blank-line noise | ~5 | high | 03 |
| Q28 | mmq-config-gfx906.cuh; mmq.cuh:229, 253-254, 288-289 | Delete dead gfx906 MMQ config and unreachable branches (no behavior change) | ~32 | high | 04 |
| Q29 | q8_repack/repack-common.cuh:84-88, 129-137 | Delete repack_qs_bytes and rp_scale_from_slot (0 callers) | ~14 | high | 04 |
| Q30 | q8_repack/mul-mat.cu:673-692; repack-common.cu:68-71 | Remove unreachable dense fused branches; fix stale comment | ~20 | high | 04 |
| Q31 | repack-common.cu:47-52; mul-mat.cu:533-600; repack-kernels.cuh:1244-1260 | Remove GGML_RP_NC_FUSION path | ~85 | med-high | 04 |
| Q32 | mul-mat.cu:297-305, 346-348 | Remove GGML_RP_NC_RPL and GGML_RP_NC_23 variants (archived "no gain") | ? | med-high | 04 |
| Q33 | mul-mat.cu:389-400; mul-mat-id.cu:145-152 | Remove GGML_RP_GENERIC_MMV hook | ~20 | med-high | 04 |
| Q34 | mmq.cuh:1560-1593 (pass J_best via mmq_args, mmq.cuh:1453) | Consolidate J-tile selection to one selector (chunked dense behavior change) | ? (about 60 duplicated) | med | 04 |
| Q35 | mmvq.cu:478-483 | Gate calc_nwarps case 5 to gfx906 or document CDNA impact (behavior change) | 0 | med-high | 04 |
| Q36 | q8_repack/flashnext-moe.cuh (59); mul-mat-id.cu:77-82 | Keep one flashnext repacked implementation | ~60 | med | 04 |
| Q37 | q8_repack/README.md line 3 and Known gaps; repack.cuh:1 | Doc fixes (7 quant types, not 2) | 0 | high | 04 |
| Q38 | tp-allreduce.cu:1021-1035; tp-allreduce.cuh:155-165 | Remove tp_custom_ar_allreduce (no callers); fix comments at .cuh:10, 50 | ~26 | high | 05 |
| Q39 | allreduce.cu:12 | Remove unused include type_traits | 1 | high | 05 |
| Q40 | allreduce.cu:386-404, 689, 867 | Drop unused slot_info.token; acquire_slot returns int (compile check) | ~6 | med-high | 05 |
| Q41 | ggml-cuda.cu:1407-1416 | Wrap GGML_TP_AR_NO_GATE and _ONESHOT reads in #ifdef GGML_USE_NCCL (no effect in non-NCCL builds) | 0 | high | 05 |
| Q42 | tp-allreduce.cu:283-300, 412-425, 284-285, 719-735, 943-956 | Trim KTIMING if broadcast profiling not used (ask first) | ? | med | 05 |
| Q43 | allreduce.cu:580-585, 619-622 | Lazily allocate dev_src only when slabs used (saves 32 MiB per GPU when off) | 0 (memory) | med | 05 |
| Q44 | ggml-cuda.cu:1218-1220; tp-allreduce.cuh:10-15 | Fix stale NCCL-fallback comments | 0 | high | 05 |
| Q45 | top-k.cu:95-96 | Delete dead greater_count, equal_count fields | 2 | high | 06 |
| Q46 | softcap.cu:63-65, 89-91 | Remove GGML_CUDA_GATE_DEBUG helper and call | ~8 | high | 06 |
| Q47 | getrows.cu:548-567 | Remove GGML_CUDA_GETROWS_CHECK (forces host sync per call) | ~20 | high | 06 |
| Q48 | fattn-common.cuh:1219-1237 | Remove GGML_CUDA_FATTN_MAX_PARALLEL_BLOCKS (keep OOM retry at ~1240-1260) | ~18 | med | 06 |
| Q49 | gated_delta_net.cu:203-211 | Demote GDN snapshot WARN banner to INFO or remove (keep LLAMA_GDN_NO_CHUNK_SNAPSHOTS) | 0 | med | 06 |
| Q50 | gated_delta_net_chunk.cu (state-load loop) | Re-indent ggml_cuda_pdl_sync() | 0 | high | 06 |
| Q51 | speculative.cpp:26 | Delete duplicate #include <cmath> | 1 | high | 07 |
| Q52 | speculative.cpp:215-216, 225, ~3902-3912 | Delete never-written counters t_encode_us, t_readnx_us, t_pp_encode_us and their prints | ? | high | 07 |
| Q53 | server-context.cpp:309, 5092 | Delete spec_i_batch_last and the std::move | 2 | high | 07 |
| Q54 | server-context.cpp:679; speculative.h:98-99 | Make common_speculative_reset void (return unused) | 0 | high | 07 |
| Q55 | speculative.cpp:3429-3431 | Fix stale comment on draft repack (or drop LLAMA_DRAFT_REPACK read) | 0 | high | 07 |
| Q56 | speculative.cpp ~3938 | Restore SPC_TRC for stats line, or keep LOG_INF deliberately | 0 | decision | 07 |
| Q57 | speculative.cpp ~1252 | Inline n_taps back to target_layer_ids_n (rename) | 0 | med | 07 |
| Q58 | common.cpp ~1752-1756 | Remove LLAMA_DFLASH_MIRROR_OUTPUT branch (DSpark mirror stays) | ? | med | 07 |
| Q59 | server-context.cpp ~4596-4612 | Remove LLAMA_NO_MIDPROMPT_CKPT | ~14 | med | 07 |
| Q60 | speculative.cpp ~1074, ~1908; server-context.cpp | Remove double blank lines | 0 | high | 07 |
| Q61 | llama-context.cpp:380-416; ~5109-5117; context.h ~151-156; llama-ext.h:206-213 | Remove dead p_end layer_inp_accum_wait and _ready (epoch variants used) | ~50 | high | 08 |
| Q62 | llama-context.cpp:332-334; ~5087-5089; llama-ext.h:198 | Remove dead layer_inp_accum lid getter (keep if out-of-tree users expected) | ~6 | med | 08 |
| Q63 | llama-context.cpp:1914-1924 | Remove LLAMA_REUSE_NO_DRAIN (restore unconditional drain; see B21) | ~10 | med-high | 08 |
| Q64 | llama-context.cpp:739-752 | Remove LLAMA_NO_FUSED_LID branch, keep resolve() call | ~11 | med | 08 |
| Q65 | llama-context.cpp:808-812 | Remove GGML_SCHED_SYNC_NONGRAPH env, keep n_ubatch < 64 rule | ~4 | med | 08 |
| Q66 | llama-context.cpp ~553-557, 569-587, 869-878, 2599-2609; llama-context.h 126-154 | Compress long comments to 1-2 lines (ASD-STE100) | ? | high | 08 |
| Q67 | llama-context.cpp ~22, ~64, ~466-467, ~1904 | Blank-line noise | 0 | high | 08 |
| Q68 | llama-context.cpp ~1520-1530 and accum setter (~300-307 diff) | Factor duplicated pinned-host-buffer fallback into one helper | ? | med | 08 |
| Q69 | llama-context.cpp:572, 584 | LLAMA_MTP_DRAFT_PIPELINE and LLAMA_DRAFT_PIPELINE: presence check to atoi parse, or document | 0 | high | 08 |
| Q70 | llama-graph.cpp:71-86 | Delete ggml_mul_mat_aux (duplicate of llama_mul_mat_hadamard, impl-h:57-74) | ~17 | high | 09 |
| Q71 | llama-model-loader.cpp:955-958, 1238-1247 | Remove GGML_OP_CPY probe and ENGRAM cast env gate (no ENGRAM_Q/K created) | ~15 | high | 09 |
| Q72 | llama-model.cpp:2909-2919 | Remove ENGRAM_EMBED pin branch (never created) | ? | high | 09 |
| Q73 | llama-arch.h ~320-335, 692-695; llama-arch.cpp name/info rows; llama-hparams.h:12-14, 393-411; llama-model.h:551-552 | Remove dead Engram enums, hparams, tensors (keep ple.* and engram.pad_token_id, engram.token_map) | ~150 | high | 09 |
| Q74 | models/models.h:1356; deepseek4.cpp:97-99 | Remove head_hc_flags (always 0) | ~3 | high | 09 |
| Q75 | llama-model.h:750-751; llama.cpp:524 | Remove llama_model::path (never read) | ~2 | high | 09 |
| Q76 | llama-model.cpp:1722-1725 | Remove orphan comment | 0 | high | 09 |
| Q77 | llama-model.cpp ~3964, ~4049; models.h ~1519; deepseek4.cpp ~255-258; loader ~1458 | Blank-line churn | 0 | high | 09 |
| Q78 | llama-model.cpp:566-572, 775-783 | Downgrade or remove debug LLAMA_LOG_ERROR and unconditional fprintf "[shexp-split]" | ~12 | high | 09 |
| Q79 | llama-model.cpp:2095-2096 | Fix MTP comment (LLAMA_MTP_REPACK defaults to 1) | 0 | high | 09 |
| Q80 | deepseek4.cpp:954-1003; llama-hparams.h:355-391 | V4.1 dead plumbing: populate in deepseek41 load or delete (decision needed; do not delete KV branches without KV auditor) | up to ~100 | high dead; med on answer | 09 |
| Q81 | llama-model.cpp:27-29, 423-462, 2551 | Remove LLAMA_PLE_PREFAULT (product decision) | ~40 | med | 09 |
| Q82 | gguf-py/gguf/constants.py:1365 | Move NEXTN_FC_HIDDEN to end of MODEL_TENSOR (avoid renumber) | 0 | med | 09 |
| Q83 | llama-kv-cache-dsv4.cpp:1417-1426 | Delete dead lambdas is_comp_owner, is_state_owner (keep comp_read) | ~10 | high | 10 |
| Q84 | llama-kv-cache-dsv4.cpp:453, 842, 854, 2301, 2304, 2307 | Delete dead pad_blocks parameter | ~8 | high | 10 |
| Q85 | tools/gguf-split/gguf-split.cpp:340 | Delete orphan comment | 0 | high | 10 |
| Q86 | .github/workflows/build-cann.yml:41-46 | Delete fork-only CANN stub (or drop from upstream diff) | ~6 | med | 10 |
| Q87 | llama-memory-recurrent.cpp:38-45, ~1074 | Shorten ring comments; drop blank line | ~5 | high | 10 |
| Q88 | allreduce.cu:1204 (base 960-963) | Restore 1-line MUSA stub comment (optional) | 0 (adds 1) | med | 05 |
