# INVENTORY: mx-llama fork audit (418f905a4 -> HEAD e04bc2e38)

Source: 10 audit reports (01-meta ... 10-kv-misc). Static reading only. Nothing built or run.
LOC are as the reports state them (approximate). "(0)" or "?" where a report gives no figure. Negative values are removals and are not in the added totals.
Rows are split where a report gives separate LOC, and merged where two reports describe the same code (both sources listed).
Theme and Class are copied from the feature rows. Class totals use the first letter of the Class field (A-F); mixed or cosmetic entries count as "other".
Table row numbers (#) are local IDs (a/b suffixes = split rows).

## 1. Feature inventory

| # | Feature | Theme | Files | ~LOC added | Class | Default | Recommendation | Confidence | Source |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Multi-stage TP (tps lanes, n_stages, per-stage comm, stage partition) | pipeline-stages | meta.cpp ctor, partition | +450 | C | on when tps < devices | keep | high | 01 |
| 2 | Stage seam placement and layer-seam cost | pipeline-stages | meta.cpp ~4000-4110 | +120 | B/C | GGML_META_LAYER_SEAM_COST on | keep | med | 01 |
| 3 | Transfer-set builder (collect_xfer_set, KV dependencies) | pipeline-stages | meta.cpp ~3720-3860 | +150 | C | multi-stage always | keep | high | 01 |
| 4 | Stage transfer (RCCL, dedicated queue, sync) and persistent-row transfer | pipeline-stages | meta.cpp transfer_lanes/tensors | +280 | B/C | RCCL rule-based; XFER_ROWS on | keep; sync fallback required | med | 01 |
| 5 | Persistent-row index stash (ggml_meta_idx_stash) | pipeline-stages | meta.cpp L1906-1968 | +70 | B | stashes every I32/I64 set | keep; guard when XFER_ROWS=0 | med | 01 |
| 6 | Fragment mode (uid==0 eval-callback path) | pipeline-stages | meta.cpp ~2030, 4370, 5160 | +110 | F | implicit | keep until reachability verified | low | 01 |
| 7 | Stage carry across graphs | pipeline-stages | meta.cpp L2519; ggml-backend.cpp | +40 | C | GGML_META_SPLIT_CARRY_STAGE on | keep | med | 01 |
| 8 | Mirror-lane mask for stage-local weights | pipeline-stages | meta.cpp L762-800 and get/set paths | +90 | C | LLAMA_STAGE_LOCAL_MIRROR | keep | med | 01 |
| 9 | Partition-time validation abort | pipeline-stages | meta.cpp ~4265-4290 | +30 | A/C | multi-stage always | keep | high | 01 |
| 10 | Split-state rules for new ops (GET_ROWS shard, batched matmul, SUM_ROWS, ROPE_BACK) | tp-core | meta.cpp L1088-1520 | +200 | C | TP always | keep | med | 01 |
| 11 | Split-state diagnostic logs | debug-trace | meta.cpp ~1070-1520 | +45 | E | log only before assert | keep or trim | med | 01 |
| 12 | GATHER closure (gather_rows, consumer repointing) | tp-core | meta.cpp ~4590-4700 | +230 | C | axis-0 into full-row op | keep | med | 01 |
| 13 | Full-row op list | tp-core | meta.cpp L497-519 | +25 | C | always | keep | high | 01 |
| 14 | TOP_K broadcast plus three env knobs | tp-core | meta.cpp ~4120-5300 | +60 | C | BCAST on; _ALL and _LOGITS off | keep BCAST; trim opt-ins | med | 01 |
| 15 | AR fold into following ADD (with get_i_delayed) | tp-core | meta.cpp L2076, ~4190 | +35 | E | always | keep one mechanism | low | 01 |
| 16 | Comm init per stage and extra AR entry points | tp-core | meta.cpp ctor ~2906-2990 | +90 | B/C | GGML_META_NO_COMM off | keep | high | 01 |
| 17 | Whole-token graph capture and replay | tp-core | meta.cpp ~2660-2700, 5010-5210 | +400 | B | GGML_META_TOKEN_GRAPH on (needs parallel dispatch) | keep; TG_LIMIT is bisect knob | med | 01 |
| 18 | Concurrent lane dispatcher threads | tp-core | meta.cpp L2357-2530 | +125 | B | GGML_META_PARALLEL_DISPATCH on | keep | high | 01 |
| 19 | Repack buffer type for meta (extra_bufts composition) | repack | meta.cpp L163-660 | +190 | C | always; active with extra bufts | keep | med | 01 |
| 20 | Event proxy (event_new, record, wait) | tp-core | meta.cpp L101-148 | +80 | C | always | keep | med | 01 |
| 21 | buft_from_simple cache key includes dev | tp-core | meta.cpp L384-406 | +20 | A | always | keep | high | 01 |
| 22 | Alloc failure returns nullptr instead of assert | tp-core | meta.cpp ~2262-2345 | +30 | C | always | keep | high | 01 |
| 23 | Multi-buffer abort guard (second copy unreachable) | tp-core | meta.cpp L1652, L1743 | +8 | E | always | keep one | med | 01 |
| 24 | Set/get async rewrite and accumulator thread | tp-core | meta.cpp L1716-1960 | +330 | B/C | always | keep chunk path; accumulator may be cut | med | 01 |
| 25 | Memset one contiguous call per device | tp-core | meta.cpp ~1900 | +5 | B | always | keep | high | 01 |
| 26 | buffer_get_tensor MIRRORED reads first live lane | tp-core | meta.cpp ~2187 | +5 | A | always | keep | high | 01 |
| 27 | Meta debug prints (PART, RUN, XFER debug) | debug-trace | meta.cpp L2737-5252 | +60 | E | all off | remove or fold into one flag | med | 01 |
| 28 | Chunk trace (GGML_META_CHUNK_TRACE, backtrace) | debug-trace | meta.cpp L5-8, ~2000, ~3369 | +40 | E | off | remove | high | 01 |
| 29 | Profiler (GGML_META_PROFILE) | debug-trace | meta.cpp L2375-5382 | +110 | E | off | move out or remove | med | 01 |
| 30 | Duplicate mmap_support line (-1) | tp-core | meta.cpp L162-163 | -1 | E | always | delete one line | high | 01 |
| 31 | Comment and cosmetic noise in meta | build-misc | meta.cpp L85, 461, 650, 1164 | +10 | E | n/a | revert noise | high | 01 |
| 32 | TODO memo note on split-state recursion | build-misc | meta.cpp L3663-3668 | +5 | E | n/a | shorten | high | 01 |
| 33 | Layout cache (gallocr layout store and restore) | context-runtime | ggml-alloc.c; ggml-backend.cpp | +230 | B | GGML_GALLOC_LAYOUT_CACHE on (0 off) | keep; review 2-buffer adoption | med | 02 |
| 34 | Realloc-check refactor (graph_fits) and realloc trace logs | context-runtime | ggml-alloc.c | +90 | B/E | trace GGML_ALLOC_REALLOC_TRACE off | keep refactor; drop logs | high | 02 |
| 35 | Graph-input staging and staging buffer pool | context-runtime | ggml-backend.cpp | +260 | B | meta-only by default | keep | med | 02 |
| 36 | Staging size knobs and helpers | context-runtime | ggml-backend.cpp | +90 | B | 7 env knobs | keep; fold F32/F16/I64/I32 knobs | med | 02 |
| 37 | Ring depth default (N_COPIES 4 to 16) | context-runtime | ggml-backend.cpp; ggml CMake | +25 | B | max(4, gpus), cap 16 | keep | high | 02 |
| 38 | Per-copy input staging events | context-runtime | ggml-backend.cpp | +30 | B | always | keep | med | 02 |
| 39a | Event query core API (ggml_backend_event_query) | context-runtime | ggml-backend.cpp; backend-impl.h | +12 | C | always; NULL returns true | keep; fix NULL default | high | 02 |
| 39b | event_query one-liners in 15 backend files | context-runtime | blas, cann, cpu, metal, vulkan and others | +15 | C | always | keep (avoids -Wextra) | high | 02 |
| 40 | Rebind guard (sched_rebind_events_complete) | context-runtime | ggml-backend.cpp | +60 | B | always | keep logic; drop trace | med | 02 |
| 41 | Share-view inputs (GGML_SCHED_SHARE_VIEW_INPUTS) | context-runtime | ggml-backend.cpp split_graph | +50 | C/F | default ON, all backends | default off or meta-only | med | 02 |
| 42 | Scheduler timing instrumentation (GGML_SCHED_TIMING) | debug-trace | ggml-backend.cpp | +190 | E | off | remove | high | 02 |
| 43 | Input trace (GGML_SCHED_TRACE_INPUTS) | debug-trace | ggml-backend.cpp | +35 | E | 0 | remove | high | 02 |
| 44 | Rebind event trace (GGML_SCHED_REBIND_EVENT_TRACE) | debug-trace | ggml-backend.cpp | +30 | E | off | remove | high | 02 |
| 45 | Sync non-graph inputs (flag, setter, env override) | context-runtime | ggml-backend.cpp; sched-impl.h; llama-context.cpp | +25 (02); +12 (08) | C/A | n_ubatch < 64, all backends | keep rule; gate to HIP if costly | med | 02,08 |
| 46 | Meta hooks in core (graph_begin, n_stages, n_backends) | pipeline-stages | ggml-backend.cpp; backend-impl.h | +15 | C | meta only | keep | high | 02 |
| 47 | Comm and dedicated-copy typedefs | tp-core | ggml/include/ggml-backend.h | +20 | C | typedef only | keep | high | 02 |
| 48 | ggml_backend_meta_device tps parameter | pipeline-stages | ggml/include/ggml-backend.h | +2 | C | always | keep | high | 02 |
| 49 | GGML_SCHED_MAX_SPLIT_INPUTS CMake knob (30 to 64) | build-misc | ggml CMakeLists; ggml-backend.cpp:1181 | +2 | F | always | confirm need; else delete knob | low | 02 |
| 50 | ggml_gallocr_alloc_graph_ids declaration | context-runtime | ggml-impl.h | +8 | C | always | keep | high | 02 |
| 51 | sched_copy_input inlined into compute_splits | context-runtime | ggml-backend.cpp | +190 (-50) | C | always | keep | med | 02 |
| 52 | Dominated-skip in layout store | context-runtime | ggml-alloc.c | +25 | B | with layout cache | keep | med | 02 |
| 53 | Blank line removed before allocate graph (-1) | build-misc | ggml-backend.cpp ~2486 | -1 | cosmetic | restore | high | 02 |  |
| 54a | Repack buft registration, supports_op and buft guards | repack | ggml-cuda.cu; q8_repack/* | +60 | A | type-based; not gfx906-gated | keep; move glue to q8_repack | high | 03 |
| 54b | Repack mul_mat and mul_mat_id dispatch, set_tensor_async repack branch | repack | ggml-cuda.cu | +15 | A | repack only | keep | high | 03 |
| 55 | Repack fused up/gate GLU and bias MMV fusion | repack | ggml-cuda.cu:5704-5800, 5934 | +90 | A/B | repack only | keep; move to q8_repack helper | high | 03 |
| 56 | Repack MoE fusion stats histogram (GGML_CUDA_REPACK_MOE_FUSION_STATS) | debug-trace | ggml-cuda.cu:13-75 | +50 | E | off | remove | high | 03 |
| 57 | Q8_1 activation cache | mmq-mmvq | ggml-cuda.cu:2740-2810; common.cuh | +140 (derived from ~165 less producer) | B | GGML_CUDA_Q8_1_CACHE on; shared with NVIDIA | keep; move to own file; review content hazard | med | 03,04 |
| 58 | Q8_1 producer target function | kernels-fusion | ggml-cuda.cu:5114-5136 | +25 | B | GGML_CUDA_Q8_1_PRODUCER on | keep | med | 03 |
| 59 | TP-overlap CUDA side (enable, graph-check getenv, fields) | tp-core | ggml-cuda.cu:1708, 1357, 3073, 3989, 829 | +25 | E | GGML_CUDA_TP_OVERLAP off | remove | high | 03 |
| 60 | TP-overlap MMQ block (mmq.cu) | mmq-mmvq | mmq.cu:296-354 | +70 (03); ~60 (04) | E | off; HIP, VEGA20, Q8_0, fixed shape | remove after allreduce owner agrees | med | 03,04 |
| 61 | tp_overlap struct fields dst and slab_elements (write-only) | tp-core | common.cuh:1486-1493 | +3 | E | overlap only | remove | high | 03 |
| 62 | Custom AR dispatch glue (CustomARPlan, comm hooks, env checks) | peer-ar | ggml-cuda.cu:1228-1252, 1366-1520, 1779 | +200 (05); ~155 (03 subset) | C/E | GGML_ENABLE_CUSTOM_AR on | keep glue; trim dead knobs | med | 03,05 |
| 63 | Comm xfer staging ring, staging_depth, set_pipeline_stages | tp-core | ggml-cuda.cu:~1260-1340, 1693 | +110 | C/F | ring used by NCCL path only | keep; move; reconcile with RCCL-off | med | 03 |
| 64 | NCCL/RCCL P2P sendrecv stage transfer | pipeline-stages | ggml-cuda.cu:1836-2045 | +215 | F | GGML_USE_NCCL only; compiled out in target | decide: move or drop | med | 03 |
| 65 | Dedicated copy stream for stage transfer | pipeline-stages | ggml-cuda.cu:3842-3960 | +140 | C | not HIP-gated; GGML_PP_DRAIN_FORCE_SYNC | keep; move to own file | med | 03 |
| 66 | cpy_tensor_async copy_from_host | pipeline-stages | ggml-cuda.cu:3749-3830 | +30 | C/F | shared; new behavior on NVIDIA | confirm or gate to HIP | med | 03 |
| 67 | set_tensor_2d_async misaligned-row workaround | tp-core | ggml-cuda.cu:3693-3735 | +30 | A | HIP, size & 3 | keep | high | 03 |
| 68 | try_fuse ADD + RMS_NORM + MUL dispatch | kernels-fusion | ggml-cuda.cu:5147-5157 | +15 | B | GGML_CUDA_FUSE_ADD_RMS_NORM on | keep | med | 03 |
| 69 | ADD + RMS_NORM + MUL kernels | kernels-fusion | norm.cu, norm.cuh | +107 | B | generic F32 path | keep | high | 06 |
| 70 | HC gate fusion dispatch (MUL/VIEW/ADD/UNARY/SCALE) | model-dsv4 | ggml-cuda.cu:4910-4925, 6018 | +45 | C | ungated, shared | keep | med | 03 |
| 71 | HC gate softcap kernel (incl. GGML_CUDA_GATE_DEBUG helper) | model-dsv4 | softcap.cu, softcap.cuh | +80 | C | ungated | keep kernel; remove debug helper | med | 06 |
| 72 | Dead try_fuse site SCALE, UNARY, SCALE, DSV4_HC_POST | kernels-fusion | ggml-cuda.cu:6008-6011 | +4 | E | unreachable | remove | med | 03 |
| 73 | MoE weighted-reduction fusion | kernels-fusion | ggml-cuda.cu:4660-5181, 6469 | +20 | A/B | GGML_CUDA_MOE_WEIGHTED_REDUCTION on | keep | high | 03 |
| 74 | Top-K fusion bounds checks | kernels-fusion | ggml-cuda.cu:~4549-4574 | +8 | A | always | keep | high | 03 |
| 75 | Graph key mixes n_nodes and nodes[0]->ne[1] | context-runtime | ggml-cuda.cu:4008-4015 | +6 | C | shared | keep | med | 03 |
| 76 | Graph update HIP recreate on kernel change | context-runtime | ggml-cuda.cu:4062-4105 | +35 | A/B | HIP block | keep; cache hipGraphGetNodes | med | 03 |
| 77 | Graph-cache instability heuristic | context-runtime | common.cuh:1309-1336; ggml-cuda.cu:6330 | +20 | F | shared | keep only if measured | low | 03 |
| 78 | Whole-token capture API (token_capture, token_graph_launch) | tp-core | ggml-cuda.cu:7750-7800, 6302 | +75 | C | meta-owned token graphs | keep; own file | med | 03 |
| 79 | GGML_CUDA_OP_SYNC per-op sync | debug-trace | ggml-cuda.cu:6218-6233 | +17 | E | off | remove | high | 03 |
| 80 | copy-only backend init (device_init_backend) | pipeline-stages | ggml-cuda.cu:7055, 7951-7970 | +20 | C | meta calls it | keep | high | 03 |
| 81 | Eager cuBLAS handle (GGML_CUDA_EAGER_BLAS_HANDLE) | context-runtime | ggml-cuda.cu:7972-7982 | +12 | B | on (0 off) | keep or default off | med | 03 |
| 82 | cuBLAS chunked source-0 conversion and alloc recovery | context-runtime | ggml-cuda.cu:2211-2330 | +95 | A | GGML_CUDA_CUBLAS_CHUNK_MIB 256 | keep | med | 03 |
| 83 | cuBLAS BF16: AMD forces F32 for all batch sizes | kernels-fusion | ggml-cuda.cu:2493 | +1 | B/F | AMD without fast bf16 | keep if measured | med | 03 |
| 84 | supports_op repack guard | repack | ggml-cuda.cu:7085-7110 | +25 | A | repacked buffers | keep | high | 03 |
| 85 | GET_ROWS IQ4_NL/MXFP4 relaxed from % QK_K to % 32 | kernels-fusion | ggml-cuda.cu:7262 | +2 | C | shared | keep if getrows bounded path verified | med | 03 |
| 86 | supports_buft accepts repack buft | repack | ggml-cuda.cu:7577-7583 | +3 | A | repack only | keep | high | 03 |
| 87 | Extra-bufts device proc (repack buft) | repack | ggml-cuda.cu:7864-7885 | +22 | A | every CUDA device | keep; confirm NVIDIA loader path | med | 03 |
| 88 | Repack scratch release on alloc failure | repack | ggml-cuda.cu:1126-1140 | +12 | A | shared | keep | med | 03 |
| 89 | event_query device interface member | context-runtime | ggml-cuda.cu:7637-7663 | +12 | C | shared | keep | high | 03 |
| 90 | Pool fallible try_alloc | context-runtime | ggml-cuda.cu:475-800; common.cuh:1217 | +60 | A/C | shared | keep | high | 03 |
| 91 | Backend context fields (mixed) | context-runtime | common.cuh:1462-1560 | +60 | mixed | shared | remove dead fields (kernel_funcs, top_k cap) | high | 03 |
| 92 | Comm init: TP overlap enable and proc table additions | tp-core | ggml-cuda.cu:1700-1800, 2100-2140 | +60 | C | custom AR env shared | keep minus overlap | med | 03 |
| 93 | Graph replay re-lookup of ggml_cuda_graph | context-runtime | ggml-cuda.cu:~6280-6295 | +3 | C | shared | keep | high | 03 |
| 94 | AR slab producer (ggml-cuda side) | tp-core | ggml-cuda.cu:2948-3019; common.cuh:1500 | +75 | B | GGML_CUDA_AR_SLABS=4 on; not HIP-gated | keep; confirm default; move | high code, med impact | 03 |
| 95 | ar_slab and tp_overlap state structs | tp-core | common.cuh:1485-1510 | +25 | B/E | shared types | keep ar_slab; trim overlap | high | 03 |
| 96 | dequantize_kq_n_t typedef | kernels-fusion | common.cuh:969-976 | +6 | C | getrows | keep | high | 03 |
| 97 | CMake q8_repack glob (CUDA and HIP lists) | build-misc | ggml-cuda/CMakeLists.txt; ggml-hip/CMakeLists.txt | +8 (4 each) | A | build-level | keep; re-run cmake | high | 03 |
| 98 | Vendor headers (cudaEventQuery, cudaErrorNotReady, musa symbols) | build-misc | vendors/hip.h; vendors/musa.h | +7 | A/C | HIP needed; MUSA unverified | keep | med | 03 |
| 99 | CUDA cosmetic hunks (blank lines, duplicate return line) | build-misc | ggml-cuda.cu diff 986, 1435, 1670 | +5 | E | n/a | clean up | high | 03 |
| 100 | Dead gfx906 MMQ config table | mmq-mmvq | mmq-config-gfx906.cuh; mmq.cuh:229, 253, 288 | +32 | E | dead; GCN table shadows it | delete; no behavior change | high | 04 |
| 101 | MMQ row-first traversal | mmq-mmvq | mmq.cuh:1057-1064, 1487 | +10 | A | gfx906 only | keep | high | 04 |
| 102 | gfx906 loader threads_per_row 8 | mmq-mmvq | mmq-load-tiles.cuh:520-552 | +11 | B | GCN5 | keep; loop rewrite could revert | med | 04 |
| 103 | vec_dot unroll 2 for Q8_0 | mmq-mmvq | mmq-vec-dot.cuh:123-134 | +14 | B | __gfx906__ | keep | med | 04 |
| 104 | calc_nwarps Q8_0 5-column case = 2 | mmq-mmvq | mmvq.cu:478-483 | +6 | B | GCN table incl. CDNA; ungated | gate or document | med | 04 |
| 105 | quantize_q8_1 block 32 on gfx906 | mmq-mmvq | quantize.cu:563-580 | +20 | B | runtime VEGA20 | keep | med | 04 |
| 106 | iq3_xxs sign-mask dp4a | mmq-mmvq | vecdotq.cuh:1235-1283 | +19 | B | gfx906 | keep | med | 04 |
| 107 | dequantize _n tail variants | kernels-fusion | dequantize.cuh:404-471 | +22 | C | generic | keep; upstream PR candidate | med | 04 |
| 108 | MMQ workspace and MoE id chunking | mmq-mmvq | mmq.cu ~330-600 | +120 | C | memory pressure only | keep; owner review | med | 04 |
| 109 | IQ2_S two-row single-token MMVQ | mmq-mmvq | mmvq.cu:643-646, 1069-1081, 1222-1235 | +30 | B | gfx906 MoE, even rows | keep; gain needs repeat | med | 04 |
| 110 | J-tile selection (three copies) | mmq-mmvq | mmq.cu:135-162, 250-277; mmq.cuh:1560-1593 | +60 | E/F | gfx906 cap inside copies | consolidate to one selector | med | 04 |
| 111 | Dense fused branches for IQ4_NL, Q5_K, Q4_K, Q6_K | repack | q8_repack/mul-mat.cu:673-692 | +20 | E | unreachable | remove | high | 04 |
| 112 | Repacked weight layout, buffer and upload | repack | q8_repack buffer.cu, repack-common, repack.cuh | +1800 | B/C | cc == VEGA20 | keep | high | 04 |
| 113 | Host vs device repack duplicated logic | repack | repack-common.cu:95-310; buffer.cu:127-375 | +450 | F/E-risk | both paths live | keep; add cross-check test | med | 04 |
| 114 | Dense narrow-batch GEMV with env geometry knobs | repack | mul-mat.cu:241-378; repack-kernels.cuh:972-1260 | +140 | B | gfx906 | keep default; drop env variants | med-high | 04 |
| 115 | MoE fused and unfused routing | repack | mul-mat-id.cu; repack-kernels.cuh:1261-1340 | +640 | B/C | gfx906 | keep | med | 04 |
| 116 | Negative-result env experiments (RPL, NC_23, FUSION, GENERIC_MMV) | repack | mul-mat.cu:297-400; repack-common.cu:47-52 | +150 | E | env, off | remove | med-high | 04 |
| 117 | Dense fused nc_glu (GGML_RP_NC_FUSION) | repack | mul-mat.cu:533-600; repack-kernels.cuh:1244-1260 | +85 | E | env off; unreachable | remove | high | 04 |
| 118 | repack_qs_bytes helper | repack | repack-common.cuh:84-88 | +5 | E | 0 callers | delete | high | 04 |
| 119 | rp_scale_from_slot helper | repack | repack-common.cuh:129-137 | +9 | E | 0 callers | delete | high | 04 |
| 120 | GGML_RP_GENERIC_MMV A/B hook | repack | mul-mat.cu:389-400; mul-mat-id.cu:145-152 | +20 | E | env | remove | med-high | 04 |
| 121 | flashnext repacked IQ4_NL grouped MoE | flashnext-moe | q8_repack/flashnext-moe.cuh; mul-mat-id.cu:77-82 | +60 | C/E | GGML_CUDA_FLASHNEXT_MOE=1 | pick one implementation | med | 04,06 |
| 122 | flashnext canonical grouped MoE (IQ2_S, IQ4_NL) | flashnext-moe | flashnext-moe.cuh (190); mmvq.cu:1474, 1617 | +190 (04); +200 (06) | C/B | opt-in; HIP, VEGA20 | keep opt-in | med | 04,06 |
| 123 | Stale q8_repack README and comments | build-misc | q8_repack/README.md; repack.cuh:1; repack-common.cu:68-71 | 0 (doc) | E | n/a | update docs | high | 04 |
| 124 | Device-side token counter in chunked AR kernel | tp-core | allreduce.cu:114-224, 929-961 | +60 (-35) | B | always | keep | high | 05 |
| 125 | Separate H2D stream (GGML_CUDA_AR_DUPLEX) | tp-core | allreduce.cu:337-907 | +50 (-7) | B | default 1 | keep | high | 05 |
| 126 | Host-staged prefill slab overlap (allreduce.cu side) | tp-core | allreduce.cu:827-927, 1006-1024 | +175 | B | GGML_CUDA_AR_SLABS=4 default on | keep; confirm default | med | 05 |
| 127 | Prepare/launch split (ggml_cuda_ar_prepare, launch_rank) | tp-core | allreduce.cu:364-372, 1161-1202 | +57 | B/C | always | keep | high | 05 |
| 128 | Copy-engine cross-stream handshakes | tp-core | allreduce.cu:342-354, 705-788 | +25 (-8) | A/B | always | keep | high | 05 |
| 129 | tp-allreduce header and types | peer-ar | tp-allreduce.cuh | +212 | C | compiled; live only with P2P | keep unless abandoned (ask user) | high | 05 |
| 130 | Barrier primitives (MUSA, CUDA, HIP) | peer-ar | tp-allreduce.cu:15-174 | +160 | C | live only with P2P | keep / ask user | high | 05 |
| 131 | One-shot kernel k_cross_device_reduce_1stage | peer-ar | tp-allreduce.cu:176-258 | +83 | C/F | reachable only if canAccessPeer=1 | keep; see bug list | high | 05 |
| 132 | Broadcast kernel k_broadcast_reduce | peer-ar | tp-allreduce.cu:261-425 | +165 | C | dead in target | keep / ask user | high | 05 |
| 133 | Two-shot kernel k_twoshot_f32 | peer-ar | tp-allreduce.cu:428-602 | +175 | C | dead in target | keep / ask user | high | 05 |
| 134 | Init, peer check, broadcast_ok | peer-ar | tp-allreduce.cu:608-716 | +110 | A/C | always when enabled | keep | high | 05 |
| 135 | Destroy and KTIMING readout | debug-trace | tp-allreduce.cu:718-769 | +52 | E | GGML_TP_AR_KTIMING off | trim KTIMING | med | 05 |
| 136 | Prepare (staging alloc, path select, RankData, grid) | peer-ar | tp-allreduce.cu:771-973 | +203 | C | reachable path | keep minus unused tuning | med | 05 |
| 137 | Launch per rank | peer-ar | tp-allreduce.cu:975-1019 | +45 | C | with prepare | keep | high | 05 |
| 138 | tp_custom_ar_allreduce non-split wrapper | peer-ar | tp-allreduce.cu:1021-1035; .cuh:155-165 | +26 | E | no callers | remove | high | 05 |
| 139 | Unused include type_traits | tp-core | allreduce.cu:12 | +1 | E | n/a | remove | high | 05 |
| 140 | Unused slot_info.token | tp-core | allreduce.cu:386-404 | +6 | E | n/a | simplify | med | 05 |
| 141 | Removed MUSA stub explanation comment (-4) | build-misc | allreduce.cu:1204 | -4 | F | n/a | restore one-line comment | med | 05 |
| 142 | GDN persistent-state chunked kernel | model-qwen | gated_delta_net_chunk.cu/.cuh; gated_delta_net.cu:194-216 | +387 | C/B | generic CUDA+HIP; n_tokens >= 2*CS | keep; demote WARN banner | high dispatch; med perf | 06 |
| 143 | GDN gfx906 launch (VEGA20 cols_per_warp, keep_rs guard) | model-qwen | gated_delta_net.cu ~396 | +30 (-15) | B | VEGA20 only | keep | med | 06 |
| 144 | Dim-0 concat with transposed src1 | model-qwen | concat.cu | +69 | B | GGML_CUDA_CONCAT_TRANSPOSED on | keep | high | 06 |
| 145 | Bitonic argsort for ncols_pad > 1024 | kernels-fusion | argsort.cu | +70 (-6) | A | non-CUB builds (HIP) | keep (bug fix) | high | 06 |
| 146 | Deterministic top-k radix gather | kernels-fusion | top-k.cu | +110 (-30) | A/F | radix path always | keep; remove dead fields | high dead; med perf | 06 |
| 147 | Dead fields greater_count, equal_count | kernels-fusion | top-k.cu:95-96 | -2 | E | n/a | remove | high | 06 |
| 148 | argmax all-NaN row returns 0 | kernels-fusion | argmax.cu:12-14 | +2 (-1) | A | always | keep | high | 06 |
| 149 | topk-MoE reduction order matched to standalone | kernels-fusion | topk-moe.cu | +50 (-20) | F | always | keep only if bit-equality required | low | 06 |
| 150 | topk-MoE single row per block on VEGA20 | kernels-fusion | topk-moe.cu:384 | +1 (-1) | B | VEGA20, n_rows == 1 | keep | low | 06 |
| 151 | GGML_CUDA_GATE_DEBUG printf in fusion matcher | debug-trace | softcap.cu:63-65, 89-91 | -8 | E | env-gated | remove | high | 06 |
| 152 | GGML_CUDA_GETROWS_CHECK debug index check | debug-trace | getrows.cu:548-567 | +20 | E | env; host sync per call | remove | high | 06 |
| 153 | get_rows IQ4_NL/MXFP4 partial super-blocks and sharded gather | kernels-fusion | getrows.cu, getrows.cuh | +150 (-10) | C | sharded mask via op_params | keep; verify unsharded mask | med | 06 |
| 154 | fattn combine zero output when sumexp == 0 | kernels-fusion | fattn-common.cuh:~977-981 | +3 (-1) | A | split-KV paths | keep | high | 06 |
| 155 | fattn workspace OOM retry (halve parallel_blocks) | context-runtime | fattn-common.cuh:~1240-1260; common.cuh:1528 | +22 (-6) | C | OOM only | keep | med | 06 |
| 156 | GGML_CUDA_FATTN_MAX_PARALLEL_BLOCKS knob | debug-trace | fattn-common.cuh:1219-1237 | +18 | E | default 0 | remove (keep OOM retry) | med | 06 |
| 157 | fattn-vec mask row stride from nb31 | kernels-fusion | fattn-vec.cuh:117, 278-279 | +4 (-1) | A | generic | keep | high | 06 |
| 158 | fattn-vec skip zero-weight V accumulation | kernels-fusion | fattn-vec.cuh:283-296, 374-380 | +30 (-10) | B | generic | keep only if measured | low | 06 |
| 159 | DFlash/DSpark deferred prompt encode (replay at first draft) | spec-decode | speculative.cpp process(), flush_pending_upto() | +250 (-40) | B | LLAMA_DFLASH_DEFER on | keep if PP matters | med | 07 |
| 160 | Lag-one async prompt replay thread | spec-decode | speculative.cpp replay_thread | +60 | B | LLAMA_DFLASH_ASYNC_REPLAY on | drop candidate (thread risk) | med | 07 |
| 161 | Flush knobs (FLUSH_ROWS, OPPORTUNISTIC) | spec-decode | speculative.cpp ~1591-1600 | +20 | E/B | 2048 / on | hardcode after tuning | med | 07 |
| 162 | DSpark draft window (sliding draft KV) | spec-decode | speculative.cpp ~1359 | +35 | C/B | LLAMA_DSPARK_DRAFT_WINDOW 4096 on | opt-in or document tradeoff | med | 07 |
| 163 | Defer short prompt tails | spec-decode | speculative.cpp ~1548 | +5 | B | LLAMA_DFLASH_DEFER_TAIL on | keep or hardcode | med | 07 |
| 164 | Lazy prefill (replay after first sampled token) | spec-decode | speculative.cpp ~1022 | +30 | B | LLAMA_SPEC_LAZY_PREFILL on | keep | med | 07 |
| 165 | MTP deferred prefill and KV-only replay | spec-decode | speculative.cpp ~2116-2330 | +170 (-10) | B | LLAMA_MTP_DEFERRED_PREFILL on | keep if win holds | med | 07 |
| 166 | MTP process_decode refactor | spec-decode | speculative.cpp process, process_decode | +80 (-40) | A/C | always | keep | med | 07 |
| 167 | MTP pending_h reset on new sequence | spec-decode | speculative.cpp process_decode | +4 | A | always | upstream bugfix | med | 07 |
| 168 | MTP adaptive disable and non-finite guard | spec-decode | speculative.cpp ~2110, ~2535 | +60 | A/C | LLAMA_MTP_ADAPTIVE_DISABLE on | keep guard; env can go | med | 07 |
| 169 | DFlash per-sequence row mapping (i_batch_rows) | spec-decode | speculative.cpp process ~1502-1540 | +30 (-15) | A | always | keep; upstream bugfix | high | 07 |
| 170 | DFlash null anchor guard (id_last < 0) | spec-decode | speculative.cpp ~1760 | +5 | A | always | keep | high | 07 |
| 171 | DSpark mask-token fallback to target vocab | spec-decode | speculative.cpp ~1190-1200 | +8 | A | only if draft lacks mask | keep | high | 07 |
| 172 | DFlash draft on first target device | spec-decode | speculative.cpp ~3440-3457 | +20 | C/B | LLAMA_DFLASH_DRAFT_TP default 0 | keep; document default | med | 07 |
| 173 | Draft repack knob LLAMA_DRAFT_REPACK | spec-decode | speculative.cpp ~3429-3433 | +6 | E | on | remove env; fix comment | high | 07 |
| 174 | DSpark target repack off by default | spec-decode | common.cpp ~1763-1769 | +6 | B/F | LLAMA_DSPARK_TARGET_REPACK 0 | keep if evidence holds | med | 07 |
| 175 | DFlash mirror-output flag via KV override key | spec-decode | common.cpp ~1746-1790 | +45 | E/C | LLAMA_DFLASH_MIRROR_OUTPUT 0 | remove DFlash branch; keep DSpark | med | 07 |
| 176 | Speculative stats breakdown and timers | debug-trace | speculative.cpp ~185-235, ~3710-3940 | +90 | E | always on (gen_perf) | remove dead counters | high | 07 |
| 177 | common_speculative_reset and server call | spec-decode | speculative.h:98; speculative.cpp ~3847; server-context.cpp:679 | +30 | A | always | keep; return value unused | high | 07 |
| 178 | llama_sched_reserve before prompt (eagle3, DFlash, MTP) | spec-decode | speculative.cpp ~645, 1258, 2080 | +8 | B | always for these drafts | keep | med | 07 |
| 179 | Server slot.sampled = LLAMA_TOKEN_NULL draft guard | spec-decode | server-context.cpp ~400, ~3899 | +6 | A | always | upstream bugfix | high | 07 |
| 180 | Server prompt_checkpoint_storage_lossless | kv-memory | server-context.cpp ~1121-2754 | +25 | A/C | DSV4 only changes behavior | keep | med | 07 |
| 181 | LLAMA_NO_MIDPROMPT_CKPT experiment knob | spec-decode | server-context.cpp ~4596-4612 | +14 | B | off | remove | med | 07 |
| 182 | reserve_prompt_graph (DSpark prompt reserve) | spec-decode | server-context.cpp ~4650-4672, ~4779 | +40 | B | LLAMA_DSPARK_PROMPT_RESERVE on; narrow gate | keep if DSpark kept; adds host sync | med | 07 |
| 183 | spec_i_batch_last dead variable | spec-decode | server-context.cpp:309, 5092 | +2 | E | n/a | delete | high | 07 |
| 184 | Slot draft-sampled guard and comment edit | spec-decode | server-context.cpp ~3899, ~5128 | +3 | A | always | keep guard; drop comment noise | high | 07 |
| 185 | examples/speculative prompt decode split by n_batch | spec-decode | examples/speculative/speculative.cpp ~197-222 | +25 (-12) | A | example binary | upstream bugfix | high | 07 |
| 186 | -tps / --tensor-parallel-size CLI | pipeline-stages | common/arg.cpp ~2852; common.h; common.cpp ~1737 | +18 | C | default 0 | keep only if pipeline topologies used | high | 07,08 |
| 187 | -tpsd / --spec-draft-tensor-parallel-size CLI | pipeline-stages | common/arg.cpp ~2863; common.h; speculative.cpp ~3360 | +15 | C | default 0 | keep if draft width must differ | high | 07 |
| 188 | load_mtp nextn and sched reserve before warmup | spec-decode | common.cpp ~1428-1431 | +5 | B | MTP-loaded contexts | keep | med | 07 |
| 189 | Speculative cosmetic noise | build-misc | speculative.cpp lines 26, ~185-190, ~1074, ~1908 | +15 | E | n/a | drop from upstream PR | high | 07 |
| 190 | MTP pre-norm accumulation buffer | spec-decode | llama-context.cpp ~1517-1545, ~2561-2610 | +110 (-10) | B | via speculative env, on | keep | high | 08 |
| 191 | Layer-input tap accumulation and epoch protocol | spec-decode | llama-context.cpp ~2860-3010 | +300 (-15) | B+C | LLAMA_TAP_ACCUM_MIN, _EMBD_SKIP, _MROPE on | keep; drop dead p_end protocol | high | 08 |
| 192 | Dead p_end wait/ready and lid getter (removable ~50) | spec-decode | llama-context.cpp:332-416, ~5087-5117; llama-ext.h:198-213 | 0 (about -50 removable) | E | none | remove | high | 08 |
| 193 | Persistent tap tensors outside galloc arena | spec-decode | llama-context.cpp ~860-915, ~3511; llama-context.h | +90 | B | LLAMA_TAP_STATIC on, all split modes | keep; fix header comment | high | 08 |
| 194 | Pipeline ring gating and TP enablement (-sm tensor) | context-runtime | llama-context.cpp ~553-590 | +30 (-6) | C+B | default-on for -sm tensor; env for drafts | keep; document default change | med | 08 |
| 195 | Graph reuse seq-layout sync and finish_compute | context-runtime | llama-context.cpp process_ubatch ~1905-1995 | +35 (-2) | A | generic | keep A; remove NO_DRAIN | med | 08 |
| 196 | Fused-op sibling-device acceptance | context-runtime | llama-context.cpp resolve_fused_ops ~691-705 | +12 | A | generic | keep | high | 08 |
| 197 | LLAMA_NO_FUSED_LID knob | debug-trace | llama-context.cpp ~739-752 | +11 | E | off | remove | med | 08 |
| 198 | Recurrent n_rs_seq clamps (qwen4exp, ubatch) | context-runtime | llama-context.cpp ~203-212, 362-372 | +20 | A | qwen4exp; small ubatch | keep | high | 08 |
| 199 | Private DFlash head (load_private_head) | spec-decode | llama-context.cpp ~3200-3285; llama.cpp:524-526 | +115 (+9 elsewhere) | A/C | LLAMA_DFLASH_PRIVATE_HEAD on | keep | high | 08 |
| 200 | MTP KV-only prefill replay flag | spec-decode | llama-cparams.h; llama-ext.h:139; qwen35.cpp:586 | +5 | B | via speculative env, on | keep | high | 08 |
| 201 | TP draft width and tensor_parallel_size plumbing (+3 public API) | pipeline-stages | llama.cpp ~156-290, ~393-410; include/llama.h:332-334 | +85 (-6; +3 API) | C | -sm tensor; DSV4 DFlash auto | keep; ABI break in llama_model_params | med | 08 |
| 202 | Multi-stage TP lane mapping (n_stages, lane_base, owning_stage) | pipeline-stages | llama-model.cpp split-state; llama-model.h | +75 (-15) | A | -tps below device count | keep | high | 09 |
| 203 | Stage-local mirrored weights (LLAMA_STAGE_LOCAL_MIRROR) | pipeline-stages | llama-model.cpp ~1316-1345 | +35 | B | default 1 | keep | high | 09 |
| 204 | MLA head split for DSV4 (LLAMA_MLA_TP) and design-A replicate | model-dsv4 | llama-model.cpp ~560-660; llama-arch.cpp ~1459-1475 | +85 (-5) | C | LLAMA_MLA_TP default 1 | keep with V4 | high | 09 |
| 205 | Shared-expert split, now global (LLAMA_SHEXP_SPLIT) | model-other | llama-model.cpp 775-792, 1160-1175 | +45 (-10) | C/B | default 1, all archs with shexp | validate on Flash Next and qwen35moe | high scope; med perf | 09 |
| 206 | Shexp split 4-granule rule | model-other | llama-model.cpp ~1160-1175 | +20 | A | -sm tensor | keep | med | 09 |
| 207 | Tied output head split | model-other | llama-model.cpp ~716-720, 906-909 | +12 | B | generic | keep | high | 09 |
| 208 | MTP block mirror under TP | model-qwen | llama-model.cpp ~680-710 | +25 | A | MTP models under -sm tensor | keep | med | 09 |
| 209 | DFlash draft split or replicate routing (LLAMA_DRAFT_SPLIT) | spec-decode | llama-model.cpp 597-605, 713-736 | +40 | C | env unset | keep; document LLAMA_DRAFT_SPLIT | med | 09 |
| 210 | PLE gather table shard (LLAMA_PLE_SHARD) | model-qwen | llama-model.cpp 411-421, 960-972, 1140-1151 | +45 | B | default 0 | keep; consider default on | med | 09 |
| 211 | PLE prefault (LLAMA_PLE_PREFAULT) | model-qwen | llama-model.cpp 27-29, 423-462, 2551 | +40 | F | default 0 | remove candidate | med | 09 |
| 212 | CPU pin of PLE and ENGRAM_EMBED tables | model-qwen | llama-model.cpp ~2885-2925 | +35 | A/E | input-class tensors | keep PLE pin; drop ENGRAM | high | 09 |
| 213 | Repacked MTP and output head (LLAMA_MTP_REPACK) | model-qwen | llama-model.cpp 1678-1697, 2095-2161 | +30 | B | default 1 | keep; fix comment at 2095 | high | 09 |
| 214 | Device extra bufts before default buft (make_gpu_buft_list) | repack | llama-model.cpp ~1648-1700 (-10) | +15 | B | any GPU with extra bufts | keep; verify non-TP runs | med | 09 |
| 215 | Weight-probe RESHAPE for attn_output_a | model-dsv4 | llama-model-loader.cpp 952-977, 1085-1094 | +25 | A | DSV4 only | keep | high | 09 |
| 216 | Weight-probe GGML_OP_CPY (LLAMA_DSV41_ENGRAM_CAST_DEVICE) | model-dsv4 | llama-model-loader.cpp 955-958, 1238-1247 | +15 | E | env default 0 | remove | high | 09 |
| 217 | load_all_data async relaxation (extra-buft, META host, copy-only) | tp-core | llama-model-loader.cpp ~1650-1710 | +20 | A | async upload path | keep | med | 09 |
| 218 | TENSOR_READ_LAZY-only context keeps ctx | model-qwen | llama-model.cpp ~2395-2415, ~2995-3000 | +20 | A | lazy tensors | keep | med | 09 |
| 219 | Engram scaffolding (new enums, tensors, hparams, helpers) | model-dsv4 | llama-arch.h/.cpp; llama-hparams.h; llama-model.h | +150 | E | none; never created or read | remove | high | 09 |
| 220 | V4.1 source-layer hparams (never populated) | model-dsv4 | llama-hparams.h 358-391; llama-arch.h/.cpp | +60 | E | none | populate in deepseek41 load or remove; flag KV | high | 09 |
| 221 | V4 v41 tier branches in deepseek4.cpp | model-dsv4 | deepseek4.cpp 954-1290 | +100 | E | dead (depends on 220) | remove or finish with 220 | high | 09 |
| 222 | dsv4_ratio_idx and dsv4_ratio_plain as hparams | model-dsv4 | llama-hparams.h 355-356 | +10 | F | values never loaded | revert to constants | high | 09 |
| 223 | head_hc_flags member (always 0) | model-dsv4 | models.h:1356; deepseek4.cpp:97-99 | +3 | E | none | remove | high | 09 |
| 224 | llama_model::path (written, never read) | model-other | llama-model.h:750-751; llama.cpp:524 | +2 | E | none | remove | high | 09 |
| 225 | NEXTN_FC_HIDDEN split MTP form | model-qwen | llama-arch.h/.cpp; gguf-py constants.py; tensor_mapping.py | +25 | F | converter never emits it | keep only if another converter ships it | med | 09 |
| 226 | tensor_mapping.py fc_embedding and nextn_hc names | model-qwen | gguf-py tensor_mapping.py 2999-3090 | +8 | E | converter only | likely redundant; verify | med | 09 |
| 227 | Compress-ratio tolerant load (qwen4exp) | model-qwen | qwen4exp.cpp 62-68 | +8 | C | qwen4exp | keep | med | 09 |
| 228 | qwen4exp MTP context filters | model-qwen | llama-model.cpp ~3700-3745 | +40 | A | MTP context | keep | med | 09 |
| 229 | qwen4exp conv-window snapshot path | model-qwen | qwen4exp.cpp 1407-1535, 1613 | +110 (-30) | B | single-sequence ring | keep | med | 09 |
| 230 | Snapshot ring in recurrent graph (build_rs, fill_s_write) | kv-memory | llama-graph.cpp/.h | +230 (-60) | B | ring_enabled; build_rs change unconditional | keep; verify non-ring topology | med | 09 |
| 231 | delta-net-base ring writes | model-qwen | models/delta-net-base.cpp | +65 (-5) | B | ring only | keep | med | 09 |
| 232 | Tap device tensors (layer_inp_dev, set_outputs, forward_expand move) | spec-decode | llama-graph.cpp ~1576-1620; deepseek4.cpp | +50 (-15) | B | LLAMA_TAP_STATIC on | keep | med | 09 |
| 233 | KV replicas in graph (build_get/cpy_k/v) | kv-memory | llama-graph.cpp ~3115-3180 | +90 (-10) | C | LLAMA_KV_REPLICA on | keep; KV audit owns | med | 09 |
| 234 | build_attn_store_kv (KV-only store for MTP replay) | spec-decode | llama-graph.cpp ~3526-3553; qwen35.cpp 586-612 | +35 | B | cparams.mtp_prefill_kv_only | keep | med | 09 |
| 235 | ggml_mul_mat_aux duplicate of llama_mul_mat_hadamard | model-other | llama-graph.cpp 71-86 | +17 | E | none | delete; call hadamard helper | high | 09 |
| 236 | kq_mask unallocated guard (KV-only graph) | spec-decode | llama-graph.cpp ~655-700 | +10 | A | KV-only graph | keep | high | 09 |
| 237 | qwen35 and qwen35moe MTP n_outputs guard | spec-decode | qwen35.cpp 553-560, 680-685; qwen35moe.cpp 592-600, 726-732 | +12 | A | MTP graph | keep | high | 09 |
| 238 | DFlash SWA pattern optional | spec-decode | dflash.cpp 144-151 | +8 | A | DFlash with SWA | keep | high | 09 |
| 239 | mimo2 SWA pattern fallback | model-other | mimo2.cpp 9-18 | +8 | C | mimo2 | keep | high | 09 |
| 240 | MiniMax-M2 TP (qk-norm split, decomposed RMS, unlisted) | model-other | minimax-m2.cpp +33; llama-model.cpp ~1100-1118; llama-arch.cpp | +75 | C | -sm tensor, minimax | keep if MiniMax in scope | med | 09 |
| 241 | deepseek4 build_hc_* API refactor and f32 mix precision | model-dsv4 | deepseek4.cpp 340-430; deepseek41.cpp 848-870; dflash.cpp | +90 (-80) | C | V4 and V4.1 | keep if V4 stays | med | 09 |
| 242 | deepseek4 layer-tap change (no forward_expand at build) | model-dsv4 | deepseek4.cpp 1313-1325, 1425-1427 | +8 (-6) | B | taps | keep | med | 09 |
| 243 | No-vocab special ids, print_info and token_to_piece guards | spec-decode | llama-vocab.cpp 2270-2293, 4100-4104, 4361-4390 | +45 (-10) | A/C | no_vocab models | keep | high | 09 |
| 244 | mxxm.tensor_mirror_output metadata key | spec-decode | llama-model.cpp 1821-1823; llama-model-loader.cpp 449 | +3 | C | DSpark or DFlash mirror | keep | high | 09 |
| 245 | Debug logs (shexp fprintf, LOG_ERROR before assert) | debug-trace | llama-model.cpp 566-572, 775-783 | +12 | E | unconditional fprintf | downgrade or remove | high | 09 |
| 246 | Model cosmetic and orphan items | build-misc | llama-model.cpp; models.h; deepseek4.cpp; loader | +25 | E | n/a | clean | high | 09 |
| 247 | Shared-layer KV replicas (llama-kv-cache) | kv-memory | llama-kv-cache.cpp ~270-320, 895, 2147; llama-kv-cache.h 336-352 | +150 (-5) | A/C | offload and LLAMA_KV_REPLICA on | keep; note Mistral-Large-4 use | med | 10 |
| 248 | Replica-aware graph helpers | kv-memory | llama-kv-cache.cpp ~1330-1500; llama-kv-cache-dsv4.cpp | +100 (-20) | A | always | keep | high | 10 |
| 249 | KV context tensor budget 2u to 4u | kv-memory | llama-kv-cache.cpp:116 | +1 (-1) | A | always | keep | high | 10 |
| 250 | DSv4 compression-plan overlap padding | model-dsv4 | llama-kv-cache-dsv4.cpp:632-690 | +55 (-30) | B | DSv4 always | keep; graph-reuse perf check | med | 10 |
| 251 | DSv4 prefill snapshot gate (LLAMA_DSV4_NO_PREFILL_SNAPSHOT) | model-dsv4 | llama-kv-cache-dsv4.cpp:722-750, 966-980 | +40 (-10) | B | default on | keep only with rollback test | med | 10 |
| 252 | DSv4 V4.1 lightning-indexer cache now populated | model-dsv4 | llama-kv-cache-dsv4.cpp:1426-1436, 1484-1488 | +4 (-6) | F/E | V4.1 only; possible VRAM cost | verify; likely revert for V4.1 | med | 10 |
| 253 | DSv4 get_can_checkpoint() == false | model-dsv4 | llama-kv-cache-dsv4.cpp:1672-1676; .h:124 | +6 | C/F | DSv4 always | keep | med | 10 |
| 254 | Recurrent snapshot ring (memory side) | kv-memory | llama-memory-recurrent.cpp 36-50, 559-600, 970-990, 1560-1585 | +230 (-40) | C/B | 4 Qwen archs, n_seq_max 1 | keep; fix docs or gate | med | 10 |
| 255 | Ring failure invalidation and finish_compute plumbing | kv-memory | llama-memory-recurrent.cpp 534-557; hybrid files; llama-memory.h | +90 (-5) | A/C | ring; hybrid always | keep | med | 10 |
| 256 | apply() returns find_slot result | kv-memory | llama-memory-recurrent.cpp:1499 | +1 (-3) | bugfix | all recurrent models | upstream candidate | med | 10 |
| 257 | Rollback guard (duplicate, max_rollback clamp, warning) | kv-memory | llama-memory-recurrent.cpp:210-266 | +40 (-15) | bugfix | all recurrent models | upstream candidate | med | 10 |
| 258 | seq_keep resets rollback index of dropped sequences | kv-memory | llama-memory-recurrent.cpp:351-360 | +6 | bugfix | all recurrent models | upstream candidate | med | 10 |
| 259 | state_read reset_ring replaces set_rs_idx | kv-memory | llama-memory-recurrent.cpp:1043-1080 | +3 (-4) | B/C | all | keep | high | 10 |
| 260 | DSv4 sync_replicas state read | model-dsv4 | llama-kv-cache-dsv4.cpp:381-433 | +5 (-20) | A | replica mode | keep | high | 10 |
| 261 | Dead DSv4 lambdas is_comp_owner, is_state_owner | model-dsv4 | llama-kv-cache-dsv4.cpp:1417-1426 | -10 | E | n/a | delete | high | 10 |
| 262 | Dead DSv4 pad_blocks parameter | model-dsv4 | llama-kv-cache-dsv4.cpp:453, 842, 854, 2301-2307 | -8 | E | n/a | delete | high | 10 |
| 263 | gguf-split chunked copy and merge | build-misc | tools/gguf-split/gguf-split.cpp:337-375, 549-580 | +40 (-25) | B/C | always | keep idea; fix compile error at 565-566 | high | 10 |
| 264 | llama-bench -tps and -nr output | build-misc | tools/llama-bench/llama-bench.cpp | +55 (-2) | C | only with -tps | keep for fork benchmarks | high | 10 |
| 265 | test-backend-ops additions | build-misc | tests/test-backend-ops.cpp | +51 (-2) | C | tests only | keep | high | 10 |
| 266 | Root CMake LLAMA_BUILD_VR_EXPERIMENTS option | build-misc | CMakeLists.txt:139, 250-252; vr/CMakeLists.txt | +5 | C | default OFF | keep in fork only | high | 10 |
| 267 | CANN CI stub job | build-misc | .github/workflows/build-cann.yml:41-46 | +6 | E/F | if: false | delete file or drop from upstream | med | 10 |
| 268 | README intro line and AGENTS.md head-merge section | build-misc | README.md:1; AGENTS.md:252-256 | +7 | C | fork docs | keep in fork only | high | 10 |
| 269 | CLAUDE.md now imports AGENTS.md | build-misc | CLAUDE.md:1 | 1 changed | F | n/a | keep | high | 10 |

## 2. Totals

Added LOC per theme (sum of positive ~LOC; "rows" = feature rows in that theme; cross-report duplicates counted once):

| Theme | Rows | ~LOC added |
|---|---|---|
| repack | 19 | 3751 |
| tp-core | 33 | 2460 |
| pipeline-stages | 20 | 1990 |
| spec-decode | 40 | 1790 |
| context-runtime | 28 | 1532 |
| peer-ar | 10 | 1379 |
| kv-memory | 12 | 966 |
| model-qwen | 15 | 937 |
| model-dsv4 | 19 | 781 |
| debug-trace | 16 | 780 |
| kernels-fusion | 20 | 630 |
| mmq-mmvq | 12 | 532 |
| flashnext-moe | 2 | 250 |
| build-misc | 18 | 241 |
| model-other | 7 | 179 |
| **Total** | 271 | **18198** |

Added LOC per class (primary letter; "other" = mixed or cosmetic):

| Class | Rows | ~LOC added |
|---|---|---|
| A | 58 | 2018 |
| B | 70 | 7648 |
| C | 69 | 5666 |
| E | 57 | 1833 |
| F | 12 | 926 |
| other | 5 | 107 |

Class key (from reports): A essential gfx906/TP compat, B perf, C functional extension, D superseded, E redundant/dead/debug, F unclear.
Rows marked "?" or with 0 added are excluded from sums. Quick-win LOC removed is listed in section 3, not subtracted here.

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

## 4. Correctness and bug findings

Each line: finding (file:line), then the report's confidence or status. "Candidate" means the report did not confirm it by test.

- B1. tools/gguf-split/gguf-split.cpp:565-566. fprintf format string split by a raw newline; does not compile. High (10).
- B2. tools/gguf-split/gguf-split.cpp:367-370. copy_file_to_file lacks the f_in truncation check the merge path has. Med-high (10).
- B3. ggml/src/ggml-backend.cpp:964-972. ggml_backend_event_query returns true when iface.event_query is NULL. Callers llama-context.cpp:1699, 1738 may treat unfinished work as ready. Fix: return false. High that fix is safe, med that it matters (02).
- B4. ggml/src/ggml-alloc.c, ggml_gallocr_layout_restore (n_buffers == 2 branch). A layout is adopted without buffer-id check; the "cannot disagree" comment is unverified. Med (02).
- B5. ggml/src/ggml-alloc.c, reserve (buffers only grow invariant). Relied on by layout cache, not checked. Med (02).
- B6. ggml/src/ggml-cuda/ggml-cuda.cu:2771-2779. q8_1 cache keys on pointer, data, strides, shape, no content version. In-place src1 write between consumers is not covered. Med (04).
- B7. ggml/src/ggml-cuda/mmq.cu:283-285 vs mmq.cuh:1560-1593. Padding from outer J_best, kernel tile from chunk J; chunk J could exceed padding and read past workspace. Medium (04).
- B8. ggml/src/ggml-cuda/mmvq.cu:478-483. calc_nwarps 5-column Q8_0 = 2 is ungated; changes results and perf on CDNA (MI100+). Med-high (04).
- B9. ggml/src/ggml-cuda/mmq.cu:351 (set), :297 (gate). tp_overlap.active is never cleared; overlap fires once per context. Low impact, default off (03).
- B10. ggml/src/ggml-cuda/ggml-cuda.cu:1410-1433 with tp-allreduce.cu. If canAccessPeer=1 on target, one-shot custom AR takes all contiguous F32 tensors: numerics switch from BF16 wire, graph capture fails (ggml-backend-meta.cpp:5197-5199). Med; decided by startup log (05 section 1c).
- B11. ggml/src/ggml-cuda/tp-allreduce.cu:626-628. CUDA_CHECK aborts if cudaDeviceEnablePeerAccess fails after canAccessPeer=1. Low-med (05).
- B12. ggml/src/ggml-cuda/getrows.cu:266-267, 579-581. Unsharded IQ4_NL/MXFP4 kq_n path always masks rows i01 >= n_rows; comment says n_rows==0 disables the mask, kernel does not implement that. If true, unsharded gathers with ne00 % 256 != 0 write zeros. Highest-impact item in 06; needs one gather test. Candidate (06).
- B13. ggml/src/ggml-cuda/ggml-cuda.cu:7262 (GET_ROWS relax to % 32) and getrows.cu bounded path. NVIDIA kernel behavior unverified. Review (03).
- B14. ggml/src/ggml-cuda/ggml-cuda.cu:2493. AMD BF16-less archs now use F32 for all batch sizes (was ne[1] > 32). Possibly unintended for non-gfx906. Med (03).
- B15. ggml/src/ggml-cuda/ggml-cuda.cu:4008-4015. Graph key change (n_nodes, ne[1]) also affects NVIDIA. Review (03).
- B16. ggml/src/ggml-cuda/ggml-cuda.cu:3672-3677 region, cpy_tensor_async copy_from_host (~3749-3830). New H2D path enabled on NVIDIA (was return false). Behavior change, confirm (03).
- B17. ggml/src/ggml-cuda/ggml-cuda.cu:6330-6345. Graph-capture instability heuristic can permanently disable capture (spec decode on NVIDIA at risk). Low-med (03).
- B18. ggml/src/ggml-cuda/ggml-cuda.cu:1218-1220; tp-allreduce.cuh:10-15. Comments say peer-write falls back to NCCL; false for non-NCCL builds. Doc bug (05).
- B19. src/llama-context.cpp extract_layer_inputs (~2571-2611, ~2946-2947). Accum branch skips host copy; host readers (speculative.cpp:725, 1672) would see stale rows if armed on same context. Also GGML_ASSERT(row_floats == n_embd) removed (H1, H3). Med (08).
- B20. src/llama-context.cpp ~2615 vs ~2622. extract_all_idxs not forced on accum branch; decode() assert on embd_batch_idxs can fire if paths mix. Verify (08 H2).
- B21. src/llama-context.cpp:1914-1924. LLAMA_REUSE_NO_DRAIN skips a synchronize guarding input overwrite. Default off, unsafe if enabled (08 H7).
- B22. src/llama-context.h (context comment) vs llama-context.cpp:869-878. Header says tap static is for non-tensor split modes; code applies to all modes. Doc bug (08 H5).
- B23. include/llama.h:332-334. tensor_parallel_size inserted mid-struct in llama_model_params: ABI break (lazy_mode, callbacks, kv_overrides shift). Inside repo fine; out-of-tree consumers break (08 H4, 01 notes).
- B24. src/models/deepseek4.cpp:955-962, 996-1003. LLAMA_DSV41_QNORM and LLAMA_DSV41_NO_COMPRESS are inert (dsv41_n_kv_source never set). FEATURES.md documents them as working. High (09).
- B25. src/llama-kv-cache-dsv4.cpp:1491-1493. V4.1 takes V4 tier branch for KV sizing because dsv41 source fields are never populated. Flag to KV auditor. High (09).
- B26. src/llama-kv-cache-dsv4.cpp:1426-1436, 1484-1488. V4.1 lightning-indexer cache allocated but probably never read or written (VRAM regression). Med (10 note 5).
- B27. src/llama-kv-cache-dsv4.cpp:973 vs 735-750. Reserve gate (max(1,n_seq_tokens) >= 64, no output check) vs fill gate; can under-reserve. Low; needs rollback test (10 note 4).
- B28. src/llama-memory-recurrent.cpp:1499 (apply returns find_slot), 210-266, 351-360. Generic fixes; upstream candidates. Med (10).
- B29. tools/server/server-context.cpp / common/speculative.cpp ~3938. Stats print level changed SPC_TRC to LOG_INF, so stats print at default verbosity. Unrequested change (07).
- B30. common/speculative.cpp:3429-3431. Comment says drafts load canonical weights; code keeps repack. Doc bug (07).
- B31. src/models/minimax-m2 and llama-arch.cpp: MiniMax-M2 removed from unsupported list under -sm tensor (llama-model.cpp:390 previously threw). Behavior change, non-target (09).
- B32. src/llama-model.cpp:775-783. Unconditional fprintf "[shexp-split]" under -sm tensor (debug). Downgrade (09).
- B33. common/speculative.cpp (MTP) pending_h zeroing on new sequence (positions[i_first]==0) and MTP non-finite guard ~2535; these are fixes; list as upstream bugfix candidates (07).

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
