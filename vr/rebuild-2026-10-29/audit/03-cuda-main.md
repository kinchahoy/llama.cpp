# Audit 03: CUDA/HIP main file (ggml-cuda.cu and related)

Scope: `ggml/src/ggml-cuda/ggml-cuda.cu` (+1578/-75; -w: +1576/-74), `common.cuh` (+148), `vendors/hip.h` (+2), `vendors/musa.h` (+5), `ggml-cuda/CMakeLists.txt` (+4/-2), `ggml-hip/CMakeLists.txt` (+4/-2).
Base: 418f905a4. Full diff saved at `/tmp/claude-1000/-home-raistlin-infer-mx-llama-cpp/552f07ca-9f17-4f87-894a-a8efaa505805/scratchpad/audit/cuda-main.diff`.
Line numbers below are HEAD (working tree, clean) unless marked "diff:".
Read-only audit. Nothing built or run.

Terms: "shared" = runs on NVIDIA CUDA too. "HIP-gated" = inside `#if defined(GGML_USE_HIP)` or only reachable on gfx906/Vega20 (`GGML_CUDA_CC_VEGA20`, `GGML_CUDA_CC_IS_AMD`).

---

## 1. Table

| Feature | Files | ~LOC +/- | Class | Upstream equiv | Gating | Recommendation | Confidence |
|---|---|---|---|---|---|---|---|
| Repacked Q8_0/MXFP4/IQ4_NL/K-quant weight integration: buft registration (`get_extra_bufts`), `supports_buft`, `supports_op` guard, `set_tensor_async` dispatch, `mul_mat`/`mul_mat_id` dispatch, alloc-retry hook | ggml-cuda.cu (+ q8_repack/*) | ~60 + ~15 dispatch | A | none (fork-only layout) | Type-based, NOT gfx906-gated in `repack-common.cu`; reachable on any CUDA device that the loader asks for extra bufts. supports_op guard is shared | Keep. Move glue to q8_repack/ (see Move-out M3) | high (for behavior), med (NVIDIA exposure, see note 2.1) |
| Repack fused up/gate MMV (GLU) and bias-ADD MMV fusion inside `ggml_cuda_try_fuse` | ggml-cuda.cu:5704-5800 (diff:1754-1820), 5934 | ~90 | A/B | none | Only fires when `ggml_cuda_repack_mul_mat_should_fire` is true (repacked src0). Code is shared but effectively repack-only | Keep. Move whole block into q8_repack helper (M3) | high |
| `GGML_CUDA_REPACK_MOE_FUSION_STATS` histogram + witness log | ggml-cuda.cu:40-75 (diff 13-55), 5742-5746 | ~50 | E (debug) | none | env, default OFF; witness log is unconditional once per type (WARN level) | Remove, or move to q8_repack (QW7). The `atexit` registration and 65x64 atomic table are debug-only | high |
| Q8_1 activation cache (+ producer hook, `ggml_cuda_q8_1_producer_target`) | ggml-cuda.cu:2740-2810, 5114-5136; common.cuh:1600-1650, 1855-1860; mmq.cu:374, mmvq.cu:1583, q8_repack/mul-mat.cu:100 | ~165 | B | none | `GGML_CUDA_Q8_1_CACHE` default ON; `GGML_CUDA_Q8_1_PRODUCER` default ON. SHARED: applies to NVIDIA mmq/mmvq too | Keep behavior. Move cache impl to its own file (M2). Confirm NVIDIA impact or gate with `#if HIP` if NVIDIA numbers were not measured | med |
| Prefill AllReduce slab overlap (`ar_slab`, `ggml_cuda_mul_mat_ar_slabs`, mark hook) | ggml-cuda.cu:2948-3013 (diff 1106-1171), 3018-3019 | ~75 | B | none | `GGML_CUDA_AR_SLABS` default 4 (0/1 off). NOT HIP-gated. Meta backend marks it whenever `tps > 1` and closure is AR (ggml-backend-meta.cpp:5261). So ON for NVIDIA multi-GPU TP too | Keep, but document NVIDIA exposure. Candidate for own file (M5) | high (code), med (NVIDIA impact) |
| TP-overlap MMQ (Vega20, Q8_0 8704x5120, 2048 tokens) | ggml-cuda.cu:3073-3081, 1708-1714, 1357, 829-831, 3989-3996; mmq.cu:290-360 (outside slice) | ~25 here, ~70 in mmq.cu | E | none | `GGML_CUDA_TP_OVERLAP` default OFF. Enable is HIP-only (1708). BUT the graph-compat check (3989-3996) reads the env with no HIP gate, and the mmq path is `#if HIP`. Shape constants hardcoded | Remove (QW9). vr/README.md:55 says the env must stay unset in the normal config. No vr doc records a win | high |
| `tp_overlap` struct fields `dst`, `slab_elements` | common.cuh:1486-1493; mmq.cu:352-353 | ~3 | E (dead write) | none | HIP overlap only | Remove with the overlap (QW3). `active` is a latch, see 2.5 | high |
| Custom AR (tp-allreduce) dispatch: prepare/launch split, eligibility gate, fallback handling | ggml-cuda.cu:~1369-1520 (diff 367-523), 1485-1502 (diff 487-523), 1468-1475 | ~155 | C | none (upstream has NCCL + host-staged only) | `GGML_ENABLE_CUSTOM_AR` default ON (0 off, 1779). NOT HIP-gated at the dispatch; tp-allreduce.cu peer-write path uses `broadcast_ok` (tp-allreduce.cu:704, set from `hw_peer_write_coherent`). NVIDIA exposure depends on that flag | Keep for gfx906. Check the NVIDIA case (is `hw_peer_write_coherent` ever true on NVIDIA?). Move dispatch glue to a comm-ext file (M1) | med |
| Comm context: xfer staging ring (`xfer_staging_*`), `staging_depth`, `set_staging_depth`, `set_pipeline_stages` | ggml-cuda.cu:~1260-1340 (diff 252-340), 1693-1700 (diff ~1690) | ~110 | C/F | none | Staging ring is used only by the NCCL sendrecv path (GGML_USE_NCCL). `set_staging_depth` proc is NCCL-only. `set_pipeline_stages` sets `custom_ar.prefer_small_twoshot`, not gated | Keep. Move (M1). Reconcile with RCCL-off production config (see 2.6) | med |
| NCCL/RCCL P2P send/recv stage transfer (`ggml_backend_cuda_comm_sendrecv_tensor`) | ggml-cuda.cu:1836-2045 (diff 617-833), proc table 2135-2142 | ~215 | F | none | `#ifdef GGML_USE_NCCL` only. Production config is RCCL OFF, so this code is compiled out there | Decide: either move to comm-ext file (M1) or drop if the dedicated-queue fallback is the one used in production (see 2.6) | med |
| Dedicated copy stream for meta stage transfer (`pp_copy_stream`, `cpy_tensor_async_dedicated_queue/drain`, `ggml_pp_drain_should_sync`) | ggml-cuda.cu:3842-3960 (diff 1329-1450), ctx fields common.cuh:1466-1474 | ~140 | C | none | Not HIP-gated. Drain host-sync is controlled by `GPU_MAX_HW_QUEUES > 4` (ROCm var, pre-existing name) and `GGML_PP_DRAIN_FORCE_SYNC` (default OFF) | Keep for stage transfer. Could be `#if HIP` if NVIDIA does not need it. Move (M7) | med |
| `cpy_tensor_async`: copy from host (`copy_from_host`) | ggml-cuda.cu:3749-3830 (diff 1279-1327) | ~30 | C/F | none | SHARED. CUDA uses `cudaMemcpyAsync` H2D on dst stream (MUSA excluded). HIP uses sync `cudaMemcpy` | Behavior change on NVIDIA (was: return false, fall back). Confirm intended, or gate to HIP (2.7) | med |
| `set_tensor_2d_async` misaligned-row workaround (pinned scratch) | ggml-cuda.cu:3693-3735 (diff 1238-1278) | ~30 | A | none | `#if defined(GGML_USE_HIP)` and `(size & 3) != 0`. gfx906/ROCm SDMA issue | Keep. Could move to a HIP-only helper file (M8) | high |
| `set_tensor_async` repacked branch | ggml-cuda.cu:3672-3677 | ~5 | A | none | Shared, but only when buft is repack | Keep | high |
| `ggml_cuda_try_fuse`: ADD + RMS_NORM + MUL fusion | ggml-cuda.cu:5147-5157 (diff 1717-1731); norm.cu helpers (other slice) | ~15 | B | none | `GGML_CUDA_FUSE_ADD_RMS_NORM` default ON. SHARED | Keep. Consider a kill switch removal later (low value) | med |
| HC gate fusion MUL/VIEW/ADD/UNARY(sigmoid)/SCALE -> `ggml_cuda_op_mul_add_sigmoid_scale` (softcap.cu) | ggml-cuda.cu:4910-4925 (diff 1620-1664), 6018-6021 (diff 1822-1826) | ~45 | C | base had a different hc_post_gate branch (base:3741-3756, 4740-4745) that the fork REPLACED in `can_fuse` | Ungated, always on, SHARED | Keep. See dead-branch item in 2.2 | med |
| DEAD: `SCALE,UNARY(sigmoid),SCALE,DSV4_HC_POST` -> `ggml_cuda_op_dsv4_hc_post` try_fuse site | ggml-cuda.cu:6008-6011 | ~4 | E | base had the matching can_fuse branch | Would be SHARED, but unreachable now: `ggml_cuda_can_fuse` no longer has a branch that returns true for this op list | Remove (QW5). Verify by re-reading `can_fuse` tail (no path returns true) | med-high |
| MoE weighted-reduction fusion: env kill switch, `n_tokens == 0` handling, alloc-dep in graph optimize | ggml-cuda.cu:4660-4680 (diff ~1592-1613), 5103-5110, 5181, 6469 | ~20 | B/A | base has the fusion (match fn at base:~3515) | `GGML_CUDA_MOE_WEIGHTED_REDUCTION` default ON. SHARED | Keep. The `n_tokens == 0` case is a crash-avoidance fix (A) | high |
| Q8_1 producer target function | ggml-cuda.cu:5114-5136 | ~25 | B | none | `GGML_CUDA_Q8_1_PRODUCER` default ON. SHARED | Keep (goes with Q8_1 cache, M2). Misplaced comment, see QW6 | med |
| Top-K fusion bounds checks (`node_idx < n_nodes`) | ggml-cuda.cu:~4549-4574 (diff 1553-1578) | ~8 | A (safety) | none | SHARED. Pure bounds guards | Keep, good | high |
| Graph key mixes `n_nodes` and `nodes[0]->ne[1]` | ggml-cuda.cu:4008-4015 | ~6 | C | none | SHARED | Keep. Note: changes cache keying on NVIDIA too | med |
| Graph update: HIP recreate-on-kernel-func-change; debug log removal | ggml-cuda.cu:4062-4105 (diff 1491-1531) | ~35 (HIP block ~27) | A/B | none | HIP block `#if defined(GGML_USE_HIP)`; the log removal is shared (cosmetic) | Keep HIP block. Performance: queries `hipGraphGetNodes` on every chunk update (see 2.8) | med |
| Graph-cache instability heuristic (`n_prop_checks/resets`, `unstable_disabled`, `disable_due_to_memory`) | common.cuh:1309-1336; ggml-cuda.cu:6330-6345 | ~20 | F | none | SHARED. Permanently disables graph capture for an entry after 4+ resets and >10% reset rate | Keep only if measured; document. Risk for spec decode on NVIDIA (2.8) | low-med |
| Whole-token capture API (`ggml_backend_token_capture_begin/end`, `token_graph_launch/free`, `graph_cache_disable`, `g_cuda_outer_capture`) | ggml-cuda.cu:7750-7800 (diff 2041-2100), 6302, 6304-6310 (graph_compute guard), proc table 2100-2130 | ~75 | C | none | Shared. Only used when meta backend owns token graphs (ggml-backend-meta.cpp:5028, n_stages==1) | Keep. Candidate for own file (M6) | med |
| `GGML_CUDA_OP_SYNC` per-op sync debug | ggml-cuda.cu:6218-6233 (diff 1834-1852) | ~17 | E (debug) | none | env, default OFF. Shared (graph exec loop) | Remove or keep as debugging tool. Low value vs. AMD_SERIALIZE_KERNEL (QW7) | high |
| `ggml_backend_cuda_init_impl` + `copy_only` backend (`device_init_backend` params "copy-only", stream() returns cudaStreamPerThread) | ggml-cuda.cu:7055-7060, 7951-7970 (diff 1945-1951, 2184-2211); common.cuh:1602-1612 | ~20 | C | none | SHARED. Called by ggml-backend-meta.cpp:2883 | Keep. Note: copy-only skips BLAS handle (eager) | high |
| Eager cuBLAS handle creation at init (`GGML_CUDA_EAGER_BLAS_HANDLE`) | ggml-cuda.cu:7972-7982 | ~12 | B | none | env default ON (0 off). SHARED: changes init on NVIDIA too | Keep only if justified; else default OFF (2.9) | med |
| cuBLAS chunked source-0 conversion (`GGML_CUDA_CUBLAS_CHUNK_MIB`, default 256 MiB) + `alloc_with_cache_recovery` | ggml-cuda.cu:2211-2300 (diff 838-883, 892-950), 2309-2330 | ~95 | A (memory) | none | Env default 256 MiB; 0 disables chunking. SHARED: any NVIDIA case with F32 compute and big non-F32 weight | Keep. Gate-check on NVIDIA (`compute_type == F32` path only) | med |
| cuBLAS BF16 fallback: `GGML_CUDA_CC_IS_AMD(cc)` now forces F32 for all AMD BF16-less archs (was only when `ne[1] > 32`) | ggml-cuda.cu:2493 (diff 955-957) | 1 | B/F | none | AMD-only (not gfx906-only; any AMD without fast bf16) | Keep if measured for gfx906; F otherwise. Confirm (2.10) | med |
| `supports_op`: repack guard block (reject repacked weights for non-MUL_MAT ops, non-src0) | ggml-cuda.cu:7085-7110 (diff 1962-1984) | ~25 | A | none | SHARED but only fires for repacked buffers | Keep (correctness) | high |
| `supports_op`: GET_ROWS for IQ4_NL/MXFP4 relaxed from `% QK_K` to `% 32` | ggml-cuda.cu:7262 (diff 1995-1999) | 2 | C | none | SHARED (all CUDA). Needs bounded path in getrows.cu (other slice) | Keep if getrows.cu bounded path is verified on NVIDIA (2.11) | med |
| `supports_buft`: accepts repack buft on same device | ggml-cuda.cu:7577-7583 | ~3 | A | none | Shared, repack only | Keep | high |
| Extra-bufts proc (`ggml_backend_dev_get_extra_bufts`) | ggml-cuda.cu:7864-7885 (diff 2156-2176) | ~22 | A | upstream has this proc name in the interface | SHARED: exposes repack buft for every CUDA device | Keep. Confirm NVIDIA loader path (note 2.1) | med |
| Repack scratch release on alloc failure (`ggml_cuda_repack_try_release_scratch`) | ggml-cuda.cu:1126-1140 (diff 1119ff: 222-229), compute hook 6309 | ~12 | A | none | SHARED (buffer alloc path on CUDA) | Keep | med |
| `event_query` device iface member (+ `cudaEventQuery`/`cudaErrorNotReady` in hip.h, musa.h) | ggml-cuda.cu:7637-7646, 7663; ggml-backend-impl.h (outside slice, HEAD:230); vendors/hip.h +2; musa.h +2 | ~12 | C | upstream iface at base has no `event_query` (base backend-impl.h has `event_synchronize` only) | SHARED (all CUDA/HIP) | Keep. Used by meta sync logic (outside slice) | high |
| Pool: fallible `try_alloc` on `ggml_cuda_pool_leg` / `_vmm`, `alloc_impl`, `ggml_cuda_pool_alloc::try_alloc/reset` | ggml-cuda.cu:475-800 (diff 67-166, 181-240ish), common.cuh:1217-1260 | ~60 | A/C | none | SHARED. Only changes behavior when `try_alloc` is called (q8 cache, cuBLAS recovery) | Keep. Note: pool_leg `invalidate_cuda_graphs` debug log removed (cosmetic) | high |
| Backend context: `pp_copy_stream`/events, `tp_overlap`, `ar_slab`, q8 cache fields, fattn/mmq/repack workspace caps, `hip_set2d_scratch` | common.cuh:1462-1560 | ~60 | mixed (see 2.4) | none | SHARED structs; HIP-only scratch fields behind `#if HIP` | Remove dead `top_k_workspace_rows_cap` (QW2), `kernel_funcs` (QW1); see table notes | high |
| Comm init: custom-AR env, TP overlap enable (HIP), `ggml_backend_comm_*` proc table additions | ggml-cuda.cu:1700-1800 (diff 1680-1800), proc table 2100-2140 | ~60 | C | none | `GGML_CUDA_TP_OVERLAP` HIP-only enable. Custom AR env SHARED | Keep (minus overlap, QW9) | med |
| `ggml_cuda_graph_check_compability`: per-node `getenv("GGML_CUDA_TP_OVERLAP")` with hardcoded 8704/5120/2048 | ggml-cuda.cu:3989-3996 | ~8 | E/B | none | SHARED, NOT HIP-gated; env read per node per graph compute | Hoist to static or delete with overlap (QW4, QW9) | high |
| `ggml_cuda_graph_evaluate_and_capture` replay: `ggml_cuda_graph * graph` re-lookup | ggml-cuda.cu:~6280-6295 | ~3 | C | none | SHARED | Keep | high |
| Hyper-connection / cuBLAS / other minor cosmetic hunks (blank lines, comment move, `use_mul_mat_vec_q` line) | diff ~986-987, 1670-1672, 1435-1436, 5111 | ~5 | E (noise) | none | - | Clean up (QW6, QW13) | high |
| Vendor headers: `hip.h` +2 (`cudaEventQuery`, `cudaErrorNotReady`); `musa.h` +5 (`cudaErrorNotReady`, `cudaEventQuery`, `cudaMemcpyFromSymbol`, `cudaMemcpyToSymbol`, `cudaStreamCaptureModeThreadLocal`) | vendors/hip.h, vendors/musa.h | +7 | A/C | none | HIP needed (event_query). MUSA: symbol macros used by tp-allreduce.cu:9-10 (already in HEAD). MUSA not built here | Keep. MUSA compile not verified | med (MUSA) / high (HIP) |
| CMake: `q8_repack/*.cu` and `*.cuh` globbed (CUDA and ROCm lists) | ggml-cuda/CMakeLists.txt:114-118, ggml-hip/CMakeLists.txt:60-63 | +4 / -2 each | A | none | Build-level, shared | Keep. Note: GLOB without CONFIGURE_DEPENDS, re-run cmake when files are added | high |
| `common.cuh`: `dequantize_kq_n_t` typedef (used by getrows.cu) | common.cuh:969-976 | +6 | C | none | SHARED (getrows) | Keep | high |
| `common.cuh`: `ggml_cuda_tp_overlap_state` and `ggml_cuda_ar_slab_state` structs | common.cuh:1485-1510 | ~25 | B / E | none | SHARED types | Keep ar_slab; trim tp_overlap (QW3) | high |

---

## 2. Per-feature notes

### 2.1 Repacked weights: NVIDIA exposure (most important open question)
- `ggml_cuda_repack_tensor_supported()` (q8_repack/repack-common.cu:12-35) is type- and shape-based. It has no `GCN`/`gfx906`/`HIP` check.
- `ggml_backend_cuda_device_supports_buft()` (ggml-cuda.cu:7577-7583) and the extra-bufts proc (ggml-cuda.cu:7864-7885) expose the repack buft for every CUDA device.
- Whether NVIDIA loads Q8_0 weights into it depends on whether the loader queries extra bufts for CUDA devices. This slice does not settle that. Check llama-model loader and `ggml-backend-meta.cpp:328`, `:414`, `:637`.
- Recommendation: add an explicit `#if defined(GGML_USE_HIP)` or a runtime `GGML_CUDA_CC_IS_CDNA/GCN` check in `ggml_backend_cuda_repack_buffer_type()` (q8_repack/buffer.cu:678) if NVIDIA should not get it.

### 2.2 Fusion sites in `ggml_cuda_try_fuse` (ggml-cuda.cu:5138 onwards)
- Ungated, always on, SHARED: ADD+RMS_NORM+MUL (5147), HC gate MUL/VIEW/ADD/UNARY/SCALE (6018), MoE weighted reduction (5181, default on), Q8_1 producer (5134).
- Repack-only (effectively gfx906-only in production): MMV GLU fusion (5719-5750), bias MMV fusion (5934), `should_fuse_mul_mat_vec_f/q` early-returns (2664, 2696).
- DEAD BRANCH: 6008-6011 (`SCALE,UNARY,SCALE,DSV4_HC_POST`). The fork removed the `hc_post_gate_ops` branch from `ggml_cuda_can_fuse` (diff 1620-1664, base:3741-3756). Walking `ggml_cuda_can_fuse` for this 4-op list: the generic `ggml_can_fuse` (4918) passes, then no specific branch matches (the only ops-list branches are 4964-5070), so it returns false at the end. The try_fuse call therefore never fires. Confidence med-high from static reading. Recommend deleting 6008-6011 or confirming with a graph dump.

### 2.3 Q8_1 cache (the largest default-on memory-touching change)
- `GGML_CUDA_Q8_1_CACHE_MAX_ENTRIES 2` (ggml-cuda.cu:~2740 region) with comment claiming hit rate is flat for bounds 1-32. Memory is `ne_padded * ubatch` per entry. Comment cites 153 MiB/device at -ub 8192 for 8 entries.
- Consumers: mmq.cu:374, mmvq.cu:1583, q8_repack/mul-mat.cu:100.
- Default ON, SHARED. Gated only by env `GGML_CUDA_Q8_1_CACHE=0`.
- `alloc_with_cache_recovery` (2246) clears the cache when a BLAS conversion buffer fails. Interacts with the pool's fallible `try_alloc`.

### 2.4 Dead or write-only members (verified by `grep` over ggml-cuda/ and ggml-backend-meta.cpp)
- `ggml_cuda_graph::kernel_funcs` (common.cuh:1315): declared, comment at 1313-1314, never read or written. Only `kernel_funcs_chunks` is used (ggml-cuda.cu:4062+). DEAD.
- `ggml_backend_cuda_context::top_k_workspace_rows_cap` and `top_k_workspace_rows_cap_logged` (common.cuh:1533-1534): declared only (`git grep` at HEAD confirms). DEAD. The comment at 1530-1532 describes behavior that does not exist.
- `ggml_cuda_tp_overlap_state::dst` and `slab_elements` (common.cuh:1486-1493): written in mmq.cu:352-353 (outside this slice), never read. Write-only.
- `ggml_cuda_tp_overlap_state::fork`, `ready[2]`: created lazily in mmq.cu:313-316, used in mmq.cu:347, destroyed in ggml-cuda.cu:829-838. Not dead, but only reachable with `GGML_CUDA_TP_OVERLAP=1`.
- All other new context members checked (`fattn_parallel_blocks_cap`, `mmq_*_cap`, `repack_workspace_cols_cap`, `hip_set2d_scratch_*`, `staging_depth`, `copy_only`, `graphs_owner_disabled`, `q8_1_cache_*`, `cublas_chunk_logged`) have at least one reader outside their declaration.

### 2.5 TP-overlap latch bug candidate
- mmq.cu:351 sets `overlap.active = true` and nothing in the tree clears it (grep: `active = false` absent). mmq.cu:297 gates on `!ctx.tp_overlap.active`, so the overlap fires at most once per context lifetime.
- Default off (needs env=1), so low impact. Still a bug or undocumented latch. Removing the overlap (QW9) removes it.

### 2.6 NCCL/staging path vs RCCL-off production config
- Brief says RCCL is off. Then `GGML_USE_NCCL` is undefined for the HIP build and these are compiled out: `ggml_backend_cuda_comm_sendrecv_tensor` (1836-2045) body, its proc registration (2135-2142), `set_staging_depth` registration.
- In production, stage transfer uses the dedicated-queue path (3842-3960). Confirm before moving or deleting the NCCL sendrecv code.
- The xfer staging ring (1287-1340) is referenced in the comm destructor (1385-1413 region) even without NCCL, so its destructor code is live.

### 2.7 `copy_from_host` in `cpy_tensor_async` (3749-3830)
- New H2D async path is enabled for CUDA (not HIP). Previously returned false. Behavior change on NVIDIA. For pageable host memory `cudaMemcpyAsync` returns after staging, so ordering is fine, but it is still a change.
- Recommend `#if defined(GGML_USE_HIP)` gating if NVIDIA was not measured, or note it as shared.

### 2.8 Graph capture/replay
- Graph key (4008): now mixes shape. Affects NVIDIA.
- Instability heuristic (6330-6345): permanently disables graph capture for an entry after >=4 property resets and >10% reset ratio. Shared. Could disable graphs for spec decode on NVIDIA. Needs evidence in vr/.
- HIP update path (4062-4105): calls `hipGraphGetNodes` twice plus `hipGraphNodeGetType`/`hipGraphKernelNodeGetParams` on every chunk update. Per-token CPU overhead on HIP. Could cache the func list at capture time.
- Token capture API: `g_cuda_outer_capture` (6302) makes `ggml_backend_cuda_graph_compute` skip per-subgraph graphs while an outer capture runs. Proc-address only, so no ABI risk.

### 2.9 Init-time memory
- `GGML_CUDA_EAGER_BLAS_HANDLE` default ON (7972). Creates cuBLAS handle at backend init on NVIDIA too. Justification in comment is for gfx906 reserve accounting. Shared.

### 2.10 cuBLAS BF16 change (2493)
- `if (GGML_CUDA_CC_IS_AMD(cc) && src1->ne[1] > 32)` became `if (GGML_CUDA_CC_IS_AMD(cc))`. Any AMD arch without fast bf16 now uses F32 compute for all batch sizes. Consider whether `ne[1] > 32` was intended for other AMD archs.

### 2.11 GET_ROWS relaxation (7262)
- IQ4_NL/MXFP4 `% QK_K` -> `% 32` on NVIDIA as well. Depends on bounded path in getrows.cu (other slice, `dequantize_kq_n_t`). Verify NVIDIA kernel behavior there.

### 2.12 Env-gated HIP-only code that is not HIP-gated at the call site
- `GGML_CUDA_TP_OVERLAP` read in graph-compat (3989) without HIP gate.
- `GGML_ENABLE_CUSTOM_AR` (1779) and custom AR dispatch (`ggml_backend_cuda_comm_allreduce_custom`) are SHARED.
- Shape constants 8704/5120/2048 (3073-3081, 3989-3996) are model-specific magic numbers in generic code.

---

## 3. Environment variables added by this slice

Default column is the behavior when the variable is unset.

| Variable | Default (unset) | Set-value effect | Location (HEAD) | Scope | Note |
|---|---|---|---|---|---|
| `GGML_TP_AR_NO_GATE` | size gate ON | any non-"0" non-empty value disables gate | ggml-cuda.cu:1410-1415 | shared (comm) | diagnostic |
| `GGML_TP_AR_ONESHOT` | OFF | forces one-shot even when no peer-write broadcast | ggml-cuda.cu:1414-1420 | shared (comm) | diagnostic |
| `GGML_CUDA_TP_OVERLAP` | OFF | =1 enables TP overlap (HIP) and forces graph-off for matching nodes | 1708-1714, 1357, 3989-3996 | HIP enable; graph-check shared | remove (QW9) |
| `GGML_ENABLE_CUSTOM_AR` | ON | =0 disables custom AR | ggml-cuda.cu:1779-1783 | shared | keep |
| `GGML_CUDA_CUBLAS_CHUNK_MIB` | 256 | 0 = no chunking; N = chunk size MiB | ggml-cuda.cu:2277-2281 | shared | keep |
| `GGML_CUDA_Q8_1_CACHE` | ON | =0 disables cache | ggml-cuda.cu:2742-2747 | shared | keep |
| `GGML_CUDA_AR_SLABS` | 4 | 0/1 disables; clamped to 0..8 | ggml-cuda.cu:2953-2960 | shared (meta gates tps>1) | keep, document |
| `GGML_PP_DRAIN_FORCE_SYNC` | OFF | =nonzero forces host sync in dedicated drain | ggml-cuda.cu:3909-3914 | shared | escape hatch |
| `GPU_MAX_HW_QUEUES` (pre-existing ROCm name, read here) | unset | >4 forces host sync in dedicated drain | ggml-cuda.cu:3916-3922 | shared | ROCm variable reused as a switch |
| `GGML_CUDA_MOE_WEIGHTED_REDUCTION` | ON | =0 disables fusion and alloc-dep | ggml-cuda.cu:5103-5109 | shared | keep |
| `GGML_CUDA_Q8_1_PRODUCER` | ON | =0 disables producer hook | ggml-cuda.cu:5116-5120 | shared | keep |
| `GGML_CUDA_FUSE_ADD_RMS_NORM` | ON | =0 disables fusion | ggml-cuda.cu:5147-5151 | shared | keep |
| `GGML_CUDA_REPACK_MOE_FUSION_STATS` | OFF | =nonzero prints histogram | ggml-cuda.cu:5742-5746 | repack | remove (QW7) |
| `GGML_CUDA_OP_SYNC` | OFF | any value syncs after every op in graph exec | ggml-cuda.cu:6223 | shared | debug, remove or keep (QW7) |
| `GGML_CUDA_EAGER_BLAS_HANDLE` | ON | =0 creates handle lazily | ggml-cuda.cu:7973-7978 | shared | review (2.9) |

Pre-existing (base) env vars in this file, not new: `GGML_CUDA_DEVICES`, `GGML_CUDA_ALLREDUCE`, `GGML_CUDA_CUBLAS_COMPUTE_TYPE`, `GGML_CUDA_GRAPH_CHUNK_SIZE`, `GGML_CUDA_GRAPH_OPT`, `GGML_CUDA_DISABLE_GRAPHS` (common.cuh:1331), `GGML_CUDA_PDL` (common.cuh:1837).

Seen in passing, in other slices (not audited here): `GGML_RP_NC_FUSION`, `GGML_CUDA_REPACK_KQUANT_MOE_FUSION`, `GGML_RP_GENERIC_MMV`, `GGML_RP_MMV_ROWS`, `GGML_RP_NC_PLANAR`, `GGML_RP_NC_L16`, `GGML_RP_NC_RPL`, `GGML_RP_NC_23`, `GGML_RP_NC_LDS`, `GGML_CUDA_FLASHNEXT_MOE` (q8_repack/*), `GGML_TP_AR_TWOSHOT`, `GGML_TP_AR_TWOSHOT_MIN_NE`, `GGML_TP_AR_BLOCKS`, `GGML_TP_AR_KTIMING`, `HSA_FORCE_FINE_GRAIN_PCIE` (tp-allreduce.cu).

---

## 4. Move-out candidates (reduce merge conflicts)

Most of the added code is not gfx906-specific. It is multi-GPU comm, fusion, and caching. Only the items marked HIP are truly gfx906/ROCm-specific.

- M1. Comm-ext file (new `ggml-cuda/comm-ext.cu` + header). Move: custom-AR prepare/launch wrappers (diff 367-523, ~155 LOC), xfer staging helpers (diff 252-340, ~90 LOC), NCCL sendrecv (diff 617-833, ~215 LOC). Needs the comm context struct in a header. Medium effort. Biggest conflict reducer.
- M2. Q8_1 cache file (new `ggml-cuda/q8-1-cache.cu`). Move `ggml_cuda_q8_1_cache_acquire`, `ggml_cuda_q8_1_cache_enabled`, `ggml_cuda_q8_1_producer_target` (~140 LOC). Struct entry stays in common.cuh (small). Low-medium effort.
- M3. Repack glue (into `q8_repack/`). Move the GLU/bias MMV fusion block (5704-5800), `ggml_cuda_repack_fusion_glu_ok` (44-57), and the MoE stats if kept, into a helper such as `int ggml_cuda_repack_try_fuse(ctx, cgraph, i)` in `q8_repack/mul-mat.cu`. ggml-cuda.cu keeps one call. Biggest gfx906-specific block; ~100 LOC. Low effort, since the repack folder already exists.
- M4. MoE stats histogram (13-75). Move to `q8_repack/repack-common.cu` or delete (QW7).
- M5. AR slab overlap (2948-3019). Move `ggml_cuda_ar_slab_count`, `ggml_backend_cuda_ar_slab_mark`, `ggml_cuda_mul_mat_ar_slabs` into `ggml-cuda/ar-slab.cu`. It recurses into `ggml_cuda_mul_mat`, so expose it via a declaration in common.cuh. Medium effort.
- M6. Token capture API (6302-6310, 7750-7800). Move to `ggml-cuda/token-graph.cu`. Low effort. `g_cuda_outer_capture` becomes an extern.
- M7. Dedicated copy stream (3842-3960, `pp_copy_stream` fields). Move to `ggml-cuda/pp-copy.cu`. Low-medium effort.
- M8. HIP-only workarounds (set_tensor_2d scratch 3693-3735, HIP graph-update block 4062-4090, HIP-only dedicated drain). Move into a `hip-compat.cu` or keep under `#if HIP` in the same file. Low effort.
- M9. Not worth moving: pool `try_alloc` changes (generic, small), fusion `can_fuse` changes (must stay with the fusion table), `supports_op` guards (must stay with `supports_op`).

---

## 5. Quick wins (high confidence, low risk)

- QW1. common.cuh:1313-1315. Remove `kernel_funcs` member and its two-line comment. Dead (only declaration). Safe. High.
- QW2. common.cuh:1530-1534. Remove `top_k_workspace_rows_cap` and `top_k_workspace_rows_cap_logged` plus their comment. Dead at HEAD (`git grep`). Safe. High.
- QW3. common.cuh:1486-1493 and mmq.cu:352-353. Remove `dst` and `slab_elements` (write-only). If overlap is removed entirely (QW9), drop the whole struct. High.
- QW4. ggml-cuda.cu:3989-3996. Replace the per-node `getenv("GGML_CUDA_TP_OVERLAP")` with nothing (if QW9) or a `static const bool` (if kept). Per-node getenv on every graph compute is wasted work and not HIP-gated. High.
- QW5. ggml-cuda.cu:6008-6011. Delete the dead `SCALE,UNARY,SCALE,DSV4_HC_POST` try_fuse block (can_fuse no longer matches). Med-high. Confirm with one graph trace before merging. Also check `ggml_cuda_op_dsv4_hc_post` still has another caller (dsv4-hc.cu:283 and ggml-cuda.cu:3612 exist, so the function itself stays).
- QW6. ggml-cuda.cu:5111 vs 5138. The comment "try and fuse nodes and return the number of nodes to skip" sits above `ggml_cuda_q8_1_producer_target` instead of above `ggml_cuda_try_fuse`. Move it. Cosmetic. High.
- QW7. Debug knobs, default OFF, no recorded win in vr/. Remove, or at least `GGML_CUDA_REPACK_MOE_FUSION_STATS` (ggml-cuda.cu:13-75 in diff, ~50 LOC including `atexit`) and `GGML_CUDA_OP_SYNC` (6218-6233). Also `GGML_TP_AR_NO_GATE`/`GGML_TP_AR_ONESHOT` (1410-1420) once the diagnostics are no longer needed. Owner decision. High for STATS and OP_SYNC.
- QW8. Remove `GGML_CUDA_TP_OVERLAP` entirely (QW9): ggml-cuda.cu:1708-1714, 1357, 3073-3081, 3989-3996, 829-831 (destructor part), plus mmq.cu:290-360 (outside slice). vr/README.md:55 requires it unset. No vr doc records a win. ~100 LOC. Medium-high. Remove after one check of `vr/old-ignore` (not done here for every doc).
- QW9. ggml-cuda.cu:1708-1714 (HIP enable) is only needed if QW8 is not done. Same change as above.
- QW10. ggml-cuda.cu:7972-7982. Consider default OFF for `GGML_CUDA_EAGER_BLAS_HANDLE` on NVIDIA, or gate to HIP. Medium confidence, needs measurement, so not a quick win on its own.
- QW11. ggml-cuda.cu:3909-3922. `GGML_PP_DRAIN_FORCE_SYNC` escape hatch: keep, but document. Not a removal.
- QW12. Hard-coded 8704/5120/2048 (3073-3081, 3989-3996). Remove with QW8. Otherwise move to a named constant or a model-config check.
- QW13. Whitespace-only/blank-line diff noise: `return use_mul_mat_vec_q;` line pair (diff 986-987, identical text) and added blank lines (diff 1435-1436, 1670-1672, 1720?). Diff hygiene only. High.
- QW14. The `ggml_cuda_graph_update_executable` HIP block (4062+) calls `hipGraphGetNodes` on every update. Not a cleanup but a perf fix: cache the func list when the instance is created. Medium confidence.

---

## 6. Summary of top risks observed

1. NVIDIA exposure of repack buft (2.1): needs a decision.
2. Several default-on shared behaviors without HIP gating: AR slab overlap (2948+), Q8_1 cache, custom AR dispatch, eager BLAS handle, MoE weighted reduction, copy_from_host, cuBLAS chunking, GET_ROWS relaxation. Each needs a conscious owner decision or a gate.
3. Dead branch (6008-6011) and dead members (QW1, QW2): static evidence only.
4. TP-overlap latch (2.5) and per-node getenv (QW4): small but real.
