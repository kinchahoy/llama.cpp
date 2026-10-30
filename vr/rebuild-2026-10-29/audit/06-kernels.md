# 06 - CUDA/HIP kernel audit (remaining slice)

Scope: ggml/src/ggml-cuda/{gated_delta_net_chunk.cu/.cuh, gated_delta_net.cu, getrows.cu/.cuh, top-k.cu, argsort.cu, argmax.cu, concat.cu, norm.cu/.cuh, softcap.cu/.cuh, topk-moe.cu, fattn-common.cuh, fattn-vec.cuh, flashnext-moe.cuh, q8_repack/flashnext-moe.cuh}.
Method: `git diff 418f905a4 HEAD -- <file>` for each file; `git grep` at HEAD for dispatch and env vars; `git grep 418f905a4` for upstream equivalents; vr/ docs and vr/old-ignore notes for evidence. No builds, no runs, no repo edits.

LOC figures are approximate, taken from `git diff --stat` for the 18 files (1400 insertions, 81 deletions in total for this slice).

## 1. Feature table

| Feature | Files | ~LOC +/- | Class | Upstream equiv | Gating | Recommendation | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F1 GDN persistent-state "chunked" kernel (per-token loop, state kept in registers, keep_rs snapshots) | gated_delta_net_chunk.cu, .cuh, gated_delta_net.cu (dispatch, line ~194-216) | +387 / 0 | C/B | None. Base has only `//TODO: Add chunked kernel` (gated_delta_net.cu:192 @418f905a4) | Generic CUDA+HIP. Only the DPP asm is gfx906-only. Always on for n_tokens >= 2*CS and S_v <= 128. Env `LLAMA_GDN_NO_CHUNK_SNAPSHOTS=1` disables only the keep_rs case. Touches shared GDN path for all backends. | Keep. Consider demoting the WARN banner (gated_delta_net.cu:207) to INFO. | high (dispatch), med (perf) |
| F2 GDN gfx906 launch: VEGA20 cols_per_warp=1 and 2-warp decode; `if (!keep_rs)` guard around MMA path | gated_delta_net.cu (launch_gated_delta_net, ~396) | ~+30 / -15 | B | Partial: base has gdn_cols_per_warp, generic launch and MMA path (gated-delta-net-mma.cuh); no VEGA20 branch | VEGA20 branch is gfx906-only (cc check). The `!keep_rs` guard is generic and changes which kernel keep_rs calls use. | Keep | med |
| F3 Fused ADD -> RMS_NORM -> MUL with Q8_1 producer | norm.cu (+101), norm.cuh (+6); dispatch ggml-cuda.cu:5147-5160 (not in slice) | +107 / 0 | B | Partial: base fuses RMS_NORM+MUL+ADD (norm.cu:696, ggml-cuda.cu:4704) which is a different order. No ADD->RMS->MUL fusion in base. | Generic F32 contiguous path. Env `GGML_CUDA_FUSE_ADD_RMS_NORM` (default 1), `GGML_CUDA_Q8_1_PRODUCER` (default 1). | Keep | high (dispatch), med (perf) |
| F4 dim-0 concat with transposed src1 (tiled transpose) | concat.cu (+69) | +69 / 0 | B | None. Base uses concat_non_cont | Generic. Env `GGML_CUDA_CONCAT_TRANSPOSED` (default 1). Only hit for GDN conv input. | Keep. Has 3 new test-backend-ops cases (per notes). | high |
| F5 Bitonic argsort for ncols_pad > 1024 (strided, 1024 threads) | argsort.cu (+~70 / -6) | +70 / -6 | A | None. Base launches ncols_pad threads (argsort.cu:249 @418f905a4), illegal above 1024 on non-CUB builds (HIP). | Generic code, effective only without GGML_CUDA_USE_CUB (HIP). Always on. | Keep (bug fix) | high |
| F6 Deterministic top-k radix gather (count / offsets / gather kernels, no atomics) | top-k.cu (~+110 / -30) | ~+110 / -30 | A/F | Base: atomic gather (top_k_radix_gather with atomicAdd, @418f905a4) | Generic radix path (nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD and fallbacks). Always on. Adds 2 extra launches per radix iteration. `top_k_radix_offsets` runs one thread per row, serial over blocks_per_row. | Keep determinism fix. Cheap cleanup: remove dead fields (see F6b). | high (dead fields), med (perf) |
| F6b Dead struct fields `greater_count`, `equal_count` in top_k_radix_state | top-k.cu:95-96 | 0 / -2 | E | n/a | n/a. Only declared, never read or written at HEAD (git grep). | Remove | high |
| F7 argmax all-NaN row returns 0 instead of -1 | argmax.cu:12-14 | +2 / -1 | A | None | Generic (CUDA+HIP). Always on. | Keep | high |
| F8 topk-MoE softmax and weight-sum reduction order matched to standalone path | topk-moe.cu (softmax_warp_inplace ~+30 / -10; wt_sum ~+20 / -10; `/=` at :352) | ~+50 / -20 | F | None | Generic fused top-k MoE path (all archs). Always on. | Keep only if fused/unfused bit-equality is a requirement. Adds ~2*experts_per_thread warp reductions per row (e.g. 16 reductions x 5 shuffles for 256 experts). Cost not measured in vr/. | low (need) / med (cost) |
| F9 topk-MoE single row per block on VEGA20 | topk-moe.cu:384 | +1 / -1 | B | Base: fixed TOPK_MOE_ROWS_PER_BLOCK (topk-moe.cuh) | gfx906-only (cc == GGML_CUDA_CC_VEGA20 && n_rows == 1) | Keep. No recorded win found in vr/. | low |
| F10 Fused MUL + ADD(bias) + SIGMOID + SCALE (DeepSeek-V4 hyper-connection gate) | softcap.cu (+~75 incl. debug helper), softcap.cuh (+5); dispatch ggml-cuda.cu:4910 and :6020 (not in slice) | +80 / 0 | C | Partial: base fuses SCALE+UNARY(SIGMOID)+SCALE+DSV4_HC_POST (ggml-cuda.cu:4740 @418f905a4). Different pattern (pre-gate, no HC_POST). | Generic (no arch gate). Always on. Debug knob GGML_CUDA_GATE_DEBUG. | Keep kernel. Remove the debug knob (F10b). | med |
| F10b GGML_CUDA_GATE_DEBUG printf in fusion matcher | softcap.cu:63-65, 89-91 | 0 / -8 | E | n/a | Env-gated; prints on each match during graph build. No vr/ reference. | Remove | high |
| F11 get_rows for IQ4_NL / MXFP4 rows that are not a multiple of QK_K (partial super-blocks) and sharded gather table (row_lo, n_rows) | getrows.cu (+~150 / -10), getrows.cuh (+1/-1); mask flag set in ggml-backend-meta.cpp:1704 (not in slice) | ~+150 / -10 | C | None. Base asserts row alignment (get_rows_cuda_kq). | Generic (all backends). Sharded mask only when meta backend sets op_params[1]=1. | Keep, but see the BUG below before merging anything | med |
| F11b Debug-only index check GGML_CUDA_GETROWS_CHECK | getrows.cu:548-567 | +20 / 0 | E | n/a | Env-gated. Each call copies indices to host and syncs the stream. No vr/ reference. | Remove | high |
| F12 fattn combine: zero output when sumexp == 0 (fully masked row) | fattn-common.cuh:~977-981 (flash_attn_combine_results) | +3 / -1 | A | None | Generic (all backends using split-KV combine). | Keep | high |
| F13 fattn workspace OOM retry (halve parallel_blocks, sticky cap) | fattn-common.cuh:~1240-1260; cap fields common.cuh:1528-1529 | ~+22 / -6 | C | None (base alloc() aborts on OOM) | Generic. Always on, only acts on allocation failure. | Keep | med |
| F13b Env knob GGML_CUDA_FATTN_MAX_PARALLEL_BLOCKS (force lower parallel_blocks) | fattn-common.cuh:1219-1237 | +18 / 0 | E | n/a | Env-gated, default 0 (off). Its comment says it reproduces the automatic OOM path, which F13 now covers. No vr/ reference. | Remove (keep F13) | med |
| F14 fattn-vec: mask row stride from nb31 (not ne11) | fattn-vec.cuh:117, 278-279 | +4 / -1 | A | Base uses ne11 (fattn-vec.cuh @418f905a4) | Generic. Matters for sequence-split fattn (KV slice length differs from mask row length). | Keep (bug fix) | high |
| F15 fattn-vec: skip KQ==-inf and skip V accumulation when all weights are 0 | fattn-vec.cuh:283-296, 374-380 | ~+30 / -10 | B | None | Generic (all archs using vec kernel). Always on. Semantics: 0*Inf/NaN in V no longer propagates for zero-weight rows. | Keep only if measured. The any_weight part appears in the flashnext-fast experiment snapshot (vr/old-ignore/2026-10-25/flashnext-fast/baseline-snapshot.patch:33441) with no isolated number. | low |
| F16 Flash Next grouped MoE (mmvq): expert grouping kernel + grouped IQ2_S / IQ4_NL dot kernels | flashnext-moe.cuh (+190, new); mmvq.cu:1474 include, :1617 dispatch | +200 / 0 | C/B | None | Opt-in GGML_CUDA_FLASHNEXT_MOE=1 (default OFF). HIP only; VEGA20 (gfx906) only; ne03==1, 2-8 tokens, <=10 slots; fusion restrictions. | Keep opt-in. Removal candidate only if the experiment is dropped. | high |
| F17 Flash Next repacked IQ4_NL grouped kernel | q8_repack/flashnext-moe.cuh (+59, new); q8_repack/mul-mat-id.cu:77-79 | +60 / 0 | C/B | None | Same env var (default OFF). Kernel body is gfx906-only; otherwise NO_DEVICE_CODE. | Keep opt-in, same as F16 | high |

## 2. Per-feature notes

**F1 GDN chunk kernel.** The kernel is not a matmul-style chunked algorithm. Each block owns (head, sequence, column group) and loops over tokens in chunks of CS (64 non-KDA, 16 KDA), keeping the S_v x S_v state shard in registers across the whole prefill. Dispatch: gated_delta_net.cu:213 calls `launch_gated_delta_net_chunk` before any other path when `n_tokens >= 2*CS && S_v <= 128`. Upstream has no equivalent. Evidence: the 2026-10-12 TP profile (vr/old-ignore/2026-10-12/tp-profile/pp-kernels.txt) lists `gated_delta_net_chunked_cuda` at 206-207 ms over 48 calls, so it is hot in PP. Only the DPP helpers (gated_delta_net_chunk.cu:4-37) are gfx906-specific; the kernel itself is general. The change applies to NVIDIA as well, and the only validation recorded is gfx906 (MERGE-GFX906-before-cleanup.md:218, 45/45 GDN/KDA cases).

**F2 GDN launch.** Restores a gfx906 launch shape. MERGE-GFX906-before-cleanup.md:209-211 shows PP 482.94 (upstream scalar launch) vs 506.89 (gfx906 launch), with TG 30.43 vs 30.01. The note says it "does not demonstrate recovery of generation throughput". The keep_rs guard means the MMA path is never used for snapshot calls.

**F3 add+rms+mul fusion.** Reported as part of the TG 34.8 -> 36.2 step (vr/old-ignore/2026-10-12/notes/2026-10-12-tp-profile-findings.md:136). Q8_1 producer side is in ggml-cuda.cu:5116 (not in this slice).

**F5 argsort.** Base launches `ncols_pad` threads for the one-element kernel, so anything above 1024 is an illegal launch on HIP. Note argsort.cu:230-237 describes the DSpark indexer crash. The fix is required on non-CUB builds.

**F6 top-k.** The atomic gather produced the right set in launch-dependent order (comment top-k.cu near line 173). The fix is a determinism change, not a speed change. The new offsets kernel is one thread per row, serial over blocks_per_row. For very large ncols this could be a bottleneck; not measured.

**F8 topk-moe.** Changes the reduction order in the fused path only to match the standalone softmax. This is a consistency requirement, and the source comment says it prevents graph-layout-dependent routing. It changes numerics on all archs. The cost is not measured.

**F10 softcap/gate.** Our pattern is MUL -> VIEW -> ADD -> UNARY(SIGMOID) -> SCALE. The VIEW node is skipped by the kernel, so the matcher must keep it as a pure view. Base has a related but different fusion (ggml-cuda.cu:4740 @418f905a4). The "bit for bit" claim relies on matching `1/(1+expf(-x))` from the unary sigmoid kernel.

**F11 getrows. BUG candidate (needs a test; not run per brief).** In getrows.cu:266-267 the quantized kq_n kernel always masks rows with `i01 < 0 || i01 >= n_rows`. For unsharded gathers `ggml_cuda_op_get_rows` passes `shard_n_rows = 0` (getrows.cu:581, op_params[1] == 0), and nothing disables the mask when n_rows == 0. The comment at getrows.cu:579 says "n_rows == 0 disables the mask", but the kernel does not implement that. If so, every unsharded IQ4_NL or MXFP4 gather with ne00 % 256 != 0 writes zeros. Please confirm with a single gather test before relying on this path. The F32/F16 paths ignore row_lo/n_rows entirely, so the sharding guarantee depends on the meta backend only using the flag for kq_n types.

**F16/F17 flashnext.** Default OFF. vr/IMPROVEMENTS.md:28 and vr/PERFORMANCE.md:29 record about 5.9-6% mean TG with grouped-only enabled, identical text and acceptance on one prompt, and PP varying "substantially". vr/PERFORMANCE.md:85 says the path requires 2-8 tokens and does not help one-token decode. The grouped path is HIP-only and VEGA20-only at dispatch (mmvq.cu:1617 and flashnext-moe.cuh:165-168). The q8_repack kernel is gfx906-only inside the kernel body.

## 3. Quick wins (safe, no behavior change at default settings)

1. Dead fields: ggml/src/ggml-cuda/top-k.cu:95-96 (`greater_count`, `equal_count`). Never referenced at HEAD. Confidence high.
2. Debug knob GGML_CUDA_GATE_DEBUG: ggml/src/ggml-cuda/softcap.cu:63-65 (function `gate_dbg`), and the call at softcap.cu:89-91. Remove the helper and the `if`. Confidence high.
3. Debug check GGML_CUDA_GETROWS_CHECK: ggml/src/ggml-cuda/getrows.cu:548-567 (whole block). Forces a host sync per get_rows when set. Confidence high.
4. Knob GGML_CUDA_FATTN_MAX_PARALLEL_BLOCKS: ggml/src/ggml-cuda/fattn-common.cuh:1219-1237. Default 0 (off). The OOM retry (F13, lines ~1240-1260) stays. Confidence medium (no vr/ reference; check with the owner).
5. WARN banner for the GDN snapshot path: ggml/src/ggml-cuda/gated_delta_net.cu:203-211. Demote to GGML_LOG_INFO or remove. Keep the LLAMA_GDN_NO_CHUNK_SNAPSHOTS knob (used in vr/old-ignore/2026-10-12 logs). Confidence medium.
6. Cosmetic: ggml/src/ggml-cuda/gated_delta_net_chunk.cu has an 8-space indented `ggml_cuda_pdl_sync();` inside the kernel body (around the state-load loop). Re-indent. Confidence high.

Do NOT remove without a decision (correctness or consistency): argsort large path (F5), top-k deterministic gather (F6), fattn-vec s31 (F14), fattn combine NaN guard (F12), argmax NaN guard (F7), topk-moe reduction order (F8, needs owner decision because it costs warp reductions).

## 4. Items to verify (not changed)

- F11 unsharded kq_n masking (see note above). Highest-impact finding in this slice if confirmed.
- F15 any_weight skip: semantic change for non-finite V and no isolated measurement.
- F8 cost: 2*experts_per_thread extra warp reductions per token on the MoE routing path.
