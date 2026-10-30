# P1 report: AR and kernels (mx-rebuild)

Files edited (all under ggml/src/ggml-cuda/): allreduce.cu, tp-allreduce.cu, tp-allreduce.cuh, getrows.cu, top-k.cu, softcap.cu, fattn-common.cuh, gated_delta_net_chunk.cu. Nothing built or run. No git state changed.

## Items

| Item | Status | Where | Approx removed |
|---|---|---|---|
| B12 getrows mask | done: `if (n_rows > 0 && (i01 < 0 || i01 >= n_rows))` | getrows.cu ~267 | 0 |
| Peer AR opt-in | done: early return unless GGML_CUDA_PEER_AR non-zero, one GGML_LOG_INFO line | tp-allreduce.cu tp_custom_ar_init ~608-614 | 0 |
| B11 peer enable abort | done: cudaDeviceEnablePeerAccess failure (not AlreadyEnabled) logs WARN and sets peer_access_ok=false; existing cudaGetLastError() clears the error | tp-allreduce.cu ~627-631 | 0 |
| Q38 remove tp_custom_ar_allreduce | done; comments at .cuh:10 and :50 repointed to tp_custom_ar_prepare / tp_custom_ar_launch_rank | tp-allreduce.cu ~1029-1041; tp-allreduce.cuh ~151-165 | ~30 |
| Q39 unused <type_traits> | done (no std::is_*/enable_if/integral_constant in allreduce.cu) | allreduce.cu:12 | 1 |
| Q40 slot_info.token | done: struct removed, acquire_slot returns int; 2 call sites updated | allreduce.cu ~385-398, 683, 861 | ~6 |
| Q44 stale NCCL comment (cuh part) | done; ggml-cuda.cu part NOT done (file not mine) | tp-allreduce.cuh 10-15 | 0 |
| Q45 dead top-k fields | done | top-k.cu:95-96 | 2 |
| Q46 GATE_DEBUG knob | done (gate_dbg helper and call removed) | softcap.cu | ~8 |
| Q47 GETROWS_CHECK block | done | getrows.cu ~548-567 | ~20 |
| Q48 FATTN_MAX_PARALLEL_BLOCKS knob | done; OOM retry loop kept | fattn-common.cuh ~1220-1231 | ~12 |
| Q50 re-indent pdl_sync | done | gated_delta_net_chunk.cu:132 | 0 |

## Left in place (another file uses or owns it)

- common.cuh:1529 `fattn_parallel_blocks_override_logged` is now written by nothing and read by nothing. Owner of common.cuh should delete it (Q48 follow-up).
- ggml-cuda.cu:1393 comment still names tp_custom_ar_allreduce (removed). Stale.
- ggml-cuda.cu:1212 and :1727 comments say custom AR is on by default. It is now off unless GGML_CUDA_PEER_AR is set. Owner of ggml-cuda.cu should update.
- ggml-cuda.cu:1218-1220 (Q44 part) still stale.
- Q41 (ggml-cuda.cu GGML_TP_AR_NO_GATE/ONESHOT wrap) not in my file list, not done.
- Q49 (gated_delta_net.cu banner) not in my file list, not done.

## Behavior change to note

GGML_CUDA_PEER_AR now gates the custom peer AR. Default is off, so the custom path no longer initializes unless the variable is set. Per 05-allreduce.md section 1, the target config should already take the host-staged path, so this matches the expected runtime, but it is a real default change for any box where peer access was reported.
