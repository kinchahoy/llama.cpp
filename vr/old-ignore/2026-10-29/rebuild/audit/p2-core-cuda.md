# P2 core/CUDA report

Tree: /home/raistlin/infer/mx-rebuild-p2 (no git state changes, no builds).

## ggml/include/ggml-backend.h
- Removed typedefs ggml_backend_comm_sendrecv_tensor_t and ggml_backend_cpy_tensor_async_dedicated_queue_t / _drain_t (~18 lines).
- ggml_backend_meta_device declaration restored to upstream signature (no tps arg); removed tps comment lines.

## ggml/src/ggml-backend-impl.h
- Removed ggml_backend_meta_n_stages and ggml_backend_meta_graph_begin declarations.
- Kept event_query in the device iface (used by the scheduler rebind check, not stage-only).

## ggml/src/ggml-backend.cpp
- ggml_backend_sched_default_n_copies: removed the meta n_stages loop; comment rewritten (no pipeline wording). Kept max(4, n_devices) and GGML_SCHED_N_COPIES.
- ggml_backend_sched_compute_splits: removed the graph_begin loop and its comment (~8 lines).
- Removed "tps4" benchmark note in input staging comment.

## ggml/src/ggml-cuda/ggml-cuda.cu (pass-1 line numbers)
- 808-816: pp_copy_event_a/b and pp_copy_stream destroy in buffer context dtor.
- 1202-1203: custom AR comment now says opt-in via GGML_CUDA_PEER_AR=1.
- 1217-1312: xfer_staging structs, staging_depth, nslots, alloc/pop/free helpers, and the staging loop in comm ctx dtor.
- 1180: "per stage" removed from ar_plan comment.
- 1634-1651: ggml_backend_cuda_comm_set_staging_depth and ggml_backend_cuda_comm_set_pipeline_stages.
- 1721: GGML_CUDA_PEER_AR comment in comm_init.
- 1777-1996: ggml_backend_cuda_comm_sendrecv_tensor (NCCL sendrecv path) plus its forward decl of pp_copy_stream_init.
- 3759-3880: ggml_backend_cuda_pp_copy_stream_init, dedicated_queue, ggml_pp_drain_should_sync, dedicated_drain.
- 7733-7749: proc-address exports for the six removed symbols.

## ggml/src/ggml-cuda/common.cuh
- Removed pp_copy_stream and pp_copy_event_a/b fields plus their comment.
- Removed fattn_parallel_blocks_override_logged (grep: no other users).

## Left in place / for other passes
- ggml-backend-meta.cpp still references the removed symbols and ggml_backend_meta_n_stages / graph_begin (lines ~2604-2841, 4474-4534, 5115-5121). Other pass; build fails until it lands.
- tp-allreduce.cuh:131 prefer_small_twoshot and tp-allreduce.cu:816 remain. Setter is gone, so the field is always false. Remove in the tp-allreduce pass.
- tp-allreduce.cu:611-613 already gates on GGML_CUDA_PEER_AR (not my file).
- Removed comments mention docs/hip-stage-transfer-event-race.md; that doc is outside my files.
- ggml-cuda.cu ~1393 had no auto-selection wording; left unchanged.
- default_n_copies still differs from upstream (parallel ? MAX_COPIES : 1). Brief asked only to drop the meta term, so not reverted.
