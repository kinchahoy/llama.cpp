# P1 CUDA report (Q17-Q24, Q27, Q41, Q44)

Files touched: ggml-cuda/ggml-cuda.cu, common.cuh, mmq.cu (all under /home/raistlin/infer/mx-rebuild/ggml/src/ggml-cuda/). Nothing built or run. No git state changed.
Diff vs index: ggml-cuda.cu +6/-111, common.cuh +0/-18, mmq.cu +0/-65.

## Items
- Q17 DONE. Removed `kernel_funcs` member and its 2-line comment (common.cuh, graph struct). `kernel_funcs_chunks` kept (used at ggml-cuda.cu HIP graph update). ~3 lines.
- Q18 DONE. Removed `top_k_workspace_rows_cap` / `_logged` fields and comment (common.cuh). Only declarations existed. ~5 lines.
- Q19 DONE (via Q24). `tp_overlap` struct (dst, slab_elements, etc.) removed with the whole feature.
- Q20 DONE (via Q24). Per-node `getenv("GGML_CUDA_TP_OVERLAP")` block in graph-compat check removed.
- Q21 DONE. Deleted `ggml_cuda_can_fuse(... SCALE, UNARY, SCALE, DSV4_HC_POST ...)` try_fuse site. Confirmed by reading all branches of `ggml_cuda_can_fuse`: no branch accepts a 4-op list of this shape (the SCALE/UNARY/SCALE branch is 3 ops, the 5-op MUL/VIEW/ADD branch differs), so it always returned false. `ggml_cuda_op_dsv4_hc_post` still used (compute switch, other fusion path). ~4 lines.
- Q22 DONE. "try and fuse nodes..." comment moved from above `ggml_cuda_q8_1_producer_target` to above `ggml_cuda_try_fuse`.
- Q23 DONE. Removed `GGML_CUDA_REPACK_MOE_FUSION_STATS` (histogram, dump, count fns, atexit, env read, call site) and `GGML_CUDA_OP_SYNC` block. ~75 lines. The `ggml_cuda_repack_fusion_glu_ok` helper and the one-time witness log were kept (not part of the knob).
- Q24 DONE. Removed GGML_CUDA_TP_OVERLAP entirely: common.cuh struct `ggml_cuda_tp_overlap_state` and `tp_overlap` member; mmq.cu `#if defined(GGML_USE_HIP)` tp_overlap block (~65 lines); ggml-cuda.cu destructor fork/ready cleanup, `enabled=false` loop in comm destructor, HIP env enable in comm init, `use_tp_overlap_mmq` in ggml_cuda_mul_mat (now plain `ggml_cuda_should_use_mmq` check), and graph-compat getenv block. Default was off, so default path unchanged.
- Q27 PARTIAL. Removed two stray blank lines (before the custom-AR "Both AR kernels" comment and before the final `return false;` of ggml_cuda_can_fuse). The "duplicate `return use_mul_mat_vec_q;`" is NOT present in the working tree (only one line at ggml-cuda.cu, same as HEAD). Nothing to remove there; the diff hunk was a no-op.
- Q41 DONE. `GGML_TP_AR_NO_GATE` / `GGML_TP_AR_ONESHOT` reads and the `s_have_nccl` static removed from the non-NCCL path. The whole gate block is now inside `#ifdef GGML_USE_NCCL`. Non-NCCL builds: both knobs were already no-ops (both conditions required `s_have_nccl`), so behavior unchanged. NCCL builds: same logic (s_have_nccl was true). ~6 lines.
- Q44 DONE (ggml-cuda.cu part). Stale comment above custom AR now reads: "Without a peer-write coherent fabric, NCCL builds defer to the regular AllReduce." tp-allreduce.cuh part NOT touched (not my file).
- Q23/Q24 env table in docs/AGENTS: not touched. vr/README.md:55 still says GGML_CUDA_TP_OVERLAP must stay unset (vr/ not edited).

## Left in place because another file uses them
- None of the removed symbols are used outside my three files (grep over mx-rebuild excluding vr/ and .git). `ggml_cuda_op_dsv4_hc_post` kept (other callers).
- `GGML_TP_AR_NO_GATE` / `GGML_TP_AR_ONESHOT` still mentioned in comments in ggml-cuda.cu and tp-allreduce.cuh:14 (comment in other file, not mine).

## Not done / flagged
- Q27 blank lines: two other double blanks (before `ggml_cuda_can_fuse`, before `// backend device`, etc.) are pre-existing HEAD and left alone.
- Compile not verified (no build per brief). Brace balance checked for all three files.
