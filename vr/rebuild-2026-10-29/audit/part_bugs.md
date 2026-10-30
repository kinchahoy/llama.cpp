
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
