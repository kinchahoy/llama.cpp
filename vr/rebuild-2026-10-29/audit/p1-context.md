# Pass 1 report: context slice (08-context.md)

Files: src/llama-context.cpp, src/llama-context.h, src/llama-ext.h (mx-rebuild). Numstat vs index: cpp +12/-127, context.h +6/-19, llama-ext.h +3/-20. Not built.

## Items

- Q61 done. Removed `layer_inp_accum_wait(p_end)` and `layer_inp_accum_ready(p_end)` (cpp), their C wrappers, the ctx.h decls, and the ext.h decls with comments. The epoch variants are kept. About -55 lines.
- Q62 done. Grep over the tree (excluding vr/, .git, build) found no caller in common/, tools/, examples/ or tests/. Removed `get_embeddings_layer_inp_accum(lid)` (ctx method, C wrapper, ctx.h decl, ext.h decl). About -10 lines. Note: this removes the public C symbol `llama_get_embeddings_layer_inp_accum`. The `_seq` form remains.
- Q63 done. Removed `LLAMA_REUSE_NO_DRAIN`. The drain before `set_inputs` now always runs when `pipeline_parallel`. About -13 lines.
- Q64 done. Removed the `LLAMA_NO_FUSED_LID` branch. The `resolve(llm_fused_op_lid_probe, ...)` path is unchanged. About -14 lines.
- Q65 done. Removed the `GGML_SCHED_SYNC_NONGRAPH` getenv. Kept `ggml_backend_sched_set_sync_non_graph_inputs(sched, cparams.n_ubatch < 64)`. About -5 lines.
- Q66 done. Shortened comments in the pipeline gating block, MTP and sidecar draft blocks, tap-static block, pre-norm drain block, llama-context.h accum API block, and llama-ext.h accum set/get block. The sidecar comment is one long line (not hard-wrapped).
- Q67 done. Removed 3 fork-added blank lines (after `#include <unordered_map>`, before the causal_attn log, and a double blank in the graph-reuse branch). The upstream double blank before `attach_threadpool` was left in place.
- B22 done. The llama-context.h comment on `ctx_layer_inp_dev` now says "all split modes unless LLAMA_TAP_STATIC=0". Code unchanged.

## Left in place

- `llama_context::layer_inp_accum_ev::pos_end` (llama-context.h:384) is still written at llama-context.cpp:2921 and 2932. Its only reader was the removed p_end wait/ready, so it is now dead. Not in Q61 scope, so kept. Remove in a later pass.
- `ggml_backend_event_query` (ext.h) is still used by `layer_inp_accum_ready_epoch`. Kept.
- `LLAMA_TAP_STATIC`, `LLAMA_MTP_DRAFT_PIPELINE`, `LLAMA_DRAFT_PIPELINE` are still read (not in this pass's list).
- llama-ext.h, llama-context.cpp: ASCII-only check passed.
