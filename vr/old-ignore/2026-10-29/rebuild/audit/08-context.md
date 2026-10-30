# 08 - llama runtime / context slice

Scope: src/llama-context.cpp, src/llama-context.h, src/llama.cpp, src/llama-ext.h (exists in base at 191 lines; fork adds 54), include/llama.h, src/llama-cparams.h, src/llama-memory.h.
Diff basis: `git diff 418f905a4 HEAD`. Saved diffs: `scratchpad/audit/ctx.diff`, `ctxh.diff`, `llama.diff`.

Diffstat (this slice): include/llama.h +3; llama-cparams.h +1; llama-ext.h +54; llama-memory.h +8; llama-context.cpp ~+815/-18; llama-context.h ~+95/-1; llama.cpp ~+97/-? (llama.diff 140 lines).

Fork-only symbols outside the slice that this slice depends on: `ggml_backend_event_query` (ggml-backend.cpp:964, not in base), `ggml_backend_sched_set_sync_non_graph_inputs` (ggml-backend.cpp:3042, not in base), `llama_memory_i::get_can_checkpoint` and `llama_memory_context_i::finish_compute` (absent at base 418f905a4 memory.h), `llama_model::path` (llama-model.h:750, not in base).

---

## 1. Public / ext API surface

### 1a. include/llama.h (public ABI)

| Change | Class | Notes |
|---|---|---|
| `int32_t tensor_parallel_size;` inserted in `struct llama_model_params` between `load_mode` and `lazy_mode` (include/llama.h:332-334 at HEAD) | C (functional) + **ABI BREAK** | Base struct is `devices, tensor_buft_overrides, n_gpu_layers, split_mode, load_mode, lazy_mode, main_gpu, tensor_split, progress_callback, progress_callback_user_data, kv_overrides, bools...`. Inserting mid-struct shifts `lazy_mode`, `main_gpu`, `tensor_split`, callbacks, `kv_overrides` and the bool block, and changes `sizeof(llama_model_params)`. The struct is passed by value to `llama_model_load_*`, so any binary built against upstream headers reads garbage. Inside this repo everything is rebuilt, so it is safe here. Downstream and upstream-ABI consumers are broken. |
| Default set in src/llama-model.cpp:3975 (`/*.tensor_parallel_size =*/ 0`) | - | Positional/designated order must match the new field. OK at HEAD. |

Recommendation: keep the feature, but it cannot be added to `llama_model_params` without an ABI break. Options: (a) accept the break and document it in the fork (the fork is not ABI-compatible with upstream anyway, but any out-of-tree consumer must rebuild), (b) move the value to `llama-ext.h` as a setter on the model params path (e.g. a side struct or `llama_model_params_ext`) so the public struct stays byte-identical to upstream. Option (b) is the low-risk route if upstream compatibility matters.

### 1b. src/llama-ext.h (WIP staging header, internal to the repo, but `LLAMA_API`-exported)

The header states breaking changes are allowed. New functions, with HEAD callers (`git grep HEAD`, excluding vr/ patches and the definitions themselves):

| Function | Declared | Callers at HEAD | Status |
|---|---|---|---|
| `llama_sched_reserve` | ext.h:181 | common/common.cpp:1431, common/speculative.cpp:647-648, 1262-1263, 2083-2084, tools/server/server-context.cpp:4677 | USED. Pre-reserves the scheduler so a topology-changing setter does not stall the next decode. |
| `llama_set_mtp_prefill_kv_only` | ext.h:139 | common/speculative.cpp:2327, 2371 | USED. Gates `src/models/qwen35.cpp:586` (cparams flag). |
| `llama_set_embeddings_pre_norm_accum` | ext.h:146 | common/speculative.cpp:2131, 2375 | USED. |
| `llama_get_embeddings_pre_norm_accum` | ext.h:147 | common/speculative.cpp:2318 | USED. |
| `llama_set_embeddings_layer_inp_accum` | ext.h:194 | common/speculative.cpp:1275 | USED. |
| `llama_get_embeddings_layer_inp_accum_seq` | ext.h:199 | common/speculative.cpp:1114 | USED. |
| `llama_get_embeddings_layer_inp_accum` (lid form, seq 0) | ext.h:198 | **none** | DEAD at HEAD (E, med). |
| `llama_layer_inp_accum_span_epoch` | ext.h:201 | common/speculative.cpp:1567 | USED. |
| `llama_layer_inp_accum_wait_epoch` | ext.h:203 | common/speculative.cpp:1386 | USED. |
| `llama_layer_inp_accum_ready_epoch` | ext.h:204 | common/speculative.cpp:1614 | USED. |
| `llama_layer_inp_accum_wait(ctx, p_end)` | ext.h:209 | **none** | DEAD at HEAD (E, high). Superseded by the epoch variant. |
| `llama_layer_inp_accum_ready(ctx, p_end)` | ext.h:213 | **none** | DEAD at HEAD (E, high). Same. |
| `ggml_backend_event_query` (GGML_API decl in ext.h:219) | ext.h:219 | used inside ggml-backend.cpp:2434/2444 and llama-context.cpp | USED. Fork-added. Declared in a llama header to avoid touching the public ggml header. Keep. |

The vr/old-ignore patch series (0003/0004) shows the wait(p_end)/ready(p_end) pair was used by an earlier speculative.cpp and was replaced by the epoch API. They are leftovers.

### 1c. llama-memory.h

- `llama_memory_context_i::finish_compute(bool)` virtual, default no-op (memory.h:98-102). Overridden in recurrent, hybrid, hybrid-iswa contexts. Called from `process_ubatch` (llama-context.cpp:1985, 1993). Class A/C (rollback staging for the recurrent snapshot ring). Needed.
- `llama_memory_i::get_can_checkpoint()` virtual default true (memory.h:141). Overridden by `llama_kv_cache_dsv4` (kv-cache-dsv4.h:124). Used by tools/server/server-context.cpp:75. Class C.

---

## 2. Logical changes in llama-context.cpp / .h / llama.cpp

Columns: ~LOC are approximate added (+) / removed (-) for the logical change in this slice.

### F1. MTP deferred prefill: pre-norm accum buffer
- Files/functions: llama-context.cpp `set_embeddings_pre_norm_accum` (~1517-1545 at HEAD), `get_embeddings_pre_norm_accum`, decode() nextn extraction block (~2561-2610). Header: `embd_pre_norm_accum*`, `buf_pre_norm_accum`.
- ~+110 / -10.
- Purpose: unmasked pre-norm extraction is also written into a position-indexed pinned host buffer, so the whole prompt hidden survives across decode calls with no per-chunk synchronize. Ring-wrap drain every `n_copies` ubatches.
- Gating: always available when the API is armed. Armed by speculative.cpp (`LLAMA_MTP_DEFERRED_PREFILL`, default on, speculative.cpp:2116-2121). Not gfx906-specific.
- Upstream equivalent: none.
- Class: **B** (perf, prefill). Keep.
- Evidence: vr/old-ignore/2026-10-12/notes/2026-10-12-tp-profile-findings.md:214 ("MTP adds about 2-3% to CLI PP") and :214 for the lazy replay cost. Recorded as a PP improvement, not a correctness fix.

### F2. Layer-input tap accumulation (DFlash / DSpark deferred prompt encode)
- Files/functions: `set_embeddings_layer_inp_accum`, `get_embeddings_layer_inp_accum_seq`, `layer_inp_accum_span_epoch`, `layer_inp_accum_{wait,ready}_epoch`, `layer_inp_accum_events_free`, and `extract_layer_inputs` (rewritten, signature now takes `llama_ubatch`). llama-context.cpp ~2860-3000 and ~2950-3010 at HEAD.
- ~+300 / -15 (the bulk of the slice).
- Purpose: single-seq, sequential-position ubatches are async-copied into a per-(lid, seq) region at position offset. Per-ubatch backend events give readiness. Epochs per row let the consumer check a span without a full sync. Lets the drafter defer its prompt gather/encode and keep target prefill overlap.
- Gating: env `LLAMA_TAP_ACCUM_MIN` (default 1), `LLAMA_TAP_EMBD_SKIP` (default on), `LLAMA_TAP_ACCUM_MROPE` (default on). Accum is armed only by the DFlash/DSpark path. Generic across backends.
- Upstream equivalent: none (upstream has layer-input taps but no accum).
- Class: **B** (perf) + **C** (feature: the defer/epoch protocol). Keep.
- Dead parts: `layer_inp_accum_wait(p_end)` and `layer_inp_accum_ready(p_end)` (+ ctx methods, llama-context.cpp ~380-416; wrappers ~5109-5117; decls ext.h:206-213, context.h ~150-156). ~50 LOC. Class **E**, high.
- Evidence: vr/old-ignore/2026-10-12/2026-10-12-three-slots.md:14 (defaults on); vr/old-ignore/2026-10-29/prior-guides/QUICKSTART.md:107-110 (documents MROPE, ACCUM_MIN, DEFER_TAIL, EMBD_SKIP as knobs with their off values); tp-profile-findings.md:241-244 (per-knob measured effect).

### F3. Persistent tap tensors outside the galloc arena
- Files/functions: `sched_reserve` (~860-915 at HEAD, the `layer_inp_dev` block); `graph_params` (`/*.layer_inp_dev*/`); `graph_compute` (tap_readback_waits consumed, ~3511-3520); header `ctx_layer_inp_dev`, `buf_layer_inp_dev`, `layer_inp_dev`, `tap_readback_waits`.
- ~+90 / -0.
- Purpose: keep tap copies out of the scheduler compute arena, so runtime graphs keep fitting the reserved plan. Measured in comment: an unexpected same-size graph realloc on every prefill chunk under `-sm layer`, 2.4x PP cost. Persistent tap uses a single-ubatch dependency (event wait on the next graph_compute) instead of a full drain.
- Gating: env `LLAMA_TAP_STATIC` (default on; `=0` restores in-arena). **Applies to all split modes**, though the header comment at context.h says "(non-tensor split modes)". Comment/code mismatch.
- Upstream equivalent: none.
- Class: **B** (perf; avoids a realloc regression). Keep. Fix the header comment.
- Evidence: 2026-10-12-three-slots.md:14 calls `LLAMA_TAP_STATIC=0` workaround unnecessary for the tested cases. This is a candidate for removing the knob (see quick wins).

### F4. Pipeline ring gating (pipeline_parallel) and TP enablement
- Files/functions: `llama_context::llama_context` (~553-590 at HEAD).
- ~+30 / -6.
- Purpose and changes:
  - `pipeline_parallel` now true for `-sm tensor` (split_mode TENSOR) when `n_gpu_layers >= n_layer_all` (equality allowed; was `>`), and for `-sm layer` only with `n_devices() > 1` (as before). `n_devices() > 1` gate dropped for TENSOR because the model exposes one Meta device.
  - MTP draft context (`LLAMA_CONTEXT_TYPE_MTP`): ring forced off unless env `LLAMA_MTP_DRAFT_PIPELINE` is set (presence check, any value enables).
  - Draft sidecar (any context with `ctx_other != nullptr`, i.e. DFlash/DSpark): ring forced off unless `LLAMA_DRAFT_PIPELINE` set (presence check).
- **Default-on behavior change for every `-sm tensor` fully-offloaded model** (not env gated). Memory use goes up because `n_copies > 1` (ring). Generic, not gfx906-only.
- Upstream equivalent: base had the `n_devices > 1` LAYER-only gate. Ours is a superset for TENSOR.
- Class: **C** (TP enablement) + **B** (draft ring off, perf, measured in comments: +6% TG MTP draft; 23.0 -> 25.5 t/s DSpark sidecar on 8x MI50 at 16k).
- Confidence in the measurement claims: not verified against vr records (only FEATURES.md:100 mentions "disabling the draft context's pipeline ring", no numbers).

### F5. Scheduler policy: sync non-graph inputs
- File: llama-context.cpp `sched_new` (~804-815 at HEAD). Uses fork-only `ggml_backend_sched_set_sync_non_graph_inputs` (ggml-backend.cpp:3042, ggml-backend-sched-impl.h:13).
- ~+12 / 0.
- Purpose: when `n_ubatch < 64`, sync non-graph inputs (ROCm race on tiny-ubatch multi-GPU pipelining). Override env `GGML_SCHED_SYNC_NONGRAPH` (value; default = `n_ubatch < 64`).
- Gating: **generic for all backends**; not `#ifdef GGML_USE_HIP`. Comment says it is ROCm-specific.
- Class: **A** (correctness on ROCm). Policy keep. The env override is **E** (diagnostic, no vr record) med.
- Recommendation: move the rule behind `GGML_USE_HIP` or a backend-type check if CUDA/SYCL pay the sync cost; otherwise keep unconditional.

### F6. Graph reuse / sequence-layout safety
- Files: `process_ubatch` (~1905-1995 at HEAD); header `graph_seq_ids`.
- ~+35 / -2.
- Changes:
  - Sync the scheduler when the sequence-id layout changes and pipeline_parallel is on (server slot join/leave). Class **A** (correctness).
  - `LLAMA_REUSE_NO_DRAIN` skips the drain before `set_inputs` on graph reuse. Default off, experimental, documented only in the code comment ("Experimental gate to measure cross-chunk prefill overlap"). No vr record. Class **E** (remove) or **F** (unclear). Risk: skipping the drain is a correctness hazard if staging assumptions break.
  - `finish_compute(success)` calls around `graph_compute`. Class **A/C** (paired with F-memory staging).
- Upstream equivalent: none for seq-layout sync.

### F7. Fused op sibling-device acceptance (resolve_fused_ops)
- File: llama-context.cpp `resolve_fused_ops` (~691-705 at HEAD).
- ~+12 / 0.
- Purpose: a fused node the scheduler placed on a sibling device of the same type that supports the op is accepted, not treated as a missing-support fallback. Case: DeepSeek-V4.1 carried hyper-connection mix under `-sm layer`.
- Gating: generic (device type check), not env gated.
- Class: **A** (fixes a false-positive warning and an unnecessary fallback). Keep.

### F8. Fused Lightning Indexer A/B knob
- File: llama-context.cpp ~739-752.
- ~+11 / 0.
- Env: `LLAMA_NO_FUSED_LID` (default off, atoi != 0).
- Purpose: diagnostic A/B for mask distribution under multi-stage TP. Only recorded in the baseline-bin of old-ignore; no notes.
- Class: **E** (diagnostic leftover). Remove after the investigation closes. Med confidence.

### F9. Recurrent rollback clamps (n_rs_seq)
- File: llama-context.cpp ~203-212 (qwen4exp multi-slot clamp) and ~362-372 (`n_ubatch >= n_rs_seq + 2` clamp).
- ~+20 / 0.
- Purpose: (a) qwen4exp with `n_seq_max > 1` clamps `n_rs_seq` to 0, because the snapshot ring has no cross-sequence plane-copy planner. (b) `n_rs_seq > 0` with `n_ubatch < n_rs_seq + 2` clamps to 0 (split_equal must keep the trailing tokens in one ubatch). Both fall back to full-state checkpoints.
- Gating: model-arch specific for (a); generic for (b) on any rollback-capable model.
- Class: **A** (correctness guards; both turn an abort or silent corruption into a logged fallback). Keep.
- Evidence: vr/IMPROVEMENTS.md:22 ("prompt state must remain per sequence") supports the per-sequence guard.

### F10. Private DFlash head (load_private_head)
- Files: llama-context.cpp `load_private_head` (~3200-3285 at HEAD), constructor call (~265-268); header members (`ctx_private_head`, `buf_private_head`, `ctx_private_out`, `buf_private_out`); llama-model.cpp (not this slice) / llama.cpp:524-526 (`model->path = path_model`); llama-model.h:750 (`path`).
- ~+115 / 0 in this slice; +5 in llama.cpp, +4 in model header.
- Purpose: a draft on devices that cannot read the target's buffers (single-GPU DFlash draft for a TP target) loads private copies of `token_embd` and `output` from the target GGUF.
- Gating: `LLAMA_DFLASH_PRIVATE_HEAD` (default on). Only runs for `LLM_ARCH_DFLASH` with `ctx_other` and when the device cannot read the target buffer type.
- Upstream equivalent: none.
- Class: **A/C** (needed for TP target + single-GPU drafter). Keep. Note `const_cast<llama_model&>` in the context: a draft-only mutation of a model owned by the draft context. Acceptable but flagged.
- Evidence: vr/old-ignore/2026-10-12/notes/2026-10-12-roofline.md:139; QUICKSTART.md:103.

### F11. MTP KV-only prefill replay flag
- Files: llama-cparams.h (+1 `mtp_prefill_kv_only`), llama-context.h (`set_mtp_prefill_kv_only` inline), llama-ext.h:139, llama-context.cpp wrapper; consumed in src/models/qwen35.cpp:586.
- ~+5 here.
- Purpose: during deferred replay, the MTP head builds and stores only K/V (skips attn/FFN/output). Enabled in speculative.cpp:2327, cleared at 2371.
- Gating: speculative.cpp `LLAMA_MTP_NO_KV_ONLY` escape hatch (default on).
- Class: **B** (perf). Keep. Good candidate for cleanup of the escape hatch after validation.

### F12. Tensor-parallel draft width selection (DSV4 DFlash) and tensor_parallel_size plumbing
- File: src/llama.cpp `llama_prepare_model_devices` (diff 156-290) and `llama_model_load` (~393-410). Also common/common.cpp:1737, common/speculative.cpp:3360.
- ~+85 / -6.
- Purpose: `tensor_parallel_size` (`-tps`) sets the TP group width; `n_stages = n_devs / tps`. `resolve_tps` validates divisibility. `draft_subset_devs` picks a TP width for a DSV4-backbone DFlash drafter from `LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT` when `-tpsd` is 0 (auto), so the serial draft stays hop-free. The auto path **runs by default** when `-sm tensor` and a DSV4 DFlash drafter is loaded.
- Gating: `-sm tensor` only; DFlash + hyper-connection count > 0. Not env gated.
- Upstream equivalent: base had `llama_prepare_model_devices` without tps; ours extends it. Superset.
- Class: **C** (functional). Default behaviour change for the DSV4 DFlash drafter under TP (draft may use fewer devices than the target). Flag: the divisor is reused from the attention output group count, which is semantically unrelated to TP width. Class **F** until the rule is documented in the code.
- Evidence: vr/old-ignore/2026-10-29/prior-guides/FEATURES.md:64-90 (multi-stage TP and `-tps`).

### F13. Persistent-tap readback waits and `graph_seq_ids` etc. (already in F3, F6)

---

## 3. Classification summary

| Feature | Files | ~LOC +/- | Class | Upstream equiv | Gating | Recommendation | Confidence |
|---|---|---|---|---|---|---|---|
| F1 MTP pre-norm accum | ctx.cpp, ctx.h, ext.h | +110/-10 | B | none | env via speculative (default on) | keep | high |
| F2 Layer-input tap accum + epoch protocol | ctx.cpp, ctx.h, ext.h | +300/-15 | B+C | none | env LLAMA_TAP_* (defaults on) | keep; drop dead p_end wait/ready (~50) | high |
| F3 Persistent tap tensors | ctx.cpp, ctx.h | +90/0 | B | none | env LLAMA_TAP_STATIC (default on, all modes) | keep; fix header comment; consider dropping knob | high |
| F4 Pipeline ring gating + TP enable | ctx.cpp | +30/-6 | C+B | base partial | not env gated (default-on for -sm tensor full offload); env for MTP/sidecar draft | keep; document the default change | med |
| F5 Sync non-graph inputs | ctx.cpp | +12/0 | A | none | generic, n_ubatch<64; env override | keep, gate to HIP if CUDA cost matters; drop env | med |
| F6 Graph reuse seq-layout sync + finish_compute | ctx.cpp, ctx.h | +35/-2 | A (+E for NO_DRAIN) | none | generic; LLAMA_REUSE_NO_DRAIN default off | keep A; remove NO_DRAIN | med |
| F7 Fused-op sibling acceptance | ctx.cpp | +12/0 | A | none | generic | keep | high |
| F8 LLAMA_NO_FUSED_LID knob | ctx.cpp | +11/0 | E | none | env default off | remove after investigation | med |
| F9 n_rs_seq clamps | ctx.cpp | +20/0 | A | none | arch/model | keep | high |
| F10 Private DFlash head | ctx.cpp, ctx.h, llama.cpp, llama-model.h | +115/0 (+9 elsewhere) | A/C | none | env LLAMA_DFLASH_PRIVATE_HEAD (default on) | keep | high |
| F11 MTP KV-only replay flag | cparams, ctx.h, ext.h, ctx.cpp | +5 | B | none | speculative env (default on) | keep | high |
| F12 TP draft width + tensor_parallel_size | llama.cpp, llama.h | +85/-6 (+3 API) | C (F for width rule) | base partial | -sm tensor, DSV4 DFlash | keep; **ABI break in llama_model_params** | med |
| D1 Dead p_end wait/ready | ctx.cpp, ctx.h, ext.h | ~-50 | E | n/a | none | remove | high |
| D2 Dead layer_inp_accum lid getter | ctx.cpp, ext.h | ~-6 | E | n/a | none | remove (or keep as documented alias) | med |
| D3 LLAMA_REUSE_NO_DRAIN | ctx.cpp | ~-10 | E | none | env default off | remove | med-high |
| D4 GGML_SCHED_SYNC_NONGRAPH env | ctx.cpp | ~-4 | E | none | env | remove env, keep rule | med |
| D5 LLAMA_NO_FUSED_LID | ctx.cpp | ~-11 | E | none | env default off | remove | med |

---

## 4. Env vars read in this slice (getenv)

Base had 1 getenv in llama-context.cpp (`LLAMA_GRAPH_REUSE_DISABLE`, line 404 at HEAD, unchanged). HEAD now has 13 call sites (12 new getenv in this slice, plus the pre-existing one).

| Env var | Read at (HEAD) | Default | Semantics | Recorded win / use | Recommendation |
|---|---|---|---|---|---|
| `LLAMA_GRAPH_REUSE_DISABLE` | ctx.cpp:404 | off | pre-existing (base) | - | keep |
| `LLAMA_MTP_DRAFT_PIPELINE` | ctx.cpp:572 | unset (ring off) | presence enables ring on MTP draft | code comment: +6% TG | keep as debug or remove |
| `LLAMA_DRAFT_PIPELINE` | ctx.cpp:584 | unset (ring off) | presence enables ring on sidecar draft | code comment: 23.0 -> 25.5 t/s | keep as debug or remove |
| `LLAMA_NO_FUSED_LID` | ctx.cpp:744-745 | off | atoi!=0 disables fused LID | none | remove (E) |
| `GGML_SCHED_SYNC_NONGRAPH` | ctx.cpp:810 | n_ubatch<64 | value overrides sync policy | none | remove env (E) |
| `LLAMA_TAP_STATIC` | ctx.cpp:876 | on | 0 restores in-arena taps | 2026-10-12-three-slots.md:14 says unneeded | consider remove (B-cleanup) |
| `LLAMA_REUSE_NO_DRAIN` | ctx.cpp:1923 | off | atoi!=0 skips graph-reuse drain | none | remove (E) |
| `LLAMA_TAP_ACCUM_MIN` | ctx.cpp:2872 | 1 | min ubatch size for accum | QUICKSTART.md:108 | keep |
| `LLAMA_TAP_EMBD_SKIP` | ctx.cpp:2878 | on | skip 2nd host copy | QUICKSTART.md:110; tp-profile-findings.md:244 | keep (see hazard H1) |
| `LLAMA_TAP_ACCUM_MROPE` | ctx.cpp:2892 | on | accept M-RoPE text rows | tp-profile-findings.md:241 | keep |
| `LLAMA_DFLASH_PRIVATE_HEAD` | ctx.cpp:3206 | on | load private head copies | QUICKSTART.md:103 | keep |

Note: `LLAMA_MTP_DRAFT_PIPELINE` and `LLAMA_DRAFT_PIPELINE` are presence checks. `LLAMA_DRAFT_PIPELINE=0` still enables the ring. Use `atoi`-style parsing like the others, or document it.

Related env in common/speculative.cpp (not this slice, noted for the doc drift in section 6): `LLAMA_MTP_DEFERRED_PREFILL` (default on), `LLAMA_MTP_NO_KV_ONLY` (escape hatch, default on), `LLAMA_MTP_ADAPTIVE_DISABLE`.

---

## 5. Correctness hazards to confirm (not cleanups)

- **H1 - host embd_layer_inp not filled on accum chunks.** `extract_layer_inputs` skips the second D2H (`embd_skip && layer_accum_ok && accum_runs.size()==1 && n_tokens>=16`). Rows exist only in the accum buffer. Host readers `llama_get_embeddings_layer_inp` (speculative.cpp:725 EAGLE3, 1672 DFlash gather) would see stale or zero data for those rows if the accum path is armed on the same context. Confirm the armed paths never call the host reader, or add an assert.
- **H2 - `extract_all_idxs` not forced in the accum branch.** In the normal nextn branch, `extract_all_idxs = extract_all_idxs || !masked` (ctx.cpp, HEAD ~2615). The accum branch (ctx.cpp ~2571-2611) only runs when `!masked`, so this line is skipped and `embd_batch_idxs` is not extended for those ubatches. decode() asserts `embd_batch_idxs.size() == n_tokens_prev` at ~2622 when `extract_all_idxs` is true later. This is safe only if tap extraction and nextn extraction never mix within one context. Verify, or keep the assignment on both branches.
- **H3 - GGML_ASSERT removed.** HEAD `extract_layer_inputs` lost `GGML_ASSERT(row_floats == model.hparams.n_embd)` (ctx.cpp ~2946-2947 at HEAD). The check is now only inside `layer_accum_ok`. The host copy path (`dst_offset = token_offset * row_floats`) still relies on the row size. Restore the assert for the non-accum path.
- **H4 - ABI break** in `llama_model_params` (section 1a).
- **H5 - Header comment mismatch.** context.h "Persistent device tensors for the layer-input taps (non-tensor split modes)" vs code that applies to every split mode unless `LLAMA_TAP_STATIC=0`.
- **H6 - Default-on TP pipeline change.** `pipeline_parallel` is now true for `-sm tensor` with full offload (F4). That increases compute-buffer memory (n_copies > 1) by default. Confirm the memory budget on 8x MI50 before merge.
- **H7 - Graph reuse drain** (LLAMA_REUSE_NO_DRAIN) removes a synchronize that guards input overwrite. Default off, but if enabled it is unsafe per its own comment's assumptions.

---

## 6. Doc drift

- vr/old-ignore/2026-10-29/prior-guides/FEATURES.md:96-100 says the MTP optimizations are opt-in via `LLAMA_ENABLE_MTP_OPT=1`. No such env var exists in HEAD code (`git grep` finds only vr/ hits). The MTP deferred prefill, KV-only replay and draft-ring-off are **on by default** in code (speculative.cpp:2116-2121, 2320-2330; context ctx.cpp:572). Update the doc or the code.
- context.h header comment "(non-tensor split modes)" vs code (H5).

---

## 7. Quick wins (safe, exact locations at HEAD)

Note: line numbers marked "~" were read from diff hunks and may drift by a few lines. Re-check with `git grep -n` before editing.

1. **Dead p_end wait/ready.** Remove `llama_context::layer_inp_accum_wait` and `layer_inp_accum_ready` (src/llama-context.cpp:380-416 at HEAD, the two non-epoch functions), their wrappers `llama_layer_inp_accum_wait` / `llama_layer_inp_accum_ready` (src/llama-context.cpp ~5109-5117), the decls in src/llama-context.h (~151-156 region, `layer_inp_accum_wait(` / `layer_inp_accum_ready(`), and src/llama-ext.h:206-213. Why safe: zero callers at HEAD outside the definitions (`git grep HEAD`). Epoch variants are used. Confidence high.
2. **Dead lid getter.** `llama_context::get_embeddings_layer_inp_accum` (src/llama-context.cpp:332-334), wrapper at ~5087-5089, decl src/llama-ext.h:198. Zero callers. Keep only if out-of-tree users are expected (the header says breaking changes are allowed). Confidence med.
3. **LLAMA_REUSE_NO_DRAIN.** Remove the `reuse_no_drain` static and the `&& !reuse_no_drain` condition at src/llama-context.cpp:1914-1924 (comment block 1916-1923 included). Restores the unconditional drain. Default was off. No recorded win. Confidence med-high.
4. **LLAMA_NO_FUSED_LID.** Remove the `no_fused_lid` branch at src/llama-context.cpp:739-752, keep the `resolve(...)` call. Default off, diagnostic only. Confidence med.
5. **GGML_SCHED_SYNC_NONGRAPH env.** Remove lines src/llama-context.cpp:808-812 (the `if (const char * value = getenv(...))` block), keep `sync_non_graph_inputs = cparams.n_ubatch < 64`. Confidence med.
6. **Comment compression (AGENTS.md: 1-2 lines, no anecdotes).** Long blocks to shorten:
   - src/llama-context.cpp ~553-557 (the 4-line `-sm tensor` / `-tps` / equality rationale; keep the 1-line invariant).
   - ~569-574 (MTP draft pipeline), ~581-587 (sidecar draft pipeline; drop the 23.0 -> 25.5 numbers).
   - ~869-878 (tap static rationale).
   - ~1914-1923 (reuse-no-drain; removed by item 3).
   - ~2599-2609 (pre-norm ring-drain rationale).
   - ~2692-2694 and ~2709-2712 not in the slice; skip.
   - src/llama-context.h ~126-154 (accum API header comments, many repeat the .cpp).
   - Confidence high (comments only).
7. **Blank-line noise** added by the fork: src/llama-context.cpp around line 22 (double blank after `#include <unordered_map>`), around line 64 (blank before `causal_attn` log), around line 466-467 (double blank inside the reuse branch, HEAD ~1930), and around line 1904 in `process_ubatch` (blank after `if` graph reuse check). Cosmetic. Confidence high.
8. **Duplicated pinned-buffer selection.** `set_embeddings_pre_norm_accum` (src/llama-context.cpp ~233-241 of the diff, HEAD ~1520-1530) and `set_embeddings_layer_inp_accum` (~300-307 of diff) duplicate the `ggml_backend_dev_host_buffer_type` fallback. Factor into one static helper. Low risk, B-cleanup. Confidence med.
9. **Env presence checks.** Convert `LLAMA_MTP_DRAFT_PIPELINE` and `LLAMA_DRAFT_PIPELINE` to the same `atoi` parse as the others, or document presence semantics (src/llama-context.cpp:572, 584). Confidence high.

Items not recommended for removal in this slice: F2, F3 (keep, but consider dropping `LLAMA_TAP_STATIC` after one validation run), F7, F9, F10, F11.

---

## 8. Upstream ABI summary

- include/llama.h: **1 field inserted mid-struct** (`llama_model_params.tensor_parallel_size`). Breaks binary compat with upstream and with any out-of-tree consumer built against the upstream header. Everything else in include/llama.h is unchanged.
- llama-ext.h, llama-memory.h, llama-cparams.h: internal headers, no ABI promise. Changes to `llama_memory_context_i` (virtual added, default impl) and `llama_memory_i` (virtual added, default impl) are source-compatible.
- llama-context.h: class layout changed (many members added). Internal.
