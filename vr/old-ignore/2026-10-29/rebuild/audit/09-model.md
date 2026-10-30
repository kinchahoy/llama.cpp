# Audit 09: model loading and graph building

Scope: src/llama-model.cpp, llama-model.h, llama-graph.cpp/.h, llama-arch.cpp/.h, llama-hparams.h, llama-model-loader.cpp, llama-vocab.cpp, src/models/* (deepseek4, deepseek41, qwen4exp, qwen35, qwen35moe, dflash, delta-net-base, minimax-m2, mimo2, models.h), gguf-py constants.py and tensor_mapping.py.
Base: 418f905a4. HEAD: e04bc2e38. Diff size in slice: +1781 / -364 (21 files). Read-only audit, nothing built or run.

Evidence keys used below: FEATURES.md = vr/old-ignore/2026-10-29/prior-guides/FEATURES.md (the only doc that describes most of these knobs), NOTES = vr/old-ignore/2026-10-12/notes/2026-10-12-tp-profile-findings.md, QS = vr/old-ignore/2026-10-29/prior-guides/QUICKSTART.md.

## Headline findings

1. **Half-finished DeepSeek-V4.1 port (non-target, but affects KV sizing).** The new hparams `dsv41_n_kv_source`, `dsv41_kv_source_layers`, `dsv41_n_index_source`, `dsv41_index_source_layers`, `dsv41_candidate_*` are declared but never written by any loader (grep HEAD: no assignment). Every `v41 = dsv41_n_kv_source > 0` branch is therefore dead. Consequences:
   - `LLAMA_DSV41_QNORM` and `LLAMA_DSV41_NO_COMPRESS` (deepseek4.cpp:955, 996) are no-ops. FEATURES.md documents them as working diagnostic gates. The actual V4.1 graph (deepseek41.cpp build_attention41) has no q-norm and never reads either env var.
   - `LLAMA_DSV41_NO_ENGRAM`, documented in FEATURES.md, does not exist in HEAD.
   - KV cache sizing for V4.1 (llama-kv-cache-dsv4.cpp:1491-1493, outside this slice) takes the `v41_tiers == false` branch, so `idx_state = 2*ratio_idx` and `idx_embd_mul = 2`. That is V4 sizing, not V4.1. Flag to the KV-cache auditor.
2. **Shared-expert TP split is now on for every architecture, not only DeepSeek.** llama-model.cpp:775-792 sits outside any arch gate. In base the shexp split existed only inside the DSV4 `is_dsv4` block. `LLAMA_SHEXP_SPLIT` defaults to 1. Flash Next (qwen4exp, llama-model.cpp path via qwen4exp.cpp:1277 `ffn_up_shexp`) and qwen35moe both have shared experts, so they change under `-sm tensor` by default. FEATURES.md only measured DSV4-Flash (+4-15%). This is the highest-priority item to validate before trusting Flash Next TP numbers. (Do not run it from this audit.)
3. **Generic graph change in build_rs.** `build_rs` lost its `state_copy` argument, and `get_state_rows` now defaults to `ggml_get_rows` (was nullptr, which meant a view). Every recurrent architecture (mamba, rwkv, delta-net, qwen35, qwen4exp, ...) now emits a dedicated gather plus a second gather for extra states. Result should be equal, but the graph topology changed for all of them. No caller passes nullptr (checked).
4. **Generic buffer-type reorder.** llama-model.cpp:1678-1696 now puts device extra buffer types (q8 repack on HIP) before the device default buffer type, for every architecture when `use_extra_bufts` is on. Base had default first. Unverified effect on non-repackable tensors beyond what the op-support check already filters.
5. **Dead Engram scaffolding.** The Engram implementation that runs is in base deepseek41.cpp (uses ENGRAM_KV / ENGRAM_GATE and the PLE keys written by conversion/deepseek41.py). The fork added ENGRAM_EMBED/WKV/Q/K tensor enums, 10 LLM_KV_ENGRAM_* enums, ~20 hparams fields and helpers, and one loader probe. None of the new names is ever created, read or written (grep HEAD). About 150 LOC, all removable.
6. **Converter never emits the split MTP form.** conversion/qwen4exp.py:148-153 always concatenates fc_embedding and fc_hidden into one NEXTN_EH_PROJ. So the `fc_hidden` branch in qwen4exp.cpp (`layer.nextn.fc_hidden`) is dead for GGUFs produced by this repo. The `eh_fused` check keeps the fused path working.
7. **Dead member.** `head_hc_flags` (models.h:1356) is never set. Its only uses pass 0 (deepseek4.cpp:97-99). deepseek41 overrides load_arch_tensors and never calls the base, so it never needed TENSOR_NOT_REQUIRED.
8. **MiniMax-M2 is newly allowed under -sm tensor.** llama-arch.cpp removes `LLM_ARCH_MINIMAX_M2` from the unsupported list (the list returns false). Before this, `-sm tensor` threw at model creation (llama-model.cpp:390). Non-target. The TP rules and the decomposed RMS norm are in minimax-m2.cpp and llama-model.cpp.

## Question-by-question answers

**Which changes are needed for our target models?**

- **Qwen3.8-27B dense (presumed arch qwen35; confirm from the GGUF arch key).** Needed: snapshot ring for single-sequence speculation (delta-net-base.cpp ring writes; llama-graph.cpp `build_rs`/`fill_s_write`/`upload_s_copy`), MTP KV-only prefill and the n_outputs guard (qwen35.cpp:553-600, 680-684), LLAMA_TAP_STATIC taps for DFlash (set_outputs), and the no-vocab special-id read if the DFlash sidecar is no_vocab. Not needed: engram, DSV4 code, MiniMax, MiMo.
- **Flash Next (qwen4exp).** Needed: conv-window snapshot path (qwen4exp.cpp `qwen4exp_conv_windows`, `build_conv_state_at` keep_snapshots), MTP context filters (llama-model.cpp create_memory, qwen35/qwen4exp branch), eh_fused MTP load, compress-ratio tolerant load (qwen4exp.cpp:62-68), MTP block mirror under TP (llama-model.cpp ~line 690, `n_layer_nextn > 0`), LLAMA_PLE_SHARD (FEATURES.md: 224-500 t/s prefill, 4x MI50 Q4), LLAMA_MTP_REPACK default on (NOTES: mtp3 n512 TG 52.9 -> 54.2 when repacked). Also active by default and unvalidated here: shexp split (headline 2).
- **Gemma 4.** No arch-specific code in this slice. It benefits from the generic tied-output split rule (llama-model.cpp:716-720, `tied_output`). FEATURES.md reports gemma-4-26B-A4B-Q8 at -tps 4: 104 -> 117 t/s, 561 MiB less per card. The buffer reorder (item 4) is also generic and reaches Gemma 4 too.
- **DFlash.** Needed: no-vocab special token ids (llama-vocab.cpp:2270-2293), SWA-pattern optional (dflash.cpp:144-151), tensor_mirror_output (only when DSpark is selected, or LLAMA_DFLASH_MIRROR_OUTPUT=1, which defaults to off per common.cpp:1755), and the DFlash draft replicate-vs-split rule (llama-model.cpp:728-736, default replicate for dense drafters). The graph_dsv4 hc_shift path is only for a V4-backbone drafter (non-target).
- **MTP.** Qwen3.5/qwen4exp MTP as above. DeepSeek V4 MTP is non-target.

**Vocab changes, why (llama-vocab.cpp):**
1. In the no_vocab branch, read the BOS/EOS/EOT/UNK/SEP/PAD/MASK ids. A DFlash or DSpark sidecar has no token list and borrows the target tokenizer. Without this, DFlash's noise block was filled with -1 and every draft decode was rejected (the comment states this). Needed for DFlash. No base equivalent: base returned before the special-token pass.
2. `print_info` uses `tok_text()`, which returns "" when `id_to_token` is empty. Without it, `.at()` throws for a no_vocab model that declares special ids.
3. `token_to_piece` returns 0 for LLAMA_VOCAB_TYPE_NONE. The speculative sidecar logs drafted tokens through this path, and the argument is evaluated even when the log level hides it.
All three are class A/C for DFlash. Keep.

**Model-specific changes for models NOT in the target list.** Generic TP enablers or experiments?

| Change | Generic TP enabler? | Verdict |
|---|---|---|
| deepseek4 / deepseek41 MLA head split (LLAMA_MLA_TP, default 1) | No: DSV4 grouped-LoRA specific | Keep as V4 feature. Its gate comment says it is DEEPSEEK4 only and the V4.1 path depends on the dead `v41` plumbing. |
| DSV4 narrow shexp -> plain buffer list (llama-model.cpp:2897) | No | Keep, DSV4 perf. |
| DSV4 weight probe RESHAPE (attn_output_a, loader:952-977) | No, but essential for V4 load (CPU_REPACK crash) | Keep. |
| deepseek4 hc_shift, carry_pre, build_head_fold, F32 mix prec (`f32_src`) | No | V4.1 only; keep if V4.1 stays. The `dspark_v41` q-norm skip is probably unreachable (graph_dsv41 does not call build_attention). Medium confidence. |
| deepseek4 v41 tier code (ratio indexes, owns_kv, idx_tier) | No | Dead (item 1). Candidate for removal after the V4.1 port is finished or dropped. |
| minimax-m2 TP (qk-norm split, decomposed RMS, unlisted from unsupported) | Yes, a TP enabler for MiniMax-M2 (FEATURES.md: MiniMax-M2.1 Q8 -tps 2, 26501 -> 26357 MiB, memory only) | Keep if MiniMax stays a supported target. Otherwise an experiment. The qk-norm rule runs for any 1-D norm with `ne[0] > n_embd_head_k`, MiniMax-only by arch check. |
| mimo2 SWA pattern fallback (try n_layer_all, else n_layer) | No, a loader fix | Keep, class C. |
| DFlash-DSV4 drafter routing (dflash_dsv4_split) | Yes for DSV4 drafter | Keep. Non-target. |

**getenv() list in this slice (defaults as coded):**

| Env var | Default | Where | Status |
|---|---|---|---|
| LLAMA_PLE_SHARD | 0 | llama-model.cpp:411-421 | Used. Flash Next TP win (FEATURES.md). Keep. Candidate for default-on after check. |
| LLAMA_PLE_PREFAULT | 0 | llama-model.cpp:429-462, call at 2551 | Used but FEATURES.md says it pays only on a fresh process and not on a warm cache. Candidate for removal (with the `<thread>`/`<atomic>` includes at :28-29). |
| LLAMA_MLA_TP | 1 | llama-model.cpp:629 | Used, DSV4/V4.1. Keep. |
| LLAMA_DRAFT_SPLIT | unset (= dflash_dsv4_split) | llama-model.cpp:730 | Used, DFlash drafter. Not documented anywhere in vr/ (grep). Add a doc line. |
| LLAMA_SHEXP_SPLIT | 1 | llama-model.cpp:776 | Used, now global (headline 2). Prints an unconditional `fprintf` once. |
| LLAMA_STAGE_LOCAL_MIRROR | 1 | llama-model.cpp:1328 | Used, multi-stage only. Documented in FEATURES.md. Keep. |
| LLAMA_MTP_REPACK | 1 | llama-model.cpp:2154 | Used. Documented in QS and NOTES. Keep. Comment at :2095 contradicts it (says MTP keeps canonical weights). |
| LLAMA_DSV41_QNORM | 0 (q-norm off when set to unset) | deepseek4.cpp:955-962 | Inert (item 1). Documented as a gate. |
| LLAMA_DSV41_NO_COMPRESS | 0 | deepseek4.cpp:996-1003 | Inert (item 1). Documented as a gate. |
| LLAMA_DSV41_ENGRAM_CAST_DEVICE | 0 | llama-model-loader.cpp:1241-1247 | Dead: only applies to ENGRAM_Q/K tensors that are never created. Remove with the GGML_OP_CPY probe (loader:955). |

Other env reads that matter for this slice but live outside it (cross-check owners): LLAMA_KV_REPLICA (llama-kv-cache.cpp:277, default on when offloaded), LLAMA_TAP_STATIC (llama-context.cpp:876, default on), LLAMA_DFLASH_MIRROR_OUTPUT (common.cpp:1755, default off), LLAMA_DSPARK_TARGET_REPACK (common.cpp:1763, default off), LLAMA_MTP_DRAFT_PIPELINE / LLAMA_DRAFT_PIPELINE (llama-context.cpp:572, 584).

**Gating notes.** The shexp split, the stage-local mirror, the MTP block mirror and the buffer reorder are all reachable only under `-sm tensor` (or, for the reorder, when `use_extra_bufts` is on). The kv-replica and tap paths are gated by offload and env. `ring_enabled` requires `n_rs_seq > 0 && n_seq_max == 1` (llama-memory-recurrent.cpp:46), so the ring is single-sequence speculation only.

## Feature table

LOC is approximate, from reading the hunks. Class: A essential gfx906/TP compat, B perf, C functional extension, D superseded by upstream, E redundant/dead/debug, F unclear. Upstream equivalent is relative to base 418f905a4.

| Feature | Files | ~LOC +/- | Class | Upstream equiv | Gating | Recommendation | Confidence |
|---|---|---|---|---|---|---|---|
| Multi-stage TP lane mapping (n_stages, lane_base, owning_stage, per-lane tensor_split) | llama-model.cpp (split-state), llama-model.h | +75/-15 | A (only for -tps < n_devices) | None (fork; base userdata had n_devices only) | -sm tensor, n_stages>1 | Keep | high |
| Stage-local mirrored weights (LLAMA_STAGE_LOCAL_MIRROR) | llama-model.cpp ~1316-1345 | +35 | B (memory) | None | -sm tensor multi-stage; env default 1 | Keep (documented, FEATURES.md) | high |
| MLA head split for DSV4/V4.1 (LLAMA_MLA_TP) + design-A replicate (llm_arch_sm_tensor_replicates_attention) | llama-model.cpp ~560-660, llama-arch.cpp ~1459-1475 | +85/-5 | C (non-target) | None | -sm tensor, DSV4 arches; env default 1 | Keep, scope with V4 | high |
| Shared-expert split (LLAMA_SHEXP_SPLIT), now global | llama-model.cpp 775-792, 1160-1175 | +45/-10 | C/B | None (base: DSV4 only) | -sm tensor, ALL archs with shexp; env default 1 | Validate on Flash Next and qwen35moe before keeping global; or re-scope to DSV4 | high (scope); med (perf effect) |
| Shexp split granularity (4-granule rule) | llama-model.cpp ~1160-1175 | +20 | A (non-dividing TP widths) | None | -sm tensor | Keep | med |
| Tied output head split (tied_output, gemma-4) | llama-model.cpp ~716-720, 906-909 | +12 | B | None | -sm tensor; generic | Keep (FEATURES.md Gemma 4 +12%) | high |
| MTP block mirror under TP (n_layer_nextn>0) | llama-model.cpp ~680-710 | +25 | A (MTP under -sm tensor) | None | -sm tensor, MTP models | Keep | med |
| DFlash draft split/replicate routing (LLAMA_DRAFT_SPLIT), tensor_mirror_output | llama-model.cpp 597-605, 713-736; common.cpp 1746-1785 (outside) | +40 | C | None | -sm tensor, DFlash; env unset | Keep. Document LLAMA_DRAFT_SPLIT | med |
| PLE gather table shard (LLAMA_PLE_SHARD) | llama-model.cpp 411-421, 960-972, 1140-1151 | +45 | B (Flash Next) | None | -sm tensor; env default 0 | Keep. Consider default-on after re-measure | med |
| PLE prefault (LLAMA_PLE_PREFAULT) | llama-model.cpp 27-29, 423-462, 2551 | +40 | F (FEATURES.md: no win on warm cache) | None | env default 0 | Remove candidate | med |
| CPU pin of PLE and ENGRAM_EMBED tables | llama-model.cpp ~2885-2925 | +35 | A (PLE) / E (ENGRAM_EMBED) | None | input-class tensors | Keep PLE pin; drop ENGRAM_EMBED branch | high |
| Repacked MTP/output head (LLAMA_MTP_REPACK) + gpu_buft_list_plain | llama-model.cpp 1678-1697, 2095-2105, 2153-2161 | +30 | B | None | GPU with extra bufts; env default 1 | Keep. Fix contradictory comment at :2095 | high |
| Device extra bufts before default (make_gpu_buft_list reorder) | llama-model.cpp ~1648-1700 | +15/-10 | B (generic) | None (base: default first) | any GPU arch with use_extra_bufts | Keep, but verify generic effect on non-TP runs (gemma4, qwen) | med |
| Weight-probe RESHAPE for attn_output_a | llama-model-loader.cpp 952-977, 1085-1094 | +25 | A (DSV4 load) | None | DSV4 only (tensor op check) | Keep | high |
| Weight-probe GGML_OP_CPY (LLAMA_DSV41_ENGRAM_CAST_DEVICE) | llama-model-loader.cpp 955-958, 1238-1247 | +15 | E (dead) | None | env default 0; targets nothing | Remove | high |
| load_all_data async relaxation (extra-buft, META host buft, copy-only backend) | llama-model-loader.cpp ~1650-1710 | +20 | A (TP load) / generic | None | async upload path | Keep; note generic effect for repacked buffers | med |
| Loader: TENSOR_READ_LAZY-only contexts keep ctx, memory_breakdown skip | llama-model.cpp ~2395-2415, 2995-3000 | +20 | A (PLE lazy table) | None | any arch with lazy tensors | Keep | med |
| Engram scaffolding (LLM_KV_ENGRAM_* new, ENGRAM_EMBED/WKV/Q/K tensors, hparams engram_* arrays, helpers, layer.engram_embed/wkv, LLAMA_MAX_ENGRAM_* defines) | llama-arch.h/.cpp, llama-hparams.h, llama-model.h | +150 | E (dead) | None (base Engram = ENGRAM_KV/GATE in deepseek41.cpp) | none | Remove | high |
| V4.1 source-layer fields (dsv41_n_kv_source, dsv41_kv_source_layers, dsv41_n_index_source, dsv41_index_source_layers, dsv41_candidate_*, dsv41_is_*_source, dsv41_kv_source_for) and LLM_KV_ATTENTION_* new names | llama-hparams.h 358-391, llama-arch.h/.cpp | +60 | E (never populated) | None (base uses dsv41_kv_src/topk_src arrays) | none | Either finish the port (populate in deepseek41.cpp load) or remove; flag KV cache | high |
| V4 v41 tier branches in deepseek4.cpp (QNORM, NO_COMPRESS, idx_tier, owns_kv) | deepseek4.cpp 954-1290 | +100 | E (dead) | None | none | Remove together with the above, or finish | high |
| dsv4_ratio_idx / dsv4_ratio_plain as hparams instead of constants | llama-hparams.h 355-356, deepseek4.cpp (replaces DSV4_*_RATIO) | +10 | F (values never loaded) | None | none | Revert to constants for the V4 path, low value either way | high |
| head_hc_flags member | models.h 1356, deepseek4.cpp 97-99 | +3 | E (always 0) | None | none | Remove | high |
| llama_model::path | llama-model.h 751; llama.cpp:524 (outside) | +2 | E (written, never read) | None | none | Remove | high |
| NEXTN_FC_HIDDEN + qwen4exp split MTP | llama-arch.h/.cpp, constants.py 1365/2250/3445, tensor_mapping.py 3002/3021, qwen4exp.cpp 320-322, 650-653 | +25 | F (converter never emits it; eh_fused covers the fused file) | None | qwen4exp MTP only | Keep if another converter ships split form; otherwise remove. Move enum to end of MODEL_TENSOR to avoid renumbering | med |
| tensor_mapping.py fc_embedding / nextn_hc_* names | tensor_mapping.py 2999-3090 | +8 | E (conversion/qwen4exp.py already maps these) | None | converter only | Likely redundant. Verify conversion path before dropping | med |
| Compress-ratio tolerant load (qwen4exp) | qwen4exp.cpp 62-68 | +8 | C | None | qwen4exp | Keep | med |
| qwen4exp MTP context filters (create_memory) | llama-model.cpp ~3700-3745 | +40 | A (MTP head-only export, MTP ctx) | None | qwen35/qwen3next/qwen4exp; ctx_type MTP | Keep. Note it also changes trunk filter (layer presence check) for target too | med |
| qwen4exp conv-window snapshot path (keep_snapshots, qwen4exp_conv_windows) | qwen4exp.cpp 1407-1535, 1613 | +110/-30 | B (MTP/speculation ring) | None (base: plane-indexed only) | uses_ring (single seq) | Keep | med |
| Snapshot ring in recurrent graph (build_rs signature, fill_s_write, upload_s_copy, can_reuse, s_write slices) | llama-graph.cpp/.h | +230/-60 | B | None (base: no ring) | ring_enabled (n_rs_seq>0, n_seq_max==1) for ring; build_rs change unconditional | Keep. Verify the unconditional build_rs gather change on non-ring archs | med |
| delta-net-base ring writes (build_conv_state, build_recurrent_attn) | models/delta-net-base.cpp | +65/-5 | B | None | ring only; plane-indexed fallback kept | Keep | med |
| Tap device tensors (layer_inp_dev, set_outputs, forward_expand moved out of graph build) | llama-graph.cpp ~1576-1620, llama-graph.h, deepseek4.cpp | +50/-15 | B (sched splits, re-reserve) | None | LLAMA_TAP_STATIC default 1 (llama-context.cpp:876) | Keep | med |
| KV replicas (build_get_k/v, build_cpy_k/v, kv_rows_written, kv_replicas_filled) | llama-graph.cpp ~3115-3180 and callers, llama-graph.h | +90/-10 | C (multi-device replica of reused layers) | None | has_replicas(); LLAMA_KV_REPLICA default on offload | Keep; owned by KV-cache audit | med |
| build_attn_store_kv (KV-only store for MTP prefill replay) | llama-graph.cpp ~3526-3553, qwen35.cpp 586-612 | +35 | B | None | cparams.mtp_prefill_kv_only (set by speculative) | Keep | med |
| ggml_mul_mat_aux (duplicate of llama_mul_mat_hadamard) | llama-graph.cpp 71-86 | +17 | E (exact duplicate) | llama-impl.h:57 `llama_mul_mat_hadamard` | none | Delete, call llama_mul_mat_hadamard | high |
| kq_mask unallocated guard (can_reuse, set_input) | llama-graph.cpp ~655-700 | +10 | A (KV-only graph) | None | KV-only graph only | Keep | high |
| qwen35/qwen35moe MTP n_outputs guard (inp_out_ids conditional) | qwen35.cpp 553-560, 680-685; qwen35moe.cpp 592-600, 726-732 | +12 | A (KV-only replay) | None | MTP graph | Keep | high |
| DFlash SWA pattern optional + set_swa_pattern(0) | dflash.cpp 144-151 | +8 | A (DFlash sidecar) | None | dflash with SWA | Keep | high |
| mimo2 SWA pattern fallback | mimo2.cpp 9-18 | +8 | C (non-target) | None | mimo2 | Keep | high |
| MiniMax-M2 TP (qk-norm split, decomposed RMS, unsupported-list removal) | minimax-m2.cpp +33, llama-model.cpp ~1100-1118, llama-arch.cpp -1 | +75 | C (non-target) | None | -sm tensor, minimax-m2 | Keep if MiniMax in scope, else experiment | med |
| deepseek4 build_hc_* API refactor (pre/mixes return values, post/comb out-params), f32 mix precision | deepseek4.cpp 340-430, deepseek41.cpp 848-870, dflash.cpp | +90/-80 | C (V4.1 needs it) | None | V4 / V4.1 | Keep if V4 stays | med |
| deepseek4 layer-tap change (no forward_expand at build time) | deepseek4.cpp 1313-1325, 1425-1427 | +8/-6 | B (measured 65 -> 51 sched splits per FEATURES/comment) | None | taps | Keep | med |
| Vocab no-vocab special ids, print_info guard, token_to_piece guard | llama-vocab.cpp 2270-2293, 4100-4104, 4361-4390 | +45/-10 | A/C (DFlash sidecar) | None | no_vocab models | Keep | high |
| Metadata override key `mxxm.tensor_mirror_output` + get_key<bool> instantiation | llama-model.cpp 1821-1823; llama-model-loader.cpp 449 | +3 | C | None | DSpark or DFlash-mirror | Keep | high |
| Cosmetic and orphan items (see Quick wins) | various | ~25 lines of noise | E | n/a | n/a | Clean | high |
| Debug logs: LLAMA_LOG_ERROR before assert (split-state lookup); unconditional fprintf "[shexp-split]" | llama-model.cpp 566-572, 775-783 | +12 | E (debug) | None | none | Downgrade or remove | high |

## Per-feature notes (brief)

- **TP split rules (llama-model.cpp).** The head-split rule for V4 attention (q_b AXIS_1, attn_sinks AXIS_0, attn_output_a AXIS_2, attn_output_b AXIS_0) is coherent: the code's own comments explain why the GROUPED LoRA output maps to group boundaries. The `is_dsv4` regex for the state cache was removed (`pattern_dsv4_state`), consistent with the design-B path. The `tied_output` rule covers the head as a duplicate of token_embd (vr doc: gemma-4 fix).
- **Shexp split.** The routed-expert and shexp split share the same axes, and the partial sums merge into the existing AllReduce. Correct by design. The risk is that Flash Next's shexp and routed experts may not be summed at the same graph point (qwen4exp.cpp:1277-1285); partial merge behaviour in the meta backend has to hold there too. Needs runtime check.
- **Ring and MTP.** The ring (`ring_enabled`) and the per-plane conv windows are the main reason MTP drafting went from a prefill tax to about a tenth (FEATURES.md). The plane-indexed fallback is kept for parallel sequences, with a comment. Good.
- **PLE shard.** Head-granular segments, a masked gather, and a single allreduce. The table is pinned to CPU when the shard is off (llama-model.cpp:2921-2929), so default behaviour is CPU-resident as in upstream lazy mode.
- **Engram.** Dead scaffolding around a working base implementation (headline 5). Converter writes PLE-named keys, not the new engram.* keys. The new keys in llama-arch are never read.
- **Graph tap change.** Moving ggml_build_forward_expand out of the build loop into set_outputs is documented in code (sched splits 65 -> 51). Plausible; keep.
- **Minor.** `llm_graph_context::build_cpy_k` etc. are templated and explicitly instantiated for the two cache types used (llama-graph.cpp ~3119-3160). OK.

## Quick wins (exact locations, why safe)

1. **Delete `ggml_mul_mat_aux`** (src/llama-graph.cpp:71-86). Body is identical to `llama_mul_mat_hadamard` (src/llama-impl.h:57-74). Replace the 2 call sites at llama-graph.cpp ~3535 and ~3538 with `llama_mul_mat_hadamard`. Safe: same code. High.
2. **Remove the GGML_OP_CPY probe and the ENGRAM cast env gate.** src/llama-model-loader.cpp:955-958 (case GGML_OP_CPY) and 1238-1247 (LLAMA_DSV41_ENGRAM_CAST_DEVICE block). No tensor ENGRAM_Q/K is ever created (grep). Safe. High.
3. **Remove the ENGRAM_EMBED pin branch.** src/llama-model.cpp:2909-2919 (`engram_cpu_only` and its use). LLM_TENSOR_ENGRAM_EMBED is never passed to create_tensor. Safe. High.
4. **Remove dead Engram enums and hparams.** src/llama-arch.h:~320-335 (new LLM_KV_ATTENTION_KV_SOURCE_LAYER_IDS, INDEX_SOURCE_LAYER_IDS, CANDIDATE_*; LLM_KV_ENGRAM_* except PAD_TOKEN_ID and TOKEN_MAP, which are used); src/llama-arch.h:692-695 (ENGRAM_EMBED/WKV/Q/K); src/llama-arch.cpp corresponding name and info rows (LLM_KV_NAMES ~319-334, LLM_TENSOR_NAMES ~654-657, LLM_TENSOR_INFOS ~939-942); src/llama-hparams.h:12-14 (LLAMA_MAX_ENGRAM_*) and :393-411 (engram_* arrays and helpers, engram_pad_token_id at :~348). Also src/llama-model.h:551-552 (engram_embed, engram_wkv). Safe: all grep count 2 (declaration and name table only). Keep the GGUF key names used by the converter (ple.* and engram.pad_token_id, engram.token_map). High.
5. **Remove `head_hc_flags`.** src/models/models.h:1356; src/models/deepseek4.cpp:97-99 (pass 0 or drop the arg). Its only value is 0. Safe. High.
6. **Remove `llama_model::path`.** src/llama-model.h:750-751 and the single assignment at src/llama.cpp:524 (outside slice). No reader. High.
7. **Remove the orphan comment.** src/llama-model.cpp:1722-1725 ("buffers over a file mapping for TENSOR_READ_LAZY tensors ...") describes a member that does not exist. High.
8. **Cosmetic blank-line churn.** Restore or remove lines so the diff stops showing noise: src/llama-model.cpp (blank before `//` interface implementation, ~3964; blank before `llama_model_n_cls_out`, ~4049, both removed); src/models/models.h (blank before `llama_model_deepseek41`, ~1519, removed); src/models/deepseek4.cpp (~255-258, two DSV4 constants removed leaving a double blank); src/llama-model-loader.cpp (~1458, one extra blank added). Pure whitespace. High.
9. **Debug logs.** src/llama-model.cpp:566-572 (LLAMA_LOG_ERROR before a GGML_ASSERT; keep the assert, drop the log or make it DEBUG) and :775-783 (unconditional `fprintf(stderr, "[shexp-split] ...")`, prints once per process under -sm tensor; use LLAMA_LOG_INFO or delete). Safe. High.
10. **Fix the MTP comment contradiction.** src/llama-model.cpp:2095-2096 says the NextN/MTP blocks keep canonical weights, but LLAMA_MTP_REPACK defaults to 1 and the code at :2153-2161 repacks them. Comment-only. High.
11. **V4.1 dead plumbing, decision needed.** src/models/deepseek4.cpp:954-1003 (QNORM/NO_COMPRESS statics and `v41`, `tier_idx`, `owns_kv`, `kv_src` logic in build_attention_impl) and src/llama-hparams.h:355-391 (dsv41_n_kv_source etc.). Either populate them in src/models/deepseek41.cpp load (lines ~40-80, the base already computes `dsv41_kv_src` and `dsv41_topk_src` arrays and the fork never copies them) or delete the dead branches. Do not delete the kv-cache branches (llama-kv-cache-dsv4.cpp:1436-1462, 1491-1493) without the KV auditor, since they change V4.1 cache sizing. Medium confidence on the right answer; high on the dead-ness.
12. **PLE prefault.** src/llama-model.cpp:27-29 (includes), 423-462 (function), 2551 (call). FEATURES.md says it helps only on a fresh process and is off by default. Removal changes no default behaviour. Medium (product decision).
13. **Move NEXTN_FC_HIDDEN to end of MODEL_TENSOR.** gguf-py/gguf/constants.py:1365 (insertion shifts values of later members). Python-only enum, GGUF stores names, so safe either way; moving it removes renumbering noise. Medium.
14. **Doc fixes (not code).** FEATURES.md lists LLAMA_DSV41_NO_ENGRAM (no such code), and documents QNORM/NO_COMPRESS as working (they are inert, item 1). LLAMA_DRAFT_SPLIT is undocumented.

## Items for other auditors

- KV cache: `dsv41_n_kv_source` is never set, so `v41_tiers` in llama-kv-cache-dsv4.cpp:1491-1493 is false for V4.1 (sizing reverts to V4). Also LLAMA_KV_REPLICA (llama-kv-cache.cpp:277) controls the replica path this slice depends on.
- Memory/recurrent: the ring condition (llama-memory-recurrent.cpp:46-49) decides whether the new ring graph code is reached.
- Context: LLAMA_TAP_STATIC (llama-context.cpp:876) decides whether the tap device tensors exist.
- Common: `mxxm.tensor_mirror_output` is set in common/common.cpp:1746-1785, and LLAMA_DFLASH_MIRROR_OUTPUT defaults off there.
- ggml backend: meta extra-buft composition (ggml-backend-meta.cpp:617+) decides which repack buffer types reach the reorder in item 4.
