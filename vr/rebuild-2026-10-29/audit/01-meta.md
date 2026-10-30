# 01 - ggml/src/ggml-backend-meta.cpp audit

Diff: `git diff 418f905a4 HEAD -- ggml/src/ggml-backend-meta.cpp` = +3062 / -144 (3815 diff lines, read in full).
Saved diff: `/tmp/claude-1000/-home-raistlin-infer-mx-llama-cpp/552f07ca-9f17-4f87-894a-a8efaa505805/scratchpad/audit/meta.diff`
Line numbers: where a reference is marked "diff L", it is a line in meta.diff; all others are HEAD line numbers in `ggml/src/ggml-backend-meta.cpp`. Items marked "~" are approximate. LOC figures are rough (+/- from hunk reading, not exact).

Base file already had: meta device/buft/buffer, split-state engine (axis 0/1/2/3, MIRRORED, PARTIAL), generic butterfly AllReduce fallback, `comm_init`/`comm_allreduce`, single-stage TP, `GGML_META_DEBUG`.

## 1. Feature table

| Feature | Files | ~LOC +/- | Class | Upstream equiv | Gating | Recommendation | Confidence |
|---|---|---|---|---|---|---|---|
| Multi-stage TP (tps lanes per stage, n_stages, per-stage comm_ctx, stage-aware subgraph partition, owning-stage detection, `tensor_stage` / `persist_buffer_stage` maps) | meta.cpp: device ctx ctor (~L56-100), meta ctx ctor (~L2880-2980), `graph_compute` partition (~L3640-4400) | +450 | C | none (base is single stage) | always on when `tps < n_devs`; ctor/`ggml_backend_meta_device(…, tps, …)` signature change; caller in `src/llama.cpp` L237/L286 | keep | high |
| Stage seam placement: persistent-buffer rewind, layer-seam cost comparison (`l_last-` prefix), `GGML_META_LAYER_SEAM_COST` | meta.cpp partition ~L4000-4110 | +120 | B (prompt-size bytes) / C | none | env `GGML_META_LAYER_SEAM_COST`, default ON | keep, but the seam-cost path only matters for multi-stage | med |
| Transfer-set builder (`collect_xfer_set`, `derive_earlier_xfer_set`) incl. persistent-view dependencies (KV write -> read) | meta.cpp ~L3720-3860 | +150 | C (correctness: DSV4 cache writes across stage seam, +1.77 ppl per commit msg) | none | always on in multi-stage | keep | high |
| Stage-to-stage transfer: 3 paths (RCCL sendrecv, dedicated copy queue+drain, sync `ggml_backend_tensor_copy`) + incremental persistent-row transfer (`transfer_tensors`, `transfer_lanes`) | meta.cpp `transfer_lanes`/`transfer_tensors`/`stage_transfer` ~L4700-4960 | +280 | B/C | none | RCCL: `GGML_META_XFER_RCCL` (unset: on if `GPU_MAX_HW_QUEUES>4` or ROCm tps>=3; `0` off; nonzero force); `GGML_META_MOE_XFER_RCCL_THRESHOLD` (default 1 MiB); dedicated: `GGML_META_NO_DEDICATED_CPY`; rows: `GGML_META_XFER_ROWS` (default ON) | keep; sync fallback is required | med |
| Persistent-row transfer via stashed index tensors (`ggml_meta_idx_stash_*`) | meta.cpp L1906-1960 (stash + range), set_tensor hooks L1968, L3120 | +70 | B | none | always-on stashing for every I32/I64 set_tensor <=1 MiB, under a global mutex; consumed only if `xfer_rows` | keep, but see QW6 | med |
| Fragment mode (uid==0 eval-callback views): `frag_last_stage`, `frag_stage_bits`, `frag_root_stage`, pre-pass cross-fragment transfer | meta.cpp fragment blocks ~L2030-2040, ~L4370-4400, pre-pass ~L5160-5210 | +110 | F (is the eval-callback path reachable with meta in any supported config?) | none | implicit (`cgraph->uid == 0`) | keep until verified; see Notes | low |
| Stage carry across graphs (`GGML_META_SPLIT_CARRY_STAGE`, `ggml_backend_meta_graph_begin`, `ggml_backend_meta_n_stages` used by `ggml-backend.cpp` L1355, L2559) | meta.cpp L2519-2530, graph_begin/n_stages at end; `ggml-backend-impl.h` L114-115; `ggml-backend.cpp` | +40 | C (DSV4.1 Engram gather) | none | env default ON | keep | med |
| Mirror-lane mask (MIRRORED tensor present only on some lanes, via `nr[1]`) + `ggml_backend_meta_mirror_lane_absent` in memset/set/get paths | meta.cpp L762-800 + memset L1894, set L2080s, get L2187, get_async ~L3330-3350, partition check ~L4281 | +90 | C (stage-local weights; enabled by `LLAMA_STAGE_LOCAL_MIRROR`) | none | set by `src/llama-model.cpp` L1326-1328 (`LLAMA_STAGE_LOCAL_MIRROR`, `=0` disables; not in this file) | keep | med |
| Partition-time validation: abort if a node on a stage reads a mirrored weight missing on a lane | meta.cpp ~L4265-4290 (GGML_ABORT names env `LLAMA_STAGE_LOCAL_MIRROR`) | +30 | A/C (guard) | none | always on in multi-stage; runs on every partition rebuild | keep | high |
| Split-state rules for new ops/arches: GET_ROWS row-shard (`handle_get_rows` axis-1 -> PARTIAL, row offset in `op_params[0..1]`), fold-split for GET_ROWS gather output, batched matmul with broadcast operand, SUM_ROWS axis-0 reduce, `handle_per_row` axis-0 -> MIRRORED, ROPE_BACK, FA "Design A", SSM_CONV/SCAN comment, `ne_set_by_rule` | meta.cpp L1088-1160 (ops), L1400-1420 (ROPE_BACK), L1480-1520 (ne-by-rule) | +200 | C (deepseek4, qwen4exp) | none | always on (TP only) | keep | med |
| Split-state diagnostics: named error log on UNKNOWN axis, divisibility log, RATIO MISMATCH log before asserts | meta.cpp ~L1070-1080, ~L1460, ~L1520 | +45 | E (debug-only, prints only on the abort path) | none | always compiled; logs only when assert about to fire | keep or trim (cheap, helpful) | med |
| GATHER closure: row-sharded gather made whole on every lane (`gather_rows`, `gathered_consumers`, `subgraph.gather`, consumer src repointing) | meta.cpp `gather_rows` ~L4590-4700, closure enum L1167, consumers ~L4260 | +230 | C | none | always (only when an axis-0 source is consumed by a full-row op) | keep | med |
| Full-row op list `ggml_backend_meta_op_needs_full_rows` | meta.cpp L497-519 | +25 | C | none | always | keep | high |
| TOP_K broadcast (BCAST closure) + 3 env knobs (`GGML_META_TOPK_BCAST`, `_ALL`, `_LOGITS`) | meta.cpp ~L4120-4160 (knobs), BCAST run ~L5300 | +60 | C (DSV4 indexer near-tie correctness) | none | `TOPK_BCAST` default ON; `TOPK_BCAST_ALL` default OFF; `TOPK_BCAST_LOGITS` default OFF | keep BCAST; trim the two opt-in knobs (QW5) | med |
| AR fold: `get_i_delayed` wrapper (existing) + call-site fold of reduce into following ADD (new) | meta.cpp L2076-2081 (comment), ~L4190-4210 (fold) | +35 | E (possibly redundant, see Notes) | base has `get_i_delayed` | always | candidate: keep one (see Notes) | low |
| Comm init per stage (`comm_ctxs[]`, `comm_set_pipeline_stages`, `comm_set_staging_depth`, `GGML_META_NO_COMM`) and extra AR entry points (`comm_allreduce_prepare`, `..._launch_rank`, `token_capture_*`, `token_graph_*`, `graph_cache_disable`, `ar_slab_mark`) | meta.cpp ctor ~L2906-2990 | +90 | B/C | base had single `comm_ctx` | `GGML_META_NO_COMM` default off; all proc addresses optional (null-checked) | keep | high |
| Whole-token graph capture/replay (`tg_cache`, `tg_run`, `tg_lookup`, warm-up, capture, replay, `GGML_META_TG_LIMIT`, `GGML_META_INNER_GRAPHS_EAGER_OFF`) | meta.cpp struct ~L2660-2700, replay/capture block ~L5010-5210 | +400 | B (perf: one host round trip per stage per token) | none | `GGML_META_TOKEN_GRAPH` default ON, but only if `parallel_dispatch` is on; backend exports in `ggml-cuda.cu` | keep; `TG_LIMIT` is a debug bisect knob (QW4) | med |
| Concurrent lane dispatch: `ggml_backend_meta_lane_dispatcher` (persistent worker threads, spin-then-block), `GGML_META_PARALLEL_DISPATCH` | meta.cpp L2357-2530 (dispatcher), dispatch in per-subgraph loop ~L5270-5300 | +125 | B | none | `GGML_META_PARALLEL_DISPATCH` default ON, only with tps>1 and every lane backend named "CUDA" or "ROCm" | keep | high |
| Repack buffer type for meta (`extra_bufts`, slot-wise composition, `repack` flag in buft ctx, `supports_op` repack gate) | meta.cpp `supports_op` L163-224, buft ctx L350-373, `buft_from_simple` L384-406, `get_extra_bufts` ~L615-660, reg L247-278 | +190 | C (repacked CUDA Q8/K-quant under TP) | base CPU has `ggml_backend_dev_get_extra_bufts`; meta is new | always; only active when a lane exposes extra bufts | keep | med |
| Event proxy (`event_new/free/sync/query`, `event_record/wait`, `caps.events=true`) | meta.cpp L101-148 (events), L289-292, L3760-3780 | +80 | C (scheduler needs events) | base: `nullptr` for all | always | keep | med |
| `buft_from_simple` cache key includes `dev` (comment explains) | meta.cpp L384-406 | +20 | A (correctness with model-local split userdata) | none | always | keep | high |
| Alloc failure becomes nullptr (instead of GGML_ASSERT) for buffer alloc and `alloc_buffer_n` | meta.cpp ~L2262-2280, ~L2328-2345 | +30 | C (OOM as load failure) | base asserts | always | keep | high |
| Multi-buffer abort guard | meta.cpp L1652-1656 AND L1743-1746 | +8 | E (second check unreachable, see QW7) | none | always | keep one (QW7) | med |
| Set/get async rewrite: chunked partial-row writes (`write_partial_row`), sync-fallback accumulator + one worker thread (`accum[2]`, `accum_worker_loop`), repacked whole-tensor path, `offset != 0` allowed, `size == 0` early return, mirror read picks owning-stage lane | meta.cpp `set_tensor_async` L1716-1906, `get_tensor_async` L1908-1960 | +330 | B/C (async upload of big weights) | base asserted `offset == 0` and used chunk fast path only | always | keep the chunk path; the accumulator thread is a substantial sub-feature (see Notes) | med |
| Memset: one contiguous memset per device replaces per-chunk loop | meta.cpp `memset_tensor` ~L1900-1915 | -5/+5 | B (cosmetic-ish; fewer calls) | base loop | always | keep | high |
| `buffer_get_tensor` MIRRORED reads first live lane | meta.cpp ~L2187 | +5 | A (needed with mirror mask) | base reads lane 0 | always | keep | high |
| Debug knobs: `dbg_part` (GGML_META_PART_DEBUG), `dbg_run` (GGML_META_RUN_DEBUG), `dbg_xfer` (GGML_META_XFER_DEBUG), prints in partition / run loop / transfers | meta.cpp fields L2737-2739, use L4056, L4352, L4743-4845, L5246-5252 | +60 | E (debug, default off; no measured win recorded) | none | all default OFF | remove or fold into one flag (QW3b) | med |
| Chunk trace (`GGML_META_CHUNK_TRACE`): enter/exit stamps, `backtrace()` on slow sync | meta.cpp L5-8 (`execinfo.h`), guard struct ~L2000-2014, sync hook L3369-3390, guard use L3422 | +40 | E (debug, default OFF, linux-only backtrace) | none | default OFF | remove (QW2) | high |
| Profiler (`GGML_META_PROFILE`): `ggml_meta_prof_now`, `ggml_meta_prof_scope`, prof_* counters, `prof_report`, timing in graph_compute / sync / token replay | meta.cpp L2375-2390, prof fields+report L2766-2815, prof init L3046/L3074, sync L3388-3390, replay L3423-3427, tg timing ~L5050-5076, per-subgraph ~L5259-5382 | +110 | E (debug, default OFF; keep only if used for ongoing tuning) | none | default OFF | move out or remove (QW3) | med |
| Duplicate `mmap_support` line | meta.cpp L162-163 | -1 | E (pure duplicate) | n/a | always | delete one line (QW1) | high |
| Comment-only/cosmetic: `host_buffer` comment reworded, `this->simple_devs` renames in ctor, blank line before "meta backend buffer" banner (~L461-462), "Design A" / "MLA head-split" / "state shape" comment edits | meta.cpp L85, L56-70, L461-462, L1164, L1171, L650 | ~+10 | E (noise) | n/a | n/a | keep (harmless) or revert noise hunks | high |
| TODO memo note for split-state recursion | meta.cpp L3663-3668 | +5 | E (comment) | n/a | n/a | keep or shorten | high |

Total: these features account for roughly the full +3062. Local estimates may be off by 10-20% per row.

## 2. Env var inventory (getenv added in this file)

Default column is the behavior when the variable is unset. Function-scope statics are read once per process; others once per meta backend construction (in `ggml_backend_meta_context` ctor).

| Variable | Read at | Default | Effect when set | Class |
|---|---|---|---|---|
| `GGML_META_SPLIT_CARRY_STAGE` | L2520 (static) | ON (`0` disables) | carry stage across splits / graphs | C |
| `GGML_META_NO_COMM` | ctor L2916-ish (`getenv` at L1500 in diff) | OFF (`!=0` disables comm_init, butterfly fallback) | debug A/B | E-ish (debug) |
| `GGML_META_XFER_RCCL` | ctor (diff L1561) | unset = rule-based (HWQ>4 or ROCm tps>=3); `0` off; nonzero force | stage transfer via RCCL sendrecv | B |
| `GPU_MAX_HW_QUEUES` (not GGML_) | ctor (diff L1575) | unset = not >4 | `>4` enables RCCL xfer path | B (HIP-specific) |
| `GGML_META_MOE_XFER_RCCL_THRESHOLD` | ctor (diff L1581) | 1048576 bytes | MoE large-xfer threshold for RCCL on ROCm tps=2 | B |
| `GGML_META_NO_DEDICATED_CPY` | ctor (diff L1608) | OFF (dedicated queue used if backend exposes it) | forces sync copy fallback | E-ish (A/B knob) |
| `GGML_META_PART_DEBUG` | ctor (diff L1630) | OFF | partition dump | E |
| `GGML_META_RUN_DEBUG` | ctor (diff L1631) | OFF | per-subgraph dump | E |
| `GGML_META_XFER_DEBUG` | ctor (diff L1632) | OFF | per-transfer dump | E |
| `GGML_META_XFER_ROWS` | ctor (diff L1633) | ON (`0` off) | persistent-row transfer + view dedup | B |
| `GGML_META_CHUNK_TRACE` | ctor (diff L1634) | OFF | entry/exit timestamps, backtrace on slow sync | E |
| `GGML_META_LAYER_SEAM_COST` | ctor (diff L1635) | ON (`0` off) | seam placement by bytes | B |
| `GGML_META_PROFILE` | ctor (diff L1645) | OFF | host-side profile counters | E |
| `GGML_META_PARALLEL_DISPATCH` | ctor (diff L1649) | ON (`0` off); only effective with tps>1 and CUDA/ROCm lanes | concurrent lane issue | B |
| `GGML_META_TOKEN_GRAPH` | ctor (diff L1652) | ON (`0` off); requires parallel dispatch | whole-token capture | B |
| `GGML_META_TOPK_BCAST` | static L2555 | ON (`0` off) | broadcast TOP_K results across lanes | C |
| `GGML_META_TOPK_BCAST_ALL` | static L2573 | OFF (`!=0` on) | broadcast also for mirrored-score TOP_K | C (opt-in) |
| `GGML_META_TOPK_BCAST_LOGITS` | static L2579 | OFF (`!=0` on) | broadcast top-k over logits | C (opt-in) |
| `GGML_META_INNER_GRAPHS_EAGER_OFF` | static L5024 | OFF | disables per-lane graph cache eagerly | B (opt-in) |
| `GGML_META_TG_LIMIT` | static L5098 | unset = all subgraphs | captures only first N subgraphs | E (bisect knob) |

Pre-existing and unchanged: `GGML_META_DEBUG` (base L448).

Related env read outside this file (not my slice, noted for completeness): `LLAMA_STAGE_LOCAL_MIRROR` in `src/llama-model.cpp` L1328; `GGML_META_*` also read in `ggml/src/ggml-cuda/ggml-cuda.cu` only in a comment (L3835), so no extra getenv there.

## 3. Upstream / API surface checks

- All proc-address names the meta backend requests (`ggml_backend_comm_set_pipeline_stages`, `..._allreduce_prepare`, `..._allreduce_launch_rank`, `ggml_backend_token_capture_begin/end`, `ggml_backend_token_graph_launch/free`, `ggml_backend_graph_cache_disable`, `ggml_backend_ar_slab_mark`, `ggml_backend_comm_set_staging_depth`, `ggml_backend_comm_sendrecv_tensor`, `ggml_backend_cpy_tensor_async_dedicated_queue/drain`) are exported by `ggml/src/ggml-cuda/ggml-cuda.cu` at HEAD and absent at base. All are null-checked in meta, so non-CUDA/ROCm builds fall through to the old paths. `ggml_backend_dev_get_extra_bufts` already existed at base for CPU.
- Public API change: `ggml_backend_meta_device(devs, n, tps, get_split_state, ud)` (new `tps` param). Callers: `src/llama.cpp` L237, L286, both updated. `vr/test-gfx906-tp.cpp` L79 already uses the 5-arg form.
- New exported functions: `ggml_backend_meta_n_stages`, `ggml_backend_meta_graph_begin` (declared in `ggml-backend-impl.h` L114-115, used by `ggml-backend.cpp` L1355, L2559). Both live.
- No dead functions found: every new static function / lambda is referenced. Events, reg, extra-bufts, mirror mask, idx stash, dispatcher, accumulator, profiler all have call sites.

## 4. Notes on specific features

- Lane dispatcher wait strategy (L2357-2370 comment): the comment argues a bounded spin of 512 iterations. The spin-count comment (`0 and 512 measure the same`) is a tuning note that will go stale. Keep only the thread-safety rationale.
- Accumulator thread (`accum_worker`, `accum[2]`, ~L1411-1466 in diff terms): a second worker thread just for `set_tensor_async` fallback. Adds shutdown ordering (`accum_shutdown` in `free`) and a `GGML_ASSERT` on slot reuse. This is a sizeable sub-feature; if the chunk path covers the real load (per-row/axis-2 weights), it may be cuttable. Confidence low, needs a loader test.
- Call-site AR fold (diff L2606-2629) and wrapper `get_i_delayed` (diff L2076-2081). The wrapper's own comment says "Measured equivalent on deepseek4 (88.0 subgraphs per token either way)", i.e. the two overlap. The call-site fold is needed for the split shared-expert shape per its own comment. Consider keeping one mechanism; not high confidence without a token-count test.
- Fragment mode (`uid==0`): the pre-pass at the top of the run path and per-fragment bit maps exist only for the scheduler eval-callback path. Need to confirm whether that path is reachable with `-sm tensor` in supported configs. If not, the whole fragment mode (~110 LOC) is dead for the supported path. Confidence low.
- `ggml_meta_idx_stash_put` runs on every I32/I64 `set_tensor`/`set_tensor_async`, including index tensors that are never used for persistent-row transfer when `GGML_META_XFER_ROWS=0`. See QW6.
- GET_ROWS `op_params[0..1]` used as a row-offset and mask flag (L1767-1790 diff, `handle_get_rows`). This overloads op params on a core op. Works, but is fragile if GET_ROWS op_params ever gain meaning upstream. Class F/technical debt, not a cleanup.
- Long comments: many new comments are multi-paragraph (token graph L2680-2690, dispatcher L2357-2375, TOPK L2549-2571, persistent-buffer L2119-2165). AGENTS.md asks for concise 1-2 line comments in ASD-STE100. Non-ASCII check: 0 non-ASCII characters in added lines. Trimming is cosmetic only.

## 5. Quick wins (high-confidence, low-risk)

Line numbers are HEAD (`ggml/src/ggml-backend-meta.cpp`).

- **QW1 - duplicate line.** L162-163: `props->caps.mmap_support = props->caps.mmap_support && tmp_props.caps.mmap_support;` appears twice. Delete one. Safe: idempotent AND. Confidence high.
- **QW2 - chunk trace + backtrace.** Remove `#if defined(__linux__) && !defined(__ANDROID__) #include <execinfo.h>` (L5-8), the `dbg_chunk` branch in `ggml_backend_meta_synchronize` (~L3369-3391 including `backtrace()`), `ggml_meta_chunk_trace_guard` (~L2000-2014 in diff, search `chunk_guard` / `ggml_meta_chunk_trace_guard`), `dbg_chunk` field and env read (L2744, L3033), and the `chunk_guard` line at ~L3422. Default OFF; no vr doc records a result. Safe to remove. Confidence high.
- **QW3 - profiler.** `GGML_META_PROFILE` is default OFF. Removing `ggml_meta_prof_now`, `ggml_meta_prof_scope`, all `prof_*` fields, `prof_report`, and the `prof_` lines in graph_compute / sync / token replay drops ~110 LOC. Locate with `grep -n 'prof' ggml/src/ggml-backend-meta.cpp`. Confidence med. Keep only if the user still profiles with it (the file's own comment says it was used for "where a token's wall time actually goes", and `vr/PERFORMANCE.md` has no mention, so the user should decide).
- **QW3b - debug prints.** `dbg_part`, `dbg_run`, `dbg_xfer` (L2737-2739, env reads L3029-3031, uses L4056, L4352, L4743-4845, L5246-5252). Default OFF; removing or folding into one flag removes ~60 LOC. Confidence med (debugging value to the user).
- **QW4 - TG_LIMIT bisect knob.** L5095-5099: `static const size_t tg_limit = getenv("GGML_META_TG_LIMIT")`. Default unset = `SIZE_MAX`. Replace with `const size_t n_cap = backend_ctx->n_subgraphs;` and remove the `tg_limit` static, the `covered` partial-capture branch can stay (it is also used when a capture is cut). Confidence high that default behavior is unchanged; medium on whether the user wants the bisect tool.
- **QW5 - opt-in TOPK knobs.** `GGML_META_TOPK_BCAST_ALL` (L4146) and `GGML_META_TOPK_BCAST_LOGITS` (L4152), both default OFF, no live vr doc. Removing them keeps the default rule (TOP_K BCAST with mirrored-score exemption). Confidence med. Only do this if no one still wants to A/B the DFlash logits path.
- **QW6 - idx stash when unused.** In `ggml_backend_meta_buffer_set_tensor` (L2080s diff L884) and `set_tensor_async` (L1968), guard `ggml_meta_idx_stash_put` with `backend_ctx->xfer_rows` (or a global check). The set_tensor path has no backend_ctx; the cheapest fix is an early return in `ggml_meta_idx_stash_put` when a static env-read flag is off. Saves a global mutex and up to 1 MiB memcpy per index write when `GGML_META_XFER_ROWS=0`. Default is ON, so there is no change in the default path. Confidence med.
- **QW7 - unreachable multi-buffer check.** L1743-1746 (`if (ggml_backend_buffer_is_multi_buffer(simple_buf)) GGML_ABORT(...)`) sits inside an `else if (simple_buf != nullptr)` branch, but the identical check runs unconditionally at L1652-1656 for the same `simple_buf` before any branch. The second check is unreachable. Delete it. Confidence med-high (verify `simple_buf` is not reassigned between the two; the diff shows it is not).
- **QW8 - stale comment.** The spin-count tuning note in the dispatcher comment (diff L965-970) names measurements that will go stale. Shorten. Cosmetic. Confidence high.
- **QW9 - whitespace noise.** Diff L461-462 adds a blank line before the "meta backend buffer" banner and L996-997 adds a double blank in the dispatcher struct. Cosmetic only. Confidence high.

Not recommended to cut (checked): the event proxy, extra-bufts/repack, mirror mask, RCCL xfer path (default ON on HWQ>4 which the fork uses), token graph, dispatcher, alloc-failure returns, `buft_from_simple` dev key (correctness, comment explains the stale-userdata bug).
