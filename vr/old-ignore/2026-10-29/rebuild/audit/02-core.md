# Audit 02: generic ggml core and 1-line backend edits

Scope: `ggml/src/ggml-backend.cpp`, `ggml/src/ggml-alloc.c`, `ggml/include/ggml-backend.h`, `ggml/src/ggml-backend-impl.h`, `ggml/src/ggml-backend-sched-impl.h` (new file), `ggml/src/ggml-impl.h`, `ggml/CMakeLists.txt`, `ggml/src/CMakeLists.txt`, and the 1-line `event_query` edits in 15 backend files.

Diff basis: `git diff 418f905a4 HEAD`. Saved diffs: `scratchpad/audit/core-backend.diff`, `scratchpad/audit/core-alloc.diff`.

LOC figures are approximate, counted from hunks.

## Headline findings

1. None of the core changes is gated to gfx906 or HIP. Almost all of them are always on for every backend. The exceptions are the staging thresholds, which default to 0 unless the device is a meta backend (see notes).
2. The 15 one-line backend edits exist only because the device interface is filled with positional `/* .name = */` initializers. A new trailing member leaves each initializer one entry short, which `-Wextra` flags (`-Wmissing-field-initializers`, set at `ggml/src/CMakeLists.txt:40`). The lines silence that warning. They do not change behavior, since the omitted member would be zero-initialized anyway. Designated initializers would not avoid the edits, because the existing style is positional. The only edit-free option is to expose `event_query` through `ggml_backend_reg_get_proc_address`, as the `comm_*` functions already are. That is a larger design change, not a cleanup.
3. Roughly 400 of the roughly 1100 added lines are debug or diagnostic code: the timing struct and prints, the input trace, the rebind event trace, and the realloc trace. None of these is needed in the default config, and none has a recorded win in `vr/`. These are the main cleanup candidates.
4. Two correctness-sensitive points need review. (a) The public `ggml_backend_event_query()` returns `true` when a backend has no `event_query`, so "ready" can be reported for unfinished work. (b) The layout cache adopts a cached layout without checking the buffer-id assignment when there are exactly 2 buffers. The comment claims this case cannot disagree, and I could not confirm that.

## 1. Feature table

| Feature | Files | ~LOC +/- | Class | Upstream equiv | Gating | Recommendation | Confidence |
|---|---|---|---|---|---|---|---|
| Layout cache (`gallocr_layout`, layout store/restore, `ggml_gallocr_alloc_graph_ids`) | ggml-alloc.c (+ backend.cpp call site, ggml-impl.h decl) | +230 / -10 | B (perf) | None. Base reserves on any realloc. | Always on; `GGML_GALLOC_LAYOUT_CACHE=0` disables. Affects every gallocr user. | Keep. Review the 2-buffer assumption (see 3.2). | med |
| Realloc-check refactor and trace (`ggml_gallocr_graph_fits`, `ggml_gallocr_realloc_trace`) | ggml-alloc.c | +90 / -35 | B refactor + E trace | Base `ggml_gallocr_needs_realloc` (ggml-alloc.c, base ~1000-1040). | Trace: `GGML_ALLOC_REALLOC_TRACE`, default off. | Keep the refactor (the cache needs `graph_fits` with a layout array). Drop the trace logs. | high (for trace) |
| Graph-input staging (graph and non-graph), staging buffer pool | ggml-backend.cpp | +260 | B (perf, +69% prefill per FEATURES.md) | None. Base copies inputs synchronously or via `cpy_tensor_async`. | Effectively meta-only by default: `should_stage_input` requires a META device unless `GGML_SCHED_STAGE_SIMPLE_GPU_INPUTS` is set. The non-graph variant requires a META destination. | Keep. | med-high |
| Staging size knobs and helpers | ggml-backend.cpp | +90 | B | None | 7 getenv knobs (list in section 4) | Keep. Consider folding the 4 F32/F16/I64/I32 knobs into one table. | med |
| Ring depth default (`default_n_copies`, `GGML_SCHED_MAX_COPIES` 4->16, CMake 16, `GGML_SCHED_N_COPIES`) | ggml-backend.cpp, ggml/CMakeLists.txt | +25 / -2 | B | Base: fixed `parallel ? 4 : 1`. | Always on. Default `max(4, n_gpu_devices)`, raised to meta stage count, capped at 16. For 1-4 GPUs this is still 4, so it is a no-op there. | Keep. Note that the 4->16 bump only changes behavior for more than 4 GPUs. | high |
| Per-(backend,copy) input staging events and `events_recorded` bookkeeping | ggml-backend.cpp | +30 | B | Base has `events[][]` only. | Always on (events are created for every backend). | Keep. | med |
| Event query API (`ggml_backend_event_query`, `iface.event_query`, 15 backend 1-liners) | backend.cpp, backend-impl.h, 15 backend .cpp files | +12 core, +15 one-liners | C (needed by rebind check and llama-context) | None in base. | Always present. Backends without it report "ready" through the public API (see 3.1). | Keep. Fix the NULL default (3.1). The 1-liners are unavoidable with positional init (see 2). | high |
| Rebind guard (`ggml_backend_sched_rebind_events_complete`) and use in `alloc_splits` | ggml-backend.cpp | +60 / -2 (+ trace ~25 counted under E) | B | None | Always on. Fast rebind only when all copy events are query-able and complete. On backends without `event_query` it always falls back to reserve, so the fast path is effectively CUDA/meta only. | Keep the logic. Remove the trace (E). | med |
| Share-view inputs (`GGML_SCHED_SHARE_VIEW_INPUTS`) in `split_graph` | ggml-backend.cpp | +50 | C/F (unclear) | None | **Generic and default ON.** Applies to any cross-backend input that is a full-size view of a root tensor, including CPU<->GPU offload. | Review. Default off, or restrict to meta backends, unless measured on non-meta paths. | med |
| Scheduler timing instrumentation (`timing` struct, `timing_*` helpers, ~80 call sites, `GGML_SCHED_TIMING`, `_INTERVAL`) | ggml-backend.cpp | +190 | E (debug) | None | Env, default off. Costs a branch per call site when off. | Remove or move to a separate debug patch. | high |
| Input trace (`GGML_SCHED_TRACE_INPUTS`, `timing_trace_input`) | ggml-backend.cpp | +35 | E (debug) | None | Env, default 0 | Remove with the timing block. | high |
| Rebind event trace (`GGML_SCHED_REBIND_EVENT_TRACE`, `ggml_sched_rebind_event_trace_on`) | ggml-backend.cpp | +30 | E (debug) | None | Env, default off | Remove. | high |
| Sync non-graph inputs (`ggml_backend_sched_set_sync_non_graph_inputs`, `sync_non_graph_inputs` flag, new header `ggml-backend-sched-impl.h`) | ggml-backend.cpp, ggml-backend-sched-impl.h (new) | +25 | C (ROCm workaround) | None | Flag is set by llama-context for ANY backend when `n_ubatch < 64`, not just HIP. Env override `GGML_SCHED_SYNC_NONGRAPH` lives in `src/llama-context.cpp:810` (another slice). | Keep, but consider gating the policy in llama-context to HIP. | med |
| Meta-backend hooks in core (`ggml_backend_meta_graph_begin`, `_n_stages`, `_n_backends` used by sched) | ggml-backend.cpp, ggml-backend-impl.h | +15 | C (needed by meta) | None (meta is in base, these hooks are new) | Only acts when a meta backend is present | Keep. | high |
| Comm and dedicated-copy typedefs (`comm_sendrecv`, `cpy_tensor_async_dedicated_queue/drain`) | ggml/include/ggml-backend.h | +20 | C | None | Typedef only, zero runtime cost | Keep. Used by meta (other slice). | high |
| `ggml_backend_meta_device` gets a `tps` parameter | ggml/include/ggml-backend.h | +2 (signature) | C | Base signature lacks `tps` | Always. Callers in `src/llama.cpp` (other slice) updated. | Keep. | high |
| Scheduler split-input capacity knob `GGML_SCHED_MAX_SPLIT_INPUTS` (CMake, default 64) | ggml/CMakeLists.txt, ggml/src/CMakeLists.txt, ggml-backend.cpp:1181 | +2 | F | Base: `#ifndef` default 30 in code only, no CMake knob. | Always. Changes effective default from 30 to 64 for every build. | Confirm need (no vr doc evidence). If not needed, delete the CMake knob and keep the code default. | low-med |
| Allocator-owned `ggml_gallocr_alloc_graph_ids` decl | ggml/src/ggml-impl.h | +8 | C | None | - | Keep. | high |
| Removed `ggml_backend_sched_copy_input` (inlined into `compute_splits` with pass loop) | ggml-backend.cpp | +190 inline / -50 | C (restructure needed for staging) | Base has the separate function. | - | Keep. Restructure is inherent to staging. | med |
| `dominated`-skip in layout store (keeps worst-case layouts) | ggml-alloc.c | +25 | B | None | Always, with the cache | Keep. | med |
| Unused debug-only blank line removal before `// allocate graph` | ggml-backend.cpp ~2486 | -1 | cosmetic | - | - | Restore the blank line (diff noise). | high |

## 2. Why each backend needed a 1-line change

The 15 backend files are: blas, cann, cpu, et, hexagon, metal, opencl, openvino, rpc, sycl, virtgpu (`ggml-backend-device.cpp`), vulkan, webgpu, zdnn, zendnn. Each adds one `/* .event_query = */ NULL` or `nullptr` entry. Each is at the end of the `ggml_backend_device_i` initializer, right after `event_synchronize`.

- Reason: the initializers are positional aggregates. With one more member than entries, GCC and Clang warn under `-Wextra` (`-Wmissing-field-initializers`). `ggml/src/CMakeLists.txt:40` enables `-Wextra` for all ggml builds, and `-Werror` applies under `GGML_FATAL_WARNINGS` / `LLAMA_FATAL_WARNINGS`. So without the edit, fatal-warning builds would fail.
- Behavior: none. The missing trailing member is zero-initialized, which is NULL. The core already treats NULL as "no query".
- Designated initializers: would avoid the warning only if every backend were rewritten to designated form, which is a larger diff than the 15 lines. Not recommended.
- Edit-free option: store `event_query` behind `ggml_backend_reg_get_proc_address` (the pattern used for `comm_*` in `ggml-backend-meta.cpp`). Only CUDA and meta would export it. That removes all 15 edits but adds a lookup and a new pattern. Medium effort. Recommend only if the team wants zero backend churn.
- Not verified by build (per brief). The warning claim is from knowledge of GCC/Clang behavior. Confirm with one compile of `ggml-cpu.cpp` with and without the line.

## 3. Per-feature notes with evidence

### 3.1 Event query NULL default returns "ready" (correctness)

- `ggml/src/ggml-backend.cpp:964-972`: `ggml_backend_event_query()` returns `true` when `iface.event_query == NULL`.
- Callers: `src/llama-context.cpp:1699` and `:1738` (`layer_inp_accum_ready` and `layer_inp_accum_ready_epoch`). On a backend without `event_query` these report ready for work that has not finished, and the drafter could then read unfinished data.
- The scheduler is safe because it checks `query_supported` separately (`ggml-backend.cpp:2430-2444`).
- Recommendation: return `false` on NULL. In the two llama-context callers, `false` only disables the early-overlap path, which is the conservative behavior. Confidence high that the change is safe; medium that it is needed in practice (depends on which backends record `layer_inp_accum` events).

### 3.2 Layout cache with 2 buffers (correctness, needs review)

- `ggml/src/ggml-alloc.c`, `ggml_gallocr_layout_restore`: for `n_buffers > 2` with `key_ids == 0` it refuses. For 2 buffers it adopts any layout with a matching topology key.
- The topology key (`ggml_gallocr_graph_key`) excludes the buffer-id assignment and float shapes. The scheduler can assign the same topology differently (op support can depend on shape). A layout reserved under one assignment could then be adopted under another.
- The code comment (`ggml-alloc.c` around the `n_buffers > 2` check) says the 2-buffer case "cannot disagree". I could not confirm this from the code.
- Also relies on the invariant that buffers only grow (`ggml-alloc.c`, comment in `ggml_gallocr_reserve_n_impl`).
- Recommendation: either pass `key_ids` on the 2-buffer path too, or document the invariant with a check. Confidence: medium.

### 3.3 Staging defaults are meta-only (informational)

- `ggml_backend_sched_should_stage_input` requires `GGML_BACKEND_DEVICE_TYPE_META` (or the env flag). The non-graph path requires a META destination.
- F32 default 4096, F16 default 512 MiB, non-graph 32 KiB, all for meta only. I64 (64 B) and I32 (16 KiB) apply to any type but still need `backend_ok`.
- So the staging machinery is inert for CUDA-only, CPU, and other non-meta setups by default. Evidence: `vr/old-ignore/2026-10-29/prior-guides/FEATURES.md:322-337` (+69% prefill with `-sm tensor -tps 4`, +37% with layer ring depth).

### 3.4 Share-view inputs is generic and on

- `ggml-backend.cpp` `split_graph` block (~lines 2136-2175 of the new file). Gated only by `src_backend_id != cur_backend_id && !buffer_supported` plus its own conditions. Default on.
- No `vr/` doc records a measurement for it (only the patch copies under `vr/old-ignore/*/patches`). Affects CPU<->GPU offload too.
- Recommendation: default off, or limit to meta, unless a non-meta measurement exists.

### 3.5 Ring depth default

- `FEATURES.md:328-333` documents the change and that `GGML_SCHED_N_COPIES` overrides it. The upstream value was fixed at 4.
- Effective change is only for more than 4 GPUs. For the dual-GPU target it is unchanged.
- The CMake default `GGML_SCHED_MAX_COPIES` 4->16 costs only fixed-size arrays (`events[MAX_BACKENDS][16]`, `input_staging_events`). Keep.

### 3.6 Debug and trace code (cleanup)

- Timing: `struct ggml_backend_sched_timing` (~35 int64 fields), `timing_now/dt/reset/maybe_print/trace_input`, about 80 `sched->timing.*` updates in the hot loop (`ggml-backend.cpp` compute_splits). Off by default, but every call site is noise in review. No `vr/` doc cites timing output.
- Rebind event trace: `ggml_sched_rebind_event_trace_on` and the `REBIND-EVENT-SUMMARY` print in `ggml_backend_sched_rebind_events_complete`. Off by default.
- Realloc trace: `ggml_gallocr_realloc_trace` plus `GGML_LOG_INFO("[realloc] ...")` calls in `graph_fits`, layout store and restore. Off by default. Keep `graph_fits`; drop the logs.
- `sched->timing.enabled` also prints a line at scheduler creation (`ggml-backend.cpp`, `ggml_backend_sched_new`) only when enabled.

### 3.7 Dead code check

- `git grep` at HEAD: every new function in the slice has at least one caller. `ggml_backend_sched_backend_is_multi_meta`, `ggml_backend_sched_stage_simple_gpu_inputs`, `ggml_backend_sched_timing_trace_input`, `ggml_gallocr_alloc_graph_ids`, `ggml_backend_meta_n_stages`, `ggml_backend_meta_graph_begin` all have callers in the slice.
- `ggml_backend_event_query` is declared only in `src/llama-ext.h:219`, not in the public `ggml/include/ggml-backend.h`. It is defined in `ggml-backend.cpp` with a duplicate forward declaration at line 962. Recommend moving the declaration to the public header and dropping the duplicate. Cosmetic, low risk.
- `GGML_SCHED_DEBUG` (`ggml-backend.cpp` `ggml_backend_sched_new`) is pre-existing, not new.

## 4. getenv() calls added in the slice (with defaults)

All are read once (static or at sched creation), not per call. `GGML_SCHED_DEBUG` is pre-existing and excluded.

| Env var | File:function | Default | Effect |
|---|---|---|---|
| `GGML_SCHED_N_COPIES` | ggml-backend.cpp: `default_n_copies` | unset: `max(4, n_gpu)`, meta stages, cap 16; 1 if not parallel | Overrides ring depth (min 1) |
| `GGML_SCHED_TIMING` | ggml-backend.cpp: `ggml_backend_sched_new` | 0 | Timing instrumentation on |
| `GGML_SCHED_TIMING_INTERVAL` | same | 64 | Print interval in calls |
| `GGML_SCHED_TRACE_INPUTS` | same | 0 | Per-input trace lines, limit |
| `GGML_SCHED_INPUT_STAGING_MAX` | ggml-backend.cpp: `input_staging_max_size` | 16384 B; 0 disables all staging | Max size for I32 staging |
| `GGML_SCHED_STAGE_SIMPLE_GPU_INPUTS` | ggml-backend.cpp: `stage_simple_gpu_inputs` | off | Also stage on plain GPU devices |
| `GGML_SCHED_INPUT_STAGING_I64_MAX` | ggml-backend.cpp: `input_staging_type_max_size` | 64 B | Max size for I64 staging |
| `GGML_SCHED_INPUT_STAGING_F32_MAX` | same | meta: 4096 B; else 0 | Max size for F32 staging |
| `GGML_SCHED_INPUT_STAGING_F16_MAX` | same | meta: 512 MiB; else 0 | Max size for F16 staging |
| `GGML_SCHED_NON_GRAPH_INPUT_STAGING_MAX` | ggml-backend.cpp: `non_graph_input_staging_max_size` | meta: 32768 B; else 0 | Non-graph staging |
| `GGML_SCHED_SHARE_VIEW_INPUTS` | ggml-backend.cpp: `split_graph` (static) | on | Share root copy for view inputs |
| `GGML_SCHED_REBIND_EVENT_TRACE` | ggml-backend.cpp: `ggml_sched_rebind_event_trace_on` | off | Rebind guard trace |
| `GGML_GALLOC_LAYOUT_CACHE` | ggml-alloc.c: `ggml_gallocr_new_n` | on (0 disables) | Layout cache |
| `GGML_ALLOC_REALLOC_TRACE` | ggml-alloc.c: `ggml_gallocr_realloc_trace` | off | Realloc-reason logs |

Outside this slice, for reference: `GGML_SCHED_SYNC_NONGRAPH` in `src/llama-context.cpp:810`, default `n_ubatch < 64`.

Build-time (not env): `GGML_SCHED_MAX_COPIES` (CMake default 4->16), `GGML_SCHED_MAX_SPLIT_INPUTS` (CMake default 64, new), `GGML_SCHED_NO_REALLOC` (pre-existing).

## 5. Quick wins

Ordered by confidence. None changes the default behavior. Line numbers refer to the HEAD working tree.

1. Remove the timing instrumentation: `ggml/src/ggml-backend.cpp` struct `ggml_backend_sched_timing` (HEAD line ~1208), its `timing` member in `struct ggml_backend_sched`, the `timing_*` helpers (HEAD lines ~1362-1400), the `sched->timing.*` lines in `ggml_backend_sched_compute_splits`, and the timing setup in `ggml_backend_sched_new` (HEAD lines ~2840-2846). Safe because it only affects stderr output when `GGML_SCHED_TIMING` is set. Also removes ~190 lines. Confidence high.
2. Remove the rebind event trace: `ggml_sched_rebind_event_trace_on` (HEAD line 2406) and the `trace` branches in `ggml_backend_sched_rebind_events_complete` (HEAD lines 2419-2460). Keep the early `return false` path. Confidence high.
3. Remove the realloc trace logs (`GGML_LOG_INFO("[realloc] ...")` in `ggml/src/ggml-alloc.c`; `ggml_gallocr_realloc_trace` at HEAD line 1152 and its call sites at lines 1214, 1259, 1301, 1327). Keep `ggml_gallocr_graph_fits`. Confidence high.
4. Restore the blank line removed before `// allocate graph` in `ggml_backend_sched_alloc_splits` (`ggml-backend.cpp` ~2486). Cosmetic. Confidence high.
5. Change `ggml_backend_event_query` NULL default from `true` to `false` (`ggml-backend.cpp:967-968`). Safety fix; see 3.1. Confidence high that it is safe, medium that it matters in practice.
6. Move `ggml_backend_event_query` declaration into `ggml/include/ggml-backend.h` and drop the duplicate at `ggml-backend.cpp:962`. Confidence high. Low value.
7. Share-view inputs default: change `GGML_SCHED_SHARE_VIEW_INPUTS` default to off, or restrict the `share_root` condition to meta backends (`ggml-backend.cpp` ~line 2140). Behavior change for non-meta paths, so measure first. Confidence medium.
8. `GGML_SCHED_MAX_SPLIT_INPUTS` CMake knob (`ggml/CMakeLists.txt:189`, `ggml/src/CMakeLists.txt:5`). Only if no measured need for 64. Confidence low-medium.

Not quick wins (keep): the 15 backend one-liners (see section 2), the layout cache, the staging, the ring depth default, the sync-non-graph setter.
