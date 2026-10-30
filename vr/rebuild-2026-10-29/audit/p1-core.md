# Pass 1 report: core (p1-core)

Tree: /home/raistlin/infer/mx-rebuild. Not built or run.

| Item | Status | Where | Approx lines removed |
|---|---|---|---|
| Q10 scheduler timing (GGML_SCHED_TIMING, _INTERVAL, GGML_SCHED_TRACE_INPUTS) | done | ggml-backend.cpp: timing struct, timing member, timing_* helpers, trace_input, ggml_backend_sched_env_int (now unused), timing statements in compute_splits, setup and print in sched_new | ~180 |
| Q11 rebind event trace (GGML_SCHED_REBIND_EVENT_TRACE) | done | ggml-backend.cpp: ggml_sched_rebind_event_trace_on, trace counters and REBIND-EVENT-SUMMARY print. Early return false kept. Behavior with trace off is unchanged. | ~30 |
| Q12 realloc trace (GGML_ALLOC_REALLOC_TRACE) | done | ggml-alloc.c: ggml_gallocr_realloc_trace, all [realloc] logs. ggml_gallocr_graph_fits kept, but its unused `trace` parameter is removed (only caller is in this file). | ~55 |
| Q13 blank line before "// allocate graph" | done | ggml-backend.cpp | -1 (restored) |
| Q14 duplicate event_query declaration | skipped | ggml-backend.cpp:962 | 0 |

Q14 reason: line 962 is the only declaration of ggml_backend_event_query in ggml-backend.cpp. The other one is in src/llama-ext.h, a different translation unit. ggml/src/CMakeLists.txt:43 enables -Wmissing-declarations for C++, so removing line 962 would warn on the definition at line 964. Moving the declaration to ggml/include/ggml-backend.h is outside my file list. NULL return value unchanged (still true).

Symbols left for other files: none. Removed symbols were grepped across the tree (excluding vr/). No other users.

Note: the file-level diffs vs the staged overlay are ggml-backend.cpp (+8/-223) and ggml-alloc.c (+3/-55).
