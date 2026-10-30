# Repository cleanup, 2026-10-29

Final branch: `master`. Cleanup was prepared from checkpoint `84932540a`, whose merged official base is `344be0a60074e6395da3c4a4f300f7361df5a67d`. The newer `418f905a4` port remains on `experiment/catchup-418f905a4` because its matched prompt-processing result regressed.

Historical branches use `archive/YYYY-MM-DD/topic`; dates are tip commit dates, not inferred benchmark dates. New unfinished work should use `experiment/topic`. Only identical backup aliases were removed; every unique branch tip was retained.

| Former branch | Archived branch |
| --- | --- |
| `backup/2026-09-old-local-master` | `archive/2026-08-13/old-local-master` |
| `backup/2026-09-old-remote-master` | `archive/2026-01-06/old-remote-master` |
| `backup/pre-mx-llama-sync-2026-09-29` | `archive/2026-09-10/pre-mx-sync` |
| `cleanup/01-mx-upstream-sync` | `archive/2026-09-09/01-mx-upstream-sync` |
| `cleanup/02-rocm-host-allreduce` | `archive/2026-09-09/02-rocm-host-allreduce` |
| `cleanup/03-mxxm-repack-moe` | `archive/2026-09-09/03-mxxm-repack-moe` |
| `personal/04-vr-bench-harness` | `archive/2026-09-09/04-vr-bench-harness` |
| `cleanup/mx-perf-wins` | `archive/2026-09-10/mx-perf-wins` |
| `cleanup/mx-upstream-sync` | `archive/2026-09-10/mx-upstream-sync` |
| `mx-llama-26-08-15-q8enhancements` | `archive/2026-08-30/q8-enhancements` |
| `mx-upstream-2026-09-01` | `archive/2026-09-01/mx-upstream` |
| `mx-upstream-2026-09-09` | `archive/2026-09-10/mx-upstream` |
| `sync/mx-and-llama-heads-2026-09-29` | `archive/2026-10-01/mx-and-llama-heads` |

Clean historical worktrees moved under `.worktrees/archive/YYYY-MM-DD/topic`. Their ignored build directories were retained. CMake caches and binaries can contain old absolute paths; reconfigure before rebuilding, and verify library resolution before using them as benchmark references. The original root `build-ilp` and external `mx-base` worktree were not moved.

`flashnext-fast` was preserved in place as unfinished Flash Next performance work. `play-qwen-off` was preserved as a separate Qwen wordiness-reduction project, outside the gfx906 performance patch chain. Neither is a discarded historical experiment. Their local changes were not merged or discarded. The stale registration for the missing `/tmp/luna-llama-head-0bc845d35` worktree was pruned.

The duplicate aliases `backup/old-local-master` and `backup/old-remote-master` pointed at the same commits as their dated counterparts. Those commits remain under the archive names above.
