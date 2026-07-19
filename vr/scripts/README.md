# gfx906 iteration tools

These tools implement a fail-fast path from a source-local kernel change to
one production-relevant model A/B. They default to the least expensive useful
action.

Any invocation likely to exceed five minutes requires explicit user permission
first. Give the exact command, expected duration, purpose, and side effects.
A `--dry-run` is inspection only and never authorizes the printed build or
benchmark.

| Tool | Purpose | Default work |
| --- | --- | --- |
| `build-gfx906-variant.sh` | Configure and build one private profile | `test-backend-ops` only |
| `check-gfx906-defines.py` | Verify private gfx906 definitions and source locations | Read-only |
| `export-graph-ops.sh` | Export exact model operation signatures | Reuse an exact cached export |
| `bench-ops.sh` | Focused operator correctness and performance A/B | No repeated correctness; one timing pass per build |
| `bench-head.sh` | Raw PP, incremental PP, or TG model A/B | One UD-Q4_K_XL raw-PP cell |
| `watch-bench.sh` | Read driver-owned status and partial results | Three-second display refresh |

## Immediate PP ablation

Build only the causal profiles:

```bash
for PROFILE in common q4 q6; do
  PROFILE="$PROFILE" vr/scripts/build-gfx906-variant.sh
done
```

Export the model graph once, then compare `common -> q4` and `common -> q6`
separately with `bench-ops.sh`. Do not build or time `combined` unless both
singletons win. The exact commands and acceptance rules are in
`../WORKTREE-BENCHMARKS.md`.

## Enforced efficiency rules

- `bench-head.sh` rejects `REPS` other than 1.
- Duplicate builds, configurations, modes, and output directories are
  rejected.
- Raw PP is one process. Incremental PP and TG at the same depth share another
  process, model load, and saved depth state.
- Raw PP skips llama-bench's duplicate full-prompt warmup by default.
  Incremental PP and TG retain only their cheap initialization warmups.
- Temperature is gated once on the used devices before an A/B pair, not before
  every build or on an idle card.
- Candidate runs second by default. Reverse the order once, in a new output
  directory, only for borderline evidence.
- The current definition-only ablation does not rerun correctness because the
  same bodies passed the 203-case gate. Set `CHECK_CORRECTNESS=1` for new body
  changes.
- Operator results are compared per shape. No unweighted mean is reported.
- Variant builds disable Vulkan, curl, the server, and examples, and compile
  only the requested target. Compiler caching remains enabled when available.
- `test-backend-ops` still uses its built-in warmup and one-second timing
  interval inside that single pass; these are measurement setup, not extra
  A/B samples.
- Model files are identified by path, size, and mtime during each run. Reuse
  the hashes already recorded in `../RESULTS.md` instead of rereading tens of
  gigabytes.

Always inspect `--dry-run` output before a model load. No helper commits,
pushes, or submits anything.

`build-gfx906-vanilla.sh` now refuses to label the private source tree as a
clean control. Use `build-gfx906-comparison.sh` for pinned `571d0d540`.
`build-gfx906-optimal.sh` remains only as a compatibility alias for the
retained `common` profile and no longer writes to the stale `optimal`
directory. Use `build-gfx906-variant.sh` with an explicit profile for new
work.
