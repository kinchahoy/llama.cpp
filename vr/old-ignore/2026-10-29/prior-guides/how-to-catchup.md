> Historical record. Commands and recommendations may be obsolete; see [the current guide](../../../README.md).

# Catch up with llama.cpp: handoff to Luna

Keep the delta from official llama.cpp small while retaining the useful MX/gfx906 improvements. Finish the merge, build it, run only the quick checks below, and refresh the replayable patch chain. Do not add features or tune performance during the catch-up.

## Starting point and authority

- Checkpoint: `84932540a0b8fbcd67128b8615da434763c185ec`.
- Merged official llama.cpp commit: `344be0a60074e6395da3c4a4f300f7361df5a67d`, 2026-10-25, `ci : update CPU job assignments (#31136)`. This is the checkpoint's second parent.
- Active historical kit: `vr/patches/llama-head-344be0a60/`. Its target is an earlier snapshot, not the complete checkpoint. In particular, preserve the checkpoint's later attention/vector-dot fixes. Use the checkpoint branch as the source of truth.
- Loose root text outputs were deliberately excluded from the commit. Leave them alone.
- The user authorized this checkpoint commit only. Leave the next merge and refreshed kit uncommitted for review. Do not push, publish, create a PR, or write reviewer responses.
- Runtime checks have a combined two-minute wall-clock budget after the build. Do not run full test suites, performance sweeps, long generation, downloads, or repeated experiments. Ask for explicit OK before any longer checks. Performance comparisons also require explicit OK; any approved comparison must stay short and focus on merge regressions.

Read `AGENTS.md`, `CONTRIBUTING.md`, `skills/code-review/SKILL.md`, `vr/QUICKSTART.md`, `vr/MERGE-GFX906.md`, and the active kit README before editing. This handoff's test limits override older instructions to benchmark automatically. Ask the user before a large redesign or a preservation tradeoff whose intent is unclear.

## Standard pre-run helper

Use `vr/catchup-script.sh` for routine setup. It operates on the current clean tracked HEAD, so verify that HEAD is the intended checkpoint first. Untracked loose outputs stay untouched. `prepare` fetches official master and creates two detached worktrees: `upstream` at that exact official SHA, and `port` at the current MX HEAD. It copies this handoff and the helper into the port and saves source identities, upstream history, the old overlay, incoming upstream changes, and current port diffs outside both worktrees.

```bash
vr/catchup-script.sh prepare /tmp/mx-catchup-session
vr/catchup-script.sh bench /tmp/mx-catchup-session /absolute/path/model.gguf
# After explicit user approval, repeat the printed command via:
vr/catchup-script.sh bench /tmp/mx-catchup-session /absolute/path/model.gguf --run-approved
vr/catchup-script.sh merge /tmp/mx-catchup-session
# Resolve conflicts manually, then refresh reports:
vr/catchup-script.sh diff /tmp/mx-catchup-session
```

The bench action previews by default. It uses a local model, PP2048/TG128, depth 0, batch/ubatch 2048, full offload, flash attention, both ROCm GPUs, 50/50 tensor split, warmup enabled and one repetition per test. Each approved invocation has a 110-second timeout plus 5 seconds for termination. A baseline/candidate pair requires approval for both invocations and their combined budget. No performance benchmark has been pre-approved by writing this helper.

The default benchmark binary is the original `build-ilp/bin/llama-bench`; pass the candidate's absolute binary path as the third positional argument to preview the matched post-merge run. Keep the original build and library directory intact. Each actual run saves its command, JSON, stderr, exit status, selected environment, binary hash, linked library paths, CMake cache when present, and source/port state in a unique session subdirectory. Model path and file metadata are recorded without hashing all weights; verify the same immutable model files, including all shards, are used for both runs. The helper records available provenance but cannot prove an existing binary was built from the checkpoint; check its build identity before approving the comparison.

After using the helper, restore the variables used by the remaining instructions:

```bash
SESSION_DIR=/tmp/mx-catchup-session
REPO=$(cat "$SESSION_DIR/source-repo")
CHECKPOINT=$(cat "$SESSION_DIR/source-sha")
UPSTREAM_HEAD=$(cat "$SESSION_DIR/upstream-sha")
UPSTREAM_SHORT=$(git -C "$REPO" rev-parse --short=9 "$UPSTREAM_HEAD")
PORT_DIR="$SESSION_DIR/port"
KIT="$REPO/vr/patches/llama-head-344be0a60"
cd "$PORT_DIR"
```

Skip the manual fetch/worktree/merge commands below; continue with source reconciliation, building, bounded correctness checks, and kit refresh. The helper does not build, resolve conflicts, export patches, commit, or publish. To retain the detached port later, create a named branch without committing. Leave worktrees and session logs in place for review.

## Fetch and merge the exact official head

Run commands from the repository root in bash. Stop on any failure. Do not use `git pull`, reset the working tree, or apply the entire overlay on a branch that already contains it.

```bash
set -euo pipefail
REPO=/home/raistlin/infer/mx-llama.cpp
cd "$REPO"
CHECKPOINT=84932540a0b8fbcd67128b8615da434763c185ec
OLD_BASE=344be0a60074e6395da3c4a4f300f7361df5a67d
KIT="$REPO/vr/patches/llama-head-344be0a60"
test -z "$(git ls-files -u)"
! git rev-parse -q --verify MERGE_HEAD >/dev/null
# Inspect tracked changes; this handoff and loose outputs may be untracked.
git status --short
git diff --exit-code
git diff --cached --exit-code
git merge-base --is-ancestor "$OLD_BASE" "$CHECKPOINT"
git fetch https://github.com/ggml-org/llama.cpp.git master
UPSTREAM_HEAD=$(git rev-parse FETCH_HEAD)
UPSTREAM_SHORT=$(git rev-parse --short=9 "$UPSTREAM_HEAD")
git show -s --format=fuller "$UPSTREAM_HEAD"
PORT_DIR="/tmp/mx-catchup-$UPSTREAM_SHORT"
test ! -e "$PORT_DIR"
git worktree add -b "catchup/llama-$UPSTREAM_SHORT" "$PORT_DIR" "$CHECKPOINT"
cd "$PORT_DIR"
```

Record `UPSTREAM_HEAD` immediately; subsequent fetches overwrite `FETCH_HEAD`. Keep these variables in the same shell, or restore them from the handoff record. Verify the fetched head descends from `OLD_BASE` with `git merge-base --is-ancestor "$OLD_BASE" "$UPSTREAM_HEAD"`; stop and investigate if it does not. If the fetched head equals `OLD_BASE`, skip the merge and only refresh the stale kit. Otherwise run:

```bash
git merge --no-commit --no-ff "$UPSTREAM_HEAD"
```

A conflict exit is expected when interfaces changed. Inspect `git status --short` and `git ls-files -u`; resolve all conflicts before proceeding. Stage resolved files individually. During this merge stage 1 is the common base, stage 2 is MX, and stage 3 is new official upstream. Do not select ours/theirs for whole files. Check automatically merged code in the same call chains. Use upstream interfaces and port only the behavior still needed; drop a local hunk only after identifying an upstream replacement and recording the reason.

Compare `git diff "$UPSTREAM_HEAD" -- src ggml common include tools tests CMakeLists.txt` to the old checkpoint delta against `OLD_BASE`. Explain additions or removals. Do not resurrect obsolete upstream code to make an old patch apply.

## Behaviors to retain

Use the source map and conflict table in `vr/MERGE-GFX906.md`. Preserve these as connected paths, including dispatch and allocation, rather than isolated kernels:

- gfx906 repack upload/layout, buffer registration, canonical versus repacked consumers, Q8 MMQ/MMVQ tuning, and supported Q4/K-quant paths.
- Normal HIP host-staged AllReduce. In `ggml/src/ggml-cuda/allreduce.cu`, HIP must still compile the first `#if !defined(GGML_USE_MUSA)` path. Do not add a HIP exclusion or require `GGML_CUDA_TP_OVERLAP=1`. Keep RCCL off.
- Concurrent TP lanes, shared-expert sharding/reduction, whole-token graph replay, fused operations, and readback/scheduler dependencies. Do not introduce per-ubatch host synchronization.
- Split KV views/copies, recurrent rollback/snapshot stride and depth, MTP planning, persistent layer taps, deferred prompt handling, and DFlash per-sequence state.
- Host PLE/CPU Engram placement, wide-row GPU argsort, graph function-update safety, and the checkpoint's masked attention/vector-dot corrections.

Retain existing upstream architecture support and reconcile converter, tensor identifiers, loader, graph, memory, and API callers together. DFlash vision remains unvalidated; do not claim this catch-up fixes it. Document genuinely redundant patches removed through upstream equivalents. Avoid unrelated cleanup.

## Source checks, build, and two-minute runtime limit

Before building, require no unmerged entries, inspect the full runtime diff, and check whitespace outside generated patch payloads:

```bash
test -z "$(git ls-files -u)"
git diff --check -- . ':!vr/patches/**'
git diff --cached --check -- . ':!vr/patches/**'
rg -n '^(<<<<<<< |=======$|>>>>>>> )' src ggml common include tools tests CMakeLists.txt || test "$?" -eq 1
```

Inspect hits: do not mistake intentional test fixtures for merge markers. Generated `.patch` context lines can trigger whitespace warnings; do not strip their spaces. Do not waive actual source whitespace errors.

Use the existing TheRock toolchain and the options in `vr/QUICKSTART.md`. Configure a separate build in the port worktree so the old binary remains available for an approved comparison. Record the compiler/ROCm version and CMake cache. Preserve any additional material settings from the old `build-ilp/CMakeCache.txt`, including `GGML_HIP_NO_VMM=ON` and `GGML_HIP_MMQ_MFMA=ON` on this checkpoint.

```bash
cmake -S . -B build-catchup \
  -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=OFF \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON \
  -DLLAMA_OPENSSL=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release \
  -DHIP_COMPILER=clang -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  -DCMAKE_HIP_FLAGS="-mllvm -amdgpu-sched-strategy=max-ilp" \
  -DCMAKE_CXX_FLAGS="-O3 -Wno-unused-command-line-argument" \
  -DLLAMA_BUILD_VR_EXPERIMENTS=ON
cmake --build build-catchup --target llama-cli llama-bench llama-server vr-test-gfx906-tp -j 8
```

If the toolchain is unavailable, report that rather than substituting an unverified compiler or fetching a new SDK. Build time is separate from the two-minute runtime-check budget. On a build failure fix the smallest merge-related cause and rebuild the affected targets.

Run this bounded correctness smoke once with both GPUs idle. It uses the existing harness to exercise Q8 TP matmul, reduction and residual output at decode and prefill widths. Its small sampled output check is limited coverage, and its printed timings are not a performance comparison.

```bash
timeout --kill-after=5s 110s bash -eu -c '
  unset GGML_CUDA_TP_OVERLAP
  ./build-catchup/bin/llama-cli --version
  ./build-catchup/bin/vr-test-gfx906-tp --n 1 --check
  ./build-catchup/bin/vr-test-gfx906-tp --n 2048 --check
'
```

The shared timeout leaves room within the two-minute budget for termination. Treat timeout, nonzero exit, missing GPUs, or absent `check=pass` as incomplete validation. Do not restart the whole check repeatedly without asking. If changed ops need a different focused check, propose replacing one of these within the same total budget; do not append an unbounded `test-backend-ops` suite. A successful build and this smoke do not establish full model/speculative correctness or unchanged performance.

## Refresh the complete patch chain

Keep historical kits immutable. Export against the exact fetched official commit and the fully resolved index, including post-checkpoint fixes. The existing exporter uses four file-disjoint groups plus a final model integration patch. Do not reuse its old intermediate tree against a new upstream base: that can reintroduce old upstream code.

Stage only reviewed runtime files and needed new files. Check `git diff --exit-code` so no unstaged runtime edits are omitted. Snapshot before writing kit metadata. The following constructs a new intermediate tree equal to the target except that `src/llama-model.cpp` comes from the new upstream base; patch 0005 then contains the complete local delta for that file. This changes the grouping, not the final replayed tree.

```bash
RESOLVED_TREE=$(git write-tree)
SNAP_REF="refs/port-snapshots/gfx906-$UPSTREAM_HEAD"
MID_REF="refs/port-snapshots/gfx906-mid-$UPSTREAM_HEAD"
# Never overwrite existing snapshot refs. Pick a documented revision suffix if needed.
! git show-ref --verify --quiet "$SNAP_REF"
! git show-ref --verify --quiet "$MID_REF"
git update-ref "$SNAP_REF" "$RESOLVED_TREE"
TASK_INDEX=$(mktemp /tmp/mx-kit-index.XXXXXX)
rm "$TASK_INDEX"
GIT_INDEX_FILE="$TASK_INDEX" git read-tree "$RESOLVED_TREE"
MODEL_BLOB=$(git rev-parse "$UPSTREAM_HEAD:src/llama-model.cpp")
GIT_INDEX_FILE="$TASK_INDEX" git update-index --add --cacheinfo "100644,$MODEL_BLOB,src/llama-model.cpp"
MID_TREE=$(GIT_INDEX_FILE="$TASK_INDEX" git write-tree)
rm "$TASK_INDEX"
git update-ref "$MID_REF" "$MID_TREE"
NEXT_KIT="$PORT_DIR/vr/patches/llama-head-$UPSTREAM_SHORT"
# A same-base refresh needs a new directory, preserving the old kit.
if test -e "$NEXT_KIT"; then
    NEXT_KIT="${NEXT_KIT}-checkpoint-refresh"
fi
test ! -e "$NEXT_KIT"
mkdir -p "$NEXT_KIT"
cp "$KIT/regenerate.py" "$KIT/verify.py" "$NEXT_KIT/"
python3 - "$NEXT_KIT/regenerate.py" "$UPSTREAM_HEAD" "$MID_TREE" "$RESOLVED_TREE" <<'PY'
import re
import sys
from pathlib import Path
path = Path(sys.argv[1])
source = path.read_text()
for key, value in zip(("DEFAULT_BASE", "DEFAULT_MID", "DEFAULT_TARGET"), sys.argv[2:]):
    source, count = re.subn(rf'^{key}\s*=.*$', f'{key} = "{value}"', source, flags=re.M)
    assert count == 1, key
path.write_text(source)
PY
python3 "$NEXT_KIT/regenerate.py"
python3 "$NEXT_KIT/verify.py"
(cd "$NEXT_KIT" && sha256sum -c SHA256SUMS)
```

The offline export/replay checks are separate from GPU runtime tests. The exporter rejects empty groups: if upstream fully absorbs a group, adapt the existing exporter/verifier minimally and document the reduced chain rather than manufacturing a dummy delta. If `src/llama-model.cpp` moves, adapt its replacement path before constructing the intermediate tree.

Inspect `MANIFEST.tsv` and `EXCLUDED.txt`. In particular the exporter currently excludes root `conversion/` Python files and other paths outside its grouping. If a required local conversion/build/runtime change appears there, extend the existing grouping and regenerate. Review the entire new upstream-to-target name list independently: a verifier that checks only manifest paths cannot detect missing manifest entries. Require every needed runtime path to be covered and all manifest paths to replay with zero mismatches. Check all final local differences, including mid-to-target paths, not just the exclusions file.

Write the new kit README with exact base, intermediate and target tree IDs; snapshot refs; ordered patch list; upstream replacements that removed local hunks; build/check commands and results; toolchain; and remaining limitations. Update `vr/README.md` and active-kit links in `vr/MERGE-GFX906.md` together. Keep this handoff available in the new worktree. Historical kits remain historical. Tree objects alone are local; patches must replay in a fresh full-history official clone without the snapshot refs, and future regeneration needs the documented target snapshot or accepted commit.

## Performance requires the user's explicit OK

Do not claim no regression from compilation, the harness timing, or old measurements. Before benchmarking, send the exact proposed commands, model path, input hash, baseline identity, and time bound to the user and wait for explicit OK. Preserve the pre-merge binary and its library directory; do not accidentally run a new binary against old shared libraries or the reverse.

Propose one baseline/candidate pair with identical local Qwen3.8-27B Q8_0 weights, 50/50 TP split, devices, environment, graphs, flash attention, batch/ubatch, context depth, warmup and repetitions. Start from a short PP2048/TG128 `llama-bench` pair with one repetition, within an explicitly approved wall-clock bound. Use the installed binary's help for supported options, include its exact command in the approval request, and avoid downloads. Only repeat or add a representative CLI check if the user authorizes it. Compare the immediate pre-merge baseline; keep historical PP/TG references separate. If performance falls, diagnose the dispatch/AllReduce change before proposing tuning.

Finish with the worktree/branch path, official SHA, diff summary, dropped redundant hunks, build and quick-check results, kit replay/checksum results, and any untested behavior. Leave the reviewed changes ready for the user to inspect and commit. State plainly when performance has not been measured.
