# Catch up with official llama.cpp

Keep upstream interfaces and the smallest local delta that preserves the gfx906 improvements. Read [the current setup](README.md), [improvements](IMPROVEMENTS.md), `AGENTS.md`, and `skills/code-review/SKILL.md` first. Historical workflows under [old-ignore](old-ignore/README.md) are reference material, not instructions.

Keep the separate `play-qwen-off` output-style project out of this performance patch chain. The accepted grouped-MoE subset from `flashnext-fast` belongs in this patch chain, including its tests and dispatch/fallback guards. Its separate GPU-feedback experiment does not. Model priorities are listed in [the current guide](README.md#priorities-and-separate-work).

Current overlay base: official `36a0e2ffc0b352b774f8d71c43668b75b3061ef5`. The active [patch kit](patches/llama-head-36a0e2ffc/README.md) exports stage D of the [2026-10-29 rebuild](old-ignore/2026-10-29/rebuild/README.md). The previous kit ([`418f905a4`](old-ignore/2026-10-29/patches/llama-head-418f905a4/README.md)) and the older checkpoint (`84932540a0b8fbcd67128b8615da434763c185ec` on `344be0a60074e6395da3c4a4f300f7361df5a67d`) are historical; do not reuse their intermediate trees for a new upstream base. Use the reviewed current branch as the source of truth.

The rebuild removed multi-stage TP (`-tps`) and the fork-only DeepSeek V4/V4.1, MiniMax-M2 and MiMo-2 code. Do not restore them during a merge; take upstream for those models. See [the removed and changed list](IMPROVEMENTS.md#2026-10-29-rebuild-removed-and-changed).

## Prepare and merge

Start from a reviewed, clean tracked HEAD. The helper fetches official master, records both SHAs, creates detached `upstream` and `port` worktrees, and saves diffs in a new session directory. Untracked loose files stay untouched. Preserve the original build and its libraries for any approved baseline comparison.

```bash
vr/catchup-script.sh prepare /tmp/mx-catchup-session
# Preview only; use an existing local model.
vr/catchup-script.sh bench /tmp/mx-catchup-session /absolute/path/model.gguf
vr/catchup-script.sh merge /tmp/mx-catchup-session
# Resolve conflicts manually, then refresh reports.
vr/catchup-script.sh diff /tmp/mx-catchup-session
```

During this merge, ours is MX and theirs is new official upstream. Inspect both sides and automatically merged callers; do not choose whole files blindly. Prefer an upstream equivalent over retaining an obsolete local implementation. Check the saved `old-overlay.patch`, `upstream.patch`, `overlay.patch`, and `merge-changes.patch` to explain additions and removals.

Preserve these connected paths:

- Repack upload/layout, dispatch, fused consumers, gfx906 MMQ/MMVQ tuning, existing supported quants, and opt-in grouped MoE weight reuse for narrow IQ2_S/IQ4_NL batches.
- Preserve gfx906 row-first MMQ traversal when reconciling upstream launch-axis changes. The first 418f port changed this traversal and showed a large PP loss; the accepted tree restores the gfx906 ordering. The active narrow IQ2_S single-token expert path is limited to one output column and even rows; odd rows retain the prior fallback.
- Upstream HIP host-staged AllReduce plus the local graph-replay, duplex/slab overlap and scheduler integration: HIP must enter the first `#if !defined(GGML_USE_MUSA)` path in `ggml/src/ggml-cuda/allreduce.cu` with no environment switch. The peer-write AllReduce stays opt-in (`GGML_CUDA_PEER_AR=1`).
- Parallel lanes, shared-expert reduction, token graphs, and scheduler/readback dependencies; avoid new per-ubatch host synchronization.
- Split KV, recurrent rollback, MTP planning, and DFlash per-sequence state. Rounded reservations can need 2049 tap rows for ubatch 2048. Chunked GDN must write all requested snapshots with the correct output stride.
- Existing host PLE placement, wide-row GPU argsort, and graph update safety where still needed.

Reconcile loader/converter/tensor identifiers and graph/memory callers together. DFlash vision remains unvalidated. Ask before a large redesign or unclear tradeoff. Keep RCCL off for the current CPU-lane/motherboard-lane topology; revisit after the hardware changes. The unreachable legacy HIP overlap-only branch has been removed from `allreduce.cu`; do not restore it.

## Build and quick validation

All conflicts and interfaces must be resolved before building. Check both staged and unstaged source diffs with `git diff --check` and `git diff --cached --check`, excluding generated patch files and archived records. Inspect `git ls-files -u` and search runtime source for conflict markers.

In the port worktree, use the existing TheRock toolchain and the [README build command](README.md), replacing `build-ilp` with `build-catchup` and adding `-DLLAMA_BUILD_VR_EXPERIMENTS=ON`. Keep its `CMAKE_HIP_FLAGS` and `CMAKE_CXX_FLAGS`; without them 27B was 2-3% slower. Preserve relevant old cache options, including `GGML_HIP_NO_VMM=ON` and `GGML_HIP_MMQ_MFMA=ON`. Build only:

```bash
cmake --build build-catchup --target llama-cli llama-bench llama-server vr-test-gfx906-tp -j 8
# One shared runtime budget; build time is separate.
timeout --kill-after=5s 110s bash -eu -c '
  unset GGML_CUDA_PEER_AR
  ./build-catchup/bin/llama-cli --version
  ./build-catchup/bin/vr-test-gfx906-tp --n 1 --check
  ./build-catchup/bin/vr-test-gfx906-tp --n 2048 --check
'
```

Require successful exits and `check=pass` for both widths. Timeout or missing GPUs means incomplete validation. These checks sample TP matmul/reduction output; they do not prove model quality, speculative correctness, or unchanged performance. Runtime checks have a combined two-minute budget. Ask before longer checks or retries; do not run full suites or download models.

## Refresh the patch chain

Keep one current kit under `vr/patches/`; move superseded kits into a dated `vr/old-ignore/` campaign when the replacement is verified. Do not alter old patch payloads.

Stage the reviewed resolution, including required new files. The following exports the resolved index against the fetched official SHA. Its temporary intermediate tree leaves `src/llama-model.cpp` at upstream so patch 0005 carries that file's complete local delta. Reusing the old intermediate tree against a new base would risk restoring obsolete upstream code.

```bash
set -euo pipefail
SESSION_DIR=/tmp/mx-catchup-session
REPO=$(cat "$SESSION_DIR/source-repo")
PORT_DIR="$SESSION_DIR/port"
UPSTREAM_HEAD=$(cat "$SESSION_DIR/upstream-sha")
UPSTREAM_SHORT=$(git -C "$PORT_DIR" rev-parse --short=9 "$UPSTREAM_HEAD")
KIT="$REPO/vr/patches/llama-head-36a0e2ffc"
cd "$PORT_DIR"
test -z "$(git ls-files -u)"
git diff --exit-code
RESOLVED_TREE=$(git write-tree)
TASK_INDEX=$(mktemp /tmp/mx-kit-index.XXXXXX)
rm "$TASK_INDEX"
GIT_INDEX_FILE="$TASK_INDEX" git read-tree "$RESOLVED_TREE"
MODEL_BLOB=$(git rev-parse "$UPSTREAM_HEAD:src/llama-model.cpp")
GIT_INDEX_FILE="$TASK_INDEX" git update-index --add --cacheinfo "100644,$MODEL_BLOB,src/llama-model.cpp"
MID_TREE=$(GIT_INDEX_FILE="$TASK_INDEX" git write-tree)
rm "$TASK_INDEX"
# Unique ref names also support a same-base refresh without overwriting history.
SNAP_SUFFIX="$UPSTREAM_HEAD-$RESOLVED_TREE"
for entry in "target:$RESOLVED_TREE" "mid:$MID_TREE"; do
    git update-ref "refs/port-snapshots/gfx906-$SNAP_SUFFIX-${entry%%:*}" "${entry#*:}" ''
done
NEXT_KIT="$PORT_DIR/vr/patches/llama-head-$UPSTREAM_SHORT"
if test -e "$NEXT_KIT"; then NEXT_KIT="${NEXT_KIT}-refresh"; fi
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

Review the full upstream-to-target file list against `MANIFEST.tsv` and `EXCLUDED.txt`. The current exporter excludes root `conversion/`; add required runtime/conversion paths to its grouping before regenerating. Replay checks only manifest files, so zero mismatches alone cannot establish completeness. If a group becomes empty or a source path moves, adapt the existing exporter minimally rather than adding dummy changes.

Document the new base, intermediate/target tree IDs and refs, build/check results, and limitations in its README. Update the active kit links here and in `vr/README.md`. The patches must replay without local snapshot refs; those refs are needed to retain the trees for regeneration. Offline export/replay is separate from the GPU runtime budget.

## Performance and handoff

Benchmarks require explicit user OK for the exact baseline/candidate commands and combined time limit. The helper's `bench` command previews PP2048/TG128 at depth 0, batch/ubatch 2048, full offload, flash attention, 50/50 TP, warmup and one repetition. Only after approval add `--run-approved`. Pass the candidate binary as the optional third positional argument for the post-merge run. Each invocation has a 110-second timeout plus 5 seconds to terminate. For this host, the requested common CPU-thread setting is `-t 20 -tb 20`; it is a comparison setting, not a claim that 20 threads are optimal.

Use the same immutable model files, GPUs, environment and settings. The helper records commands, binary hash, library paths, cache and output, but cannot prove a binary matches a source tree. Check build identity and keep each binary with its own libraries. Do not compare different context depths or workloads as if matched. Diagnose a regression before proposing tuning.

Leave the merge and kit uncommitted. Report worktree path, official SHA, meaningful delta changes, build/smoke/replay results, and untested behavior. Do not claim preserved performance without measurements. Do not commit, push, or publish without the user's separate instruction.
