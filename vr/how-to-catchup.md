# Catch up with official llama.cpp

Keep upstream interfaces and the smallest local delta that preserves the gfx906 improvements. Read [the current setup](README.md), the [feature index](IMPROVEMENTS.md#merge-maintenance-index), `AGENTS.md`, and `skills/code-review/SKILL.md` first. Read detailed evidence only for affected features; do not load archived investigations on every merge. Historical workflows under [old-ignore](old-ignore/README.md) are reference material, not instructions.

Keep the separate `play-qwen-off` output-style project out of this performance patch chain. The accepted grouped-MoE subset from `flashnext-fast` belongs in this patch chain, including its tests and dispatch/fallback guards. Its separate GPU-feedback experiment does not. Model priorities are listed in [the current guide](README.md#priorities-and-separate-work).

Current overlay base: official `36a0e2ffc0b352b774f8d71c43668b75b3061ef5`. The active [patch kit](patches/llama-head-36a0e2ffc/README.md) exports stage D of the [2026-10-29 rebuild](old-ignore/2026-10-29/rebuild/README.md). The previous kit ([`418f905a4`](old-ignore/2026-10-29/patches/llama-head-418f905a4/README.md)) and the older checkpoint (`84932540a0b8fbcd67128b8615da434763c185ec` on `344be0a60074e6395da3c4a4f300f7361df5a67d`) are historical; do not reuse their intermediate trees for a new upstream base. Use the reviewed current branch as the source of truth.

The rebuild removed multi-stage TP (`-tps`) and the fork-only DeepSeek V4/V4.1, MiniMax-M2 and MiMo-2 code. Do not restore them during a merge; take upstream for those models. See [the removed and changed list](IMPROVEMENTS.md#2026-10-29-rebuild-removed-and-changed).

## Prepare and merge

Use the normal checkout. The human first commits and pushes the reviewed current state to the personal fork, including any local work that must be kept. Confirm the push succeeded and start with a clean tracked working tree. Agents do not commit or push. Keep unrelated untracked files untouched.

Keep the previous successful build in its original directory with all its shared libraries. Do not rebuild it, copy only its executable, or rename it: build paths can be embedded in library lookup. Set `BASE` to that build; as of 2026-11-11 it is `build-mslot-ring` (`build-play-b1` is kept as the previous one). Build each candidate in a new directory and keep the last two successful builds for quick comparisons. After acceptance, that candidate becomes the next merge's baseline. Record its path and source identity with the results.

Run from the repository root in Bash, keeping these variables for the later steps:

```bash
# One-time repository setup. Git invokes rerere automatically during merges.
git config rerere.enabled true
git config rerere.autoupdate false

OLD_HEAD=$(git rev-parse HEAD)
BASE="$PWD/build-mslot-ring" # The last accepted build; update after each accepted merge.
CAND="$PWD/build-catchup-$(date +%Y%m%d-%H%M%S)"
OUT="$CAND/validation"
mkdir -p "$OUT"
printf '%s\n' "$OLD_HEAD" > "$OUT/baseline-source.txt"
printf '%s\n' "$BASE" > "$OUT/baseline-build.txt"

git fetch upstream
UPSTREAM_HEAD=$(git rev-parse upstream/master)
printf '%s\n' "$UPSTREAM_HEAD" > "$OUT/upstream-source.txt"
git merge --no-commit --no-ff "$UPSTREAM_HEAD"
```

If the merge stops, review reused resolutions and resolve the remaining conflicts. Run `git rerere` to record the resolutions, then `git add` only the resolved paths and required new files. Review automatically merged callers too; rerere cannot detect semantic errors. Prefer upstream equivalents over obsolete local code and take new model support from upstream. Do not choose whole files blindly.

The worktree-based `catchup-script.sh` remains an optional alternative; it is not required for this routine.

A background agent session cannot write to the normal checkout. Create a worktree from the reviewed HEAD under `.claude/worktrees/catchup-<upstream-short>` on a branch of the same name, and run the merge there. Keep `BASE` and `CAND` as absolute paths under the main repository root, so the builds survive when the worktree is removed. The human commits the merge from that branch.

Haiku-class agents are fine for triage and additive hunks. MMQ launch, recurrent rollback and DFlash/MTP state need a stronger model.

Preserve these connected paths:

- Repack upload/layout, dispatch, fused consumers, gfx906 MMQ/MMVQ tuning, existing supported quants, and default-on grouped MoE weight reuse for narrow IQ2_S/IQ4_NL batches.
- Preserve gfx906 row-first MMQ traversal when reconciling upstream launch-axis changes. The first 418f port changed this traversal and showed a large PP loss; the accepted tree restores the gfx906 ordering. The active narrow IQ2_S single-token expert path is limited to one output column and even rows; odd rows retain the prior fallback.
- Upstream HIP host-staged AllReduce plus the local graph-replay, duplex/slab overlap and scheduler integration: HIP must enter the first `#if !defined(GGML_USE_MUSA)` path in `ggml/src/ggml-cuda/allreduce.cu` with no environment switch. The peer-write AllReduce stays opt-in (`GGML_CUDA_PEER_AR=1`).
- Parallel lanes, shared-expert reduction, token graphs, and scheduler/readback dependencies; avoid new per-ubatch host synchronization.
- Split KV, recurrent rollback, MTP planning, and DFlash per-sequence state. Rounded reservations can need 2049 tap rows for ubatch 2048. Chunked GDN must write all requested snapshots with the correct output stride.
- Existing host PLE placement, wide-row GPU argsort, and graph update safety where still needed.

Reconcile loader/converter/tensor identifiers and graph/memory callers together. DFlash vision remains unvalidated. Ask before a large redesign or unclear tradeoff. Keep RCCL off for the current CPU-lane/motherboard-lane topology; revisit after the hardware changes. The unreachable legacy HIP overlap-only branch has been removed from `allreduce.cu`; do not restore it.

## One final validation pass

Finish the merge and fixes before building. Run one final pass; repeat only affected checks if something fails or subsequent code changes affect the result. Intermediate checks are for diagnosing a specific problem.

Check staged and unstaged source diffs with `git diff --check` and `git diff --cached --check`, excluding generated patches and archives. Require no entries from `git ls-files -u` and inspect runtime source for conflict markers. Confirm HIP still compiles the default host-staged AllReduce path without an environment switch.

Configure and build once with the existing comparison flags. Leave `BASE` untouched.

```bash
source /home/raistlin/amd-clean/env/therock.sh
cmake -S . -B "$CAND" \
  -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=OFF \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON \
  -DLLAMA_OPENSSL=ON -DLLAMA_BUILD_VR_EXPERIMENTS=ON \
  -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release \
  -DHIP_COMPILER=clang -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  -DCMAKE_HIP_FLAGS="-mllvm -amdgpu-sched-strategy=max-ilp" \
  -DCMAKE_CXX_FLAGS="-O3 -Wno-unused-command-line-argument" \
  > "$OUT/configure.log" 2>&1
# Continue only if configuration succeeds.
cmake --build "$CAND" --target llama-cli llama-completion llama-server llama-bench vr-test-gfx906-tp -j 8 \
  > "$OUT/build.log" 2>&1
```

The binaries find their libraries through a RUNPATH set to the build directory at build time. `build-ilp` was built as `/home/raistlin/infer/mx-llama.cpp/build-ilp`, and that directory no longer exists. Run its binaries with `LD_LIBRARY_PATH="$BASE/bin"`; this loads the libraries stored with that build. Do not move or rename later builds, so they do not need this.

Continue only after a successful build. Record the resolved source and build settings:

```bash
git diff --binary HEAD > "$OUT/candidate-changes.patch"
cp "$CAND/CMakeCache.txt" "$OUT/CMakeCache.txt"
sha256sum "$BASE/bin/llama-bench" "$CAND/bin/llama-bench" > "$OUT/binaries.sha256"
ldd "$BASE/bin/llama-bench" > "$OUT/baseline-libraries.txt"
ldd "$CAND/bin/llama-bench" > "$OUT/candidate-libraries.txt"
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP

timeout --kill-after=5s 110s bash -eu -c '
  "$1/bin/vr-test-gfx906-tp" --n 1 --check
  "$1/bin/vr-test-gfx906-tp" --n 2048 --check
' bash "$CAND" > "$OUT/tp.log" 2>&1
```

Require exit zero and `check=pass` at both widths. These synthetic tests need no model or `-hf`: Q8 weights and F32 activations contain ones, the dot-product dimension is split across ROCm0/ROCm1, and the reduced output should be 17408. The harness executes the graph twice and checks only the first 16 output values of the final result, within an absolute tolerance of 2. It is a narrow matmul/reduction smoke test, not full model or speculative correctness coverage. Its `time_us` is not model PP/TG performance.

The two TP checks share the existing two-minute runtime budget (110 seconds plus termination allowance). Timeout, missing GPUs or missing pass output means incomplete validation. Generation and benchmarks below require explicit user OK for the exact commands and combined time limit; no standing budget has been agreed. Do not download models or run full suites. Ask before longer checks or retries.

After approval, run one short Qwen3.8-27B Q8_0 generation and matched PP2048/TG128 benchmarks. Both builds must resolve `-hf` to the same cached model files; keep the cache unchanged between runs. `--offline` prevents downloads. Use the same environment and no competing GPU workload.

```bash
# llama-cli is chat-only and rejects -no-cnv; plain completion is llama-completion.
timeout --kill-after=5s 110s "$CAND/bin/llama-completion" \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline \
  -ngl 999 -dev ROCm0,ROCm1 -sm tensor -ts 1,1 -fa on \
  -b 2048 -ub 2048 -c 4096 -t 20 -tb 20 \
  --temp 0 --top-k 1 -s 42 -n 32 -no-cnv --simple-io --no-display-prompt \
  -p 'The history of the printing press begins in' \
  < /dev/null > "$OUT/generation.txt" 2> "$OUT/generation.log"
# Continue only if generation succeeds; inspect the saved text.
for label in baseline candidate; do
  if [ "$label" = baseline ]; then BUILD="$BASE"; else BUILD="$CAND"; fi
  timeout --kill-after=5s 110s "$BUILD/bin/llama-bench" \
    -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline \
    -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa on \
    -b 2048 -ub 2048 -t 20 -p 2048 -n 128 -d 0 -r 1 -o json \
    > "$OUT/$label.json" 2> "$OUT/$label.log" || {
      BENCH_STATUS=$?
      printf '%s\n' "$BENCH_STATUS" > "$OUT/$label.exit"
      exit "$BENCH_STATUS"
    }
  printf '0\n' > "$OUT/$label.exit"
done
```

These are three separate runtime invocations, each capped at 110 seconds plus 5 seconds to terminate; this is not a claim they fit the TP checks' two-minute budget. The benchmarks use warmup and one repetition. Require successful exits and complete JSON for both builds. One pair is a quick regression screen, not repeatability evidence. Add other models or targeted speculative/MoE checks only when the affected feature needs them and the user approves their budget.

Report baseline and candidate PP2048 and TG128 in one table, with tokens/second and percentage change `(candidate / baseline - 1) * 100`. Keep the JSON, stderr, generation text, TP pass/fail and source/build identities in `OUT`. Compare measured results from this session; historical numbers are context, not a matched baseline. Mark skipped checks as not run and performance as not measured if either benchmark fails. Diagnose regressions before tuning; no automatic regression threshold has been agreed.

## Optional patch kit export

Git merges are the primary update mechanism. Skip this section during routine weekly merges. Export a kit when a portable snapshot or release checkpoint is needed; export and replay are not required before accepting a validated merge. A kit describes its recorded base and target, and can lag behind the branch. Record the new upstream base in the merge handoff even when no kit is exported.

Keep one current kit under `vr/patches/`; move superseded kits into a dated `vr/old-ignore/` campaign when the replacement is verified. Do not alter old patch payloads.

Stage the reviewed resolution, including required new files. The following exports the resolved index against the fetched official SHA. Its temporary intermediate tree leaves `src/llama-model.cpp` at upstream so patch 0005 carries that file's complete local delta. Reusing the old intermediate tree against a new base would risk restoring obsolete upstream code.

```bash
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
# UPSTREAM_HEAD is the pinned SHA from the merge step.
UPSTREAM_SHORT=$(git rev-parse --short=9 "$UPSTREAM_HEAD")
KIT="$REPO/vr/patches/llama-head-36a0e2ffc"
cd "$REPO"
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
NEXT_KIT="$REPO/vr/patches/llama-head-$UPSTREAM_SHORT"
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

## Handoff and human push

Report the old and new upstream bases, meaningful overlay changes, candidate and baseline build paths, TP results, generation result, side-by-side PP/TG table, patch replay results if exported (otherwise say export skipped), log directory, and untested behavior. Keep this short; link detailed logs instead of pasting them. Do not claim preserved performance without measurements.

Leave agent work uncommitted. After reviewing the passing final checks, the human commits and pushes the merge to the personal fork. Record that accepted commit with the candidate's validation logs. Retain this build and the previous successful build in their original directories for quick comparisons; use this candidate as `BASE` next week. Never rebuild the retained baseline during the next merge. Agents do not commit, push, or publish.
