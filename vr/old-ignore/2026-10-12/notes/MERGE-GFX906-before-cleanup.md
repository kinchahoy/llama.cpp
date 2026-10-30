> Historical record. Commands and recommendations may be obsolete; see [the current guide](../../../README.md).

# Merging llama.cpp heads for MI50/MI60 (gfx906)

Historical record. Earlier recommendations and task deadlines describe the recorded tree; use the [current vr guide](../../../README.md) for operating instructions. Exact pre-cleanup bytes are preserved in `before-cleanup.tar.gz`.

Short procedure: [active patch kit README](../patches/llama-head-08ffba21e/README.md). This page is the long reference.

Keep the accepted MX runtime overlay when moving to a new ggml-org/llama.cpp head. The active kit is linked in [README.md](../../../README.md); `OLD-IGNORE/patches` is archived and must not replace it. Resolve all source interfaces before configuring, building, or testing. A text-clean merge is not evidence that the fast kernels still dispatch.

## Inject into a future upstream head

Use this route for a clean upstream checkout. Use a normal upstream merge when the destination already contains the MX overlay; do not apply the complete kit twice. The five patches are one runtime overlay grouped by file ownership, so applying only the GPU patch is insufficient. They include model/API integration beyond the gfx906 kernels.

Set these three absolute paths. `LLAMA_REPO` is an existing Git checkout; `PORT_DIR` must not exist. The new detached worktree isolates the port from any work in the original checkout. These commands fetch the official head, retain its exact commit, ensure the kit's upstream base objects are available, and stop at the first patch conflict. They do not build, benchmark, or commit.

```sh
KIT=/home/raistlin/infer/mx-llama.cpp/vr/patches/llama-head-08ffba21e
LLAMA_REPO=/path/to/llama.cpp
PORT_DIR=/path/to/llama-gfx906
(
    set -e
    test ! -e "$PORT_DIR"
    (cd "$KIT" && sha256sum -c SHA256SUMS)
    git -C "$LLAMA_REPO" fetch https://github.com/ggml-org/llama.cpp.git master
    UPSTREAM_HEAD=$(git -C "$LLAMA_REPO" rev-parse FETCH_HEAD)
    KIT_BASE=08ffba21e5c2f1728b6b405c05ff3913f65706d4
    if ! git -C "$LLAMA_REPO" cat-file -e "$KIT_BASE^{commit}"; then
        git -C "$LLAMA_REPO" fetch https://github.com/ggml-org/llama.cpp.git "$KIT_BASE"
    fi
    git -C "$LLAMA_REPO" worktree add --detach "$PORT_DIR" "$UPSTREAM_HEAD"
    for patch in "$KIT"/patches/*.patch; do
        printf 'Applying %s\n' "$patch"
        git -C "$PORT_DIR" apply --3way "$patch"
    done
)
```

The upstream head is saved before the optional base fetch, which also replaces `FETCH_HEAD`. No MX target tree object is needed to apply the patches. When using a refreshed kit, set `KIT_BASE` to that kit's documented upstream commit. Use an ordinary full-history clone or fetch the needed history if a shallow clone lacks the base blobs.

### Resolve and resume

Read the failing patch name, `git -C "$PORT_DIR" status --short`, and `git -C "$PORT_DIR" ls-files -u`. Resolve each affected call chain using the preservation rules and conflict table below. Inspect stages with `git -C "$PORT_DIR" show :1:path/to/file`, and likewise `:2:` and `:3:`.

| Operation | Stage 1 (base) | Stage 2 (ours) | Stage 3 (theirs) |
| --- | --- | --- | --- |
| Applying this kit with `git apply --3way` | Kit's upstream base | New upstream source | Base plus the MX patch |
| Merging upstream into an MX branch | Merge base | Existing MX source | New upstream source |

The meaning of ours and theirs reverses between these workflows. Reconcile the source behavior; do not choose a side for every conflict. Stage only the resolved files with `git -C "$PORT_DIR" add -- path/to/resolved-file`. Check that the index has no unmerged entries before continuing to the next patch.

If `0002-ggml-hip-gfx906-kernels.patch` landed with conflicts, finish its resolution and then apply only the remaining patches:

```sh
(
    set -e
    test -z "$(git -C "$PORT_DIR" ls-files -u)"
    for name in 0003-llama-model-runtime.patch 0004-common-tools-tests-harness.patch 0005-tp-token-graph-fusions-ar-overlap.patch; do
        git -C "$PORT_DIR" apply --3way "$KIT/patches/$name"
    done
)
```

Adjust the remaining list to the patch that actually stopped. Do not replay a patch that already landed. A missing-blob or rejected-hunk error with no unmerged entries needs separate inspection; it does not mean the failing patch was installed. Fetch missing base objects or adapt the rejected change before proceeding. Keep all remaining builds and tests deferred until every patch and its API reconciliation is complete.

Review the resulting overlay with `git -C "$PORT_DIR" diff HEAD --stat` and `git -C "$PORT_DIR" diff HEAD -- path/to/file`. Review automatically merged changes too. Then check `git diff --check`, `git diff --cached --check`, conflict markers, and stale tensor/API identifiers inside the port worktree. Only after this source review should you build and measure. Use the kit's pinned command to reproduce its historical result and the [current CLI workflow](../../../README.md#benchmark-workflow) to assess the real workload.

## Standard workload and performance priority

Optimize Q8_0 first, then preserve Q4_0 and Q4_K. Use the [current benchmark workflow](../../../README.md#benchmark-workflow): the trusted quick check is the real CLI prompt with 256 output tokens, and longer CLI generation provides the strongest confirmation. Explore batch/ubatch 2048/2048 and 4096/4096 separately. PP and TG are reported in that order.

The older pinned kit used a synthetic PP2048/TG256 workload with full GPU offload, flash attention, tensor split, batch/ubatch 4096/4096 and populated depth 16384. Its reference was over 400 PP and about 30 TG tokens/s on two gfx906 GPUs. The one-repetition/no-warmup and single-final-run instructions were limits for those port tasks, not a general performance-validation policy. Preserve that exact command when reproducing those records; do not compare its numbers directly with the current CLI workload. Agree on any broader sweep before running it.

Carry both the immediate pre-port result and the older performance reference into the next kit. The September 10 reference was 414.24 PP / 30.28 TG; the recorded checkpoint before the October 1 merge was 431.13 / 27.81, and the resolved October 1 run was 424.57 / 28.86. The latest run improved TG over the immediate checkpoint but remains 1.42 TG (4.7%) below the older reference. Do not describe that historical TG gap as recovered. These runs are historical comparisons with different repetition counts; the cause of the remaining gap is unproven.

## Preserve the performance paths

- Keep `q8_repack/` upload layout, buffer registration, per-lane buffer selection, dense/MoE dispatch, single-token fused FFN dispatch, narrow-batch dispatch, and occupancy thresholds together. Canonical kernels cannot consume the repacked two-plane layout. Preserve the checks on both gate and up weights before fusing. `--no-repack` disables this path.
- Keep both host and device gfx906 MMQ selectors in `mmq.cuh`, the 512-thread Q8_0 configuration in `mmq-config-gfx906.cuh`, and the J > 64 occupancy/MoE gate. Keep the GCN MMVQ table and the special five-column Q8 topology. Q4_0 and Q4_K continue through the canonical tuned paths; do not widen the Q8-only repack predicates to other quants.
- HIP must enter the first `#if !defined(GGML_USE_MUSA)` branch of `allreduce.cu`. This branch contains the host-staged pipeline used by normal tensor parallelism. A later `#elif defined(GGML_USE_HIP)` branch contains an overlap-only experiment, but HIP already matched the first branch. Do not infer the active path from that later branch or add `!defined(GGML_USE_HIP)` to the first guard. Normal execution must not require `GGML_CUDA_TP_OVERLAP=1`.
- Preserve concurrent per-lane graph dispatch, repack-aware allocation, shared-expert sharding, and its output reduction. Keep the scheduler dependencies and persistent tap readback events; adding host synchronization to each ubatch can erase the PP gain.
- Keep noncontiguous large-row argsort on the GPU. The padded row must fit shared memory; the 64 KiB gfx906 limit allows 16384 indices. An old 1024-column gate sends sparse attention through synchronous CPU transfers. Retain wide-row strided kernels and 64-bit byte addressing.

Use this source map when upstream moves an integration point:

| Behavior | Current source locations |
| --- | --- |
| Repack layout, upload, supported shapes and dense/MoE consumers | `ggml/src/ggml-cuda/q8_repack/`; registration, dispatch and fusion in `ggml-cuda.cu`; source globs in CUDA and HIP `CMakeLists.txt` |
| Q8 and canonical Q4 MMQ/MMVQ selection | `mmq-config-gfx906.cuh`, `mmq-config-rdna2.cuh`, `mmq.cuh`, `mmvq.cu`, `mmq-load-tiles.cuh`, `vecdotq.cuh` |
| Normal HIP host-staged AllReduce and communicator dispatch | `allreduce.cu`, `allreduce.cuh`, `ggml-cuda.cu`, `vendors/hip.h` |
| Tensor-parallel lane buffers, concurrent issue and reductions | `ggml/src/ggml-backend-meta.cpp`; shared-expert placement in `src/llama-model.cpp` |
| Split KV helpers, tap readback dependencies and snapshots | `src/llama-graph.cpp`, `src/llama-context.cpp`, `src/llama-memory-recurrent.cpp`, corresponding headers and model helpers |

GPU filenames in this table are relative to `ggml/src/ggml-cuda/`. Preserve each behavior at its new upstream location rather than copying an obsolete interface back into place. A successful build alone cannot prove that repack or AllReduce executes.

## Resolve interfaces as units

Use the three-way base and inspect both sides of each conflict. Avoid choosing ours or theirs across a complete file merely because one hunk concerns performance. Inspect automatic merges in the same call chain as well.

| Conflict | Typical resolution |
| --- | --- |
| Meta backend inactive AllReduce lanes | Keep MX lane indexing and dispatch. Take upstream `GGML_OP_FILL` zeroing and the nonempty guard; multiplying NaNs by zero does not clear them. Translate upstream lane variables to the MX loop variable. |
| Attention training bypass | Keep upstream direct K/V reads for training. For inference retain MX `build_cpy_k/v` and `build_get_k/v` helpers so split KV buffers use the correct views and copies. |
| Layer-input and NextN extraction | Carry upstream full-token batch-index mapping through the MX extraction signature with `llama_ubatch`. Return whether ordinary rows were copied, keep deferred position/sequence accumulation and readback events, and reorder unmasked rows independently of selected logits. Retain support for wider layer taps instead of adding an unconditional n_embd-width assertion. |
| Recurrent memory | Preserve the MX snapshot ring, validity depths, convolution windows, and rollback handling. Carry upstream empty-memory detection so a draft with no recurrent layers disables snapshots. |
| MTP planning and GDN snapshots | Keep the [fixes incorporated into the main patch series](../../2026-10-01/patches/llama-head-e358d-to-mx-20261001/README.md#mtp-planning-and-snapshot-correctness). Enable target extraction before warmup and complete target/draft reservations before prompt processing. The custom chunked GDN kernel must store every requested trailing snapshot and use the supplied output slot stride; its input has one state per sequence, not K. Preserve the chunked GDN/KDA and direct-cache regression cases. A fast PP run does not prove rollback correctness. |
| Tensor identifiers | Rename C++ enum, tensor names/roles, layer fields, Python enums, mappings, and model tensor lists together. Keep GGUF wire names stable and old source aliases where needed; remove duplicate mappings and declarations. |
| Qwen4Exp QSA/MTP refactors | Reconcile loader, converter, graph declarations, trunk graph, separate MTP graph, and k-pool memory access together. Retain MX convolution snapshot helpers. Accept old split fc_embedding/fc_hidden GGUF projections and new concatenated eh_proj tensors. Permit trunk-only compress-ratio arrays with a zero default for missing MTP entries, but honor explicit new MTP ratios. |
| Application batch API | Use upstream `common_batch` and `llama_process` interfaces while keeping MX prompt chunking and separate final-token handling. |
| Added architectures | Combine upstream architecture conditions with MX-only architectures rather than replacing either list. Keep helper declarations consistent with all implementations. |

## Finish and export

Once all source reconciliation is complete, check for conflict markers, stale identifiers, and whitespace errors. Configure the existing gfx906 Release/HIP build with RCCL off and build the benchmark tools. Compare pre-merge and post-merge binaries with identical `llama-bench` settings, then confirm with the [CLI workflow](../../../README.md#benchmark-workflow) within the requested test budget. Record actual values and the environment. Compare with the recorded pre-merge checkpoint, and identify that comparison as historical if no fresh baseline was requested. A Q8 run does not measure Q4 performance or validate every optional model path.

Export a new kit against the exact upstream commit. Keep the old kit as a pinned reference. Runtime groups are file-disjoint and must all apply; inspect `EXCLUDED.txt` for omitted source paths. A Git tree snapshot can pin an uncommitted resolution without creating a commit. Replay the exported patches on a clean upstream archive and compare their runtime files with the snapshot before treating the kit as reproducible. Leave commits and publication to the user.

### Create the next reusable kit

Run this after the source has been reviewed and the requested benchmark checks have finished. First stage any remaining reviewed source edits, including new files; `write-tree` captures only the index. The clean-worktree route leaves `HEAD` at the exact upstream base; these commands require all source edits to be staged and all conflicts resolved. For an existing MX branch merge, replace the `UPSTREAM_HEAD` assignment below with its saved upstream commit instead of `HEAD`.

```sh
(
    set -e
    test -z "$(git -C "$PORT_DIR" ls-files -u)"
    git -C "$PORT_DIR" diff --quiet
    UPSTREAM_HEAD=$(git -C "$PORT_DIR" rev-parse HEAD)
    UPSTREAM_SHORT=$(git -C "$PORT_DIR" rev-parse --short=8 "$UPSTREAM_HEAD")
    RESOLVED_TREE=$(git -C "$PORT_DIR" write-tree)
    git -C "$PORT_DIR" update-ref "refs/port-snapshots/gfx906-$UPSTREAM_HEAD" "$RESOLVED_TREE"
    NEXT_KIT="$PORT_DIR/vr/patches/llama-head-$UPSTREAM_SHORT"
    test ! -e "$NEXT_KIT"
    mkdir -p "$NEXT_KIT"
    cp "$KIT/regenerate.py" "$KIT/verify.py" "$NEXT_KIT/"
    python3 - "$NEXT_KIT/regenerate.py" "$UPSTREAM_HEAD" "$RESOLVED_TREE" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
source = path.read_text()
for key, value in zip(("DEFAULT_BASE", "DEFAULT_TARGET"), sys.argv[2:]):
    source, count = re.subn(rf'^{key} = "[^"]+"$', f'{key} = "{value}"', source, flags=re.M)
    assert count == 1, key
path.write_text(source)
PY
    python3 "$NEXT_KIT/regenerate.py"
    python3 "$NEXT_KIT/verify.py"
    (cd "$NEXT_KIT" && sha256sum -c SHA256SUMS)
)
```

The exporter expects the kit under `vr/patches/<kit-name>/` in the target checkout. Updating the defaults matters: otherwise a later invocation without arguments silently exports the old snapshot. The snapshot ref retains the uncommitted tree without making a commit. Patch replay is an offline source check, separate from the single GPU performance run.

Review `MANIFEST.tsv` and `EXCLUDED.txt`; if an excluded path is required runtime code, update the existing export grouping and regenerate before calling the kit complete. Write the new kit's README with its upstream commit, resolved tree/ref, changed interfaces, exact build/benchmark command and environment, raw result paths, both historical references, remaining performance gaps, and unmeasured quant/model paths. Copy the merge guide into a new checkout if it is not already there. Update `vr/README.md` to point at the new kit only after accepting the port; retain the older kits and measurements.

## Copyable agent handoff

Send this with the active kit and this guide available in the workspace:

```text
Port the active gfx906 patch kit linked from vr/README.md to the current official ggml-org/llama.cpp head. Follow vr/MERGE-GFX906.md. Use an isolated clean worktree for injection; use an upstream merge if MX is already installed. Apply all five patch groups. Prioritize MI50/MI60 Q8_0 performance, then preserve Q4_0 and Q4_K. Retain repack layout and dispatch, gfx906 MMQ/MMVQ tuning, normal HIP host-staged AllReduce without a TP-overlap flag, parallel lane dispatch, shared-expert sharding and scheduler dependencies. Reconcile all conflicts and automatically merged interfaces before any build or test. Then build llama-bench and llama-cli. Compare the pre-port and post-port binaries with identical llama-bench settings and confirm with the CLI workload in vr/README.md; use longer generation when the test budget permits. Keep historical depth-16384 results separate from the current CLI workload. Report the immediate baseline and the older 30.28 TG reference separately; do not declare the historical gap recovered without evidence. Export and document a new pinned kit for the next port. Leave changes uncommitted and do not push or submit a PR.
```

## 2026-10-01 reconciliation

Upstream head: `e358d59178377be4c58ba567925e05faadbccb57`. Pre-merge MX head: `8535054eb819d4a4a3faf267a238df792ebc2f4f`. The merge has 15 conflicting files and retains the active runtime overlay. Q8 repack files, gfx906 MMQ configuration, and AllReduce are byte-identical to the pre-merge source. Upstream CUDA/MUSA/BF16 changes are carried alongside the gfx906 dispatch. The conflict decisions follow the table above.

The final single standard run measured 424.57 PP2048 / 28.86 TG256 tokens/s at depth 16384. The gfx906 Release benchmark target built successfully. The new port kit contains the measured source overlay and raw results, and its 114 runtime files replay exactly against the resolved tree. See the link in [README.md](../../../README.md).


## 2026-10-11 reconciliation

Pre-merge MX head: `5d6e2c4121e73ff39ac54802e33093ba16fe7752`. Upstream head: `08ffba21e5c2f1728b6b405c05ff3913f65706d4`. This is an ordinary upstream merge with local conflict reconciliation. The October 1 patch kit remains a pinned historical snapshot; it does not reproduce this tree.

The merge keeps the HIP host-staged AllReduce, Q8 repack kernels, gfx906 MMQ configuration, activation cache, parallel scheduler staging, recurrent snapshot ring, and deferred draft prefill. Upstream changes include the mixed token/embedding batch API, speculative rejection sampling, MoE weight-copy callback, MMQ precision/padding changes, and DFlash output-head sharing correction. The Q8 repack buffer interface adds the two new optional interface slots. The recurrent graph keeps the local explicit copy inputs and gather path. Wide TOP_K keeps stable tie selection while using upstream row chunking. GDN keeps upstream kernels on other devices and the previous one-column/two-warp generation launch on gfx906; the local chunked snapshot kernel stays in use for its supported cases.

### Build and measurements

Build on the two-MI60 server with the same ILP flags as the pre-merge build:

```sh
cmake -S . -B build-ilp \
  -DGGML_HIP=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_RCCL=OFF \
  -DLLAMA_OPENSSL=ON \
  -DAMDGPU_TARGETS=gfx906 \
  -DCMAKE_BUILD_TYPE=Release \
  -DHIP_COMPILER=clang \
  -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  -DCMAKE_HIP_FLAGS="-mllvm -amdgpu-sched-strategy=max-ilp" \
  -DCMAKE_CXX_FLAGS="-O3 -Wno-unused-command-line-argument"
cmake --build build-ilp --config Release -j16
```

The full build passes with ROCm SDK Clang 24. The comparison below uses the same command before and after the merge, with the default AllReduce and overlap unset:

```sh
env -u GGML_CUDA_TP_OVERLAP build-ilp/bin/llama-bench \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline \
  -ngl 999 -fa on -sm tensor -dev ROCm0/ROCm1 \
  -b 2048 -ub 2048 -p 2048 -n 256 -r 1 -o jsonl --no-warmup
```

| Tree | PP2048 tokens/s | TG256 tokens/s |
| --- | ---: | ---: |
| Pre-merge | 499.66 | 31.54 |
| Merged with upstream scalar GDN launch | 482.94 | 30.43 |
| Merged with gfx906 GDN launch preserved | 506.89 | 30.01 |

These are single measurements without warmup, at depth 0. The final measurement is +1.4% PP and -4.8% TG relative to the fresh pre-merge baseline. Restoring the gfx906 GDN launch recovers prompt throughput in this check; it does not demonstrate recovery of generation throughput. Raw records are in [results/2026-10-11-head-merge](../../2026-10-11/head-merge). The merged binary still reports the pre-merge build commit because the merge is not committed when built.

The requested CLI workload (`-f flake.nix -st -b 2048 -ub 2048 -c 150000 --spec-type none --spec-draft-n-max 2 --temp 0.0 --top-k 1 -s 42 -n 256 --simple-io`, with the same model, devices, tensor split and flash attention) measures 452.2 PP / 29.7 TG. The user's earlier 461.5 / 30.8 is historical, rather than a fresh paired CLI baseline. The synthetic depth-0 benchmark and CLI prompt/context numbers are separate workloads. No Q4, MTP, DFlash, vision, or long-context scaling performance claim is made from these runs.

### Validation and remaining limits

- Existing GPU GDN/KDA reference cases: 45/45 pass after final kernel reconciliation, including snapshot and direct-cache cases.
- Existing wide TOP_K reference cases: 76/76 pass, including tied values and infinities.
- Existing gfx906 tensor-parallel harness: `--n 1 --check` and `--n 2048 --check` pass with `GGML_CUDA_TP_OVERLAP` unset; both report `allreduce=default overlap=off wire=f32`.
- Allocator, batch allocator, and server component CTest cases pass.
- The CTest model-generation fixture fails on missing `deepseek41.expert_feed_forward_length`, so the dependent recurrent test does not run through CTest. Direct execution against the generated Qwen3.5 dense, Qwen3.5 MoE, and Qwen3Next fixtures passes seven checks and skips two unsupported Qwen3Next rollback checks with normal scheduler allocation.
- The strict `GGML_SCHED_DEBUG_REALLOC=1` diagnostic aborts on an unexpected graph reallocation for the Qwen3.5 dense fixture. This allocation diagnostic remains unresolved; the normal-allocation replay checks do not establish that graph reservations are optimal.
- The upstream `vendor/nlohmann/json-deep-nesting.patch` contains trailing spaces in patch context. Preserve that vendored patch verbatim; the other staged paths pass the whitespace check.

### DFlash2 vision follow-up

The linked [vision-position fix](https://github.com/z-lab/llama.cpp-fork/pull/4/changes/46d78b7fa5de4a6526c9b950a15d48682285f1d7) is not included in this merge. It gives injected target feature rows consecutive draft-cache positions and starts the next draft block after the draft cache's maximum position. This avoids copying repeated vision M-RoPE positions directly into a one-dimensional draft cache.

A separate follow-up patch is needed for this fork. Its DFlash path has deferred feature accumulation keyed by target positions, skips pinned image chunks, and uses target positions for draft-block/rollback alignment. The merged mixed-batch API also permits tokens and embeddings together, which the current DFlash process guard skips. Applying the old patch alone would leave these cases unresolved. Keep target feature row identity separate from draft-cache positions throughout eager injection, deferred flush, drafting and rollback, then validate a real screenshot request and its text continuation against non-speculative output. The upstream DFlash output-head sharing correction carried by this merge addresses a different problem. DFlash2 vision remains unvalidated and should not be treated as fixed by this update.
