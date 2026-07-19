# gfx906 A/B runbook

Last updated: 2026-07-19

This is the minimal validation path for a raw kernel or dispatch change. The
production goal is fast 25B-35B agentic inference on one MI60 plus one MI50 at
long context, not merely the highest isolated pp8192 number.

Do not start any command likely to take more than five minutes without explicit
user permission. Before asking, provide the exact command, expected duration,
why it is the cheapest decision-relevant gate, and whether it builds, loads a
model, changes files, or only measures. Dry runs and source/static inspection
do not authorize the real long-running command.

## Build matched trees

Build clean upstream and one current candidate from `571d0d540`:

```bash
MAINLINE_REF=571d0d540 \
TARGETS=llama-bench \
GGML_VULKAN=OFF \
CANDIDATE_BUILD="$PWD/build/gfx906-merge-review" \
CMAKE_EXTRA_ARGS="-DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF" \
vr/scripts/build-gfx906-comparison.sh
```

The helper now defaults to pinned `571d0d540` and builds only
`test-backend-ops` unless `TARGETS` is set. Keep the explicit ref when copying
commands into a result record. Do not benchmark the existing
`build/gfx906-optimal` directory. Its compile database still contains disabled
experiment definitions.

Current review outputs:

```text
build/head-control/bin/llama-bench
build/gfx906-merge-review/bin/llama-bench
```

Build only `llama-bench` for screening. Add `test-backend-ops` when focused
correctness or the final gate is due.

## Current PP ablation

The causal full-model baseline is a private `common` build with the retained
Q4_K MMVQ change and Q8_0 dispatch, but neither unresolved PP definition.
Clean upstream is still useful as an absolute reference. This distinction
does not change isolated PP Q4_K/Q6_K code paths, but it matters for a model
that also contains Q8_0 tensors.

`GGML_HIP_GFX906_PROFILE` now selects the existing source-local definitions.
It does not change kernel bodies and defaults to `combined` to preserve the
pre-audit source behavior. The exact matrix is:

| Build | `mmvq.cu` | Q4_K MMQ instance | Q6_K MMQ instance |
| --- | --- | --- | --- |
| common | `GGML_CUDA_MMVQ_Q4K_GFX906_BRANCHLESS_SCALES` | none | none |
| q4 | `GGML_CUDA_MMVQ_Q4K_GFX906_BRANCHLESS_SCALES` | `GGML_CUDA_MMQ_Q4K_GFX906_PRECOMPUTE` | none |
| q6 | `GGML_CUDA_MMVQ_Q4K_GFX906_BRANCHLESS_SCALES` | none | `GGML_CUDA_MMQ_Q6K_GFX906_MIN_BLOCKS_1` |
| combined | `GGML_CUDA_MMVQ_Q4K_GFX906_BRANCHLESS_SCALES` | `GGML_CUDA_MMQ_Q4K_GFX906_PRECOMPUTE` | `GGML_CUDA_MMQ_Q6K_GFX906_MIN_BLOCKS_1` |

Q8_0 dispatch is identical source logic in all four private builds. Leave
`GGML_CUDA_GFX906_Q8_MMQ_MAX_NE11` unset during this ablation. The Q4_K PP
definition is one bundle; do not describe q4-only as a precompute-versus-stride
test.

The matching portable source compositions are
`patches/retained-gfx906.series`, `patches/q4-gfx906.series`,
`patches/q6-gfx906.series`, and `patches/combined-gfx906.series`. The retained
series maps to profile `common`. Do not select a profile whose numbered body
patch is absent.

Build only the three causal inputs first. Unchanged objects are recovered from
the compiler cache, and only `test-backend-ops` is requested:

```bash
for PROFILE in common q4 q6; do
  PROFILE="$PROFILE" vr/scripts/build-gfx906-variant.sh
done
```

Do not build `combined` yet. The helper runs
`check-gfx906-defines.py`, which checks the exact macro set and source location
and rejects every other private `GGML_CUDA_*GFX906*` definition. It catches the
stale `optimal` directory. Profile `none` is not clean upstream because the
private Q8_0 dispatch is source logic. Separate directories prevent an old
object from masquerading as a new variant.

## Cheap correctness

The current Q4_K and Q6_K kernel bodies already passed the 203-case focused
gate in the combined build. Switching the same translation-unit-local
definitions off and on does not justify repeating correctness for each
ablation build.

For a new kernel-body change, request one exact-shape candidate correctness
pass from `bench-ops.sh` with `CHECK_CORRECTNESS=1`. It rejects a missing ROCm0
backend and a zero-case filter. Use the broader focused command only when the
changed code is shared across types:

```bash
source vr/scripts/setup-therock-env.sh
BUILD=build/gfx906-final
"$BUILD/bin/test-backend-ops" test \
  -b ROCm0 -o MUL_MAT \
  -p 'q4_0|q4_1|q4_K|q5_K|q6_K|q8_0' -j 1
```

Use a narrower type or shape filter while iterating. Plain `MUL_MAT` does not
exercise fused gate/value graph paths; test those separately before retaining
a fusion change.

Never accept a run that reports the ROCm backend as skipped. The reduced
candidate passed 203 of 203 cases with the command above.

## Exact model-shape replay

The built-in performance shapes are generic and do not match Qwen3.6-27B.
Export the real PP and TG graph without loading weight data:

```bash
cmake --build build/head-control --target test-export-graph-ops -j"$(nproc)"

MODEL="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/ac393bc3d23fd5a929a85e2f33c7c4fd5be02d43/Qwen3.6-27B-UD-Q4_K_XL.gguf"
OPS=/tmp/qwen36-ud-q4-k-xl-pp8192-ub512-ops.txt

MODEL="$MODEL" OUT="$OPS" vr/scripts/export-graph-ops.sh

OPS="$OPS" TYPES=q4_K N=512 \
CONTROL_BUILD=build/gfx906-ablate-common \
CANDIDATE_BUILD=build/gfx906-ablate-q4 \
OUT=vr/bench-results/gfx906-op-q4-pp-2026-07-18 \
vr/scripts/bench-ops.sh

OPS="$OPS" TYPES=q6_K N=512 \
CONTROL_BUILD=build/gfx906-ablate-common \
CANDIDATE_BUILD=build/gfx906-ablate-q6 \
OUT=vr/bench-results/gfx906-op-q6-pp-2026-07-18 \
vr/scripts/bench-ops.sh
```

Choose new dated output names when these commands are actually run.
`bench-ops.sh` performs one timed pass per build, gates temperature once before
the pair, compares latency row by row, and rejects the pair if any selected
shape loses. It does not compute an unweighted aggregate. Set
`ORDER="candidate control"` only for one deliberate reversed check in a new
output directory.

The checked export reported 57 unique PP signatures; TG then added 48
signatures not already present, for a 105-signature union. The second number
is not the total unique TG count. The file preserves tensor types, dimensions,
strides, and operation parameters. It deduplicates across layers and phases,
does not use the node name as part of the key, and stores neither phase nor
call frequency. The exporter is CPU-only; `-ngl` has no effect.

For PP8192, dense matmul runs at physical microbatch size `n=512`, not
`n=8192`. Single-sequence TG uses `n=1`; context depth 8192 changes attention
and KV work, not dense weight matrix shapes. Use `n=2/4/8` only for batched
decode or parallel sequences.

The TG reserve graph does not contain a populated depth-8192 KV cache. Replay
is exact for dense `n=1` shapes, but it cannot measure the growing attention,
KV, or graph-scheduling share at long context.

The old Q4_K_M trace gives rough prioritization only: Q4_K MMQ 54.07 percent,
Q6_K MMQ 19.80 percent, Q5_K MMQ 6.16 percent, and about 80 percent quantized
matmul total. Do not use those shares as precise UD-Q4_K_XL Amdahl weights.

Do not replay clean upstream or the pre-existing combined build. They do not
answer the immediate causal questions, and exact source-local verification
already guards the variant composition. If a singleton wins every relevant
shape, multiplicity cannot reverse the decision. If results are mixed, never
sum or average the unique timing rows. Extend the CPU exporter with a sidecar
count keyed by the same signature and phase, or collect one current-model
trace. Predict time with `sum(count * time_us)` and combine changes in latency,
not throughput percentages. Only then pay for one target-model cell.

Keep the existing test-file format unchanged. For a count sidecar, increment a
map keyed by the existing comparator, where `name` is excluded, before
insertion into the exporter's de-duplication set. Maintain separate PP and TG
maps and emit the representative test line, phase, and count. This preserves
compatibility with `test-backend-ops` and makes layer multiplicity explicit.
Record that the PP count represents one physical microbatch; for a raw pp8192
run with `-ub 512`, the same dense graph is launched repeatedly.

If both singletons win, build `PROFILE=combined` and replay both types in one
process:

```bash
PROFILE=combined vr/scripts/build-gfx906-variant.sh

OPS="$OPS" TYPES="q4_K q6_K" N=512 \
CONTROL_BUILD=build/gfx906-ablate-common \
CANDIDATE_BUILD=build/gfx906-ablate-combined \
OUT=vr/bench-results/gfx906-op-combined-pp-2026-07-18 \
vr/scripts/bench-ops.sh
```

If only one singleton wins, that singleton is already the composed profile and
does not need another operator replay. If neither wins, `common` is final.

When refining the Q8_0 crossover, export the Q8_0 model at only `-ub 128`,
`-ub 256`, and `-ub 512`. Set `UB` and `OPS` to the matching export, and
exclude the TG rows with the same exact-`n` filter used above:

```bash
OPS=/tmp/qwen36-q8-pp-ub128-ops.txt \
TYPES=q8_0 N=128 \
CONTROL_BUILD=build/gfx906-ablate-common \
CANDIDATE_BUILD=build/gfx906-ablate-common \
CONTROL_ENV=GGML_CUDA_GFX906_Q8_MMQ_MAX_NE11=1 \
CANDIDATE_ENV=GGML_CUDA_GFX906_Q8_MMQ_MAX_NE11=1048576 \
vr/scripts/bench-ops.sh
```

Repeat with the matching 256 and 512 exports. The threshold is read once per
process and applies only to non-expert Q8_0 MMQ selection. MMVQ dispatch still
takes precedence for its small-column range. Compare each matrix shape
individually: a single `ne11` threshold is only justified if their crossovers
agree. Otherwise the next dispatch should include the weight-matrix dimensions
rather than averaging unlike rows.

## Minimal model screen

The model driver defaults to one causal cell, not a suite. Inspect it, then
run only after operator evidence passes:

```bash
vr/scripts/bench-head.sh --dry-run
vr/scripts/bench-head.sh
```

Current defaults:

- UD-Q4_K_XL on ROCm0;
- raw pp8192 at depth zero;
- control followed by candidate;
- exactly one timed sample, with the duplicate pp8192 warmup disabled;
- a 55 C edge-temperature gate on the used devices once before the A/B pair.

The script rejects `REPS` other than 1, duplicate cells, and an existing output
directory. It records exact commands, binary hashes, model stat data, build
order, and before/after GPU telemetry. It does not cool between control and
candidate by default; candidate runs second, so a clear win is conservative.
`WARMUP=auto` retains llama-bench's cheap initialization for incremental PP
and TG but adds `--no-warmup` to raw PP, where the built-in warmup would repeat
the entire prompt. Use a new reversed A/B only for a borderline result.

Build `llama-bench` only for the profile that survives operator replay, then
run its one affected model cell:

```bash
PROFILE=q4 TARGETS="test-backend-ops llama-bench" \
vr/scripts/build-gfx906-variant.sh

OUT=vr/bench-results/gfx906-model-q4-pp-2026-07-18 \
CONFIGS=ud_q4_k_xl_single MODES=raw_pp FAIL_BELOW_PCT=0 \
CONTROL_BIN_DIR="$PWD/build/gfx906-ablate-common/bin" \
CANDIDATE_BIN_DIR="$PWD/build/gfx906-ablate-q4/bin" \
vr/scripts/bench-head.sh --dry-run
```

Remove `--dry-run` only after the printed job is correct. Choose the surviving
profile and a new dated `OUT`; the example profile is not a prediction.

Incremental PP is a timed prompt chunk after a separately populated context.
Select one chunk from the observed production request distribution. When
incremental PP and TG use the same depth, request both together:

```bash
OUT=vr/bench-results/gfx906-model-q4-context-2026-07-18 \
CONFIGS=ud_q4_k_xl_dual MODES="inc_pp tg" \
INC_PROMPT=256 CONTEXT_DEPTH=8192 GEN_TOKENS=64 \
CONTROL_BIN_DIR="$PWD/build/gfx906-ablate-common/bin" \
CANDIDATE_BIN_DIR="$PWD/build/gfx906-ablate-q4/bin" \
vr/scripts/bench-head.sh
```

This creates one populated-context process per build. `llama-bench` reuses the
loaded model and saved depth state between the incremental-PP and TG cells, so
the depth prefill is not paid twice. The depth setup is outside each timed
interval. Do not add a list of chunks or depths.

To monitor a run without process-name guessing:

```bash
vr/scripts/watch-bench.sh vr/bench-results/<current-output>
```

The driver writes its own status, current cell, result table, commands, and
telemetry under `OUT`.

Run only the path an edit can affect:

```bash
# MMQ or PP dispatch
CONFIGS=ud_q4_k_xl_single MODES=raw_pp vr/scripts/bench-head.sh --dry-run

# Incremental PP dispatch at a populated context
CONFIGS=q8_0_dual MODES=inc_pp vr/scripts/bench-head.sh --dry-run

# MMVQ
CONFIGS=ud_q4_k_xl_single MODES=tg vr/scripts/bench-head.sh --dry-run
```

PP selects MMQ or rocBLAS at larger `n`; single-sequence TG selects MMVQ at
`n=1`. Separate optimized implementations are sensible. Test both PP and TG
only when shared quantization, vec-dot helpers, layouts, or crossover dispatch
change.

For the Q4_0/Q4_1 format question, first verify that the older files match the
current model revision. Export each graph and replay it on the candidate build.
These are different quantized models, so a control/candidate A/B is not the
relevant comparison. Load only a materially different candidate:

```bash
OUT=vr/bench-results/gfx906-model-format-screen-2026-07-18 \
BUILDS=candidate CONFIGS="q4_0_single q4_1_single" MODES="raw_pp tg" \
CANDIDATE_BIN_DIR="$PWD/build/gfx906-merge-review/bin" \
vr/scripts/bench-head.sh
```

Report the PP and TG deltas without a fixed threshold. Quantization quality,
memory use, and the magnitude of the speed difference determine whether the
UD-Q4_K_XL accuracy advantage is worth its cost.

## Production split mode

Use `-sm tensor`; it is already established as fastest on this MI60+MI50
system. Do not spend time resweeping layer, row, main-GPU placement, or basic
topology.

A later experiment may be worthwhile only if tensor mode can assign layers or
shards asymmetrically while preserving its execution model. Verify what
`--tensor-split` changes before treating it as layer placement, and test one or
two justified ratios rather than a broad sweep. The current Q8_0 PP result
files used layer mode, so the frozen Q8_0 candidate needs one tensor-mode
confirmation.

Make the final production validation one matched run per affected primary
format, not another parameter sweep. Keep `-fa on`, both devices, and
`-sm tensor`. For UD-Q4_K_XL, group incremental PP and TG at one populated
depth. For Q8_0, run raw PP and the one incremental-PP cell; do not add Q8_0 TG
when its path did not change. `ud_q4_k_xl_dual` and `q8_0_dual` encode the
production split and device list. Use pinned `build/head-control` against the
fresh final profile here; the private `common` build is only the causal
baseline for the PP singleton decision.

## Resolve only uncertain results

Do not increase repetitions; the driver refuses to do so. A large result that
agrees with operator data or prior model evidence can be accepted with a
one-sample/order caveat. For a borderline result or a conflict with prior
evidence, run one reversed A/B of only the affected cell in a new directory:

```bash
OUT=vr/bench-results/gfx906-model-q4-tg-reverse-2026-07-18 \
BUILDS="candidate control" CONFIGS=ud_q4_k_xl_single MODES=tg \
CONTROL_BIN_DIR="$PWD/build/head-control/bin" \
CANDIDATE_BIN_DIR="$PWD/build/gfx906-merge-review/bin" \
vr/scripts/bench-head.sh
```

Accept only when:

- the delta is large enough to survive the recorded order and telemetry caveat;
- exact-shape operator data or prior model evidence supports the direction;
- the executed kernel path is known.

For a small win, retain it when focused correctness is applicable,
`test-backend-ops` reports a positive exact-shape result over its internal
timing interval, and resource use does not worsen. If its full-model effect is
below one-sample resolution, record it as operator-proven with unresolved model
impact instead of paying for repeated model runs. If a reversed A/B changes
the conclusion, classify the result unresolved. Do not add prompt sizes to
average away drift.

## Wall-clock order

Use this order and stop at the first failed gate:

1. Inspect the source diff and worktree status.
2. Build only `test-backend-ops` for the needed profiles and let the exact
   compile-definition checker reject stale configurations.
3. If a kernel body changed and is not already covered by current correctness,
   run one focused exact-shape correctness pass. Skip it for definition-only
   ablations of the already-proven bodies.
4. Run one operator A/B for only the changed type and physical `n`.
5. Compose winners. Replay once only when two singleton winners were combined.
6. Build `llama-bench` and run one full-model cell only if every important
   replay shape wins or
   phase-weighted replay predicts a useful gain.
7. Run one reversed affected A/B only when the evidence is borderline or
   contradictory; otherwise record the sample/order caveat and proceed.
8. Run the full ROCm0 `MUL_MAT` gate once after the candidate is frozen.
9. Run only the affected production tensor-mode cells before calling the set
   complete.

Do not sweep contexts or matrix sizes. Batch independent operator cases in one
`test-backend-ops` invocation. Keep upstream `-fa on` for graph export and all
model runs; disabling custom FA code does not mean benchmarking with FA off.
The 55 C gate controls gross thermal bias without the long waits caused by the
old 45 C gate.

## Operator attribution

Use an operator benchmark before a model run when the code changes one kernel
specialization. Select the dominant real shape and one dispatch-boundary shape.
Do not sweep every matrix size.

For PP resource work, record compiler output before using the GPU:

- VGPR and SGPR per lane;
- LDS per workgroup;
- scratch bytes;
- workgroup size;
- expected waves per SIMD.

For TG loader work, use one counter run only when end-to-end data cannot tell
whether the kernel is latency or bandwidth limited. Record achieved HBM
bandwidth, memory busy/stall, VALU busy, and occupancy.

Use rooflines to choose work, not to declare a bottleneck from format size.
For TG, compare effective streamed bytes per token times tokens/s with measured
HBM traffic/bandwidth. For PP, compare measured operation throughput with both
compute and memory ceilings. If a small optimization underperforms its Amdahl
prediction, investigate path selection, actual operation share, launch count,
dependencies, clocks, and split synchronization before discarding the idea.

## Final gate

After the candidate set is frozen:

```bash
PROFILE=common BUILD_DIR=build/gfx906-final \
TARGETS="test-backend-ops llama-bench" \
vr/scripts/build-gfx906-variant.sh

FINAL_BUILD=build/gfx906-final
"$FINAL_BUILD/bin/test-backend-ops" test -b ROCm0 -o MUL_MAT
```

Replace the example `common` profile with the selected profile. Build
`FINAL_BUILD` fresh from the selected modular series; compile-disabled
experiment bodies have already been removed from live source. The helper
verifies its compile-definition allowlist before compiling. Replay the retained
paths, then run only the production tensor-mode cells described above. Do not
reuse the stale `optimal` directory or run the full matrix gate after each
edit.

## Result record

Store new output under:

```text
vr/bench-results/<experiment>-<date>/
```

Include:

- control and candidate commit IDs;
- exact build flags and compile definitions;
- binary paths;
- model path and the hash already recorded in `RESULTS.md`; the tools record
  size and mtime rather than rereading a 18-29 GB model for every run;
- GPU identity;
- commands and raw output;
- graph-export phase/count provenance when a weighted prediction was needed;
- resource report or profiler CSV when used;
- one short conclusion: retain, reject, or unresolved.

This revision's `test-backend-ops --output csv` is support-only and omits
timings. `bench-ops.sh` therefore records and parses the console performance
stream. Do not call the support-only CSV a performance result.
