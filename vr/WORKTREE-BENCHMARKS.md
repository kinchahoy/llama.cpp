# gfx906 A/B runbook

Last updated: 2026-07-18

This is the minimal validation path for a raw kernel or dispatch change. The
production goal is fast 25B-35B agentic inference on one MI60 plus one MI50 at
long context, not merely the highest isolated pp8192 number.

## Build matched trees

Build clean upstream and the current candidate from `571d0d540`:

```bash
TARGETS=llama-bench \
GGML_VULKAN=OFF \
CANDIDATE_BUILD="$PWD/build/gfx906-merge-review" \
CMAKE_EXTRA_ARGS="-DGGML_CCACHE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF" \
vr/scripts/build-gfx906-comparison.sh
```

Current review outputs:

```text
build/head-control/bin/llama-bench
build/gfx906-merge-review/bin/llama-bench
```

Build only `llama-bench` for screening. Add `test-backend-ops` when focused
correctness or the final gate is due.

## Cheap correctness

For Q4_K/Q8_0 matmul changes:

```bash
source vr/scripts/setup-therock-env.sh
build/gfx906-merge-review/bin/test-backend-ops test \
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
cmake --build build/head-control \
  --target test-export-graph-ops test-backend-ops -j"$(nproc)"
cmake --build build/gfx906-merge-review \
  --target test-backend-ops -j"$(nproc)"

MODEL="$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/ac393bc3d23fd5a929a85e2f33c7c4fd5be02d43/Qwen3.6-27B-UD-Q4_K_XL.gguf"
OPS=/tmp/qwen36-ud-q4-k-xl-pp8192-ub512-ops.txt

build/head-control/bin/test-export-graph-ops \
  -m "$MODEL" -c 8192 -b 8192 -ub 512 -fa on \
  -o "$OPS"

for BUILD in build/head-control build/gfx906-merge-review; do
  "$BUILD/bin/test-backend-ops" perf -b ROCm0 \
    --test-file "$OPS" -o MUL_MAT \
    -p 'q4_0|q4_1|q4_K|q5_K|q6_K|q8_0' \
    --output csv
done
```

The checked export produced 57 unique PP operations, 48 unique TG operations,
and 105 total. It preserves tensor types, dimensions, strides, and operation
parameters. It deduplicates operations, so it does not preserve layer or call
frequency. The exporter is CPU-only; `-ngl` has no effect.

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
Graph replay is a cheap rejection and attribution tool; one final target-model
cell is required for operation frequency, long-context, and whole-graph effects.

For the current PP regression, use four variants: upstream control, Q4_K-only,
Q6_K-only, and the existing combined build. Replay both singleton types before
another model load. Source-specific compile definitions should rebuild the
affected MMQ instance; verify `compile_commands.json` rather than assuming the
gate changed. Combine only singleton winners, replay that combination once,
then pay for one UD-Q4_K_XL PP model cell.

When refining the Q8_0 crossover, export the Q8_0 model at only `-ub 128`,
`-ub 256`, and `-ub 512`. Run each PP shape in a separate process with:

```bash
# Force rocBLAS for PP shapes above n=1.
GGML_CUDA_GFX906_Q8_MMQ_MAX_NE11=1 \
  build/gfx906-merge-review/bin/test-backend-ops perf \
  -b ROCm0 --test-file "$OPS" -o MUL_MAT -p q8_0 --output csv

# Force MMQ for the tested PP shapes.
GGML_CUDA_GFX906_Q8_MMQ_MAX_NE11=1048576 \
  build/gfx906-merge-review/bin/test-backend-ops perf \
  -b ROCm0 --test-file "$OPS" -o MUL_MAT -p q8_0 --output csv
```

The threshold is read once per process. MMVQ dispatch still takes precedence
for its small-column range. Tune from these exact boundary shapes instead of
sweeping full-model prompt sizes.

## Minimal model screen

Inspect the jobs, then run them:

```bash
vr/scripts/bench-head.sh --dry-run
vr/scripts/bench-head.sh
```

Current defaults:

- UD-Q4_K_XL on ROCm0 and Q8_0 on ROCm0/ROCm1;
- pp8192;
- tg128 at depth 8192;
- control followed by candidate;
- one timed repetition with built-in warmup.

This is a cheap core screen, not the production workload definition. For a
frozen candidate, add one incremental prompt size representative of tool
turns and one TG depth near the longest production context that fits. Do not
sweep many contexts.

Use a 55 C edge-temperature gate and explicit matched build directories:

```bash
COOL_TEMP=55 REPS=1 \
CONTROL_BIN_DIR="$PWD/build/head-control/bin" \
CANDIDATE_BIN_DIR="$PWD/build/gfx906-merge-review/bin" \
vr/scripts/bench-head.sh
```

Run only the path an edit can affect:

```bash
# MMQ or PP dispatch
CONFIGS=ud_q4_k_xl_single MODES=pp vr/scripts/bench-head.sh --dry-run

# MMVQ
CONFIGS=ud_q4_k_xl_single MODES=tg vr/scripts/bench-head.sh --dry-run

# Q8_0 MMVQ
CONFIGS=q8_0_dual MODES=tg vr/scripts/bench-head.sh --dry-run
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
BUILDS=candidate REPS=1 COOL_TEMP=55 \
CONFIGS="q4_0_single q4_1_single" MODES="pp tg" \
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
two justified ratios rather than a broad sweep. The current Q8_0 result files
used layer mode, so confirm the final candidate once in tensor mode.

## Resolve only uncertain results

Do not run three repetitions by default. A large result that agrees with
operator data or prior model evidence can be accepted with a one-sample/order
caveat. For a borderline result or a conflict with prior evidence, run one
reversed sample of only the affected cell:

```bash
COOL_TEMP=55 BUILDS="candidate control" REPS=1 \
CONFIGS=ud_q4_k_xl_single MODES=tg \
CONTROL_BIN_DIR="$PWD/build/head-control/bin" \
CANDIDATE_BIN_DIR="$PWD/build/gfx906-merge-review/bin" \
vr/scripts/bench-head.sh
```

Accept only when:

- the delta exceeds within-build spread;
- exact-shape operator data or prior model evidence supports the direction;
- the executed kernel path is known.

For a small win, retain it when focused correctness passes, exact-shape
operator replay repeats, and resource use does not worsen. If its full-model
effect is below benchmark noise, record it as operator-proven with unresolved
model impact instead of paying for repeated model runs. If a reversed run
changes the conclusion, classify the result unresolved. Add clock, power, and
temperature capture only then; do not add prompt sizes to average away drift.

## Wall-clock order

Use this order and stop at the first failed gate:

1. Inspect the source diff and active compile definitions.
2. Compile only changed targets and inspect VGPR, SGPR, LDS, and scratch.
3. Run focused correctness for changed types and representative shapes.
4. Replay exported operations for only the changed type/path.
5. Run one full-model cell only if weighted replay predicts a useful gain.
6. Run one reversed affected cell only when the evidence is borderline or
   contradictory; otherwise record the sample/order caveat and proceed.
7. Run the full ROCm0 `MUL_MAT` gate once after the candidate is frozen.
8. Run one production split and long-context scenario before calling the set
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
build/gfx906-merge-review/bin/test-backend-ops -b ROCm0 -o MUL_MAT
```

Then run one affected model cell and one production long-context/split scenario
once more. Do not run the full matrix gate after each edit.

## Result record

Store new output under:

```text
vr/bench-results/<experiment>-<date>/
```

Include:

- control and candidate commit IDs;
- exact build flags and compile definitions;
- binary paths;
- model path and hash;
- GPU identity;
- commands and raw output;
- resource report or profiler CSV when used;
- one short conclusion: retain, reject, or unresolved.

Do not call support-only CSV from `test-backend-ops` a performance result when
it lacks timing columns.
