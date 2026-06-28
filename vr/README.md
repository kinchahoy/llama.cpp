# gfx906 optimization work

Last updated: 2026-06-28

This directory contains private-fork work for AMD Vega 20 GPUs: MI50, MI60,
and Radeon VII. This is the sole operational and technical guide. Measured
results and their confidence levels are recorded in `RESULTS.md`.

The work is for a private fork. Upstream llama.cpp contribution rules still
apply if any part is later proposed upstream.

## Current state

The current best build is the exact single-target `gfx906` HIP build from this
tree, produced by:

```bash
vr/scripts/build-gfx906-optimal.sh
```

It includes the accepted code-level changes only:

- `Q8_0` selective rocBLAS dispatch for wide dense prefill.
- `Q4_K` MMQ metadata precompute plus stride-9 LDS layout for prefill.
- `Q6_K` MMQ `min_blocks=1` launch bound for the accepted Q6_K prefill path.
- `Q4_K` MMVQ branch-free scale decoding plus Q8_1 sum reuse for depth-0 TG.

The build script prints this list at startup, configures `gfx906` exactly, and
checks `compile_commands.json` for the gfx906-only compile definitions before
building. The default output is `build/gfx906-optimal`; override it with
`BUILD_DIR=/path/to/build`. The build-tree binaries are configured with
`CMAKE_INSTALL_RPATH=$ORIGIN`, `CMAKE_BUILD_WITH_INSTALL_RPATH=ON`, and
`CMAKE_BUILD_RPATH_USE_ORIGIN=ON` so the final build directory can be moved and
still find its local shared libraries. ROCm runtime libraries may still require
the normal ROCm environment or `LD_LIBRARY_PATH`.

The optimal patch order is recorded in `vr/patches/optimal-gfx906.series`:

```text
gfx906-q8-rocblas-dispatch.patch
gfx906-q4k-metadata-precompute.patch
gfx906-q6k-min-blocks-1.patch
gfx906-q4k-mmvq-branchless-scales.patch
```

The rejected Q5_K metadata experiment is retained under
`vr/archive/rejected-patches/` and is intentionally not part of the optimal
series.

Known remaining gaps:

- The latest head-versus-patches run is still triage-only.
- Long-context TG is unresolved and order-sensitive.
- The direct Q6_K scratch-spill trace remains an open validation artifact.
- Flash Attention tile tuning has not started; profile first.

## Scope

gfx906 has no MFMA matrix cores. Quantized matrix multiplication uses the
vector ALU, principally `V_DOT4_I32_I8` through `ggml_cuda_dp4a`.

This creates two distinct optimization problems:

- Prompt processing (PP or prefill) uses MMQ and is usually limited by DP4A,
  LDS behavior, register pressure, or occupancy.
- Token generation (TG or decode) normally uses MMVQ and is usually limited by
  effective HBM2 bandwidth plus long-context attention and KV-cache traffic.

Do not transfer occupancy conclusions between quant types without profiling.
The accepted Q4_K and Q6_K changes deliberately make opposite tradeoffs.

The active scope is code-level GPU optimization:

- Quantized MMQ/MMVQ kernels, launch geometry, memory layout, and fusion.
- Flash Attention kernel and tile tuning.
- Profiling and correctness work required to validate those changes.

Runtime or model configuration changes are out of scope for now. This includes
KV-cache type or precision changes, speculative decoding, layer/row split
tuning, tensor placement, and similar settings-only comparisons. Existing
historical results may mention them, but they are not current work items.

## Repository layout

The source changes remain in upstream paths because they modify existing code:

- `ggml/src/ggml-cuda/mmq.cu`: selective Q8_0 rocBLAS dispatch
- `ggml/src/ggml-cuda/mmq.cuh`: Q4_K and Q6_K gfx906 specializations
- `ggml/src/ggml-cuda/vecdotq.cuh`: Q4_K MMVQ branch-free scale decoding
- `ggml/src/ggml-hip/CMakeLists.txt`: exact-gfx906 compile gates

Everything added specifically for this work is under `vr/`:

- `patches/`: standalone source changes and retained experiment patches
- `scripts/`: current build, benchmark, and monitoring tools
- `bench-results/`: raw JSONL, text, and profiler output
- `assets/`: benchmark prompt fixtures
- `archive/`: superseded documentation, historical scripts, and rejected patches

The intended branch delta from upstream is the four scoped source changes plus
the `vr/` tree. New optimization source changes should have a corresponding
standalone patch under `vr/patches/`.

## Accepted source changes

### Q8_0 selective rocBLAS dispatch

For dense Q8_0 matmuls on Vega 20, `ggml_cuda_should_use_mmq` keeps MMQ through
`ne11 <= 256` and uses the existing rocBLAS path above that threshold. Expert
`MUL_MAT_ID` operations remain on MMQ.

The observed benefit is in wide prefill operations. The current evidence does
not establish a decode benefit. Patch:
`vr/patches/gfx906-q8-rocblas-dispatch.patch`.

### Q4_K metadata precompute and stride-9 LDS layout

The gfx906 Q4_K MMQ specialization computes final `dm * scale` and `dm * min`
metadata while loading the weight tile instead of repeatedly unpacking it in
the dot-product loop. Metadata rows use stride 9 to reduce LDS bank conflicts
for the existing wave64 geometry.

Profiling showed Q4_K to be LDS-wait and occupancy sensitive. Keeping occupancy
is important. Applying `min_blocks=1` to Q4_K caused a large regression. Patch:
`vr/patches/gfx906-q4k-metadata-precompute.patch`.

### Q6_K minimum launch occupancy of one block

The gfx906 Q6_K MMQ specialization uses `__launch_bounds__(..., 1)`. This gives
the compiler more VGPR headroom at the cost of occupancy. It improved the
profiled Q6_K workload because that kernel was register-spill and VALU limited,
not LDS-wait limited.

The direct scratch trace that proves removal of the previously observed
52-byte-per-thread spill remains an open validation item. Patch:
`vr/patches/gfx906-q6k-min-blocks-1.patch`.

### Q4_K MMVQ scale decoding and Q8_1 sum reuse

The TG specialization replaces the per-lane Q4_K scale-layout branch with safe
0/2/4 word loads and a mask select. It also reuses the scaled Q8_1 sum already
stored in `ds.y` for the minimum term instead of recomputing partial sums with
four DP4A operations per vec-dot call. It is compiled only for an exact single
gfx906 HIP target. The original branch-free decoder improved n=1 latency by
about 4.7 percent and depth-0 TG by 5.4 to 6.0 percent. Q8_1 sum reuse added
about 0.9 percent depth-0 TG in alternating tests.

Focused correctness and the full ROCm0 `MUL_MAT` gate passed after both changes.
Depth-8192 results were
order-sensitive and do not establish a long-context gain; repeat that cell
only when long-context behavior becomes decision-relevant. Patch:
`vr/patches/gfx906-q4k-mmvq-branchless-scales.patch`.

## Critical build gate

The Q4_K MMQ/MMVQ and Q6_K specializations compile only when the HIP target list
contains exactly one target and that target is `gfx906`. A multi-architecture build
silently uses the generic implementations. The Q8_0 dispatch uses a runtime
compute-capability check and is not subject to this compile gate.

Configure and build the current optimal with:

```bash
vr/scripts/build-gfx906-optimal.sh
```

The `vr/scripts/*.sh` entrypoints use one rule: the `vr` directory must live
inside the llama.cpp tree being built. The scripts derive the llama.cpp root as
two directories above themselves, so moving `vr/` into another checkout is
enough to make the same commands build that checkout.

Shared ROCm detection, CMake options, RPATH settings, and common build helpers
live in `vr/scripts/_gfx906_build.sh`. Edit that file when changing core build
flags; the vanilla, optimal, and comparison build scripts all use it.

Common overrides:

```bash
BUILD_DIR=/mnt/fast/gfx906-optimal vr/scripts/build-gfx906-optimal.sh
TARGETS="llama-cli llama-server llama-bench" vr/scripts/build-gfx906-optimal.sh
GGML_VULKAN=OFF vr/scripts/build-gfx906-optimal.sh
CONFIGURE_ONLY=1 vr/scripts/build-gfx906-optimal.sh
INSTALL_DIR=/opt/llama-gfx906 vr/scripts/build-gfx906-optimal.sh
```

The script verifies that the specializations were compiled rather than assuming
the CMake option was honored. Manual verification is:

```bash
rg 'GGML_CUDA_MMQ_Q4K_GFX906_PRECOMPUTE|GGML_CUDA_MMVQ_Q4K_GFX906_BRANCHLESS_SCALES' \
  build/gfx906-optimal/compile_commands.json
```

## Correctness gate

Run the full ROCm matrix-multiplication tests before accepting any kernel
change:

```bash
build/gfx906-optimal/bin/test-backend-ops -b ROCm0 -o MUL_MAT
```

For a new quant-specific implementation, first use a focused test during
development, then run the full gate before recording an accepted result.

## A/B build rules

A valid full-patch comparison is:

- Control: clean selected upstream commit, with no gfx906 patches applied.
- Candidate: the same upstream commit plus the intended patch set.

`vr/scripts/build-gfx906-comparison.sh` now defaults `CONTROL_PATCH` to empty,
so the control is clean unless explicitly overridden. Preserve that default for
a full head-versus-optimal comparison:

```bash
CONTROL_PATCH="" vr/scripts/build-gfx906-comparison.sh
```

A benchmark is not a clean head-versus-optimal comparison unless its provenance
confirms that `CONTROL_PATCH` was empty.

Before a run, also verify that `build/.mainline-src` is at the intended commit
and contains no unexpected changes. Do not silently reuse a dirty comparison
worktree.

## Benchmark policy

Use `-p N -d D` for prompt processing and `-n M -d D` for generation. Do not use
a blended prompt-plus-generation result when evaluating a PP- or TG-specific
change.

The primary harness is `vr/scripts/bench-head.sh`. Optimize for short feedback
cycles:

- Use `REPS=1` plus warmup for the default screen.
- Run one relevant operator sample and one end-to-end sample when both paths
  exercise the change.
- Repeat only a promising result, an ambiguous result near the decision line,
  or a result being promoted to accepted.
- Use alternating independent invocations and telemetry only when noise is
  preventing a decision.
- Always run the appropriate correctness gate before accepting source changes.

The current harness interleaves control and candidate and keeps warmup enabled.
It does not yet write a complete provenance and telemetry manifest. Until that
is added, record the source commits, patch hashes, command-line environment,
and `rocm-smi` state beside each accepted result.

### Single-GPU and dual-GPU measurements

Use single GPU with `-sm none -dev ROCm0` as the primary kernel baseline.
Dual-GPU measurements are only needed as a regression check when a code change
affects multi-GPU execution.

Do not spend optimization cycles comparing layer, row, or tensor split modes.
Those are configuration choices and currently out of scope.

Q8_0 does not fit the same long-context single-GPU matrix used by the current
benchmark, so its present end-to-end baseline is dual GPU.

### Thermal and power controls

The two installed cards are not equivalent. MI50-0 has a 225 W cap and direct
CPU PCIe attachment; MI50-1 has a 178 W cap and chipset PCIe attachment. Do not
average them as interchangeable devices.

Measure active memory clock and power before comparing a TG result with a
theoretical 1024 GB/s HBM2 figure. Use measured sustainable bandwidth at the
actual clock as the practical roofline. Any power-limit change requires normal
hardware safety review and should be reported separately from kernel results.

## Current plan

The current focus is stable, measurable PP and TG improvement from code-level
GPU changes, primarily Q4_K before extending proven techniques to Q5_K or
Q6_K.

1. Establish a reproducible TG baseline.
   - Build a verified clean control and candidate from the same upstream commit.
   - Add or manually capture a provenance manifest.
   - Use one timed sample for screening and repeat only ambiguous or promoted
     results.
2. Profile the actual decode kernels.
   - Confirm which MMVQ kernels dominate Q8_0 and Q4_K_M generation.
   - Measure effective bandwidth, occupancy, and memory-clock state.
   - Inventory the model's Q4_K and Q6_K tensor bytes before sizing Q6_K work.
   - Repeat depth-8192 only if a later change makes long-context behavior
     decision-relevant.
3. Evaluate Q4_K MMVQ first.
   - Start from the existing wave64 GCN path and profile the actual Q4_K decode
     shapes before changing vector width or launch geometry.
   - Gate any specialization to exact gfx906 builds.
   - Require full correctness and a repeatable depth-0 TG gain. Test long
     context only when it is decision-relevant.
4. Extend a validated MMVQ technique to Q5_K or Q6_K only when their profile
   shows the same bottleneck.
5. Tune Flash Attention tiles when profiling shows attention is material to
   the target workload.
6. Revisit Q8_0 MMVQ separately; it has different storage and model-fit
   constraints.

The practical bandwidth target should be derived from a measured bandwidth
microbenchmark. The previous aspirational target of more than 85 percent of
advertised peak is not an acceptance gate.

## Do not reopen without new evidence

- Q4_K `min_blocks=1`: pp8192 regressed by 33.74 percent.
- Q4_K y64 tile: pp8192 regressed by 9.27 percent.
- Forced Q4_K rocBLAS/F16 conversion: operator-level regression of 30 to 34
  percent.
- Q4_K stride 10 or 11: both lost in the tested configurations.
- Q4_K group2pad1 layout: LDS conflicts returned to the original level.
- Q6_K prefill metadata precompute: pp8192 regressed by 5.3 percent.
- Q5_K `min_blocks=1`: representative n=512 operator throughput regressed by
  about 17 percent.
- Q4_K two-accumulator DP4A: narrow MMQ improved, but the n=512 PP operator
  regressed by 2.4 percent.

These conclusions apply to the recorded prefill experiments. They do not by
themselves reject a distinct Q6_K decode MMVQ design.

## Script index

- `_gfx906_build.sh`: shared ROCm/CMake setup and core build flags
- `build-gfx906-optimal.sh`: build the current optimized gfx906 tree
- `build-gfx906-vanilla.sh`: build an unpatched upstream-style gfx906 tree
- `setup-therock-env.sh`: configure the TheRock compiler/runtime environment
- `build-gfx906-comparison.sh`: build upstream control and current candidate
- `bench-head.sh`: primary PP/TG A/B harness
- `watch-bench.sh` and `_bench_table.py`: live result display
- `rocm-max-temps.sh`: simple temperature logging helper

Historical one-off experiment drivers were moved to
`vr/archive/scripts-2026-06/`. They may use older repetition and warmup
policies. Their outputs are historical evidence and should not define the
current acceptance policy.

## Documentation archive

The complete pre-consolidation Markdown tree is preserved at:

```text
vr/archive/docs-pre-consolidation-2026-06-20.tar.gz
```

SHA-256:

```text
8dcbf659820e9cac33ba7caae6155d0558cb4544f5ca97a84fbc1f6d3f521c2e
```

List or extract it from the repository root with:

```bash
tar -tzf vr/archive/docs-pre-consolidation-2026-06-20.tar.gz
tar -xzf vr/archive/docs-pre-consolidation-2026-06-20.tar.gz -C /tmp
```

The archive contains the previous guides, benchmark document, optimization
summary, plan, and generated Markdown reports. Raw benchmark and profiler data
remain in their original `vr/bench-results/` locations.

For future material rewrites, create a new dated archive before editing these
two files. Do not replace an existing archive; each archive is a documentation
checkpoint for resolving later discrepancies.

The stable PP ledger immediately before beginning the Q4_K TG phase is stored
in `vr/archive/docs-pp-stable-pre-tg-2026-06-20.tar.gz`, SHA-256
`005ff52bfae076783fa9a3d2045309d95afdc036f782cd0772418f7a37978ccc`.

The candidate-stage ledger before the Q4_K TG change passed its full gate is
stored in `vr/archive/docs-pre-q4k-tg-acceptance-2026-06-20.tar.gz`, SHA-256
`cf99dd6a0db8af3c8b69415ce62de62d2076da719ada2f6ff3966168936b1393`.

The ledger before adopting the kernel-only optimization scope is stored in
`vr/archive/docs-pre-kernel-only-scope-2026-06-20.tar.gz`, SHA-256
`3ec7c6f18b499a2b74129a0b828a85ef0481c8a847caf28d012223794ed7cf67`.

The human-oriented guide and ledger before adding the following restart
handoff are stored in
`vr/archive/docs-pre-fused-q4k-handoff-2026-06-20.tar.gz`, SHA-256
`a75d336d418a12690a3e2176f122f43f56a94d6a3009c3f8092b9a68136b4c19`.

## Fresh-session execution handoff

This section is an execution checklist for the next engineer or LLM. The
earlier sections are the design and evidence context and should remain the
primary human-readable guide.

### Stable starting point

The working source has four accepted gfx906 changes: Q8_0 selective rocBLAS
dispatch, Q4_K PP metadata precompute with stride 9, Q6_K PP
`min_blocks=1`, and Q4_K TG branch-free scale/min decoding. Do not revert the
branch-free decoder before starting the next experiment. Its standalone patch
is `vr/patches/gfx906-q4k-mmvq-branchless-scales.patch`; a reverse apply check
should succeed against the current source.

The next target is Q4_K TG only. A tg32 trace attributes 53.83 percent of GPU
kernel time to Q4_K MMVQ and 40.68 percent to the fused Q4_K MMVQ kernel alone.
Q6_K and Q5_K MMVQ account for 19.46 and 4.57 percent respectively. The trace
summary is in
`vr/bench-results/gfx906-q4k-tg-profile/trace/q4_k_m_tg32_d0/summary.txt`.

### Next experiment: paired fused Q4_K dot product

Test a Q4_K-specific fused MMVQ helper that computes the value and gate dot
products together. The existing fused loop calls the same Q4_K/Q8_1 dot helper
twice with different weight matrices but the same Q8 activation data. The
hypothesis is that a paired helper can load or decode that activation data once
and accumulate both results.

Relevant source locations at this checkpoint are:

- `ggml/src/ggml-cuda/mmvq.cu`, generic `mul_mat_vec_q` around lines 475-673;
  its fused inner loop is around line 580.
- `ggml/src/ggml-cuda/vecdotq.cuh`,
  `vec_dot_q4_K_q8_1_impl_vmmq` around line 505 and its wrapper around line 864.

Before changing code, inspect generated code or make a minimal experiment to
check whether the compiler already eliminates the duplicate Q8 work. Keep the
generic path unchanged. Initially specialize only Q4_K with `ncols_dst == 1`
and `has_fusion`, and compile it only for the existing exact single-gfx906
gate. Do not combine this experiment with launch-geometry, vector-width, or
independent-accumulator changes; those directions have already been screened.
If retained, add a standalone patch under `vr/patches/`.

### Fast validation loop

Leave `build/q4k-ilp2` untouched as the accepted incremental control. Create a
separate candidate build such as `build/q4k-fused-pair` from the current source
plus only this experiment. Both builds must target exact gfx906.

Use this local model:

```text
/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-Q4_K_M.gguf
```

Run one depth-0 TG screen from each build on ROCm0:

```bash
BIN -m /home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-Q4_K_M.gguf \
  -ngl 99 -fa on -sm none -dev ROCm0 -p 0 -n 64 -d 0 -r 1 -o jsonl
```

Replace `BIN` with each build's `bin/llama-bench`. One sample per build is the
default decision screen. Repeat only if the change is promising (roughly more
than 2 percent), contradictory, or ready for promotion. Do not rerun depth
8192 unless long-context behavior becomes relevant to the decision.

The ordinary focused `MUL_MAT` test does not necessarily exercise the fused
gate path. Find an existing fusion graph test, export a representative fused
graph, or otherwise verify that path directly before acceptance. Also run the
focused Q4_K operator cases for helper coverage. In all cases, run the complete
ROCm0 `MUL_MAT` gate and require 1103 of 1103 before retaining the patch. Record
raw data and the result classification in `RESULTS.md`.

If paired fusion does not win, the next profiling-led options are Q6_K MMVQ,
then a Q5_K port of the branch-free decoder, then broader reuse of Q8 activation
quantization. Flash Attention tile tuning remains in scope when a trace shows
attention is material. Settings changes remain out of scope.
