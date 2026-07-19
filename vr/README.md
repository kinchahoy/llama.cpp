# gfx906 kernel optimization

Last updated: 2026-07-18

This directory tracks private llama.cpp work for Vega 20 (`gfx906`): MI50,
MI60, and Radeon VII. The scope is raw GPU implementation performance:
quantized matmul kernels, launch geometry, data layout, and dispatch. Runtime
techniques that change the workload are out of scope.

## Goal

Maximize real agentic inference speed for 25B-35B models on one MI60 plus one
MI50 at the longest practical context. Q8_0 and UD-Q4_K_XL are the primary
formats because model publishers commonly provide them and they preserve
useful quality. Track prompt processing, incremental prompt ingestion, and
long-context token generation separately; all matter.

Keep MTP and similar workload-changing features out of core optimization
measurements. They can be enabled later in production, after the underlying
PP, TG, attention, KV, dispatch, and multi-GPU paths are understood.

## Read this first

- `RESULTS.md`: measured results and rejected ideas.
- `WORKTREE-BENCHMARKS.md`: the cheapest useful A/B procedure.
- `bench-results/gfx906-pp8192-kernel-profile-2026-07-02/FINDINGS.md`:
  profiler evidence behind the PP diagnosis.
- `patches/README.md`: current patch contents and status.

Raw benchmark and profiler output stays under `bench-results/`. Old code
experiments stay under `patches/legacy/` and `archive/rejected-patches/`.
Historical prose snapshots and handoff documents were removed because they
duplicated or contradicted these live documents.

## Current tree

The private branch is based on upstream commit `571d0d540`. Build the exact
single-target candidate with:

```bash
vr/scripts/build-gfx906-optimal.sh
```

The historical name `optimal` is only a build-directory name. The matched
review builds are `build/head-control` (clean `571d0d540`) and
`build/gfx906-merge-review` (private candidate).

The active candidate retains selective Q8_0 MMQ/rocBLAS dispatch, Q4_K MMQ
metadata precompute and stride 9, Q6_K MMQ `min_blocks=1`, and Q4_K branchless
MMVQ scale decode with stored Q8_1 sum reuse. Q4_K `min_blocks=3`, paired
MMVQ, Q8_1 DPP, and custom Flash Attention experiments are disabled. Focused
ROCm0 correctness passed 203 of 203 target `MUL_MAT` cases after this
reduction.

The current model screen proves the Q8_0 PP dispatch win. It shows that the
combined Q4_K MMQ precompute/stride-9 plus Q6_K `min_blocks=1` PP candidate
regresses on current head; it does not identify which component is responsible.
Disabled experimental implementation code still exists in the patch and
should be removed after the retained set is frozen.

## What the current code does

### Prompt processing

- Q4_K uses DP4A MMQ with Q8_1 activations.
- Q8_0 uses MMQ only through a configurable batch threshold. Wide dense
  matrices dequantize Q8_0 weights to F16 and use rocBLAS.
- At pp8192 on the profiled Q4_K_M model, quantized matmul is 80 percent of GPU
  kernel time: Q4_K 54 percent, Q6_K 20 percent, and Q5_K 6 percent.
- Upstream Flash Attention remains enabled for every model benchmark and graph
  export with `-fa on`. Only experimental custom FA patch changes are disabled.
  FA was about 4.1 percent of PP8192 kernel time, so custom FA optimization is
  lower priority than quantized matmul.

An older Q4_K_M build measured Q4_K at 128 VGPR per lane, two waves per SIMD,
60 percent VALU busy, about 11 percent memory-unit busy, and negligible LDS
bank conflicts. This supports an occupancy/dependency-latency hypothesis for
that build. It does not prove the current UD-Q4_K_XL kernel is register-limited;
recheck current ISA resources and one counter trace before designing around it.

### Token generation

- Q4_K and Q8_0 use MMVQ for the small-column path.
- At one generated token, each weight matrix is streamed once. Q4_K costs
  about 0.5625 bytes per weight and Q8_0 about 1.0625 bytes per weight before
  cache-line and other model traffic.
- The fused gate/value MMVQ path already quantizes the shared activation once.
  Sharing another activation load cannot remove weight bytes, but fusion may
  still reduce activation traffic, launches, and synchronization. Measure its
  total share before estimating the ceiling.

Q8_0 TG has a large unavoidable weight-traffic floor, but achieved HBM
bandwidth has not been measured. Do not label it bandwidth-bound yet: launch
latency, dependency chains, tensor-mode synchronization, or insufficient
memory-level parallelism may explain the gap to the byte roofline.
Wider VDR was neutral in one implementation. Q4_K TG also decodes packed 6-bit
scale/min metadata and therefore has additional instruction-side opportunities.

## Current decision state

| Change | Status | Reason |
| --- | --- | --- |
| Q8_0 MMQ/rocBLAS crossover | Proven on current head | Two pp8192 screens: +29.0 and +27.9 percent; earlier TG neutral |
| Q4_K MMQ precompute plus stride 9 | Combined result rejects current form | Reduced PP candidate including Q6_K is -5.5 percent; attribution is unresolved |
| Q6_K `min_blocks=1` | Historical support; unresolved on head | Must be replayed alone on current Qwen shapes |
| Q4_K branchless MMVQ plus stored sum | Retain with caveat | Current tg128@d8192 is +21.7 percent in one ordered sample; historical depth-zero tests were +5 to +6 percent |
| Q4_K `min_blocks=3` | Unproven/disabled | Present in a combined set that lost 17.1 percent; it was not isolated |
| Paired MMVQ, Q8_1 DPP, custom FA | Unproven/disabled | Removed from active compile gates |
| Q8_0 VDR 4 | Rejected/disabled | `+0.19%` TG is neutral |

## Important unknowns

- Current UD-Q4_K_XL operation frequencies and kernel counters have not been
  profiled; old Q4_K_M shares are only directional.
- Achieved HBM bandwidth, launch share, and split synchronization cost are
  unknown for long-context TG.
- Tensor split mode is already established as the fastest production mode.
  Internal asymmetric layer/shard placement remains a possible later question.
- Graph replay covers exact dense shapes but not a populated long-context KV
  cache.
- Q4_0/Q4_1 files are from an older snapshot; no fair speed/quality comparison
  exists yet.

## Immediate handoff

Do these in order:

1. Build Q4_K-only and Q6_K-only PP variants. Keep the independent Q4_K MMVQ
   and Q8_0 dispatch changes identical. The existing control and combined
   result complete the four-way ablation.
2. Replay current UD-Q4_K_XL Q4_K and Q6_K shapes for both singletons. This
   separates individual effects and reveals whether the combined set has an
   interaction. Do not assign the old -5.5 percent result to precompute,
   stride 9, or Q6_K without this isolation.
3. Combine only singleton winners, replay once, then run one UD-Q4_K_XL
   `pp8192` cell if the predicted net result is positive. Do not rerun Q8_0
   pp8192: its 28-29 percent dispatch gain has repeated.
4. Retain Q4_K branchless MMVQ with a one-sample/order caveat. Its +21.7
   percent current result is directionally supported by historical +5 to +6
   percent depth-zero tests; do not spend three runs reconfirming it.
5. Treat Q8_0 TG as neutral with a caveat. The reduced-candidate cell was
   interrupted, but the earlier current-head pair was -1.5 percent and the
   Q8_0 dispatch change targets PP. Rerun TG only if its code path changes.
6. Freeze the set, delete disabled experiment code from the generated patch,
   rebuild, run the full ROCm0 `MUL_MAT` gate once, then one final production
   `-sm tensor` configuration at a long context that actually fits.

Commands are in `WORKTREE-BENCHMARKS.md`. Current raw data is under
`bench-results/gfx906-head-{core-screen,q8-screen,retained-screen}-2026-07-18/`.

## Separate PP and TG paths

Separate optimized paths are appropriate, but dispatch is a continuum rather
than a binary PP/TG split. Large PP selects MMQ or rocBLAS, single-sequence TG
selects MMVQ at `n=1`, and incremental/batched agent work exercises boundary
sizes between them. The old PP trace suggests occupancy/dependency latency;
TG has a weight-traffic floor plus metadata and launch costs.

Keep the public dispatch and quant formats shared, but use narrowly gated PP
and TG kernels. A PP-only MMQ change does not justify rerunning TG, and an
MMVQ-only change does not justify rerunning PP. Test both only when changing
shared quantization, vec-dot helpers, type layout, or the MMQ/MMVQ crossover.

## Further improvements

### 1. Reduce Q4_K/Q6_K PP live state

This was the highest-share kernel in an older Q4_K_M trace, but the current
UD-Q4_K_XL shares are unknown. Do not treat stronger `launch_bounds` or an
84-VGPR target as a solution by itself; either can force spills or
recomputation when values remain live.

Use exported Qwen shapes rather than the generic `4096 x 14336` built-in
performance case. Dominant dimensions are approximately `5120 <-> 17408`,
with `n=512` for PP because pp8192 is executed as `-ub 512` microbatches.
Compile the dominant Q4_K and Q6_K specializations and record VGPR, SGPR, LDS,
scratch, and waves/SIMD. Then compare two structural variants:

1. A larger workgroup that distributes the same output tile over more lanes,
   reducing accumulators per lane while preserving weight reuse.
2. A smaller output-column tile, accepting less weight reuse in exchange for
   fewer live accumulators and more resident blocks.

Lower VGPR without scratch is a useful experiment only if current ISA and
counters confirm occupancy is limiting. A variant with more resident waves but
more instructions, barriers, or dependencies may still lose. Test only the
dominant current operator shape first, then pay for one model cell.

Defer double-buffering, larger tiles, or more independent accumulators until
current counters show enough register and occupancy headroom. They add live
state and were poor fits for the older measured kernel.

Also compare the existing integer DP4A path with one tile-local
dequantize-to-half2 design if current instruction counters justify it. gfx906
does not use llama.cpp's `AMD_MFMA_AVAILABLE` path; that starts at CDNA
gfx908. It does provide packed FP16 `v_dot2`. The previously rejected
whole-matrix Q4-to-F16 rocBLAS path paid global conversion and scratch traffic,
so it does not rule out fused tile-local conversion. Estimate conversion,
LDS, and `v_dot2` cost against DP4A before implementing a full kernel.

### 2. Make Q4_K TG metadata wave-cooperative

Within each Q4_K block, many lanes load and decode overlapping scale/min words.
First confirm redundant scale/min loads and decode instructions in current ISA.
If material, test selected lanes loading and decoding the eight pairs, then
broadcasting within each 32-lane half wave using DPP or `ds_bpermute`.

This keeps the compressed format and attacks redundant instructions without
adding weight bytes. It is preferable to a decoded sidecar, which increases
TG traffic. Check generated ISA and VGPR count before any model run; a shuffle
sequence that raises VGPR pressure can erase the decode saving.

If counters show unused HBM bandwidth after this change, test a two-way
strip-mined K loop with independent partial sums to expose more outstanding
loads. Do not retry VDR 4: that reduced participating work groups and was
already neutral.

### 3. Refine Q8_0 PP dispatch, not the 8K GEMM

At pp8192, one Q8_0-to-F16 conversion is amortized across a large rocBLAS GEMM.
A new compressed GEMM must beat tuned F16 rocBLAS, not merely remove the
conversion kernel. The likely near-term gain is a shape-aware crossover using
`M`, `N`, and `K`, replacing the current scalar `ne11` threshold.

The current default is MMQ at `ne11 <= 256` and rocBLAS above it. Only the
pp8192/`n=512` side is proven; agentic incremental prompt sizes near the
boundary are not. Measure a few exact Qwen shapes around 128/256/512 under
forced-MMQ and forced-rocBLAS dispatch. A persistent F16 weight cache nearly
doubles weight storage and is unsuitable as the long-context default. A direct
block-scaled Q8 GEMM is justified only if profiling shows conversion and
scratch traffic remain material.

### 4. Measure the Q8_0 TG roofline

Strict single-token Q8_0 TG cannot avoid reading roughly 1.0625 bytes per
weight. First calculate effective bytes/s from model traffic and measure HBM
counters. Then choose among:

- verify aligned, fully coalesced Q8 block loads and achieved HBM bandwidth;
- load and broadcast the per-block scale cooperatively if it reduces
  instructions without extra shuffles or registers;
- try a two-way prefetched K loop only when memory-level parallelism is low;
- use streaming cache policy only if weight traffic is evicting useful
  activation or KV data.

Do not assume only incremental gains remain until the measured roofline rules
out launch, synchronization, split balance, and memory-level-parallelism gaps.
Fusion of different projection matrices does not reduce their weight bytes,
but it may still reduce activation traffic and launches; value it with an
Amdahl calculation rather than rejecting it categorically.

### 5. Decide whether Q4_0 or Q4_1 is worth the quality trade

Do not assume a simpler quant is faster enough to matter. Export and replay
the Q4_0 and Q4_1 model graphs on the same candidate build, then compare their
PP `n=512` and TG `n=1` shapes with UD-Q4_K_XL. Load only the best predicted
format for one `pp8192` and one `tg128@d8192` model screen.

Report PP, incremental PP, long-context TG, memory use, and quality together;
do not impose a fixed speed threshold in advance. UD-Q4_K_XL is expected to be
more accurate, so the magnitude and location of any Q4_0/Q4_1 speed advantage
determines the trade. Compare only matching model revisions.

## Exact implementations closed by evidence

- Q4_K forced rocBLAS/F16: large PP regression.
- Q4_K `min_blocks=1`, the larger activation K tile, and two-accumulator PP
  variants: the tested mappings regressed; this does not close other tile or
  accumulator mappings.
- Q4_K stride 10/11 and group2 padding: regressions.
- Q6_K metadata precompute: regression.
- Q5_K metadata precompute: synthetic win did not survive the model gate.
- Q8_0 VDR 4: neutral TG.
- The tested custom Flash Attention changes: unproven and disabled. Upstream
  FA remains on. The old trace assigned FA about 4 percent at pp8192, but its
  share can grow with context.

Do not retry an identical implementation without new evidence. The broader
optimization areas remain open when current target traces or rooflines justify
them.

## Acceptance rule

Use `WORKTREE-BENCHMARKS.md`. Reject candidates first with exact graph-derived
operator shapes. Pay for a full-model cell only after the profiler-weighted
operator result predicts a useful gain. Accept a large, coherent signal with
an explicit sample/order caveat. Use one reversed sample only for borderline
or contradictory evidence, not three repetitions by default. Run focused
correctness during development and the full ROCm0 `MUL_MAT` gate only after
the final candidate is frozen.

Small repeatable wins are valid. If a theoretically meaningful optimization
produces a tiny result, compare predicted and observed impact before moving on:
verify the executed path, operation share, resource change, launch count,
clocks, and roofline assumption. The discrepancy is evidence about the real
bottleneck.
