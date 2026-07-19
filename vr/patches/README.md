# gfx906 patch directory

## Use this for the next bigbang run

Final current patch:

```text
current/gfx906-current-bigbang.patch
```

Final current series entrypoints:

```text
all-gfx906.series
optimal-gfx906.series
```

Both series files currently contain the same single patch:

```text
current/gfx906-current-bigbang.patch
```

Use either top-level series file if a script expects a series file. Use the
patch file directly if you want the simplest manual apply target.

## What is in the current bigbang patch

- Q8_0 selective MMQ/rocBLAS dispatch with sweepable `GGML_CUDA_GFX906_Q8_MMQ_MAX_NE11`.
- Q8_0 MMVQ wide-VDR candidate.
- Q8_0 MMVQ paired value+gate activation-load sharing.
- Q8_1 MMQ quantization DPP reductions.
- Q4_K MMQ metadata precompute.
- Q4_K MMQ `min_blocks=3` occupancy experiment.
- Q6_K MMQ `min_blocks=1`.
- Q4_K MMVQ branchless scales and Q8_1 sum reuse.
- Q4_K MMVQ paired value+gate activation-load sharing.
- FlashAttention vector Q8_1 quantization DPP reductions for `-fa on`.
- FlashAttention vector quantized KQ 8-lane candidate for `-fa on`.

## Legacy reference patches

Older per-feature patches and handoff notes live here:

```text
legacy/2026-06-old-base/
```

Those files are retained for attribution and design notes. They target an older
upstream base and should not be treated as the runnable patch series for the
current checkout.

## Current status

The bigbang patch was regenerated against upstream commit `571d0d540` after
porting the private MMQ changes across upstream's config/load/vector-dot
refactor. A fresh exact-gfx906 build succeeded and the complete ROCm0 `MUL_MAT`
gate passed 1134 of 1134 cases. Both series entrypoints resolve to the current
patch, and it applies cleanly to a fresh archive of `571d0d540`. The gate does
not directly exercise fused value-plus-gate paths. Treat the patch as a
candidate until the remaining benchmark checklist in
`vr/WORKTREE-BENCHMARKS.md` is complete.
