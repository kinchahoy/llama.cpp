# gfx906 patches

## Current artifact

Both series files currently resolve to:

```text
current/gfx906-current-bigbang.patch
```

The patch targets upstream `571d0d540`, but it is currently stale relative to
the reduced working tree. It still contains disabled and rejected experiments.
Do not use it as the handoff candidate until the active set is frozen, dead
experiment code is removed, the patch is regenerated, and the final gate
passes.

## Contents

| Area | Status |
| --- | --- |
| Q8_0 MMQ/rocBLAS dispatch and threshold instrument | Proven PP direction on current head |
| Q4_K MMQ metadata precompute and stride 9 | Combined Q4_K/Q6_K PP set regresses; isolate |
| Q6_K MMQ `min_blocks=1` | Historical support; current-head contribution unresolved |
| Q4_K branchless MMVQ scales and stored Q8_1 sum | Retain with one-sample/order caveat |
| Q4_K MMQ `min_blocks=3` | Unproven in a regressing combined set; compile gate disabled |
| Q4_K/Q8_0 paired MMVQ helpers | Unproven; compile gate disabled |
| Q8_1 and Flash Attention DPP paths | Unproven; compile gates disabled |
| Flash Attention 8-lane quantized KQ | Unproven; compile gate disabled |
| Q8_0 MMVQ VDR 4 | Rejected; source retained, compile gate disabled |

Do not infer acceptance from `optimal-gfx906.series`; the filename is
historical. The live decision state and exact next steps are in `../README.md`.
All graph exports and model tests keep upstream Flash Attention enabled with
`-fa on`; only custom FA patch code is disabled.

## Legacy

`legacy/2026-06-old-base/` contains code patches against the older source
layout. They are porting references only. Their old handoff prose was removed;
current decisions live in `../README.md` and `../RESULTS.md`.
