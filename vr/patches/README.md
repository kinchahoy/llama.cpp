# gfx906 patch stack

Last updated: 2026-07-19

The runtime-impacting gfx906 delta is split into five current-head patches.
They target upstream `571d0d540`, which was also the official `origin/master`
head when verified on 2026-07-19.

The private branch is three commits ahead and zero commits behind that head.
Most branch volume is `/vr` evidence and tooling. The portable GGML delta is
only seven files: 197 insertions and 11 deletions. Apply the patches here, not
the branch commits, when moving the optimization work to a newer llama.cpp.

## Patches

| Patch | Scope | Evidence state |
| --- | --- | --- |
| `0001-gfx906-q8-rocblas-dispatch.patch` | Q8_0 MMQ/rocBLAS crossover | Proven PP direction; final tensor-mode confirmation remains |
| `0002-gfx906-q4k-mmvq-branchless-sum.patch` | Branchless Q4_K scale decode plus stored Q8_1 sum | Retain with sample/order caveat |
| `0003-gfx906-q4k-mmq-precompute-stride9.patch` | Q4_K PP metadata precompute plus stride 9 | Unresolved; test as one Q4_K bundle |
| `0004-gfx906-q6k-min-blocks1.patch` | Q6_K PP `min_blocks=1` | Historical support; unresolved on current head |
| `0005-gfx906-profile-wiring.patch` | Exact gfx906-only CMake profiles | Infrastructure; apply last |

The patches have disjoint files except for their intentional dependency on the
profile macros. This keeps conflicts attributable to one optimization area.
Patch 0005 does not make Q8 dispatch optional: patch 0001 is source dispatch
logic shared by every private profile.

## Series

| Series | Composition | Use |
| --- | --- | --- |
| `retained-gfx906.series` | 0001 + 0002 + 0005 | Best evidence-backed set before final validation |
| `q4-gfx906.series` | retained + 0003 | Isolate the Q4_K PP bundle |
| `q6-gfx906.series` | retained + 0004 | Isolate Q6_K PP occupancy |
| `combined-gfx906.series` | all five | Exact active source composition in this worktree |
| `all-gfx906.series` | same as combined | Historical compatibility name |
| `optimal-gfx906.series` | same as retained | Historical compatibility name; not a final validation claim |

With the retained series, build only profile `common`. Profile `q4` requires
0003, profile `q6` requires 0004, and profile `combined` requires both.
Profile `none` is not clean upstream because patch 0001 remains active source
logic.

## Port to a new head

Start from a clean checkout of the new upstream head and copy this patch
directory into it. Apply one chosen series atomically:

```bash
vr/patches/apply-series.sh retained-gfx906.series
```

The helper passes every listed patch to one `git apply` invocation; it does not
perform a redundant preflight/apply pair. On a conflict, nothing should be
treated as accepted. Do not apply the archived monolith as a fallback.

For each conflict:

1. Compare the upstream file with the corresponding numbered patch.
2. Re-express only that patch's behavior in the new local structure.
3. Regenerate that numbered patch against the new base.
4. Apply profile wiring last and verify the active definitions in
   `compile_commands.json`.

Do not forward-port 0003 or 0004 merely because they exist in the combined
worktree. Carry them only while they remain useful ablation candidates, then
delete a loser from the selected series and live source.

After a mechanical conflict-free port, use the fail-fast gates in
`../WORKTREE-BENCHMARKS.md`. A source-layout-only rebase does not justify
repeating old screens. Recheck the affected exact operator shapes, then pay
for one model cell only if executed code or dispatch changed materially.

## Historical code

The former current-head monolith is preserved at
`../archive/rejected-patches/gfx906-bigbang-pre-modular-2026-07-19.patch`.
It contains compile-disabled and rejected experiments and is not an accepted
candidate. Older-layout modular references remain under
`legacy/2026-06-old-base/`.

Disabled DPP, custom Flash Attention, wide-VDR, paired-MMVQ, and Q4_K
`min_blocks=3` implementations are no longer present in live GGML source.
Their code remains available only in those historical artifacts.
