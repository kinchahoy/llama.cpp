# gfx906 kit for official llama.cpp 418f905a4

Experimental candidate kit; not promoted to the current kit. The build and focused op checks passed. A matched one-run Flash Next comparison completed, with PP 345.09 -> 247.31 tok/s (-28.3%) and TG 37.68 -> 40.22 tok/s (+6.7%). Output and draft acceptance matched, but the PP regression remains unexplained. See [validation records](../../old-ignore/2026-10-29/head-merge-validation/README.md).

- Official base: `418f905a4efe910d2a75636bbe65d48eca38790a` (2026-10-29).
- Runtime target tree: `12229543ad2d128b4663a2f9839401dfb7bec6cb`, ref `refs/port-snapshots/gfx906-418f905a4-20261029-target`.
- Intermediate tree: `2fe701f8a0bdcea569d8281954c0f980b50b1b2b`, ref `refs/port-snapshots/gfx906-418f905a4-20261029-mid`.

Apply all five patches in numeric order. Groups 0001-0004 are file-disjoint; 0005 carries the final `src/llama-model.cpp` delta. This kit includes accepted opt-in grouped MoE and removes the unreachable legacy HIP AllReduce block. GPU feedback is excluded.

The port preserves upstream MMQ workspace/chunking and architecture additions alongside gfx906 selectors, fusion and scheduler hooks. DSV4 cache constructors and HC-mix callers use current upstream interfaces. HIP Release builds passed for CLI, bench, server, backend-op tests and the TP harness; the canonical grouped filter passed 18/18 on ROCm0. Those checks do not establish performance or correctness of every model path.

Regenerate with `python3 vr/patches/llama-head-418f905a4/regenerate.py`; replay with `python3 vr/patches/llama-head-418f905a4/verify.py`; check `(cd vr/patches/llama-head-418f905a4 && sha256sum -c SHA256SUMS)`. All runtime differences from this base are covered by `MANIFEST.tsv`; `EXCLUDED.txt` is empty for the pinned candidate tree. Local tree refs retain regeneration inputs; patch application does not need those refs.
