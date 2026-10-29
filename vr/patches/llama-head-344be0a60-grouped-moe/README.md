# gfx906 overlay with accepted grouped MoE

Official base: `344be0a60074e6395da3c4a4f300f7361df5a67d`.
Source: checkpoint `84932540a` plus the seven grouped-MoE kernel/dispatch/test files and removal of the unreachable legacy HIP AllReduce block, exported without creating a commit. Other uncommitted documentation/archive changes are not part of this runtime snapshot.

- Target tree: `5f6a45b574595e809c83a03f343ba8b801431111`, ref `refs/port-snapshots/gfx906-grouped-moe-clean-20261029-target`.
- Intermediate tree: `00224c01e87a837c7630b024219a0224a43e1398`, ref `refs/port-snapshots/gfx906-grouped-moe-clean-20261029-mid`.

Apply all five patches in numeric order. Patches 0001-0004 are file-disjoint; 0005 applies the complete local `src/llama-model.cpp` delta after them. The kit includes the checkpoint's attention/vector-dot fixes, which the previous export lacked.

| Patch | Scope |
| --- | --- |
| 0001 | ggml core and TP meta backend |
| 0002 | HIP/gfx906 kernels, repack, AllReduce, grouped MoE |
| 0003 | Model runtime and interfaces |
| 0004 | Common/tools/tests/build and both focused VR harnesses |
| 0005 | Final model placement/runtime integration |

Grouped MoE reuses decoded expert weights across tokens while keeping original routing, precision and separate outputs. It remains opt-in via `GGML_CUDA_FLASHNEXT_MOE=1`. The separate `LLAMA_QWEN4_MTP_DEVICE` GPU-feedback experiment is not included.

[Historical evidence](../../old-ignore/2026-10-25/flashnext-fast/README.md): grouped-only mean TG 50.45 versus 47.65 on two short runs per variant, matching text and acceptance, 18 canonical and 24 repacked CPU-reference cases passed. Those results predate this integration. The wider MMQ fallback, sampled generation, concurrency and other GPU architectures are not established by that evidence.

Regenerate and replay from this repository:

```sh
python3 vr/patches/llama-head-344be0a60-grouped-moe/regenerate.py
python3 vr/patches/llama-head-344be0a60-grouped-moe/verify.py
(cd vr/patches/llama-head-344be0a60-grouped-moe && sha256sum -c SHA256SUMS)
```

The snapshot refs are local retention aids. Patches can replay without them; regeneration needs the documented tree objects. Review `MANIFEST.tsv` and `EXCLUDED.txt` against the full runtime delta, because replay only checks manifest paths. Follow [the catch-up guide](../../how-to-catchup.md) for the next update and [the current guide](../../README.md) for build settings.

[Integration validation](../../old-ignore/2026-10-29/grouped-moe-integration/README.md): HIP build and 121-file replay passed; 18 canonical cases passed. The repacked harness exceeded the shared 110-second limit, so current repacked/TP runtime validation is incomplete. No new performance run was made.

Cleanup removed the unreachable 290-line HIP overlap-only `#elif`; the active host-staged, graph-counter, duplex/slab and scheduler paths remain. The validation linked above predates this dead-code removal.
