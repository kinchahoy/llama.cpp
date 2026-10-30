# gfx906 patch kit on 36a0e2ffc

Official base: `36a0e2ffc0b352b774f8d71c43668b75b3061ef5`. This is stage D of the 2026-10-29 rebuild: upstream plus the cleaned overlay, without multi-stage TP (`-tps`), and with DeepSeek V4/V4.1, MiniMax-M2 and MiMo back on upstream code. RCCL stays off and host-staged AllReduce remains the default; peer AllReduce is opt-in (`GGML_CUDA_PEER_AR=1`).

- Runtime target tree: `3875d3850b45825cc12d5aecb22487b7377ad598`, ref `refs/port-snapshots/gfx906-36a0e2ffc-rebuild-20261029-target`.
- Intermediate tree: `14fb69bd91d3048dffd7bc20f489e607f28fc8fe`, ref `refs/port-snapshots/gfx906-36a0e2ffc-rebuild-20261029-mid`.
- Full stage D tree (runtime plus fork docs): `b8f809e2fae2cbc929196479cc60fe29d4c7fd0d`, ref `refs/port-snapshots/gfx906-36a0e2ffc-rebuild-20261029-staged-full`.

Apply all five patches in numeric order. Groups 0001-0004 are file-disjoint; 0005 carries the final `src/llama-model.cpp` delta. The group rules and patch names are the same as in `../llama-head-418f905a4/`. The runtime target excludes fork documentation (`EXCLUDED.txt` lists the paths, taken from the full tree). Local tree refs retain regeneration inputs; applying the patches does not require those refs.

Performance, output checks and the list of cut features: [rebuild README](../../rebuild-2026-10-29/README.md).

Regenerate: `python3 vr/patches/llama-head-36a0e2ffc/regenerate.py`.
Replay: `python3 vr/patches/llama-head-36a0e2ffc/verify.py`.
Checksums: `(cd vr/patches/llama-head-36a0e2ffc && sha256sum -c SHA256SUMS)`.
