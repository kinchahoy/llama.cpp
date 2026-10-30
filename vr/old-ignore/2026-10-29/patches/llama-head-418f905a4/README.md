# Current gfx906 patch kit

Official base: `418f905a4efe910d2a75636bbe65d48eca38790a`. This is the accepted overlay for master, including opt-in grouped MoE, restored gfx906 MMQ traversal, and single-token IQ2_S expert tuning. GPU feedback is excluded; RCCL stays off and host-staged AllReduce remains the default for our topology.

- Runtime target tree: `fb7da801711c102d7d6bebec3b35c2f5030d3769`, ref `refs/port-snapshots/gfx906-418f905a4-final-20261029-target`.
- Intermediate tree: `342d4ab70e1b6042f36beaba46ea924921b65ad0`, ref `refs/port-snapshots/gfx906-418f905a4-final-20261029-mid`.

Apply all five patches in numeric order. Groups 0001-0004 are file-disjoint; 0005 carries the final `src/llama-model.cpp` delta. The runtime target excludes fork documentation and archives. Local tree refs retain regeneration inputs; applying the patches does not require those refs.

The port preserves upstream MMQ chunking and current model/cache interfaces. The gfx906 non-stream-K traversal restores prompt throughput in the measured Flash Next workload. The two-row IQ2_S variant is limited to gfx906 single-token expert operations with even row counts; dense Q8_0 is unaffected. Its +2.37% TG result is one matched pair, not a repeatability claim. See [performance evidence](../../../../PERFORMANCE.md) and [exact kernel experiment](../../../../old-ignore/2026-10-29/iq2s-single-token-experiment/README.md).

HIP builds, focused reference checks and matched CLI runs passed on the source represented here. These checks do not establish every model, shape, or GPU architecture. The first port had a PP regression; acceptance includes the subsequent traversal fix, not that original snapshot.

Regenerate: `python3 vr/old-ignore/2026-10-29/patches/llama-head-418f905a4/regenerate.py`.
Replay: `python3 vr/old-ignore/2026-10-29/patches/llama-head-418f905a4/verify.py`.
Checksums: `(cd vr/old-ignore/2026-10-29/patches/llama-head-418f905a4 && sha256sum -c SHA256SUMS)`.

`MANIFEST.tsv` covers all runtime differences from the pinned official base; `EXCLUDED.txt` is empty for this runtime snapshot.
