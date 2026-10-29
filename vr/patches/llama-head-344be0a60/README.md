# gfx906 overlay against llama.cpp 344be0a60

Upstream base: `344be0a60074e6395da3c4a4f300f7361df5a67d` (ggml-org/llama.cpp, 2026-10-25). The resolved target tree is `252e58f45c368fb9b7208f10a3ec07e1ab1b7d6e`, pinned by `refs/port-snapshots/gfx906-344be0a60074e6395da3c4a4f300f7361df5a67d`. The intermediate tree is `560be2a07763631fe14f431ff6bedb6a66636ad2`, pinned by `refs/port-snapshots/gfx906-mid-head-344be0a60`. No commit was created.

| Patch | Files | Content |
| --- | ---: | --- |
| `0001-ggml-core-and-tp-meta-backend.patch` | 24 | ggml core, tensor-parallel meta backend and interfaces |
| `0002-ggml-hip-gfx906-kernels.patch` | 43 | HIP/GPU runtime, gfx906 tuning, repack and AllReduce |
| `0003-llama-model-runtime.patch` | 36 | models, placement, split KV and speculative paths |
| `0004-common-tools-tests-harness.patch` | 13 | common, tools, tests, root build and VR harness |
| `0005-tp-token-graph-fusions-ar-overlap.patch` | 1 | final `src/llama-model.cpp` integration changes |

Apply patches in numeric order. The four main groups are file-disjoint; patch 0005 carries the final model placement/runtime file after those groups. The older `llama-head-797de63b2` kit remains as its pinned historical reference.

The merge reconciled upstream's chunked HIP/CUDA graphs with the gfx906 kernel-function update safety check, retained the host-resident PLE table and CPU Engram placement, and adopted upstream's current DSV4.1 implementation and server interfaces. The merged HIP build passed for `llama-cli` and `llama-bench` using TheRock ROCm 10.2.0:

```sh
cmake --build build-ilp --target llama-cli llama-bench -j 8
```

The Flash Next IQ3_XXS run measured 552.3 PP / 36.7 TG with a 4,092-token prompt and 4,096-token output limit. Context and quant limits are in [the gfx906 Qwen3.8 Flash Next report](../../results/2026-10-25-qwen3.8-flash-next-gfx906.md). Patch replay compared 117 runtime files against the resolved target with zero mismatches; all five SHA256 checks passed. This merge did not run the full project test suite, CUDA, SYCL, or a full-context 262K conversation benchmark.

Regenerate with `python3 vr/patches/llama-head-344be0a60/regenerate.py`; verify with `python3 vr/patches/llama-head-344be0a60/verify.py` and `(cd vr/patches/llama-head-344be0a60 && sha256sum -c SHA256SUMS)`. The pinned references above preserve the uncommitted target and intermediate trees.
