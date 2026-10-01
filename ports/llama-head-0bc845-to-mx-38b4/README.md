# MX runtime port kit: llama.cpp head to 2026-10-01 checkpoint

This kit exports the committed runtime difference from clean llama.cpp `0bc845d356f437d5ce4fe975c36428f7522829cb` to MX checkpoint `38b4da7ba6714919ccdb8a19ac82489e34ee7855`. It is an exact source overlay for 115 changed runtime files. The five patches have disjoint file lists, apply in filename order, and include full blob IDs for three-way application on later upstream commits. They are grouped by file ownership, not by independently selectable optimization.

The patches omit general documentation, the upstream CI workflow, historical `vr` notes, old experimental patches, and generated `vr/build` files. `MANIFEST.tsv` maps every exported file to a patch; `EXCLUDED.txt` lists every changed file outside the runtime export. The repository source files, especially `ggml/src/ggml-cuda/q8_repack/README.md`, are the implementation reference. The archived campaign notes are in `vr/OLD-IGNORE`.

| Patch | Contents |
| --- | --- |
| `0001-core-ggml.patch` | ggml core, backend meta, CPU and other backend integration |
| `0002-gpu-backends.patch` | CUDA/HIP kernels, Q8 repack, host-staged AllReduce, GPU build files |
| `0003-model-runtime.patch` | model loading, tensor placement, graph code, conversion and GGUF support |
| `0004-common-tools.patch` | common options, applications, bench, and existing backend test changes |
| `0005-build-harness.patch` | root CMake and the active gfx906 tensor-parallel harness |

## Apply to the pinned llama.cpp head

Start from the pinned commit in a clean checkout or worktree. Set `KIT` to this directory's absolute path. The commands stop on the first conflict.

```sh
KIT=/home/raistlin/infer/mx-llama.cpp/ports/llama-head-0bc845-to-mx-38b4
(
    set -e
    cd /path/to/clean/llama.cpp
    test "$(git rev-parse HEAD)" = 0bc845d356f437d5ce4fe975c36428f7522829cb
    test -z "$(git status --porcelain)"
    (cd "$KIT" && sha256sum -c SHA256SUMS)
    for patch in "$KIT"/patches/*.patch; do
        git apply --check "$patch"
        git apply "$patch"
    done
)
```

The expected result is a working tree whose files in `MANIFEST.tsv` exactly match checkpoint `38b4da7ba`. Run `python3 "$KIT/verify.py"` from the source repository to replay the patches in a temporary clean archive and compare every exported file byte for byte with the checkpoint. This check does not build or benchmark.

## Port to a newer llama.cpp head

Create a clean worktree at the new upstream commit and apply the same patches with `git apply --3way` in order. Resolve conflicts in the source, especially where upstream has changed an interface, then build and benchmark. A clean apply only proves that text was carried forward; it does not prove that the fast dispatch path still runs. For each new checkpoint, record the upstream commit, inspect `git diff` against it, and run `regenerate.py --base <upstream-commit> --target <accepted-mx-commit>` from the copied kit directory. The exporter fails if one of its five runtime groups becomes empty. Review `EXCLUDED.txt` for new paths that need an explicit group before treating a new export as complete.

The highest value post-merge checks are these integration seams:

1. Q8_0 weights must enter the gfx906 repack buffer at upload and reach only repack-aware `MUL_MAT`, `MUL_MAT_ID`, and supported fused single-token kernels. `--no-repack` is the control. The weight layout and dispatch thresholds are documented in `ggml/src/ggml-cuda/q8_repack/README.md`.
2. `ggml/src/ggml-cuda/allreduce.cu` must compile for HIP (`#if !defined(GGML_USE_MUSA)`). Its host-staged pipeline must execute on the normal tensor-parallel path without `GGML_CUDA_TP_OVERLAP=1`. Pipeline initialization alone is not proof of execution. On PCIe-only gfx906, use `GGML_CUDA_ALLREDUCE=internal` when specifically checking this transport. `GGML_CUDA_AR_BF16_THRESHOLD=0` is the F32 wire control; the default BF16 wire path changes precision.
3. `ggml/src/ggml-backend-meta.cpp` must preserve per-lane repack buffer selection and concurrent lane dispatch. `GGML_META_PARALLEL_DISPATCH=0` is the serial control.
4. `src/llama-model.cpp` must split shared-expert up/gate and down weights across tensor-parallel lanes and merge their partial output before the existing reduction. `LLAMA_SHEXP_SPLIT=0` is the mirrored-weight control.

## Build and measure on the two-gfx906 host

```sh
cmake -S . -B build -DAMDGPU_TARGETS=gfx906 -DGGML_HIP=ON -DGGML_HIP_RCCL=OFF -DGGML_CCACHE=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_VR_EXPERIMENTS=ON
cmake --build build -j 16 --target llama-bench vr-test-gfx906-tp
MODEL=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/fe1e2a23d973adb629709749dc4f6756df66ef10/Qwen3.8-27B-Q8_0.gguf
HIP_VISIBLE_DEVICES=0,1 ./build/bin/llama-bench -m "$MODEL" -ngl 999 -fa on -sm tensor -dev ROCm0/ROCm1 -b 4096 -ub 4096 -p 0 -n 0 -pg 2048,0 -pg 0,256 -d 16384 -r 3 -o jsonl --progress --no-warmup
```

Record PP2048 and TG256 separately and compare with the pre-merge checkpoint using identical build flags, environment, model, and benchmark settings. The 2026-10-01 checkpoint measured 431.13 PP / 27.81 TG tokens/s in one default run. A separate two-repetition `GGML_ENABLE_CUSTOM_AR=0` run averaged 435.14 PP / 28.43 TG. Those numbers are context for this host, not a portable threshold. Use `--no-repack`, `GGML_META_PARALLEL_DISPATCH=0`, `LLAMA_SHEXP_SPLIT=0`, and `GGML_ENABLE_CUSTOM_AR=0` only as one-change-at-a-time controls when a regression needs isolation.

The old exact-shape `GGML_CUDA_TP_OVERLAP` experiment is optional and archived under `vr/OLD-IGNORE`. It is not required to reproduce the standard benchmark above.

## Regenerate this export

`regenerate.py` reads committed trees only, writes the five patches, `MANIFEST.tsv`, `EXCLUDED.txt`, and `SHA256SUMS`, and leaves source files alone. Its default refs are the two commits named above. Its output path is the directory containing the script. Pass the same `--base` and `--target` values to `verify.py` after any regeneration. The current files have passed that check.
