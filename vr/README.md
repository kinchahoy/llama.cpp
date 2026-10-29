# gfx906 guide

Local enhancements for two 32 GiB gfx906 GPUs (MI50/MI60), primarily tested with Qwen3.8-27B Q8_0. Built on [MX](https://github.com/mixa3607/mx-llama.cpp) and [iacopPBK's gfx906 work](https://github.com/iacopPBK/llama.cpp-gfx906); see [credits](IMPROVEMENTS.md#built-on). The merged official base is `344be0a60`; source snapshots and benchmark identities are recorded in the performance reports.

| Guide | Purpose |
| --- | --- |
| [Improvements](IMPROVEMENTS.md) | What changed, why it helps, and what to revisit |
| [Performance](PERFORMANCE.md) | Representative measurements and their limits |
| [Catch up](how-to-catchup.md) | Merge upstream, validate briefly, and refresh patches |
| [Current patch kit](patches/llama-head-344be0a60-grouped-moe/README.md) | Replayable checkpoint overlay plus accepted grouped MoE |
| [Historical records](old-ignore/README.md) | Dated benchmarks, experiments, and superseded workflows |

## Priorities and separate work

1. Qwen3.8-27B at Q8_0 is the primary performance target.
2. Gemma 4 at Q8 and Qwen Next at a quantization that fits the two GPUs are secondary targets.

`flashnext-fast` produced the accepted grouped-MoE weight-reuse path, now included in the source and active patch set. Enable it with `GGML_CUDA_FLASHNEXT_MOE=1` for eligible gfx906 MoE verification; it remains opt-in. Its separate GPU-feedback experiment stays in the original worktree. See [the evidence](old-ignore/2026-10-25/flashnext-fast/README.md).

`play-qwen-off` is a separate project exploring how to reduce Qwen's wordiness. Keep its branch and worktree separate from the gfx906 performance patch chain; assess it on output behavior rather than PP/TG. Both worktrees remain in place with their local changes intact.

## Build

Use the local TheRock environment with gfx906 support. From the repository root:

```sh
source /home/raistlin/amd-clean/env/therock.sh
cmake -S . -B build-ilp \
  -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=OFF \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON \
  -DLLAMA_OPENSSL=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release \
  -DHIP_COMPILER=clang -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  -DCMAKE_HIP_FLAGS="-mllvm -amdgpu-sched-strategy=max-ilp" \
  -DCMAKE_CXX_FLAGS="-O3 -Wno-unused-command-line-argument"
cmake --build build-ilp --target llama-cli llama-server llama-bench -j 8
```

One GPU is on CPU PCIe lanes and the other on motherboard lanes. Keep RCCL off and use host-staged AllReduce on this topology. Revisit RCCL after fixing the lane placement; this is a host constraint, not a permanent gfx906 rule.

## Run

This example requires the cached model. Keep a 50/50 tensor split while comparing changes.

```sh
./build-ilp/bin/llama-cli \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline \
  -ngl 999 -dev ROCm0,ROCm1 -sm tensor -ts 1,1 -fa on \
  -b 2048 -ub 2048 -c 32768 --temp 0 --top-k 1 -s 42
```

For single-request MTP, append `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling`. This is a useful starting point, not a universal winner; concurrency and prompt length change the result. Use the same model/GPU flags with `llama-server` for serving, setting host and port explicitly.

Keep HIP graphs enabled and `GGML_CUDA_TP_OVERLAP` unset. The normal HIP AllReduce path does not need that switch. Most local optimizations are enabled by default; historical environment sweeps are in the archive.

## Validation

The focused TP harness is built with `-DLLAMA_BUILD_VR_EXPERIMENTS=ON`, target `vr-test-gfx906-tp`. Follow the catch-up guide's combined two-minute runtime-check budget after a build. Longer checks and performance comparisons require explicit user OK. `catchup-script.sh bench` previews its command unless approved execution is requested.

`master` is the final branch. Use `experiment/topic` for unfinished work and `archive/YYYY-MM-DD/topic` for retained historical branches; see the [cleanup record](old-ignore/2026-10-29/cleanup.md).
