# gfx906 guide

Local enhancements for two 32 GiB gfx906 GPUs (MI50/MI60), primarily tested with Qwen3.8-27B Q8_0. Built on [MX](https://github.com/mixa3607/mx-llama.cpp) and [iacopPBK's gfx906 work](https://github.com/iacopPBK/llama.cpp-gfx906); see [credits](IMPROVEMENTS.md#built-on). The active overlay targets official llama.cpp `36a0e2ffc0b352b774f8d71c43668b75b3061ef5` (upstream master on 2026-10-29). It is stage D of the [2026-10-29 rebuild](old-ignore/2026-10-29/rebuild/README.md): upstream plus cleanup and fixes, without pipeline stages, and with DeepSeek V4/V4.1, MiniMax-M2 and MiMo-2 back on upstream code. Source snapshots and benchmark identities are recorded in the performance reports.

| Guide | Purpose |
| --- | --- |
| [Improvements](IMPROVEMENTS.md) | What changed, why it helps, and what to revisit |
| [Performance](PERFORMANCE.md) | Representative measurements and their limits |
| [Catch up](how-to-catchup.md) | Merge upstream, validate briefly, and refresh patches |
| [Current patch kit](patches/llama-head-36a0e2ffc/README.md) | Replayable overlay for the `36a0e2ffc` base |
| [Rebuild record, 2026-10-29](old-ignore/2026-10-29/rebuild/README.md) | Stages A-D, cumulative patches, results, status log and what was dropped |
| [Benchmark scripts](scripts/README.md) | Build flags, llama-bench, output and MTP checks used for the rebuild |
| [Previous patch kit](old-ignore/2026-10-29/patches/llama-head-418f905a4/README.md) | Historical overlay for the old `418f905a4` base |
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

Keep the `CMAKE_HIP_FLAGS` and `CMAKE_CXX_FLAGS` values above for every comparison build. A build without them was 2-3% slower on 27B in the 2026-10-29 rebuild.

One GPU is on CPU PCIe lanes and the other on motherboard lanes. Keep RCCL off and use host-staged AllReduce on this topology. Revisit RCCL after fixing the lane placement; this is a host constraint, not a permanent gfx906 rule.

## Run

This example requires the cached model. Keep a 50/50 tensor split while comparing changes.

```sh
./build-ilp/bin/llama-cli \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline \
  -ngl 999 -dev ROCm0,ROCm1 -sm tensor -ts 1,1 -fa on \
  -b 2048 -ub 2048 -c 32768 -t 20 -tb 20 --temp 0 --top-k 1 -s 42
```

Twenty CPU threads are the requested comparison setting for this host, not a claim that 20 is optimal for every workload.

For single-request MTP, append `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling`. This is a useful starting point, not a universal winner; concurrency and prompt length change the result. Use the same model/GPU flags with `llama-server` for serving, setting host and port explicitly.

Keep HIP graphs enabled and RCCL off for the current mixed PCIe-lane topology. HIP uses the upstream host-staged AllReduce path plus the local graph-safe counters and slab/duplex overlap hooks; the old `GGML_CUDA_TP_OVERLAP` switch is removed. The peer-write AllReduce is opt-in with `GGML_CUDA_PEER_AR=1`. Most local optimizations are enabled by default; historical environment sweeps are in the archive. Multi-stage TP (`-tps`) is removed; see [the rebuild changes](IMPROVEMENTS.md#2026-10-29-rebuild-removed-and-changed).

## Validation

The focused TP harness is built with `-DLLAMA_BUILD_VR_EXPERIMENTS=ON`, target `vr-test-gfx906-tp`. Follow the catch-up guide's combined two-minute runtime-check budget after a build. Longer checks and performance comparisons require explicit user OK. `catchup-script.sh bench` previews its command unless approved execution is requested.

`master` is the final branch. Use `experiment/topic` for unfinished work and `archive/YYYY-MM-DD/topic` for retained historical branches; see the [cleanup record](old-ignore/2026-10-29/cleanup.md).
