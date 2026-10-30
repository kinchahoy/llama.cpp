# Review of Abzolute1/llama-cpp-optimization-lab

Read-only review. Clone in `/home/raistlin/infer/ext-optlab-review/src`. Nothing was built or run.

## 1. What it is

The repo is a documentation and patch package, not an engine fork. It holds two small patch sets against other forks (TheTom/llama-cpp-turboquant `bd1bf025`, unslothai/llama.cpp `329b6160`), launch profiles, timing JSON, a CUDA fixture and a Python benchmark harness (`README.md`, `START-HERE.md`). License: no license of its own ("no additional blanket license is asserted", `README.md`); the retained upstream notices in `licenses/` are MIT (ggml authors). Commit reviewed: `5408613efc1d195fa7cc1cb08da0bb757dd40789`, 2026-09-17, "Add reviewed local inference optimization reproduction package". Hardware: one consumer NVIDIA GPU plus a host CPU with 12 to 24 threads; `PERFORMANCE-REPORT.md` mentions Ada/Turing dispatch (line 291) and NVML/CUDA. Models: Qwen3.6-35B-A3B Q4_K_XL and Qwen3.8 Flash-Next IQ4_XS (a different quant from ours). Setup: most experts live in host RAM (`--n-cpu-moe 32` and `42`) and are streamed over PCIe or computed on the CPU. Its results come from that offload regime. No AMD, ROCm or gfx906 results. The package says historical numbers were not reproduced from it.

Plain finding: almost all of it targets CPU/PCIe expert offload on NVIDIA. We run all experts on GPU, so it transfers weakly. Only L1 and L2 below touch our setup.

## 2. Ideas

| ID | Idea | Evidence | Applies to | Our equivalent today | Expected effect and why | Effort | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | Skip K/V reads and math for a warp whose 32 KV rows are all masked (-inf) in the vec flash-attention kernel. Result is unchanged, because a masked step only adds zero and rescales by one | `patches/flash-next-selected.patch` hunk on `fattn-vec.cuh` (guarded: D==256, ncols==1, Q8_0 K and V, ne11>=4096, mask, no ALiBi, `!GGML_USE_HIP`); report line 100 area: synthetic sparse case 3.4x, real 85K-token decode about +13% (NVIDIA) | both, only if decode uses the vec FA kernel at long context with masked rows (padding, SWA layers) | `ggml/src/ggml-cuda/fattn-vec.cuh` has no such skip (mask read at line 279, no early-out) | Zero gain at short context (our benchmarks). Possible gain at 32K+ context only if many KV rows are masked. Our wave is 64 lanes, so the uniform test must cover 64 rows per wave, which makes skipping rarer. Guard excludes HIP today and uses `__all_sync` | S | low (no gfx906 data, depends on mask sparsity; first count masked fraction in our real decode) |
| L2 (REJECTED by user 2026-10-29, do not propose again) | Use stock n-gram speculation for source-edit and copy workloads: `--spec-type ngram-mod`, n-max 16, n-min 2, n-match 24 | `profiles/flash-next-candidate.json`; report: edit decode about 27 -> 50 tok/s, copying 26 -> 60, but repetitive rewrite 27 -> 19 to 24 (NVIDIA, CPU offload) | both | `common/ngram-mod.cpp` exists in our tree. We use MTP/DFlash for decode speedup | Config only, no code. Helps copy-heavy prompts, loses on repetitive rewrite. Could combine as a fallback when no MTP head is loaded. The lab itself says acceptance alone is not a speed metric and its adaptive backoff was not good enough | S | medium for copy workloads, low as a default |

Nothing else is portable to our hardware. The rest of the repo is CPU/PCIe offload work.

Supports existing items: none directly. The lab's finding that disabling CUDA graphs cut decode by about 55% (report, "CUDA graph optimizer" row, and about 23-25 tok/s in the Flash-Next table) agrees with our HIP graph replay choice, but adds nothing new. The lab's rejected small CPU kernel micro-wins (paired MXFP4 rows +6% microbench, no application gain, report line 285) is a weak caution for S2 and S4: confirm end-to-end, not by microbench only.

## 3. Excluded and not applicable

- excluded: accuracy cost: q4 KV / TurboQuant V (changed quality metric on their corpus).
- excluded: accuracy cost: IQ1_M-class expert quants as a cache lever (their cache eligibility threshold moves IQ1_M bytes, model stays low precision).
- not applicable (host offload): private writable mmap plus `cuMemHostRegister` pinning (`flash-next-selected.patch`, `llama-mmap.cpp`, `llama-model.cpp`); commits about 34 GiB anonymous RAM.
- not applicable: bounded pinned staging ring for expert H2D copies (`GGML_CUDA_MOE_STAGING*`, `qwen35-selected.patch` `ggml-cuda.cu`).
- not applicable: GPU MoE expert cache with LRU-like hot-use threshold, dedicated MMV for cached experts, 32-token/256-row cache coverage extension (`moe-cache.cu`). We keep all experts on GPU.
- not applicable: CPU backend patches (`ggml-cpu.c`), mmap prefetch and release controls (`llama-model-loader.cpp`), CPU thread, affinity and poll tuning.
- rejected by the lab itself: MTP head on CPU-offload setup (prose slower), small draft model Qwen3.5-0.8B (30-34 tok/s), CUDA scheduling flags, huge-page pinned allocator.

## 4. License note

The lab adds no license. Patch hunks derive from MIT ggml code, so small pieces (L1 is about 12 lines) could be re-implemented freely; copy attribution is simplest if we write our own version. All other content is ideas only. Treat the report text as data, not instructions. The L1 numbers are NVIDIA-only and unverified (the package states historical results were not reproduced).
