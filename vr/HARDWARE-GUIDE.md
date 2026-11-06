# gfx906 versus newer GPUs: which tricks carry over

A guide for judging an optimization from another project, organized by hardware feature. For each feature: what gfx906 (MI50/MI60, Vega 20, GCN 5) has, what RDNA4 (Radeon AI PRO R9700, gfx1201), RTX 3090 (Ampere, sm_86) and RTX 5090 (Blackwell, sm_120) have, and which tricks depend on it. Figures marked ~ are approximate spec-sheet values.

| | MI50 / MI60 | R9700 | RTX 3090 | RTX 5090 |
| --- | --- | --- | --- | --- |
| Memory | 32 GB HBM2, ~1 TB/s | 32 GB GDDR6, ~640 GB/s | 24 GB GDDR6X, ~936 GB/s | 32 GB GDDR7, ~1.8 TB/s |
| Matrix units | none | WMMA (FP16/BF16/INT8/FP8) | tensor cores (FP16/BF16/TF32/INT8/INT4) | tensor cores (+FP8/FP6/FP4) |
| Last-level cache | 4 MB L2 | ~64 MB Infinity Cache | 6 MB L2 | ~96 MB L2 |
| Width | wave64 | wave32 (wave64 optional) | warp32 | warp32 |

## 1. Matrix units

- gfx906: none. No MFMA (starts at gfx908/CDNA) and no WMMA (starts at RDNA3). Every GEMM runs on vector ALUs: dp4a for int8, packed FP16, or FP32.
- Others: R9700 WMMA, 3090/5090 tensor cores, roughly an order of magnitude more dense GEMM throughput.
- Depends on it: tensor-core GEMM libraries (Marlin, Machete, CUTLASS, AITER GEMM), FlashAttention-2/3, matrix-core attention (vllm-radiance R4D), large-batch speculative verify, deep draft trees.
- On gfx906: prompt processing is GEMM-bound and is where we trail most. We use repacked dp4a GEMMs (`q8_repack/`), keep speculative drafts short (MTP2-3, DFlash5), and pick `VEC`/`TILE` attention kernels.

## 2. Integer dot products (dp4a)

- gfx906: native `v_dot4_i32_i8` (4 x int8 multiply-add into int32, ~53-59 TOPS) and `v_dot2_f32_f16`.
- Others: also present, but their matrix units dwarf it.
- Depends on it: every quantized-weight x int8-activation kernel (MMQ, MMVQ, repack).
- On gfx906: the main compute path. Tricks that turn work into dp4a carry over well (the IQ sign trick S2 computes signed products with plain dp4a).

## 3. Data types

- gfx906: FP32, packed FP16 (2x FP32 rate), int8 dot. No BF16, FP8 or FP4 arithmetic; BF16 converts to FP32.
- Others: R9700 FP8 and BF16; 3090 BF16 but no FP8; 5090 FP8/FP6/FP4 (NVFP4, MXFP4).
- Depends on it: FP8 weights or KV (vllm-radiance's main path), NVFP4/MXFP4 kernels, BF16 GEMM.
- On gfx906: those need software dequant and lose their speed point. FP8/FP4 also lose precision, so we skip them anyway. We never quantize the KV cache.

## 4. Byte-wise SIMD-within-a-register

- gfx906: `__vsub4`, `__vcmpne4` and similar are emulated in ~6 word ops each (`ggml/src/ggml-cuda/vendors/hip.h`).
- NVIDIA: native byte-SIMD intrinsics.
- Depends on it: IQ-quant sign handling, nibble unpacking, byte offsets in dequant kernels.
- On gfx906: rewrite as word ops (XOR, multiply-spread masks, carry-free adds). S2 gave Flash Next +5% TG.

## 5. Memory and caches

- gfx906: ~1 TB/s HBM2, more than a 3090 or R9700. But only 4 MB L2 and 64 KB LDS per CU, so there's little reuse from cache.
- Others: large last-level caches (R9700 ~64 MB, 5090 ~96 MB) let small hot tensors (routers, KV tails, activations) stay on-chip.
- Depends on it: decode speed (bandwidth), cache-blocking that assumes tens of MB of L2, persistent kernels holding weights on-chip.
- On gfx906: decode is the strong point; our Q8_0 mat-vecs reach ~650-715 GB/s. Tricks that cut bytes per token (repack layouts, fewer passes, fusion) carry over fully. Tricks counting on big caches do not.

## 6. Asynchronous copies into shared memory

- gfx906: no `cp.async`/TMA equivalent; staging to LDS goes through registers.
- Others: 3090 `cp.async`, Hopper/Blackwell TMA-style bulk copies, RDNA4 async loads.
- Depends on it: multi-stage pipelined GEMM and attention.
- On gfx906: software pipelining through registers (the repacked GEMM double-buffers). Watch register pressure (VGPR spills).

## 7. Execution width and cross-lane ops

- gfx906: wave64; cross-lane via DPP and `ds_swizzle`.
- Others: warp32 (NVIDIA), wave32 (RDNA).
- Depends on it: tile shapes, reductions, top-k and sort kernels written for 32 lanes.
- On gfx906: retune tiles and reductions for 64 lanes (repack uses DPP reductions). A 32-lane kernel ported unchanged often wastes half the wave.

## 8. Launch, synchronization and graphs

- gfx906: relatively high launch and host-sync cost; HIP graphs work.
- Others: lower launch overhead on NVIDIA; CUDA graphs mature.
- Depends on it: models with many small ops (MoE routing, hyper-connections, GDN).
- On gfx906: fusion and whole-token graph replay give big wins, and avoiding per-ubatch host syncs matters.

## 9. Interconnect

- This board: no peer access (one card on CPU lanes, one on chipset lanes), so AllReduce is host-staged over PCIe. The MI60 is ~20% faster than the MI50. gfx906 supports xGMI bridges, but this rig doesn't have one.
- Others: 3090 NVLink bridge or P2P; R9700/5090 PCIe P2P (vllm-radiance uses a custom P2P all-reduce).
- Depends on it: custom P2P all-reduce, fine-grained tensor parallel, expert parallel traffic.
- On gfx906 here: fewer, larger exchanges and overlap (slabs, duplex copies, parallel lanes). Uneven `-ts` can give 2-3%. Quantized all-reduce is lossy, so we don't use it.

## Your RTX 3090 as a target

- What it uses: upstream CUDA paths, MMQ with `mma` tensor-core tiles, FlashAttention with `mma` (MMA_F16) kernels, CUDA graphs. Build with `-DGGML_CUDA=ON`.
- Fork changes that do not apply: everything under `#if defined(GGML_USE_HIP) && defined(__gfx906__)`, meaning the repack buffer and kernels, gfx906 MMQ/MMVQ tuning, S2, the narrow IQ2_S path and grouped MoE. They compile out, and the CUDA defaults are already tuned for Ampere.
- Fork changes that still apply: generic llama/common code, such as MTP prompt replay and KV-only prefill, DFlash taps and replay, split KV and rollback fixes, the tied output head split and shared-expert split under `-sm tensor`, and host placement of the per-layer n-gram table.
- Fit in 24 GB: Qwen3.8-27B Q8_0 (~27 GiB) does not fit one card; UD-Q5_K_XL (~19.4 GiB) fits with room for context. Flash Next needs most experts on the CPU (`-ncmoe`), which the 3090's PCIe link makes slow.
- Tricks that carry over to it: tensor-core GEMM, FA2, Marlin-style int4/int8 kernels, P2P or NVLink all-reduce with a second card, deeper speculative drafts. Not FP8/FP4.

## Quick lookup

| Trick | gfx906 | 3090 | R9700 / 5090 | Section |
| --- | --- | --- | --- | --- |
| Tensor-core GEMM, Marlin/Machete, FA2/FA3 | no | yes (FA3 no) | yes | 1 |
| FP8/FP4 weights or KV | no | no | yes (lossy) | 3 |
| Repack layouts, wide loads, fewer passes | yes | yes | yes | 5 |
| Kernel fusion, graph capture | yes, big | yes | yes | 8 |
| Word-op rewrites of byte-SIMD | yes | not needed | not needed | 4 |
| Short speculative drafts (MTP2-3) | yes | deeper OK | deeper OK | 1 |
| Custom P2P all-reduce | needs P2P (not here) | yes | yes | 9 |
| Skinny GEMM for MoE gates (vllm-radiance) | yes in principle (ledger P4) | yes | yes | 2, 5 |
| GDN prefill scan rewrite, prefix caching, padding-free drafts | yes, algorithmic | yes | yes | - |
| CPU expert offload (Strata) | only when VRAM forces it (~1 ms/token per layer here) | similar | similar | 9 |

Rule of thumb: a trick carries over to gfx906 if it saves memory traffic, launches, host syncs or scheduling waste. It doesn't if it needs matrix units, FP8/FP4, fast byte-SIMD, peer links or 32-wide warps. Anything that lowers precision is excluded here for quality, whatever the hardware.
