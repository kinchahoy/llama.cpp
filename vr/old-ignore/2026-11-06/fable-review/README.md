# Fable review, 2026-11-06 (verbatim)

Read-only review by Claude Fable 5.1 of the Flash Next and 27B Q8_0 paths, given the 2026-11-06 profiles, memory map and ledger. Estimates are its own, not measured. Ledger rows: F1, F2, Q1 (its "D1"), Q2 (its "D2") in [PERFORMANCE.md](../../../PERFORMANCE.md).

---

## Flash Next (qwen4exp) - two proposals

### F1. Remove the blit-kernel host traffic (`copyBufferRectAligned`: 19% of MTP3 kernel time, 15.6% of PP)

What the profile shows: decode 37.8k calls at 65 us (the AR kernel itself is only 43 us), PP 545 calls at 4.1 ms avg / 35 ms max (~2.5 GB/s for ~10 MB, i.e. a blit kernel pulling host memory over PCIe on the compute queue, serialized with compute). It is not `ggml_cuda_cpy` (ledger) and it is not the decode AR (`ggml_cuda_ar_kernel` uses mapped pinned memory, no memcpy). Three remaining producers, all in the fork's own code:
1. `ggml_backend_cuda_cpy_tensor_async` (ggml/src/ggml-cuda/ggml-cuda.cu:3548) and `ggml_backend_cuda_buffer_cpy_tensor` (:986) call `cudaMemcpyPeerAsync`; with no P2P on this topology ROCm stages it through its own bounce buffer with blit kernels. Callers: meta `gather_rows` (ggml-backend-meta.cpp:3711, also does a host `ggml_backend_synchronize` on every lane), `BCAST` (:4074), `push_data` fallback (:3582), and the MTP sidecar hidden-state hand-off to ROCm1.
2. `ggml_backend_cuda_set/get_tensor_2d_async` (:3480, :3491) = `hipMemcpy2DAsync`, used by meta `set_tensor` for every AXIS-split graph input (:2954, "one strided 2D dispatch per device"). Only the misaligned-pitch case already packs into pinned scratch + 1D copy (:3447-3478).
3. Memcpys captured inside the whole-token HIP graph: ROCm executes graph memcpy nodes as rect blits.

Change: route all cross-device and split-input traffic through the AR pipeline's pinned buffers (`host_large`, SDMA 1D `cudaMemcpyAsync` + events, exactly like `ggml_cuda_ar_allreduce_copy_impl` stage 1/2), i.e. add a `ggml_cuda_ar_send/gather` primitive and point `gather_rows`, `BCAST` and the draft hand-off at it; drop the host synchronizes (events instead); make the 2D-input path always pack-then-1D (persistent double-buffered pinned scratch, no `cudaStreamSynchronize`). Bit-exact (pure data movement).
Win: decode ~180 copies/step x (65 -> ~10-15 us) is 4-5 ms per GPU per step on a ~20-25 ms critical path (>15% TG, plain and MTP); PP 2.25 s of blit time becomes overlapped SDMA (~10% PP).
Risk: ordering bugs across streams; run `test-backend-ops` TP smoke plus byte-identical CLI output.
First test (10 min): `rocprofv3 --hip-trace --kernel-trace` on 32 decode tokens; correlate each `copyBufferRectAligned` with its HIP API call to confirm which of the three producers dominates before writing code. Also try `HSA_ENABLE_SDMA=1` explicitly once.

### F2. One fused hyper-connection mix kernel (re-does P1/P2/S3/S4 as one change)

`build_hc_mix` (src/models/qwen4exp.cpp:366-424) is 7 launches per call, 2 calls per layer: rms_norm, mul, F32 mat-vec `down_inject` [10240 x (hc_lr+4)], scale, silu, F32 mat-vec `up` [hc_lr x 10240], `dsv4_hc_pre_gated`. Per token that is ~670 launches of 4-15 us (profile: 288 float mat-vecs 3.1 ms at ~100 GB/s on 41 blocks, 303 scales 1.8 ms, plus norm/silu/quantize) - about 7-8 ms of the 24 ms token on both GPUs, mirrored, so -ts cannot help it.
Change: new CUDA op for T <= 8 tokens: one block per (token, K-slice), stage the 40 KB stream row in LDS, compute rms, low-rank down + inject, silu, up, gate, mean, and write `mixed` + `inject` (optionally also emit q8_1 of `mixed` for the following projection). Keep F32 everywhere; fix the summation order so the result is deterministic. Register it as a fused node like `LLM_FUSED_OP_DSV4_HC_PRE`.
Win: ~500 fewer launches/token plus the 15 MB/layer of hc weights streamed at 600+ GB/s instead of ~100: 4-5 ms/token -> plain TG 41.7 -> ~50; same per-step saving in MTP verify.
Risk: not bit-identical to rocBLAS/mmvf (F32 order change); validate with PPL on 3 texts, greedy output match, `test-backend-ops` vs CPU.

Also noted: F32 router SGEMM (P4, 15% PP) - vllm-radiance's only applicable hint is a skinny split-K gate GEMM; a custom F32 kernel for [2560 x 512] x T is the fix, keep F32. Strata PDF: glyph-encoded, not text-extractable; nothing beyond the S1-S8 review. vllm-radiance targets gfx1201 only (custom P2P AR, fp8), not applicable here.

## 27B Q8_0 (qwen35) - two proposals

### D1. Fuse AR + residual add + RMS norm + q8_1 quantize into the AR kernel (TG, plain and MTP2)
Every layer ends in AR -> add residual -> rms_norm -> mul -> quantize_q8_1 before the next mat-vec, twice per layer: ~400-500 small launches/token (profile pattern as in the Flash Next TG trace: quantize 2.3 ms, scale/norm/add similar). `ggml_cuda_ar_kernel` (allreduce.cu:114) already holds the summed F32 row; one block per token can finish the row (n_embd reduction) and emit q8_1 directly. Same F32 math and order as `rms_norm_f32`/`quantize_q8_1`, so bit-exact is achievable. Win: 2.5-3.5 ms of a 27 ms token (10-13%). Risk: scheduler must recognize the fused chain (reuse `add_fused_node`). Test: byte-identical output, r3 alternating TG.

### D2. PP: SDMA-driven AR copies + two-half pipelining (A6 -> N3)
The Flash Next PP profile suggests the host-staged copies on this box run as blit kernels at ~2.5 GB/s. Re-profile 27B PP2048 (A6); if `copyBufferRectAligned` shows up, F1 applies directly. Then N3: split the 2048 ubatch into two 1024-token halves per layer so half A's reduce (host SDMA, add kernel) runs under half B's GEMM; slabs already overlap D2H but not the H2D + add. Bit-exact. Win: Oct 12 showed ~20% of the ubatch in AR copies; expect +8-12% PP. Risk: pool memory for two in-flight halves. Test: PP2048 r3 vs `GGML_CUDA_AR_SLABS=4`.

Asymmetry note: the MI60 waits in AR (5.2 vs 1.0 ms/token). Mirrored fixed costs (F2, D1) are where the MI50 lags on every op, so shrinking them helps more than `-ts`; after F2/D1 re-sweep `-ts 0.55/0.45` for PP, where the 178 W cap bites.
