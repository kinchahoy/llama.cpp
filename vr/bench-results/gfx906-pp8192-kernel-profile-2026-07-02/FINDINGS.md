# gfx906 pp8192 kernel-time split + bottleneck counters (2026-07-02)

Model: Qwen3.6-27B-Q4_K_M (qwen35 hybrid). Build: build/gfx906-optimal
(b3fed31b9, Jun 28, stale but valid for ratios). Single GPU, -sm none -dev
ROCm0 -fa on. Tool: rocprofv3. Raw CSVs beside this file.

Caveats: ratios only (kernel-trace serializes; warmup included). pp8192 =
PREFILL regime; decode is MMVQ/bandwidth-bound and NOT characterized here.

## 1. Kernel-time split at pp8192 (the measurement that was never run)

| kernel                          | %      | note |
| ------------------------------- | -----: | ---- |
| mul_mat_q Q4_K (type 12)        | 54.07  | MMQ prefill |
| mul_mat_q Q6_K (type 14)        | 19.80  | MMQ prefill |
| gated_delta_net (linear attn)   |  8.13  | the "SSM" mixer |
| mul_mat_q Q5_K (type 13)        |  6.16  | MMQ prefill |
| flash_attn_tile                 |  4.11  | + combine 0.13 |
| Cijk_... (rocBLAS/Tensile gemm) |  2.72  | Q8_0 dispatch path |
| concat, quantize_q8_1, norms... |  ~5    | rest |

**Quantized matmul (Q4_K+Q6_K+Q5_K) = 80.0% of prefill kernel time.**
Flash Attention = 4%. DeltaNet = 8%.

Consequences:
- ARCH-REVAMP-NOTES 5.5 ("FA is the Amdahl ceiling at pp8192") is FALSE here:
  FA is 4%. FA tile tuning is not a prefill priority in the pp8192 range.
- The matmul focus (5.1) is correct and well-targeted. The 5.5-vs-5.1
  self-contradiction resolves in favor of 5.1.
- Q6_K is 20% of prefill and per-layer (2048 calls, heaviest per-call at
  6.26us), yet only ever got min_blocks=1. It is the largest under-worked
  target after Q4_K.

## 2. WHY Q4_K/Q6_K MMQ is slow (counters, pp512, same per-chunk shape)

| counter          | Q4_K (128 VGPR) | Q6_K (140 VGPR) |
| ---------------- | --------------: | --------------: |
| VALUBusy %       | 60.0            | 44.5            |
| VALUUtilization %| 100.0           | 100.0           |
| MemUnitBusy %    | 11.3            | 8.9             |
| MemUnitStalled % | 0.06            | 0.02            |
| LDSBankConflict %| 0.12            | 0.58            |
| VGPR / lane      | 128             | 140             |
| waves / SIMD     | 2  (256/128)    | 1  (256/140)    |

Diagnosis (measured, not inherited):
- NOT VALU-saturated: VALU busy only 60% / 44%, but 100% lane utilization
  (no divergence, no wasted lanes).
- NOT memory/bandwidth-bound: MemUnitBusy ~10%, stalled ~0%.
- NOT LDS-bank-conflict-bound: ~0.1-0.6%. (Stride-9 layout is polishing a
  non-bottleneck; LDS_Block_Size even reports 0 for these dispatches.)
- IS register-pressure-limited OCCUPANCY. Q4_K 128 VGPR -> 2 waves/SIMD
  (20% of the 10-wave hw max); Q6_K 140 VGPR -> 1 wave/SIMD (10%). With so
  few concurrent waves there is nothing to hide instruction/load-return
  latency, so the VALU sits idle ~40-56% of the time.

This explains every rejected prefill experiment in RESULTS.md: min_blocks=1
(-34%), y64 tile (-9%), 2-accumulator (-2.4%) all ADD live VGPR state, which
lowers occupancy further -- the exact wrong direction. The one lever the data
supports is the opposite: REDUCE VGPRs to raise waves/SIMD.

## 3. Recommended experiments, ranked by leverage/effort

1. CHEAP TEST of the whole thesis: force higher occupancy on Q4_K MMQ via
   `__launch_bounds__` / `__attribute__((amdgpu_waves_per_eu(3,...)))` to cap
   VGPRs at <=85 (3 waves/SIMD). If it helps without spill, occupancy is
   confirmed as THE bottleneck and justifies (4). If it spills and regresses,
   the VGPRs are genuinely live and only a repack can shed them. ~1 line +
   one A/B. Do this before any big build.
2. Same occupancy sweep for Q6_K (currently 1 wave/SIMD -- most starved,
   20% of prefill). Watch scratch: min_blocks=1 was accepted here to avoid
   spill, so pushing occupancy up may reintroduce it.
3. Offline gfx906-native repack (ARCH-REVAMP-NOTES 5.1) -- justified IF (1)
   confirms occupancy-bound. Mechanism per the data is VGPR reduction (drop
   in-loop nibble/6-bit-scale decode temporaries), NOT LDS-conflict removal.
   Realistic gain: 2->3 waves/SIMD (~1.2-1.4x on the Q4_K portion), not the
   1.5-1.7x implied by 189->280-320. Apply to Q4_K, Q6_K, Q5_K (all share the
   k-quant super-block structure) to cover the full 80%.
4. DROP as prefill priorities: Flash Attention (4%), DeltaNet (8%).

## 3b. Occupancy experiment location (2026-07-02)

Running in a dedicated git worktree so the main tree stays clean:
- path:   `/home/raistlin/infer/gfx906-q4k-occupancy`
- branch: `gfx906-q4k-occupancy-2026-07-02` (off 0a8fe8255)
- find it: `git worktree list | grep occupancy`
- change: adds `GGML_CUDA_MMQ_Q4K_GFX906_MIN_BLOCKS_3` on
  `mmq-instance-q4_k.cu` (CMake option `GGML_GFX906_Q4K_MIN_BLOCKS_3`, default
  OFF) -> forces Q4_K MMQ to <=85 VGPR / 3 waves/SIMD. Control build = option
  OFF (min_blocks=2), candidate = ON (min_blocks=3), same source.

## 4. Reproduce
```
rocprofv3 --kernel-trace --stats -f csv -d OUT -- \
  build/gfx906-optimal/bin/llama-bench -m $MODEL -ngl 99 -fa on -sm none \
  -dev ROCm0 -p 8192 -n 0 -r 1
rocprofv3 --pmc VALUBusy VALUUtilization MemUnitBusy MemUnitStalled \
  LDSBankConflict Wavefronts --kernel-trace -f csv -d OUT2 -- \
  build/gfx906-optimal/bin/llama-bench -m $MODEL -ngl 99 -fa on -sm none \
  -dev ROCm0 -p 512 -n 0 -r 1
```
