# A7: repacked Q8_0 GEMM occupancy variants (negative result)

Branch play-a7. Variants selected by GGML_RP_GEMM_VAR (dense Q8_0 GEMM only):
0 = baseline 64x128 tile, 256 thr, 126 VGPR, 29.7 KB LDS, 2 waves/SIMD.
1 = 64x64 tile, 256 thr, launch_bounds(256,3): 84 VGPR (15 spilled), 19.2 KB LDS, 3 waves/SIMD.
2 = 64x64 tile, 512 thr, launch_bounds(512,4): 64 VGPR (60 spilled), 19.2 KB LDS, 4 waves/SIMD.

Harness rpgemm.cpp (test-backend-ops does not use the repack buffer, so it cannot reach this kernel):
weights in the repack buft vs the stock Q8_0 buffer, k=5120, m in {5120, 8704, 17408, 5152}, n in {128,129,191,512,2048}.

Correctness: all 20 shapes per run give NMSE 0 vs the stock path (bit-identical) on ROCm0 and ROCm1.
VAR=1 and VAR=2 outputs are byte-identical to VAR=0 (cmp) on both GPUs.

GEMM perf, n=2048, k=5120, ms (3 rounds agree within 1%):

| m | GPU | VAR0 | VAR1 | VAR2 |
|---|---|---|---|---|
| 17408 | ROCm0 | 14.02 | 18.47 | 22.2 |
| 8704 | ROCm0 | 6.81 | 9.10 | 10.84 |
| 5120 | ROCm0 | 4.14 | 5.47 | 6.54 |
| 17408 | ROCm1 | 16.1 | 21.25 | 25.3 |
| 8704 | ROCm1 | 7.93 | 10.55 | 12.42 |
| 5120 | ROCm1 | 4.85 | 6.35 | 7.51 |

Baseline is about 26 TOPS (ROCm0) and 22.7 TOPS (ROCm1). VAR1 is 24-31% slower, VAR2 37-57% slower.
Llama-bench PP/TG not run: a 25%+ GEMM loss cannot give a PP gain.

Verdict: more waves per SIMD does not help. The kernel is VALU/LDS-issue bound, not latency bound, and halving the
column tile halves weight reuse and x/W LDS amortisation (plus spills). Default stays 0. Not worth repeating:
TN=1 tiles, 512-thread blocks, VGPR caps below 128. Untried: larger tile (TN=3-4, needs LDS < 32 KB per block
by shrinking BK), fewer per-(r,n) conversion ops.
