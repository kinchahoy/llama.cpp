# 2026-11-10 Q3: 27B Q8_0 mat-vec ceiling

Go/no-go for idea Q3: is there bandwidth headroom above the repacked Q8_0 mat-vec at the 27B per-GPU decode shapes? Kind: microbenchmark and profile.

- Source: `0f7ce1b04` (exp-q3, no source change); build `build-exp-q3` (catch-up flags, `-j 6`).
- Model: 27B Q8_0 rev `4ca720788d1e`. Settings: standard 27B.
- Microbench: `mvbench.hip` (standalone, includes the real `q8_repack/repack-kernels.cuh`). It allocates 16 layers of the per-GPU 1/1 split shapes (3.24 GB, 108 mat-vecs: fused gate/up 5120x8704 x2, down 8704x5120, GDN qkv 5120x5120, gate 5120x3072, alpha/beta 5120x24, ssm_out 3072x5120, attn q/k/v/out 1 layer in 4), runs them back to back on one stream, 5 passes, hipEvent timing. Build: `mk.sh mvbench.hip mvbench`; run `flock <gpu.lock> ./mvbench <dev>`.

## Streaming ceiling vs the real kernel (GB/s, weight bytes incl. scale plane)

| Test | ROCm0 (MI60, 64 CU) | ROCm1 (MI50, 60 CU) |
| --- | ---: | ---: |
| Flat 2 GiB, 1 KB contiguous per wave | 905 | 937 |
| Flat 2 GiB, grid-stride uint4 x4 | 764 | 753 |
| D2D memcpy (read+write) | 828 | 744 |
| Output head 5120x124160: stream 1 KB/wave | 903 | 931 |
| Output head: real kernel ROWS=8 | 775 | 761 |
| Output head: row-shaped stream, no math, all loads in flight | 797 | 788 |
| Token sequence: best stream (1 KB/wave) | 795-800 | 761-767 |
| Token sequence: real kernel ROWS=8 (default) | 715-718 | 682-689 |
| Token sequence: real kernel ROWS=4 | 720 | 685 |

At the real per-matrix sizes (17-95 MB, 25-130 us per call) even a pure stream reaches only about 760-800 GB/s; the 900+ figure needs very large reads. The kernel sits at 89-90% of that best stream. A row-shaped stream with no math at all reaches about the same rate as the real kernel on the head matrix (770-797 vs 775), so the remaining gap is the one-row-per-wave access pattern plus launch ramp/tail, not dp4a or load depth.

## Kernel variants tried (token sequence, ROCm0 / ROCm1, GB/s)

| Variant | ROCm0 | ROCm1 | Output |
| --- | ---: | ---: | --- |
| baseline `<8,8>` (2 loads in flight per lane) | 715 | 682 | - |
| U loads in flight per lane, predicated tail, same sum order: U=2 / 3 / 4 / 5 | 710 / 687 / 668 / 590 | 675 / 644 / 618 / 544 | bit-identical (checked all shapes) |
| U=4, 4 rows per block | 709 | 666 | bit-identical |
| U=4, 16 rows per block | 209 | 204 | bit-identical |
| k-split WPR=2/4/8 (existing template) | 697 max | 631 max | not bit-identical |
| interleaved block read (stream only, 4/8/16 rows) | +0-3% over row stream | - | needs LDS re-reduce |
| row stride 5120/5136/5184/5248 (stream only) | 785-795, no effect | - | - |

More loads in flight or more rows per block costs occupancy (VGPRs) and loses. k-split loses and changes sum order.

llama-bench TG128 r3, `GGML_RP_MMV_ROWS` alternating 8/4/8/4 (existing env knob): 37.45 +-0.21, 37.60 +-0.19, 37.44 +-0.20, 37.52 +-0.20. ROWS=4 is +0.3%, inside 27B noise. Not changed.

## Remaining per-token costs (rocprofv3 kernel trace, llama-bench tg64, profiled 33.8 t/s, 29.55 ms/token)

| Item | ROCm0 | ROCm1 |
| --- | ---: | ---: |
| Q8_0 mat-vec, 433 calls (fused gate/up 128 us at about 740 GB/s; head 877 us at 770 GB/s) | 19.33 ms (65%) | 19.54 ms |
| `ggml_cuda_ar_kernel`, 128 calls | 3.40 ms (26.6 us avg) | 1.59 ms (12.4 us) |
| `add_rms_norm_mul_f32`, 128 calls | 1.39 ms (10.9 us avg) | 1.40 ms |
| `quantize_q8_1` + gated unary + `rms_norm_f32` + cpy/get_rows/concat/bcast (about 600 calls at 4-5 us) | about 2.6 ms | about 3.0 ms |
| Idle (no kernel) | 1.07 ms (0.43 before the first kernel of the token, 0.54 around `copyBuffer`) | 1.96 ms (1.18 before the first kernel) |
| Launches per token | 1596 | 1596 |

Verdict: Q3 no-go, validated wrong for the planned changes (2-deep or deeper prefetch, more rows per block, k-split): the best stream at the real shapes is only 10-11% above the kernel and every bit-exact variant was slower. The upper bound left in the mat-vec is about 1.9 ms/token and it needs a different access pattern (block-contiguous reads with an LDS per-row reduce to keep the order), not more loads in flight; expected gain is a few percent at best, so it is low priority. Larger targets: the AllReduce skew (ROCm0 waits 1.8 ms/token longer than ROCm1 in the AR kernel; P3/AR-LL), the 10.9 us `add_rms_norm_mul` after each AR (Q1 fusion, 1.4 ms), and the about 600 small glue launches (about 2.6 ms). Not covered: the interleaved-read mat-vec with LDS reduce was only tested as a stream; MI60 vs MI50 TG split. Raw: `mv_dev0_final.txt`, `mv_dev1_final.txt`, `rows.txt`, `analyze.py`, trace in `trace/` (not kept, 100 MB).
