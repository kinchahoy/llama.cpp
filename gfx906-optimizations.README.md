# gfx906 optimization package

Status date: 2026-06-16

This private branch carries three distinct gfx906 optimizations for AMD Vega
20 / MI50 / MI60. The patches are intentionally separated by quant so each
change can be reviewed, applied, and reverted independently.

## Patch Set

Apply from a clean llama.cpp checkout near `origin/master` commit
`6e9007ae61f4`.

```bash
git apply --check gfx906-q8-rocblas-dispatch.patch
git apply gfx906-q8-rocblas-dispatch.patch

git apply --check gfx906-q4k-metadata-precompute.patch
git apply gfx906-q4k-metadata-precompute.patch

git apply --check gfx906-q6k-min-blocks-1.patch
git apply gfx906-q6k-min-blocks-1.patch
```

Patch order matters only for Q6_K: `gfx906-q6k-min-blocks-1.patch` assumes
the gfx906 HIP target gate introduced by the Q4_K patch already exists.

| Patch | Files | Status |
| --- | --- | --- |
| `gfx906-q8-rocblas-dispatch.patch` | `ggml/src/ggml-cuda/mmq.cu` | accepted |
| `gfx906-q4k-metadata-precompute.patch` | `ggml/src/ggml-cuda/mmq.cuh`, `ggml/src/ggml-hip/CMakeLists.txt` | accepted |
| `gfx906-q6k-min-blocks-1.patch` | `ggml/src/ggml-cuda/mmq.cuh`, `ggml/src/ggml-hip/CMakeLists.txt` | accepted |

Build HIP for exact `gfx906`. The Q4_K and Q6_K specializations are enabled
only when the configured HIP target list contains exactly one target and that
target is `gfx906`.

## Q8_0

Change: on Vega20, keep dense Q8_0 `MUL_MAT` batches through `ne11 <= 256` on
MMQ and route larger dense Q8_0 matmuls to the existing rocBLAS path. Expert
`MUL_MAT_ID` is unchanged.

Measured on Qwen3.6-27B Q8_0:

| Test | Result |
| --- | ---: |
| pp512 | +21% to +25% |
| pp2048 | +53% to +57% |
| tg128 | neutral |

Operator-level `m=4096,k=14336,n=512` improved from 8.49 to 10.80 TFLOPS on
ROCm0 and from 6.96 to 8.63 TFLOPS on ROCm1.

## Q4_K

Change: for the gfx906 DP4A Q4_K MMQ translation unit, precompute the
`dm * scale/min` half2 metadata at tile-load time. The final accepted layout
uses stride9 rows for the precomputed metadata:

```cpp
mmq_y*9
```

This keeps the existing y128, x64, four-wave geometry and moves arithmetic out
of the inner dot loop without enabling the failed Q4_K `min_blocks=1` path.

Correctness:

| Test | Result |
| --- | --- |
| ROCm0 `MUL_MAT` Q4_K/F32 focused cases | 41/41 passed |
| Full ROCm0 `MUL_MAT` after stride9 | 1103/1103 passed |

End-to-end Qwen3.6-27B Q4_K_M, three alternating runs:

| Test | Gain |
| --- | ---: |
| pp512 | +8.73% |
| pp2048 | +9.36% |
| pp8192 | +11.02% |
| pp20000 | +5.31% |
| tg128 | +6.03% |

Stride9 follow-up:

| Test | Gain |
| --- | ---: |
| pp512 | +2.15% |
| pp2048 | +2.34% |
| pp8192 | +6.62% |
| pp20000 | -0.03% |
| tg128 | +1.72% |

The stride9 layout cut aggregate pp2048 LDS bank conflicts from 4.332B to
1.999B. The dominant Q4_K shape dropped from 1.783B to 0.357B.

Rejected Q4_K alternatives:

| Variant | Result |
| --- | --- |
| `min_blocks=1` | pp8192 -33.74% |
| y64 four-wave | pp8192 -9.27% |
| forced rocBLAS / F16 conversion | operator-level -30% to -34% |
| stride10 | quick pp8192 -3.86%, pp20000 -6.76% |
| stride11 | quick pp8192 -7.12%, pp20000 -5.87% |
| group2pad1 compact skew | aggregate bank conflicts back to 4.332B |

The group4pad1 compact skew was not accepted. It reduced aggregate bank
conflicts further to 1.416B, but throughput samples were order-sensitive and
mixed: pp8192 ranged from +3.02% to -11.12%, while pp20000 ranged from -7.41%
to +1.54%. Keep it as a profiling clue, not as a final patch.

## Q6_K

Change: for the gfx906 Q6_K MMQ translation unit only, use
`__launch_bounds__(..., 1)`.

Rationale: Q6_K profiling showed a VALU/spill-bound shape with low LDS wait
and 52 B/thread scratch. Unlike Q4_K, Q6_K can trade occupancy for VGPR
headroom profitably.

Validation on Qwen3.6-27B Q6_K, six alternating pp8192 runs:

| Metric | Result |
| --- | ---: |
| median, all six runs | +4.00% |
| steady-state median, runs 2-6 | +4.14% |

The direct scratch-memory trace for the dominant Q6_K kernel is still a useful
follow-up, but the end-to-end pp8192 gate passed cleanly.

Rejected Q6_K alternative:

| Variant | Result |
| --- | --- |
| Q4_K-style metadata precompute port | pp8192 -5.3% |

## Local Artifacts

Raw benchmark and profiler outputs are under `bench-results/`. The most useful
directories are:

| Directory | Purpose |
| --- | --- |
| `bench-results/gfx906-qwen36/quick/` | Q8_0 quick model gate |
| `bench-results/gfx906-q4k-experiments/q4k-metadata-precompute-20260615/` | accepted Q4_K precompute gate |
| `bench-results/gfx906-q4k-experiments/q4k-stride9-20260616/` | accepted Q4_K stride9 gate |
| `bench-results/gfx906-q6k-experiments/20260615-q6k-mb1-n6/` | accepted Q6_K gate |

The current source tree contains the Q4_K and Q6_K accepted code state. Q8_0 is
already present in this branch history and is also preserved as a standalone
patch.
