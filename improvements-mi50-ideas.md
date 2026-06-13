# gfx906 Q4_K and Q8_0 optimization record

This note records the evidence behind the two source changes in this branch. See
`whatsnext.md` for the current validation workflow and runnable commands.

## Goal and scope

The target is prompt processing on AMD Vega 20 (`gfx906`, MI50/MI60) without
regressing token generation. gfx906 supports wave64 and DP4A, but it does not
have the MFMA accumulator path used by gfx908 and newer CDNA devices.

The changes are deliberately narrow:

- dense Q8_0 `MUL_MAT` dispatch on exact gfx906;
- Q4_K MMQ launch geometry in the Q4_K HIP translation unit on exact gfx906.

CUDA, other AMD architectures, other quant types, and Q8_0 `MUL_MAT_ID` are
outside the changed paths.

## Baseline result

The representative operator shape was
`m=4096,k=14336,n=512,type_b=f32`. Values are single-run measurements used to
select the implementation, not final model-level results.

| Type | Path | ROCm0 | ROCm1 | Result |
| --- | --- | ---: | ---: | --- |
| Q8_0 | fused DP4A MMQ | 8.52 TFLOPS | 6.94 TFLOPS | baseline |
| Q8_0 | F16 conversion + rocBLAS | 10.85 TFLOPS | 8.68 TFLOPS | 25-27% faster |
| Q4_K | fused DP4A MMQ | 15.91 TFLOPS | 11.90 TFLOPS | baseline |
| Q4_K | F16 conversion + rocBLAS | 10.52 TFLOPS | 8.35 TFLOPS | 30-34% slower |

The formats need different solutions. Large Q8_0 batches benefit from rocBLAS
after conversion to F16. Q4_K benefits from staying compressed and fused in
MMQ; forcing rocBLAS discards that advantage.

## Q8_0: selective rocBLAS dispatch

`ggml_cuda_should_use_mmq()` now returns MMQ for dense Q8_0 batches through 256
on exact Vega 20 and returns the existing rocBLAS path above 256:

```cpp
if (cc == GGML_CUDA_CC_VEGA20 && type == GGML_TYPE_Q8_0 && n_experts == 0) {
    return ne11 <= 256;
}
```

This works because the decision is made after the force-MMQ build option and
before the generic AMD fallback. Small batches retain the low-overhead fused
path, while larger dense batches use a GEMM large enough to amortize conversion
and launch costs. Excluding expert matmuls avoids changing `MUL_MAT_ID` without
supporting measurements.

Measured at batch 512:

| Device | MMQ | Selective rocBLAS | Gain |
| --- | ---: | ---: | ---: |
| ROCm0 | 8.49 TFLOPS | 10.80 TFLOPS | 27.2% |
| ROCm1 | 6.96 TFLOPS | 8.63 TFLOPS | 23.9% |

Both devices passed the available Q8_0/F32 `MUL_MAT` correctness cases. Batch
2048 completed at 11.68 TFLOPS on ROCm0 and 10.12 TFLOPS on ROCm1. Batch 1 and
Q4_K controls were unchanged in the operator checks.

## Q4_K: eight-wave MMQ workgroup

The baseline `128x64` Q4_K tile uses four wave64 waves. Each thread owns 32
float accumulators, and the exact gfx906 kernel reached 128 VGPRs with 44 bytes
of private scratch per thread.

The selected design keeps the same output tile and doubles the workgroup to
eight waves. Each thread owns half the output cells, reducing accumulator and
instruction pressure without changing the quantization math or output layout.

| Property | Four waves | Eight waves |
| --- | ---: | ---: |
| Threads/workgroup | 256 | 512 |
| Output tile | 128x64 | 128x64 |
| Accumulators/thread | 32 | 16 |
| VGPRs/thread | 128 | 111 |
| Scratch/thread | 44 bytes | 0 bytes |
| VGPR spills | 10 | 0 |
| Static instructions/thread | 3319 | 1834 |

The Q4_K template translation unit receives a private compile definition. Host
and device wave-count helpers both select eight waves only when that definition
and exact gfx906 are present. Keeping the two decisions aligned is required for
correct launch geometry.

Compilation, linking, resource metadata, and ISA inspection passed. Runtime
correctness and model-level performance remain the acceptance gate; lower VGPR
or scratch use alone does not prove a speedup.

## Rejected and fallback experiments

Reducing the row tile from 128 to 64 removed spills but doubled row-tile work.
ROCm0 fell from about 15.89 to 14.45 TFLOPS, so that design was rejected.

At `mmq_y=128`, a four-wave `mmq_x=40` tile was the largest spill-free fallback:

| X tile | ROCm0 | ROCm1 | Scratch/thread |
| ---: | ---: | ---: | ---: |
| 32 | 16.37 TFLOPS | 13.78 TFLOPS | 0 bytes |
| 40 | 16.24 TFLOPS | 15.05 TFLOPS | 0 bytes |
| 48 | 16.34 TFLOPS | 15.00 TFLOPS | 20 bytes |
| 56 | 13.93 TFLOPS | 11.36 TFLOPS | 36 bytes |
| 64 | 15.89 TFLOPS | 11.88 TFLOPS | 44 bytes |

Use X=40 only if the eight-wave candidate is correct but slower. Do not combine
the two changes: eight waves are intended to make the normal X=64 tile
spill-free.

## What to try next

Only continue kernel work after model-level validation.

1. Profile the selected Q4_K batch-512 kernel and confirm 512 threads, about 111
   VGPRs, and zero scratch.
2. If eight waves win, test aligned Q4_K LDS reads using the existing
   `ggml_cuda_memcpy_1<16>` pattern from Q4_0/Q4_1.
3. Keep staging groups small enough that scratch remains zero.
4. Reject any change that improves isolated prefill but regresses generation or
   long-context model throughput.

Do not start with a custom Q4_K-to-INT8 rocBLAS path. Q4_K block scales and
minimums do not map directly to a plain INT8 GEMM, and the measured F16 fallback
already loses substantially to fused MMQ.
