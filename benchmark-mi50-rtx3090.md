# llama.cpp MUL_MAT GPU Benchmark

Benchmark command:

```bash
./build/bin/test-backend-ops perf \
  -b <backend> -o MUL_MAT \
  -p 'type_a=(f16|q4_0|q8_0|q4_K|q6_K),type_b=f32,m=4096,n=(1|512),k=14336'
```

The MI50 results were collected with the ROCm build at commit `d8a24ccee`.
The RTX 3090 results were supplied from the equivalent CUDA benchmark.

## Hardware

| Label | GPU | Backend | VRAM | Notes |
|---|---|---|---:|---|
| MI50-0 | AMD Radeon Instinct MI50, gfx906 | ROCm0 | 32 GiB | 225 W power cap, CPU-attached PCIe |
| MI50-1 | AMD Radeon Instinct MI50, gfx906 | ROCm1 | 32 GiB | 178 W power cap, chipset-attached PCIe |
| RTX 3090 | NVIDIA GeForce RTX 3090, CC 8.6 | CUDA0 | 24 GiB | 24575 MiB reported VRAM |

## Decode, n=1

Higher throughput and lower latency are better.

| Weight type | MI50-0 latency | MI50-0 throughput | MI50-1 latency | MI50-1 throughput | RTX 3090 latency | RTX 3090 throughput | 3090 vs MI50-0 | 3090 vs MI50-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F16  | 199.78 us | 0.588 TFLOPS | 200.33 us | 0.586 TFLOPS | 133.19 us | 0.882 TFLOPS | 1.50x | 1.50x |
| Q4_0 | 72.96 us  | 1.61 TFLOPS  | 80.04 us  | 1.47 TFLOPS  | 41.31 us  | 2.84 TFLOPS  | 1.76x | 1.93x |
| Q8_0 | 101.17 us | 1.16 TFLOPS  | 111.14 us | 1.06 TFLOPS  | 74.54 us  | 1.58 TFLOPS  | 1.36x | 1.49x |
| Q4_K | 68.53 us  | 1.71 TFLOPS  | 78.68 us  | 1.49 TFLOPS  | 43.14 us  | 2.72 TFLOPS  | 1.59x | 1.83x |
| Q6_K | 97.96 us  | 1.20 TFLOPS  | 111.89 us | 1.05 TFLOPS  | 68.69 us  | 1.71 TFLOPS  | 1.43x | 1.63x |

## Prefill, n=512

| Weight type | MI50-0 latency | MI50-0 throughput | MI50-1 latency | MI50-1 throughput | RTX 3090 latency | RTX 3090 throughput | 3090 vs MI50-0 | 3090 vs MI50-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F16  | 5222.55 us | 11.51 TFLOPS | 6606.26 us | 9.10 TFLOPS  | 737.33 us | 81.55 TFLOPS | 7.09x | 8.96x |
| Q4_0 | 3058.84 us | 19.66 TFLOPS | 4308.35 us | 13.96 TFLOPS | 723.35 us | 83.13 TFLOPS | 4.23x | 5.96x |
| Q8_0 | 7058.67 us | 8.52 TFLOPS  | 8662.66 us | 6.94 TFLOPS  | 762.32 us | 78.88 TFLOPS | 9.26x | 11.37x |
| Q4_K | 3779.61 us | 15.91 TFLOPS | 5053.89 us | 11.90 TFLOPS | 863.83 us | 69.61 TFLOPS | 4.38x | 5.85x |
| Q6_K | 6182.09 us | 9.73 TFLOPS  | 7557.69 us | 7.96 TFLOPS  | 893.33 us | 67.31 TFLOPS | 6.92x | 8.46x |

## Summary

- The RTX 3090 is about 1.4x to 1.9x faster for these decode-shaped matrix-vector cases.
- The RTX 3090 is about 4.2x to 11.4x faster for these prefill-shaped matrix-matrix cases.
- MI50-1 is materially slower than MI50-0, especially for prefill. Its lower power cap and chipset PCIe attachment make the two MI50 results non-equivalent.
- These are isolated backend operation results. They do not directly measure model token throughput, multi-GPU scaling, sampling, KV-cache work, or host-device transfer overhead.
