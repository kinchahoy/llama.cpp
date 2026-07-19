# gfx906 pp8192 profile

Date: 2026-07-02

Model: Qwen3.6-27B Q4_K_M. Device: ROCm0, single GPU, `-fa on -sm none`.
Tool: `rocprofv3`. Raw CSV files are beside this document.

The build predates the July upstream merge, but the result remains useful for
bottleneck classification. Kernel tracing serializes execution, so use shares
and counters rather than the traced wall-clock throughput.

## Kernel-time share

| Kernel group | Share |
| --- | ---: |
| Q4_K MMQ | 54.07 percent |
| Q6_K MMQ | 19.80 percent |
| Q5_K MMQ | 6.16 percent |
| DeltaNet | 8.13 percent |
| Flash Attention | 4.11 percent |
| Flash Attention combine | 0.13 percent |
| rocBLAS GEMM | 2.72 percent |
| Other | about 5 percent |

Quantized matmul accounts for 80 percent of pp8192 kernel time. Flash
Attention is not a priority for this workload.

## Counters

Counters used the pp512 chunk shape executed inside the same PP path.

| Counter | Q4_K | Q6_K |
| --- | ---: | ---: |
| VGPR per lane | 128 | 140 |
| Waves per SIMD | 2 | 1 |
| VALU busy | 60.0 percent | 44.5 percent |
| VALU lane utilization | 100 percent | 100 percent |
| Memory unit busy | 11.3 percent | 8.9 percent |
| Memory stalled | 0.06 percent | 0.02 percent |
| LDS bank conflicts | 0.12 percent | 0.58 percent |

## Conclusion

For this older Q4_K_M build, both kernels are consistent with low occupancy
and exposed instruction/load latency:

- lanes are fully utilized, but the VALU is idle 40 to 55 percent of the time;
- memory traffic is low and almost never stalled;
- LDS bank conflicts are negligible;
- VGPR allocation allows too few resident waves to hide latency.

This does not prove that current UD-Q4_K_XL kernels have the same bottleneck.
Check current ISA resources and a target-model counter trace before using these
numbers as a design constraint. Merely forcing higher launch occupancy can
spill; a resource change matters only when operator throughput improves.

Q6_K was more register-heavy in this trace, and historical `min_blocks=1`
improved an older model run. Its current-head effect remains unresolved. Do
not infer that either more or fewer Q6_K waves will help without isolated
current-shape replay.

## Reproduce

```bash
rocprofv3 --kernel-trace --stats -f csv -d OUT -- \
  build/gfx906-optimal/bin/llama-bench -m "$MODEL" -ngl 99 -fa on \
  -sm none -dev ROCm0 -p 8192 -n 0 -r 1

rocprofv3 --pmc VALUBusy VALUUtilization MemUnitBusy MemUnitStalled \
  LDSBankConflict Wavefronts --kernel-trace -f csv -d OUT_COUNTERS -- \
  build/gfx906-optimal/bin/llama-bench -m "$MODEL" -ngl 99 -fa on \
  -sm none -dev ROCm0 -p 512 -n 0 -r 1
```
