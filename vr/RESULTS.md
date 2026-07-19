# gfx906 results

Last updated: 2026-07-18

This ledger contains the results that still affect engineering decisions.
Raw files under `bench-results/` are authoritative.

## Confidence labels

- Retained: repeated performance evidence plus relevant correctness.
- Supporting: useful evidence that does not independently justify a change.
- Triage: useful directionally, but insufficient repetitions or provenance.
- Rejected: missed the performance gate or regressed the target workload.
- Unresolved: conflicting measurements or uncertain executed path.

## Hardware

| Device | GPU | VRAM | Note |
| --- | --- | ---: | --- |
| ROCm0 | gfx906 | 32 GiB | 225 W cap; observed PCIe 8.0 GT/s x8 |
| ROCm1 | gfx906 | 32 GiB | 178 W cap; observed PCIe 16.0 GT/s x4 |

The system has one MI60 and one MI50, but the ROCm device-to-board mapping has
not been recorded reliably. Tensor split mode has already been established as
the fastest production mode, so topology calibration is not a current
priority. Do not combine single-device results. Advertised HBM bandwidth is
not a measured roofline.

## Current candidate

The tree and clean control are based on `571d0d540`. The reduced exact-gfx906
candidate passed 203 of 203 focused ROCm0 `MUL_MAT` cases for Q4_0, Q4_1,
Q4_K, Q5_K, Q6_K, and Q8_0.

Active gates are Q4_K MMQ precompute, Q6_K `min_blocks=1`, Q4_K branchless
MMVQ, and selective Q8_0 rocBLAS dispatch. Q4_K `min_blocks=3`, paired MMVQ,
Q8_1 DPP, and custom Flash Attention gates are disabled.

### Matched current-head screens

All model runs used upstream `-fa on`; only custom FA experiments were
disabled. Each row is one timed repetition after warmup with adjacent control
and candidate runs. The reduced screen used a 55 C edge-temperature gate.
The recorded dual-GPU Q8_0 screens used layer split mode; confirm the frozen
candidate once in the established production tensor mode.

| Candidate | Configuration | Test | Control | Candidate | Delta | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Combined gates | UD-Q4_K_XL single | pp8192 | 217.02 | 179.98 | -17.1 percent | Reject combined gates |
| Combined gates | UD-Q4_K_XL single | tg128 d8192 | 18.25 | 19.21 | +5.3 percent | Triage only |
| Combined gates | Q8_0 dual | pp8192 | 277.29 | 357.77 | +29.0 percent | Retain Q8 dispatch |
| Combined gates | Q8_0 dual | tg128 d8192 | 16.69 | 16.44 | -1.5 percent | Neutral at one sample |
| Reduced gates | UD-Q4_K_XL single | pp8192 | 215.33 | 203.52 | -5.5 percent | Reject combined Q4_K/Q6_K PP set; isolate |
| Reduced gates | UD-Q4_K_XL single | tg128 d8192 | 16.98 | 20.66 | +21.7 percent | Retain with one-sample/order caveat |
| Reduced gates | Q8_0 dual | pp8192 | 267.73 | 342.52 | +27.9 percent | Repeats Q8 dispatch win |

The reduced Q8_0 TG cell was interrupted and remains unrun. Do not infer it
from the PP result.

### Target model inventory

| Format | Size | Tensor type counts |
| --- | ---: | --- |
| UD-Q4_K_XL | 17,909,097,600 bytes | f32 456, q4_K 225, q5_K 70, q6_K 66, q8_0 49 |
| Q8_0 | 29,047,084,160 bytes | f32 360, q8_0 506 |
| Q4_0 | 16,056,476,800 bytes | f32 456, q4_0 352, q4_1 8, q5_K 48, q6_K 1, q8_0 1 |
| Q4_1 | 17,540,703,360 bytes | f32 456, q4_1 360, q5_K 48, q6_K 1, q8_0 1 |

Current snapshot hashes:

```text
UD-Q4_K_XL  4085665ee36d82a672a238a43f0e5643f2f0e39f2d7bd5d373f0ef10ecf53095
Q8_0        9408dcb356cc061a05c139e5647cbde0698ff980c6a69f7fc214e9989f86cfa8
```

Q4_0 and Q4_1 throughput against the current candidate remains untested. The
files are from an older snapshot, so verify matching model revision metadata
before comparing. Report speed and quality as a trade curve; do not use a
fixed throughput threshold.

## Operator baseline

Historical shape: `m=4096`, `k=14336`; `n=1` for TG-like MMVQ and `n=512`
for PP-like MMQ.

| Type | n=1 latency | n=1 TFLOPS | n=512 latency | n=512 TFLOPS |
| --- | ---: | ---: | ---: | ---: |
| F16 | 199.78 us | 0.588 | 5222.55 us | 11.51 |
| Q4_0 | 72.96 us | 1.61 | 3058.84 us | 19.66 |
| Q8_0 | 101.17 us | 1.16 | 7058.67 us | 8.52 |
| Q4_K | 68.53 us | 1.71 | 3779.61 us | 15.91 |
| Q6_K | 97.96 us | 1.20 | 6182.09 us | 9.73 |

Classification: historical baseline, not a current end-to-end result.

## PP profile

The July 2 pp8192 trace used an older Q4_K_M model/build on ROCm0 and
attributed GPU kernel time as follows:

| Kernel group | Share |
| --- | ---: |
| Q4_K MMQ | 54.07 percent |
| Q6_K MMQ | 19.80 percent |
| Q5_K MMQ | 6.16 percent |
| DeltaNet | 8.13 percent |
| Flash Attention plus combine | 4.24 percent |
| rocBLAS GEMM | 2.72 percent |
| Other | about 5 percent |

Q4_K counters at the matching pp512 chunk shape:

| Counter | Q4_K |
| --- | ---: |
| VGPR per lane | 128 |
| Waves per SIMD | 2 |
| VALU busy | 60.0 percent |
| VALU lane utilization | 100 percent |
| Memory unit busy | 11.3 percent |
| Memory stalled | 0.06 percent |
| LDS bank conflicts | 0.12 percent |

Conclusion for that build: counters support an occupancy or dependency-latency
limit and do not support HBM or LDS-conflict limits. They are prioritization
evidence, not a current UD-Q4_K_XL roofline or exact Amdahl weighting. Recheck
current kernel resources and one target-model trace before redesigning tiles.

## Historical PP evidence

These measurements predate the current upstream merge. Current-head evidence
overrides them where it conflicts.

### Q8_0 MMQ/rocBLAS crossover

Repeated historical model comparisons:

| Test | Gain |
| --- | ---: |
| pp512 | 21 to 25 percent |
| pp2048 | 53 to 57 percent |
| tg128 | Approximately neutral |

Retained conclusion: use MMQ only for smaller batches and rocBLAS for wide
dense Q8_0 PP. The exact crossover remains shape dependent.

### Q4_K metadata precompute

Three alternating single-GPU runs:

| Test | Control | Candidate | Gain |
| --- | ---: | ---: | ---: |
| pp512 | 176.00 | 191.36 | 8.73 percent |
| pp2048 | 215.01 | 235.14 | 9.36 percent |
| pp8192 | 198.19 | 220.03 | 11.02 percent |
| pp20000 | 168.56 | 177.50 | 5.31 percent |

Classification: historical signal contradicted by the current combined Q4_K
precompute/stride-9 plus Q6_K result. The current result cannot distinguish
precompute, layout, Q6_K, or interactions. Isolate before retaining.

### Q4_K stride-9 metadata layout

| Test | Control | Candidate | Gain |
| --- | ---: | ---: | ---: |
| pp512 | 178.82 | 182.67 | 2.15 percent |
| pp2048 | 214.45 | 219.46 | 2.34 percent |
| pp8192 | 211.77 | 225.79 | 6.62 percent |
| pp20000 | 179.62 | 179.56 | -0.03 percent |

The layout reduced historical aggregate bank conflicts, but current counters
show conflicts are no longer the limiting resource. Classification: supporting.

### Q6_K `min_blocks=1`

Six alternating pp8192 runs improved the median by 4.00 percent; excluding the
first run gave 4.14 percent. Classification: historical signal; current-head
contribution is unresolved because it is combined with the regressing Q4_K PP
change.

## Historical TG evidence

A tg32 trace attributed 53.83 percent of kernel time to Q4_K MMVQ, 19.46
percent to Q6_K MMVQ, and 4.57 percent to Q5_K MMVQ. The fused Q4_K kernel
alone was 40.68 percent.

### Q4_K branchless scale decode

| Test | Control | Candidate | Gain |
| --- | ---: | ---: | ---: |
| n=1 operator | 69.27 us | 66.02 us | 4.7 percent lower latency |
| tg64 d0, order 1 | 23.60 t/s | 25.01 t/s | 6.0 percent |
| tg64 d0, reverse order | 23.60 t/s | 24.86 t/s | 5.4 percent |

Focused Q4_K correctness passed 41 of 41 and the full gate passed 1103 of
1103. Classification: retain. The current depth-8192 result is one ordered
sample, but its large positive direction agrees with the smaller repeated
historical gain. Record the caveat instead of spending more full-model runs.

### Stored Q8_1 sum reuse

Alternating tg64 d0 invocations improved mean throughput from 25.03 to 25.26
t/s, about 0.9 percent. Focused correctness passed 41 of 41 and the full gate
passed 1103 of 1103. Classification: retained small signal.

## Rejected work

| Experiment | Result | Decision |
| --- | ---: | --- |
| Q4_K `min_blocks=1` | pp8192 -33.74 percent | Reject |
| Q4_K larger activation K tile | pp8192 -9.27 percent | Reject tested mapping |
| Forced Q4_K rocBLAS/F16 | -30 to -34 percent | Reject |
| Q4_K stride 10 | pp8192 -3.86 percent | Reject |
| Q4_K stride 11 | pp8192 -7.12 percent | Reject |
| Q4_K group2 padding | conflicts returned to 4.332 billion | Reject |
| Q4_K two accumulators | n=512 -2.4 percent | Reject for PP |
| Q6_K metadata precompute | pp8192 -5.3 percent | Reject |
| Q5_K metadata precompute | model pp512 neutral; one pp8192 pair -17.96 percent | Reject |
| Q8_0 MMVQ VDR 4 | tg32 d0 +0.19 percent | Neutral; keep VDR 2 |

## Triage-only combined run

The June 20 clean-versus-patched run used one timed sample. It is useful only
as a consistency check:

| Configuration | Test | Control | Candidate | Delta |
| --- | --- | ---: | ---: | ---: |
| Q4_K_M single | pp8192 | 180.34 | 189.48 | +5.1 percent |
| Q4_K_M single | tg128 d8192 | 17.53 | 17.31 | -1.2 percent |
| Q4_K_M dual | pp8192 | 372.04 | 390.38 | +4.9 percent |
| Q8_0 dual | pp8192 | 248.64 | 321.64 | +29.4 percent |
| Q8_0 dual | tg128 d8192 | 19.12 | 18.94 | -1.0 percent |

The PP direction matches isolated experiments. The TG deltas do not establish
a result.

## Evidence locations

Raw data retained in this tree:

- `bench-results/gfx906-pp8192-kernel-profile-2026-07-02/`
- `bench-results/gfx906-q4k-experiments/`
- `bench-results/worktree-comparison-2026-07-13/`
- `bench-results/gfx906-head-core-screen-2026-07-18/`
- `bench-results/gfx906-head-q8-screen-2026-07-18/`
- `bench-results/gfx906-head-retained-screen-2026-07-18/`

Several historical results summarized above no longer have raw directories in
this tree. Their classification is intentionally no stronger than the surviving
record supports. Do not recreate missing evidence from memory; rerun a
decision-relevant cell when it becomes load-bearing.
