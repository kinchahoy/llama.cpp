# gfx906 results ledger

Last updated: 2026-06-20

This is the sole summary of measured gfx906 results. Raw JSONL, comparison
text, and profiler output under `bench-results/` remain authoritative. Results
are classified as confirmed, triage-only, rejected, or unresolved so that an
old screening run is not mistaken for an acceptance gate.

## Result standards

- Confirmed: correctness passed and the performance result has repeated,
  appropriately ordered measurements.
- Triage-only: useful for selecting follow-up work, but based on insufficient
  repetitions, incomplete provenance, or uncontrolled telemetry.
- Rejected: the tested hypothesis missed its performance gate or regressed.
- Unresolved: measurements conflict or the executed kernel path is unclear.

Historical reports used different repetition and warmup policies. Their raw
numbers are retained, but confidence is assigned using the policy above rather
than the label originally written by the harness.

## Active scope

Current work covers code-level GPU optimization: quantized kernels, fusion,
launch geometry, memory layout, and Flash Attention tile tuning. Settings-only
changes are excluded, including KV-cache precision, speculative decoding,
layer/row split selection, and tensor placement. Historical configuration
results remain in this ledger only as prior evidence.

## Test hardware

| Label | GPU | Backend | VRAM | Relevant limitation |
| --- | --- | --- | ---: | --- |
| MI50-0 | AMD Vega 20, gfx906 | ROCm0 | 32 GiB | 225 W cap, CPU-attached PCIe |
| MI50-1 | AMD Vega 20, gfx906 | ROCm1 | 32 GiB | 178 W cap, chipset-attached PCIe |
| RTX 3090 | NVIDIA CC 8.6 | CUDA0 | 24 GiB | Comparison device only |

The two MI50 cards are not interchangeable. Results must identify the device or
split topology. Advertised HBM2 peak is not the practical bandwidth denominator
unless active clocks and sustainable bandwidth were also measured.

## Original operator baseline

The original `MUL_MAT` comparison used `m=4096`, `k=14336`, and either `n=1`
for decode shape or `n=512` for prefill shape. MI50 data came from ROCm commit
`d8a24ccee`; the RTX 3090 used the equivalent CUDA test.

### Decode shape, n=1

| Weight | MI50-0 latency | MI50-0 TFLOPS | MI50-1 latency | MI50-1 TFLOPS | RTX 3090 latency | RTX 3090 TFLOPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F16 | 199.78 us | 0.588 | 200.33 us | 0.586 | 133.19 us | 0.882 |
| Q4_0 | 72.96 us | 1.61 | 80.04 us | 1.47 | 41.31 us | 2.84 |
| Q8_0 | 101.17 us | 1.16 | 111.14 us | 1.06 | 74.54 us | 1.58 |
| Q4_K | 68.53 us | 1.71 | 78.68 us | 1.49 | 43.14 us | 2.72 |
| Q6_K | 97.96 us | 1.20 | 111.89 us | 1.05 | 68.69 us | 1.71 |

### Prefill shape, n=512

| Weight | MI50-0 latency | MI50-0 TFLOPS | MI50-1 latency | MI50-1 TFLOPS | RTX 3090 latency | RTX 3090 TFLOPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F16 | 5222.55 us | 11.51 | 6606.26 us | 9.10 | 737.33 us | 81.55 |
| Q4_0 | 3058.84 us | 19.66 | 4308.35 us | 13.96 | 723.35 us | 83.13 |
| Q8_0 | 7058.67 us | 8.52 | 8662.66 us | 6.94 | 762.32 us | 78.88 |
| Q4_K | 3779.61 us | 15.91 | 5053.89 us | 11.90 | 863.83 us | 69.61 |
| Q6_K | 6182.09 us | 9.73 | 7557.69 us | 7.96 | 893.33 us | 67.31 |

Classification: historical baseline. It identifies candidate paths but is not
an end-to-end model result and does not establish a current bandwidth roofline.

## Accepted prefill changes

### Q8_0 selective rocBLAS dispatch

Historical Qwen3.6-27B comparisons reported:

| Test | Gain | Classification |
| --- | ---: | --- |
| pp512 | 21 to 25 percent | Confirmed historical signal |
| pp2048 | 53 to 57 percent | Confirmed historical signal |
| tg128 | Approximately neutral | Historical only |

The June 20 single-sample baseline also showed +27.6 percent at pp512 and +29.4
percent at pp8192 for the dual-GPU Q8_0 model. That run is triage-only and used
a different matrix from the earlier experiment.

### Q4_K metadata precompute

Three alternating single-GPU runs, warmup disabled:

| Test | Control | Candidate | Gain | Classification |
| --- | ---: | ---: | ---: | --- |
| pp512 | 176.00 | 191.36 | 8.73 percent | Confirmed historical signal |
| pp2048 | 215.01 | 235.14 | 9.36 percent | Confirmed historical signal |
| pp8192 | 198.19 | 220.03 | 11.02 percent | Confirmed historical signal |
| pp20000 | 168.56 | 177.50 | 5.31 percent | Confirmed historical signal |
| tg128 | 18.32 | 19.42 | 6.03 percent | Unresolved path/noise |

The reported TG result conflicts with the prefill-only scope of the MMQ change.
Retain it as an anomaly until profiling proves that generation executes the
modified path and a current repeated benchmark reproduces the gain.

### Q4_K stride-9 follow-up

Three alternating single-GPU runs, warmup disabled:

| Test | Control | Candidate | Gain | Classification |
| --- | ---: | ---: | ---: | --- |
| pp512 | 178.82 | 182.67 | 2.15 percent | Supporting evidence |
| pp2048 | 214.45 | 219.46 | 2.34 percent | Supporting evidence |
| pp8192 | 211.77 | 225.79 | 6.62 percent | Confirmed historical signal |
| pp20000 | 179.62 | 179.56 | -0.03 percent | Neutral |
| tg128 | 17.60 | 17.90 | 1.72 percent | Unresolved/noise-sized |

Profiler evidence:

- Aggregate pp2048 LDS bank conflicts: 4.332 billion to 1.999 billion.
- Dominant Q4_K shape: 1.783 billion to 0.357 billion.
- Full ROCm0 `MUL_MAT` correctness after stride 9: 1103 of 1103 passed.

### Q6_K minimum launch occupancy

Six alternating pp8192 runs reported:

| Metric | Gain | Classification |
| --- | ---: | --- |
| Median, all six runs | 4.00 percent | Confirmed historical signal |
| Median, runs 2 through 6 | 4.14 percent | Confirmed historical signal |

The direct scratch-memory trace remains outstanding, so spill removal is still
a profiled explanation rather than a completed validation artifact.

### Q5_K metadata precompute and stride-9 experiment

The Q4_K metadata technique was ported to Q5_K as an isolated gfx906
translation-unit specialization. Focused Q5_K/F32 correctness passed 11 of 11
cases on ROCm0, followed by the full ROCm0 `MUL_MAT` gate at 1103 of 1103.

At `m=4096`, `n=512`, and `k=14336`, four sequential measurements per build
gave median throughput of approximately 12.25 TFLOPS for control and 13.19
TFLOPS for precompute/stride 9, a gain of about 7.7 percent.

| Variant | Representative throughput | Delta | Classification |
| --- | ---: | ---: | --- |
| Control | 12.25 TFLOPS | - | Quick operator baseline |
| Q5_K precompute plus stride 9 | 13.19 TFLOPS | +7.7 percent | Synthetic-only signal |
| Q5_K `min_blocks=1` | 10.13 TFLOPS | about -17 percent | Rejected |

The local Unsloth Qwen3.6-27B Q4_K_M dynamic quant contains 48 Q5_K tensors
totaling 0.967 GiB, in addition to 294 Q4_K tensors and 67 Q6_K tensors. It is
therefore a valid model-level gate for this change.

The quick single-GPU model comparison did not reproduce the synthetic gain:

| Test | Control | Candidate | Candidate delta | Classification |
| --- | ---: | ---: | ---: | --- |
| pp512, first order | 257.27 t/s | 242.20 t/s | -5.86 percent | Order-sensitive |
| pp512, reverse order | 255.87 t/s | 256.52 t/s | +0.26 percent | Neutral |
| pp8192, one pair | 237.82 t/s | 195.10 t/s | -17.96 percent | Regression signal |

These runs are intentionally noisy and insufficient to estimate a small
model-level delta, but they are sufficient to reject enabling the patch: the
real dynamic-quant workload showed no repeatable gain and one material
regression. The standalone patch and raw experiment summary are retained for
reference, while the source tree continues to use the generic Q5_K path.

## Rejected experiments

| Experiment | Result | Conclusion |
| --- | ---: | --- |
| Q4_K `min_blocks=1` | pp8192 -33.74 percent | Q4_K needs occupancy to hide LDS waits |
| Q4_K y64 | pp8192 -9.27 percent | Additional tile work outweighed register reduction |
| Forced Q4_K rocBLAS/F16 | -30 to -34 percent | Keep the fused MMQ path |
| Q4_K stride 10 | pp8192 -3.86 percent, pp20000 -6.76 percent | Reject tested layout |
| Q4_K stride 11 | pp8192 -7.12 percent, pp20000 -5.87 percent | Reject tested layout |
| Q4_K group2pad1 | Conflicts returned to 4.332 billion | Reject tested layout |
| Q6_K metadata precompute | pp8192 -5.3 percent | Wrong prefill bottleneck |
| Q5_K metadata precompute | pp512 neutral/order-sensitive; pp8192 -17.96 percent in one pair | Synthetic shape did not pass the model gate |
| Q4_K two-accumulator DP4A | n=1..8 +1.8 to +3.8 percent; n=512 -2.4 percent | Narrow-shape ILP gain did not pass the PP gate |
| Q8_0 MMVQ VDR 4 | tg32 d0 +0.19 percent | Neutral; keep VDR 2 |

The Q4_K group4pad1 result was order-sensitive and mixed. It remains a profiling
clue, not an accepted optimization.

## Accepted Q4_K TG changes

A tg32 kernel trace attributed 53.83 percent of GPU kernel time to Q4_K MMVQ,
19.46 percent to Q6_K MMVQ, and 4.57 percent to Q5_K MMVQ. The fused Q4_K MMVQ
kernel alone accounted for 40.68 percent, confirming Q4_K as the primary TG
target for this model.

The branch-free Q4_K scale/min decoder produced:

| Test | Control | Candidate | Delta | Classification |
| --- | ---: | ---: | ---: | --- |
| n=1 operator latency | 69.27 us median | 66.02 us median | -4.7 percent | Repeated signal |
| tg64 d0, candidate run 1 | 23.60 t/s | 25.01 t/s | +6.0 percent | Accepted quick signal |
| tg64 d0, reverse-order candidate | 23.60 t/s | 24.86 t/s | +5.4 percent | Accepted quick signal |
| tg32 d8192 | 18.09 t/s | 17.76 to 22.17 t/s | Order-sensitive | Unresolved |

Focused Q4_K/F32 correctness passed 41 of 41 and the full ROCm0 `MUL_MAT` gate
passed 1103 of 1103. The change is accepted for gfx906 TG. The depth-8192 cell
remains noisy and should not be used to claim a long-context gain.

The follow-up reuses the scaled Q8_1 sum in `ds.y` for the minimum term. Four
lanes partition each Q8_1 block and use the same minimum, so each lane adds one
quarter of the stored sum and avoids four DP4A operations per vec-dot call.

Alternating single-GPU `tg64 d0` runs with three timed samples per invocation
produced 25.30 and 25.21 t/s for the candidate, bracketed by controls at 25.05
and 25.02 t/s. Mean invocation throughput improved from 25.03 to 25.26 t/s,
about 0.9 percent. Focused Q4_K/F32 correctness passed 41 of 41 and the full
ROCm0 `MUL_MAT` gate passed 1103 of 1103.

## June 20 head-versus-patches baseline

Source directory:

```text
vr/bench-results/gfx906-head-2026-06-20/
```

These results used one timed sample with warmup. They are triage-only. The
result files do not contain a complete source/patch provenance manifest or
clock/power telemetry.

| Configuration | Test | Control | Candidate | Delta |
| --- | --- | ---: | ---: | ---: |
| Q4_K_M dual | pp512 | 224.71 | 237.30 | +5.6 percent |
| Q4_K_M dual | pp8192 | 372.04 | 390.38 | +4.9 percent |
| Q4_K_M dual | tg128 d0 | 23.18 | 22.93 | -1.1 percent |
| Q4_K_M dual | tg128 d8192 | 22.44 | 22.02 | -1.9 percent |
| Q4_K_M dual | tg128 d16384 | 21.14 | 21.20 | +0.3 percent |
| Q4_K_M single | pp512 | 234.75 | 241.07 | +2.7 percent |
| Q4_K_M single | pp8192 | 180.34 | 189.48 | +5.1 percent |
| Q4_K_M single | tg128 d0 | 17.59 | 17.42 | -1.0 percent |
| Q4_K_M single | tg128 d8192 | 17.53 | 17.31 | -1.2 percent |
| Q4_K_M single | tg128 d16384 | 16.14 | 16.23 | +0.6 percent |
| Q8_0 dual | pp512 | 145.45 | 185.54 | +27.6 percent |
| Q8_0 dual | pp8192 | 248.64 | 321.64 | +29.4 percent |
| Q8_0 dual | tg128 d0 | 17.35 | 18.98 | +9.3 percent |
| Q8_0 dual | tg128 d8192 | 19.12 | 18.94 | -1.0 percent |
| Q8_0 dual | tg128 d16384 | 16.87 | 17.39 | +3.1 percent |

Interpretation:

- The PP results reinforce the accepted Q8_0 and Q4_K directions.
- The TG deltas are internally inconsistent and cannot support a conclusion.
- The Q8_0 d0 TG increase is especially suspect because it disagrees with both
  long-context cells and the documented scope of the source change.
- Repeat only a decision-relevant TG cell with a verified clean control. Start
  with one sample per build; add alternating invocations and telemetry only
  when noise prevents a decision or the result is being promoted.

The directory `gfx906-head-2026-06-20-r3-aborted/` is incomplete and must not be
used as an acceptance result.

## Historical Qwen quick report

The June 13 Qwen quick comparison used one repetition, disabled built-in
warmup, and tested an intermediate candidate. It successfully identified the
Q8_0 prefill opportunity but showed Q4_K_M prefill regressions in that candidate
and skipped the long benchmark.

Classification: historical triage only. The raw JSONL remains under
`bench-results/gfx906-qwen36/quick/`. It is not the current baseline.

## Evidence locations

- `bench-results/gfx906-head-2026-06-20/`: current single-sample screening run
- `bench-results/gfx906-qwen36/quick/`: early Qwen matrix
- `bench-results/gfx906-q4k-experiments/q4k-metadata-precompute-20260615/`:
  accepted precompute experiment
- `bench-results/gfx906-q4k-experiments/q4k-stride9-20260616/`: stride-9 data
- `bench-results/gfx906-q4k-experiments/q4k-dp4a-ilp2-20260620/`:
  rejected independent-accumulator experiment
- `bench-results/gfx906-q6k-experiments/20260615-q6k-mb1-n6/`: Q6_K data
- `bench-results/gfx906-q5k-experiments/q5k-precompute-stride9-20260620/`:
  Q5_K synthetic and model-gate summary
- `bench-results/gfx906-q4k-profile/`: original Q4_K profiler capture
- `bench-results/gfx906-q4k-profile-precompute/`: precompute profiler capture
- `bench-results/gfx906-q4k-profile-stride9/`: stride-9 profiler capture
- `bench-results/gfx906-q4k-tg-profile/`: TG kernel trace and quant time share
- `bench-results/gfx906-q4k-tg-experiments/branchless-scales-20260620/`:
  accepted Q4_K TG change
- `bench-results/gfx906-q4k-tg-experiments/stored-q8-sum-20260620/`:
  accepted Q8_1 sum-reuse follow-up
- `bench-results/gfx906-q4k-tg-experiments/mmvq-geometry-20260620/`:
  rejected MMVQ geometry sweep
- `bench-results/gfx906-q8-tg-experiments/vdr-20260620/`:
  Q8_0 TG profile and VDR experiment

## Superseded reports

The exact Markdown files consolidated into this ledger are stored in:

```text
vr/archive/docs-pre-consolidation-2026-06-20.tar.gz
```

Its SHA-256 is:

```text
8dcbf659820e9cac33ba7caae6155d0558cb4544f5ca97a84fbc1f6d3f521c2e
```

Use the archive when a consolidated number does not match an older narrative.
Raw data should resolve the discrepancy; do not silently overwrite this ledger
to match a stale generated report.

The README and results ledger immediately before the Q5_K dynamic-quant model
gate were preserved in:

```text
vr/archive/docs-pre-q5-model-gate-2026-06-20.tar.gz
```

Its SHA-256 is `12cd89aebb0d0b373ac8b6faa744aabf60399ef63a8fac361853b72d68f351f2`.

The stable PP ledger before the Q4_K TG phase is preserved in:

```text
vr/archive/docs-pp-stable-pre-tg-2026-06-20.tar.gz
```

Its SHA-256 is `005ff52bfae076783fa9a3d2045309d95afdc036f782cd0772418f7a37978ccc`.

The candidate-stage ledger before full Q4_K TG acceptance is preserved in:

```text
vr/archive/docs-pre-q4k-tg-acceptance-2026-06-20.tar.gz
```

Its SHA-256 is `cf99dd6a0db8af3c8b69415ce62de62d2076da719ada2f6ff3966168936b1393`.

The ledger before the kernel-only scope decision is preserved in:

```text
vr/archive/docs-pre-kernel-only-scope-2026-06-20.tar.gz
```

Its SHA-256 is `3ec7c6f18b499a2b74129a0b828a85ef0481c8a847caf28d012223794ed7cf67`.

The guide and ledger before the fused-Q4_K restart handoff are preserved in:

```text
vr/archive/docs-pre-fused-q4k-handoff-2026-06-20.tar.gz
```

Its SHA-256 is `a75d336d418a12690a3e2176f122f43f56a94d6a3009c3f8092b9a68136b4c19`.
