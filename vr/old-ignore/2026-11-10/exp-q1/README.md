# Q1: post-AllReduce add + RMS norm + q8_1 fusion on 27B TG (2026-11-10)

Kind: profile and kernel experiment. Result: no code change kept.

- Source: `0f7ce1b04` (fork merged with upstream `aa75ff323`), branch `exp-q1-work`.
- A: `build-catchup-20261110-151625` (same dense TP decode code as `0f7ce1b04`). B: `build-exp-q1`, flags from [how-to-catchup.md](../../../how-to-catchup.md).
- Model: Qwen3.8-27B Q8_0 rev `4ca720788d1e`, `-ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa on -b 2048 -ub 2048 -t 20`.
- Profile: `rocprofv3 --kernel-trace` of `llama-bench -p 0 -n 64 -r 1` ([prof.sh](prof.sh)), window after the last gap above 20 ms, summed by [summ.py](summ.py). rocprofv3 slows TG (31 ms/token instead of about 27).

## What exists already

Q1 option (a) is already in the tree. `ggml_cuda_try_fuse` runs ADD + RMS_NORM + MUL as one kernel, `add_rms_norm_mul_f32` (norm.cu), and `ggml_cuda_q8_1_producer_target` lets that kernel also write the Q8_1 row into the q8_1 cache for the next mat-vec (`GGML_CUDA_Q8_1_PRODUCER=0` disables). So after each AR there is one kernel, not 3-4. The Fable estimate of 400-500 launches/token assumed the chain was unfused.

## Profile (A, per token per GPU, ROCm0 / ROCm1)

| Kernel | Calls | Avg us | ms/token |
| --- | ---: | ---: | ---: |
| `mul_mat_vec_q8_0_repacked` (both variants) | 440 | 30 / 128 | 19.7 / 19.9 |
| `ggml_cuda_ar_kernel` | 130 | 30.0 / 12.6 | 3.90 / 1.64 |
| `add_rms_norm_mul_f32<1024>` (with q8_1 epilogue) | 130 | 10.6 / 10.9 | 1.37 / 1.42 |
| `quantize_q8_1` (attention/SSM out and FFN down inputs, not post-AR) | 131 | 4.2 / 5.0 | 0.55 / 0.65 |

With `GGML_CUDA_Q8_1_PRODUCER=0`: `add_rms_norm_mul_f32` 7.1 us plus a second `quantize_q8_1` 4.2 us = 11.3 us, against 10.6 us fused. The fused q8_1 epilogue costs 3.5 us inside the kernel, almost as much as a separate launch. Kernels run back to back on one queue (no gaps), so only kernel time can be recovered. The whole post-AR kernel is 1.4 ms/token (about 4.5%); a perfect fold into the AR kernel would recover at most about 1 ms (the floor for a tiny kernel here is about 4.2 us). The ROCm0 AR time is mostly waiting for ROCm1 (P3).

## Tried

| Variant | Post-AR kernel avg us (ROCm0 / ROCm1) | Profiled ms/token | Greedy 64 tokens |
| --- | ---: | ---: | --- |
| A | 10.6 / 10.9 | 31.49 | reference |
| B1: unrolled epilogue, row kept in registers, all Q8_1 group reductions in flight ([patch](unrolled-v1.patch)) | 12.6 / 13.2 | 31.00 | identical |
| B2: split output columns over 5 blocks, each block recomputes the full-row sum of squares in the same order ([patch](split-v2.patch)) | not taken | 30.94 | identical |

The ms/token column moves with the AR wait on ROCm0 (27-30 us per call between runs) and is noise here.

B1 is slower: the shuffle chains are not the bottleneck. B2 is never taken: a debug print showed the residual ADD runs in place (`sum == a`) and the MUL output reuses the buffer of an add input (`dst == a` or `dst == b`). With several blocks, a block that writes its columns would corrupt rows that other blocks still read. B2 needs a grid barrier between the reads and the writes, which costs about as much as it saves and needs a per-device counter that is safe under concurrent streams and graph replay. A guard keeps B2 off when outputs alias inputs, so its greedy output matched only because it ran the old kernel.

Checks on B builds: `vr-test-gfx906-tp --n 1 --check` and `--n 2048 --check` pass; `test-backend-ops -b ROCm0` ADD_RMS_NORM 25/25, RMS_NORM 61/61, ADD 167/167, MUL_MAT 2520/2520 (these do not cover the q8_1 producer path; the greedy check does). Greedy check: [greedy.sh](greedy.sh), two prompts, 64 tokens, sha `65cbbfedfad1` and `e0aff3500f30` for A, B1, B2.

No llama-bench A/B was run: B1 is slower by kernel time and B2 runs the A kernel. A short A run with and without the producer gave TG64 34.61 +- 0.79 and 35.91 +- 0.28 (r2, while another agent was building), inside noise.

## Verdict

Q1 (a) is already done; the Q1 estimate (2.5-3.5 ms) is wrong for this tree. What is left is about 1.4 ms/token in one kernel per AR, of which at most about 0.8 ms is above the launch floor. Not worth more work alone. The only route to most of it is option (b): do the residual add, norm and q8_1 in the AR kernel's phase 3. That needs a cross-block step in `ggml_cuda_ar_kernel` (8 blocks per call) and should be done together with AR-LL, which rewrites the same kernel. Raw profiles: `prof-*.txt`.
