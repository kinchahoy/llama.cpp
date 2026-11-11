# play-b1: one-warp MoE routing kernel (2026-11-10)

Change: `topk_moe_wave_cuda<512,10>` in ggml/src/ggml-cuda/topk-moe.cu, gfx906 only (no bias, no groups, no delayed softmax, rows <= 8).
WARP_SIZE is 32 in this fork, so two logical rows share one wave64.

Findings
- The old kernel had no scratch use (0 bytes) and 116 VGPRs; the 28 us was not a scratch problem.
- Isolated microbench (warm caches, 1 row): old 17.5 us, new 8.8 us, empty kernel 1.9 us.
- In the decode trace the first version (fully unrolled k loop) took 25.0 us per call, the rolled-k version 14.9 us (old 28.8 us). Cold instruction fetch of a single-wave kernel is a large cost, so code size matters.
- Two bugs found on the way: readlane over a 64-lane assumption (reviewer), and the 64-bit ballot of a wave64 shared by two rows returned no bits for the upper row in the real build (not reproduced in a standalone harness). Replaced the ballot by a DPP min-reduction of the winning expert index.

Tests (build-play-b1, ROCm0)
- test-backend-ops test -o TOPK_MOE, 5 runs (and 2 more after the last change): 0 failures in all n_expert_groups=1 cases, including the new {512,R,1,1} k=10 cases for R=1..9.
- The n_expert_groups>1 cases (256/128/512 with groups) fail with and without the new kernel (94 failures with the new kernel disabled by a temporary env switch); not caused by this change.
- Greedy llama-completion, Flash Next, 200 tokens, temp 0: output byte-identical to build-exp-spec (971 bytes, cmp equal), for the first and the last version.
- KLD not run: build-exp-spec has no llama-perplexity.

Flash Next TG128 @ d8192 (llama-bench -r 5, -ts 1/1, alternating, 60 s idle)
- A build-exp-spec: 45.27 42.17 45.71 45.83 45.83 45.40 -> mean 45.03 +- 1.42 t/s
- B build-play-b1 (final): 46.31 46.85 47.61 -> mean 46.93 +- 0.65 t/s
- change +4.2 percent (about 0.9 ms per token; the trace predicts 48 x 13.9 us = 0.67 ms).
- Earlier version (25 us/call): 44.29 46.29 46.30, mean 45.62.

Trace (rocprofv3 kernel trace, TG64 @ 8192): topk_moe_cuda<512> 28.84 us/call (7648 calls) vs topk_moe_wave_cuda<512,10> 14.89 us/call (6256 calls).
