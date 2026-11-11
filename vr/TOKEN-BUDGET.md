---
title: "Token budget: where one decode token and one prompt ubatch go"
description: Per-step time budget of decode (TG) and prefill (PP) for the target models on two gfx906 GPUs, from rocprofv3 traces of llama-bench
date: 2026-11-10
sources:
  columns: vr/token-budget-columns.csv
  steps: vr/token-budget-steps.csv
---

# Token budget: where one decode token and one prompt ubatch go

One decode token (TG) and one prompt ubatch (PP), split into model steps, side by side for the target models on two gfx906 GPUs (`-sm tensor`, host-staged AllReduce). Each column comes from one `rocprofv3` trace of llama-bench plus an unprofiled llama-bench run, cut into steps by [token_budget.py](scripts/token_budget.py). Use it to see which step is worth attacking before you start an experiment. For model limits and ideas see [PERFORMANCE.md](PERFORMANCE.md#what-limits-each-model).

This is a growing record. The source of truth is two append-only CSV files, [token-budget-columns.csv](token-budget-columns.csv) (one row per datapoint) and [token-budget-steps.csv](token-budget-steps.csv) (one row per datapoint, step and GPU). The tables between `<!-- tb:... -->` markers are generated from them; never edit their numbers by hand (see [How to update](#tb-update)).

Best viewed in VS Code with the MyST-Markdown extension (executablebooks.myst-highlight / the MyST VS Code extension); GitHub shows the same tables, minus the dropdowns.

(tb-tg)=

## Decode (TG)

### Main table (critical path, ms per token, profiled)

Each cell is `busy + idle-before (share of the profiled token)`. Busy is kernel time on the GPU that is on the critical path for that part of the token. Idle-before is GPU time with no kernel, charged to the step that waited to start after it. Step rows are means over steady tokens; the warm-up row adds the extra time of the first few tokens, so each column sums to the all-token profiled mean. The last row scales the total to the unprofiled llama-bench speed; scale step rows by the same factor if you need estimates, but the profiler inflates small kernels and host gaps more than mat-vecs (see [limits](#tb-method)).

<!-- tb:tg-main -->
| Row | 27B@0 2026-11-10 | 27B@32K 2026-11-10 | FN@8K 2026-11-10 | G4@0 2026-11-10 | FN@32K 2026-11-10 | 27B 1-GPU@0 2026-11-10 |
| --- | --- | --- | --- | --- | --- | --- |
| Measured | 2026-11-10 16:46 PST | 2026-11-10 17:50 PST | 2026-11-10 17:48 PST | 2026-11-10 18:28 PST | 2026-11-10 18:48 PST | 2026-11-10 18:48 PST |
| Source commit | 0f7ce1b04 | 0f7ce1b04 | 0f7ce1b04 | 0f7ce1b04 | 0f7ce1b04 | 0f7ce1b04 |
| Build (sha256 llama-bench / libggml-hip.so) | `build-exp-q3` (`246887e7bebf / d67316b02bbe`) | `build-exp-spec` (`e4e83771fc55 / 1ac1232870d5`) | `build-exp-spec` (`e4e83771fc55 / 1ac1232870d5`) | `build-exp-spec` (`e4e83771fc55 / 1ac1232870d5`) | `build-exp-spec` (`e4e83771fc55 / 1ac1232870d5`) | `build-exp-spec` (`e4e83771fc55 / 1ac1232870d5`) |
| Model, quant, file revision | Qwen3.8-27B Q8_0, rev 4ca720788d1e | Qwen3.8-27B Q8_0, rev 4ca720788d1e | Qwen3.8 Flash Next UD-IQ3_XXS rev d96201194f10 + Q8_0 n-gram table | Gemma 4 31B-it Q8_0, rev c1ac76e99d55 | Qwen3.8 Flash Next UD-IQ3_XXS rev d96201194f10 + Q8_0 n-gram table | Qwen3.8-27B Q8_0, rev 4ca720788d1e |
| Settings | -ts 1/1 -b 2048 -ub 2048 -fa on, depth 0, tg64 | -ts 1/1 -b 2048 -ub 2048 -fa on, depth 32768, tg32 | -ts 1/1 -b 1024 -ub 512 -fa on, depth 8192, tg64 | -ts 1/1 -b 2048 -ub 2048 -fa on, depth 0, tg64 | -ts 1/1 -b 1024 -ub 512 -fa on, depth 32768, tg64 | -dev ROCm0 (1 GPU, no -sm tensor) -b 2048 -ub 2048 -fa on, depth 0, tg64 |
| Unprofiled TG t/s (ms/token) | 37.45 +- 0.21 (26.70) (TG128 r3, exp-q3 record, same build) | 33.48 +- 0.97 (29.87) | 43.68 +- 0.60 (22.90) | 32.68 +- 0.15 (30.60) | 41.32 +- 0.97 (24.20) | 23.42 +- 0.01 (42.70) |
| Profiled ms/token: all tokens; steady; llama-bench | 29.44; 28.58; - | 34.29; 32.88; 35.12 | 27.18; 25.63; 28.39 | 33.08; 32.84; 33.09 | 27.93; 27.23; 28.64 | 43.99; 43.99; 44.02 |
| Kernels / AllReduces per token | 1595 / 128 | 1594 / 128 | 2908 / 96 | 1463 / 120 | 2870 / 96 | 1594 / 0 |
| Token start: host gap, no kernel | 0.00 + 1.02 (3%) | 0.00 + 1.20 (3%) | 0.00 + 2.09 (8%) | 0.00 + 1.07 (3%) | 0.00 + 2.51 (9%) | 0.00 + 0.35 (1%) |
| Token start: input copies, embedding | 0.04 + 0.00 (0%) | 0.03 + 0.00 (0%) | 0.08 + 0.00 (0%) | 0.06 + 0.00 (0%) | 0.07 + 0.00 (0%) | 0.02 + 0.00 (0%) |
| Warm-up tokens: excess over steady tokens, spread over all | 0.00 + 0.86 (3%) | 0.00 + 1.41 (4%) | 0.00 + 1.55 (6%) | 0.00 + 0.24 (1%) | 0.00 + 0.70 (3%) | - |
| Residual add + norm (Flash Next: hyper-connection combine + mix) | 1.39 + 0.00 (5%) | 1.47 + 0.00 (4%) | 3.88 + 0.00 (14%) | 1.66 + 0.00 (5%) | 3.86 + 0.00 (14%) | 0.63 + 0.00 (1%) |
| Per-layer n-gram embedding (Flash Next) | - | - | 0.25 + 0.00 (1%) | - | 0.22 + 0.00 (1%) | - |
| Mixer: input projections (mat-vec) | 4.60 + 0.00 (16%) | 4.74 + 0.00 (14%) | 2.69 + 0.00 (10%) | 4.51 + 0.00 (14%) | 2.69 + 0.00 (10%) | 8.07 + 0.01 (18%) |
| Attention: q/k norm, rope, KV write, gate | 0.66 + 0.00 (2%) | 0.67 + 0.00 (2%) | 0.50 + 0.00 (2%) | 1.67 + 0.00 (5%) | 0.49 + 0.00 (2%) | 0.59 + 0.01 (1%) |
| Attention: flash attention | 0.45 + 0.00 (2%) | 3.34 + 0.00 (10%) | 0.65 + 0.00 (2%) | 1.96 + 0.00 (6%) | 1.46 + 0.00 (5%) | 0.38 + 0.00 (1%) |
| Attention: QSA indexer (Flash Next) | - | - | 2.45 + 0.00 (9%) | - | 2.99 + 0.00 (11%) | - |
| Recurrent: conv, gated delta net, gates, state | 3.23 + 0.00 (11%) | 3.24 + 0.00 (9%) | 2.32 + 0.00 (9%) | - | 2.29 + 0.00 (8%) | 3.46 + 0.03 (8%) |
| Mixer: output projection (mat-vec) | 1.57 + 0.00 (5%) | 1.62 + 0.00 (5%) | 0.76 + 0.00 (3%) | 2.40 + 0.00 (7%) | 0.76 + 0.00 (3%) | 3.05 + 0.00 (7%) |
| AllReduce after mixer: exchange | 0.83 + 0.03 (3%) | 0.84 + 0.07 (3%) | 0.50 + 0.01 (2%) | 0.69 + 0.02 (2%) | 0.50 + 0.02 (2%) | - |
| AllReduce after mixer: wait for other GPU | - | - | - | - | - | - |
| FFN dense: gate/up mat-vec | 8.19 + 0.00 (28%) | 8.78 + 0.00 (26%) | - | 10.11 + 0.00 (31%) | - | 16.12 + 0.00 (37%) |
| FFN dense: activation | - | - | - | 0.04 + 0.00 (0%) | - | - |
| FFN dense: down mat-vec | 4.28 + 0.00 (15%) | 4.57 + 0.00 (13%) | - | 5.24 + 0.00 (16%) | - | 8.41 + 0.00 (19%) |
| MoE: router + top-k | - | - | 1.82 + 0.00 (7%) | - | 1.86 + 0.00 (7%) | - |
| MoE: routed experts | - | - | 2.58 + 0.00 (9%) | - | 2.61 + 0.00 (9%) | - |
| MoE: shared expert + gate | - | - | 1.41 + 0.00 (5%) | - | 1.42 + 0.00 (5%) | - |
| MoE: weighted reduce + combine | - | - | 1.03 + 0.00 (4%) | - | 0.87 + 0.00 (3%) | - |
| AllReduce after FFN: exchange | 0.73 + 0.02 (3%) | 0.72 + 0.05 (2%) | 0.49 + 0.01 (2%) | 1.16 + 0.02 (4%) | 0.49 + 0.01 (2%) | - |
| AllReduce after FFN: wait for other GPU | - | - | - | - | - | - |
| Output: final norm (+ softcap) | 0.01 + 0.00 (0%) | 0.01 + 0.00 (0%) | 0.08 + 0.00 (0%) | 0.03 + 0.00 (0%) | 0.08 + 0.00 (0%) | 0.01 + 0.00 (0%) |
| Output: LM head mat-vec | 0.88 + 0.00 (3%) | 0.95 + 0.00 (3%) | 0.52 + 0.00 (2%) | 1.00 + 0.00 (3%) | 0.52 + 0.00 (2%) | 1.75 + 0.00 (4%) |
| q8_1 quantize of mat-vec inputs | 0.63 + 0.00 (2%) | 0.62 + 0.00 (2%) | 1.50 + 0.00 (6%) | 1.18 + 0.00 (4%) | 1.49 + 0.00 (5%) | 1.11 + 0.01 (3%) |
| Glue: copies, crops, other | 0.01 + 0.00 (0%) | 0.01 + 0.00 (0%) | 0.01 + 0.00 (0%) | 0.01 + 0.00 (0%) | 0.01 + 0.00 (0%) | - |
| Critical-path rounding (GPU switch at AllReduce ends) | - | - | - | - | - | - |
| Total, profiled | 29.44 | 34.29 | 27.18 | 33.08 | 27.93 | 43.99 |
| Total, scaled to unprofiled | 26.70 | 29.87 | 22.90 | 30.60 | 24.20 | 42.70 |
<!-- /tb:tg-main -->

Column notes:

- 27B@0: the trace is the Q3 trace ([exp-q3](old-ignore/2026-11-10/exp-q3/README.md)); no unprofiled TG64 JSON was kept, so the unprofiled figure is the Q3 TG128 r3 result of the same build (`GGML_RP_MMV_ROWS=8`). Its profiled llama-bench figure was 33.84 t/s (29.55 ms) in `trace-bench.txt`.
- 27B@32K and FN@8K: traces from the P3/Q1 follow-up profile runs of `build-exp-spec`; G4@0 was traced for this record with [token_budget_trace.sh](scripts/token_budget_trace.sh).
- FN@32K: same build and settings as FN@8K except depth; traced with [token_budget_trace.sh](scripts/token_budget_trace.sh). Flash attention grows 0.65 -> 1.46 ms and the QSA indexer 2.45 -> 2.99 ms (it scores and selects over all pooled keys); the rest stays flat. (Corrected 2026-11-10: the first text said the QSA indexer stays flat.)
- 27B 1-GPU@0: ROCm0 (MI60) alone, no `-sm tensor`, `-ub 2048` fits. There is no AllReduce, so the script ends each segment before the residual norm where TP would reduce (AllReduce rows empty). On one GPU the residual add is fused into `rms_norm_f32` without the q8_1 producer, so quantize is a separate row (1.11 ms). Mat-vec is 37.4 ms (85%, about 767 GB/s); TP halves it to 19.3 / 19.5 per GPU (0.6-0.8 ms above half). The other kernels do not shrink under TP: 6.2 ms on one GPU against 6.4 ms per GPU under TP (recurrent 3.46 vs 3.23, norm + quantize 1.73 vs 2.02), because they are latency-bound. TP adds 1.6 ms of AllReduce exchange and about 0.7 ms more token-start gap (mostly profiler). Steady profiled 28.58 ms against 22.0 ms for a perfect halving; unprofiled 1.60x (42.70 -> 26.70 ms). (Clarified 2026-11-10: the first text said TP "adds about 3.5 ms" and did not say that the non-mat-vec work does not halve.)
- The critical path has no "wait" rows by construction: the GPU that arrives last at an AllReduce does not wait. The waits are in the per-GPU view below.
- Times are PST (the host clock); the trace file time is the end of the profiled run.

:::{dropdown} Per-GPU view (steady tokens, ms per token)

<!-- tb:tg-gpu -->
| Datapoint | GPU | Mat-vec (calls) | flash_attn* | AR exchange | AR wait | Token-start gap | Other idle | Busy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 27B@0 | ROCm0 | 19.30 (433) | 0.36 | 1.40 | 1.89 | 0.25 | 0.03 | 28.30 |
| 27B@0 | ROCm1 | 19.51 (433) | 0.45 | 1.57 | 0.00 | 1.02 | 0.05 | 27.50 |
| 27B@32K | ROCm0 | 20.70 (433) | 3.08 | 1.43 | 1.85 | 0.32 | 0.06 | 32.49 |
| 27B@32K | ROCm1 | 19.57 (433) | 3.34 | 1.65 | 0.84 | 1.20 | 0.12 | 31.56 |
| FN@8K | ROCm0 | 7.86 (592) | 0.67 | 0.87 | 3.22 | 0.90 | 0.02 | 24.71 |
| FN@8K | ROCm1 | 8.25 (592) | 0.65 | 0.99 | 0.00 | 2.09 | 0.02 | 23.51 |
| G4@0 | ROCm0 | 22.95 (360) | 1.54 | 1.68 | 2.20 | 0.25 | 0.03 | 32.56 |
| G4@0 | ROCm1 | 23.26 (360) | 1.96 | 1.86 | 0.00 | 1.07 | 0.05 | 31.72 |
| FN@32K | ROCm0 | 7.89 (592) | 1.45 | 0.89 | 3.25 | 1.22 | 0.02 | 25.99 |
| FN@32K | ROCm1 | 8.28 (592) | 1.46 | 1.00 | 0.01 | 2.50 | 0.03 | 24.70 |
| 27B 1-GPU@0 | ROCm0 | 37.40 (433) | 0.38 | - | - | 0.35 | 0.05 | 43.59 |
<!-- /tb:tg-gpu -->

"Busy" includes AR wait (the AR kernel spins). ROCm1 starts each token about 0.8-1.3 ms after ROCm0 under the profiler (one slow `hipGraphLaunch` per lane, see [P3](old-ignore/2026-11-10/exp-p3/README.md)), so ROCm0 waits at the first AllReduces of the token. Without the profiler this lag is about 20 us, so most of the token-start gap is profiler cost. The rest of ROCm0's AR wait is ROCm1 being slower inside the token: ROCm1 busy minus AR wait is above ROCm0's by 1.09 ms (27B@0), 2.02 ms (FN@8K), 1.95 ms (FN@32K) and 1.36 ms (G4@0), almost all in small kernels; P3 counts this part as real. At 27B@32K the two ranks are even (0.08 ms) because ROCm0's mat-vec is 1.1 ms slower in that run. (Corrected 2026-11-10: the first text said most of the ROCm0 AR wait is profiler cost; the start lag is about 41% of it on 27B@0 and 37% on G4@0 and FN@8K.) Inside the token there is almost no idle: the GPUs run back to back between AllReduces. The full per-GPU tables (busy + idle per step) are in [old-ignore/2026-11-10/token-budget/](old-ignore/2026-11-10/token-budget/) (`*.txt`, `*.json`).

Sanity checks against today's records: 27B@0 mat-vec 19.30 / 19.51 ms (Q3: 19.33 / 19.54); 27B@32K flash attention 3.08 / 3.34 ms (profile summary: 3.08 / 3.33); Flash Next `topk_moe_cuda` 1.35 ms (48 calls x 28 us; the earlier summary showed 1.26 because it divided 60 tokens of calls by 64).

:::

### Per-model notes (decode)

- Qwen3.8-27B Q8_0 (`qwen35`): 64 layers, 48 gated delta net (GDN) + 16 full attention (every 4th), dense SwiGLU FFN with gate/up fused into one mat-vec and the activation fused into it (so "activation" is empty). 2 AllReduces per layer (128). Mat-vec is 68% of the token (19.3 ms); the post-AR kernel is the fused `add_rms_norm_mul` with the q8_1 epilogue ([Q1](old-ignore/2026-11-10/exp-q1/README.md)). The GDN "recurrent" row is about 13 small kernels per layer (state gather, conv, l2 norms, gates, GDN, gated norm): 1595 kernels minus 433 mat-vec, 128 AR, 130 post-AR norms and about 131 quantize leave about 775, of which attention layers take about 110. (Corrected 2026-11-10: the first text said about 25, which does not fit the kernel count.) It takes 3.23 ms per GPU under TP against 3.46 ms on one GPU, so it does not halve. At 32K flash attention grows 0.45 -> 3.34 ms; the critical mat-vec rows also rose 1.1 ms (ROCm0 mat-vec 19.30 -> 20.70 ms, ROCm1 flat 19.51 -> 19.57; different builds and one run each, cause not known).
- Qwen3.8 Flash Next UD-IQ3_XXS (`qwen4exp`): 48 layers, 36 GDN + 12 attention with the QSA indexer, MoE with top-10 routed experts plus a shared expert, 4 hyper-connection streams in place of residual + norm, and one per-layer n-gram embedding layer. 2 AllReduces per layer (96). 2908 kernels per token at about 8 us each: mat-vec is only 31% (7.9 ms). The large rows are fixed-cost glue: hyper-connection mix 3.9 ms, QSA indexer 2.5 ms (lightning indexer, radix top-k, top-k mask; on HIP a one-row top-k over more than 1024 pooled keys takes the radix path, 12 launches per call in `top-k.cu`: init, 4 x histogram + select, count, offsets, gather; corrected 2026-11-10 from "10-kernel"), router + top-k 1.8 ms, q8_1 quantize 1.5 ms. The final hyper-connection mix runs unfused before the LM head (output norm row).
- Gemma 4 31B-it Q8_0 (`gemma4`): 60 layers, 50 sliding-window (head 256) + 10 global (head 512) attention, dense GELU FFN (51 layers fused gate/up + GELU, 9 with separate gate and up), attention and FFN post-norms, final logit softcap. 2 AllReduces per layer (120). It ran under `-sm tensor` without problems. Mat-vec is 70% (23.0 ms): gate/up alone is 10.1 ms, the largest single row of the TP columns. Q/K norm and RoPE are fused (`rms_norm_mul_rope`). Global layers take 26.6 us per `flash_attn_tile<512>` against 20.4 us for SWA layers at depth 0.

### Speculative modes (per verify step)

No speculative traces exist. Measured: ms per step, tokens per step and acceptance from the llama-server 5-prompt workload ([spec_accept.sh](scripts/spec_accept.sh), 256 greedy tokens, about 2K-token and short prompts). The split inside a step is an estimate, marked `est`, except where marked `meas`; `derived` = measured step minus the estimates.

```
27B Q8_0, MTP2 (2026-11-10, build-exp-spec, exp-spec record, r1/r2)
  step 40.6 ms (meas, 40.81 / 40.34), 2.55 tokens/step (meas, server mean len), accept 0.777
  -> 15.9 ms/token (62.8 t/s; the server mean TG was 62.1, it includes per-request cost; corrected 2026-11-10) vs 26.7 plain
  +-----------------------------------------------+----------+
  | verify batch, 3 tokens (1 + 2 drafts), TP     | ~31  est |  plain 26.7 x 1.15-1.19 (3-token/1-token kernel time at 32K: 1.19, meas)
  | MTP draft decode x2 (1 dense layer + LM head) | ~4.5 est |  ~0.45 layer + 0.88 head + ~0.9 launch per draft
  | host: CPU sampling of 5 rows x 248K logits,   | ~5 derived |  backend sampling falls back to CPU under llama-server
  |   accept, GDN state rollback, batch build     |          |
  +-----------------------------------------------+----------+

27B Q8_0, DFlash5 (2026-11-10, build-exp-spec, exp-spec record, r1/r2)
  step 54.9 ms (meas, 55.03 / 54.79), 3.88 tokens/step (meas), accept 0.581
  -> 14.2 ms/token (70.7 t/s; the server mean TG was 69.3; corrected 2026-11-10)
  step grows with context: 48-49 ms on 22-98 token prompts, 60-67 ms on ~1.75K token prompts (meas, per prompt)
  +-----------------------------------------------+----------+
  | verify batch, 6 tokens, TP                    | ~35  est |  plain 26.7 x ~1.3 (6-token/1-token kernel time at 32K: 1.42, meas)
  | DFlash draft on ROCm0 (block 8, 5 taps, 2 GB) | ~20 derived |  draft forward + context catch-up + tap copies; context-dependent part ~12-18 (meas spread)
  |   + host sampling/accept                      |          |
  +-----------------------------------------------+----------+

Flash Next UD-IQ3_XXS, MTP3 (2026-11-06, 52bcba210, ctx 100352, -ts 0.52,0.48, Q8_0 sidecar on ROCm1)
  step 49.5 ms (meas), ~3.10 tokens/step (meas: 62.7 t/s x 49.5 ms), accept 0.721
  -> 15.9 ms/token (62.7 t/s) vs 22.9 plain @8K
  ROCm0 idle 22% of the step (meas, ~11 ms): drafts run on ROCm1 alone
  +-----------------------------------------------+----------+
  | verify batch, 4 tokens, TP, grouped MoE       | ~32  est |  plain 22.9 + routed experts for ~4x picks + attention at 100K
  | MTP draft decode x3 on ROCm1 (1 MoE layer +   | ~7.5 est |  ~2.5 per draft
  |   HC + LM head)                               |          |
  | host: sampling, accept, state rollback        | ~10 derived |
  +-----------------------------------------------+----------+
```

To replace the estimates, trace one server run per mode (as in the [N2 host-gap record](old-ignore/2026-10-29/tg-host-gap/diagnosis.md)) and split by graph: the verify graph has the batch-N mat-vecs (`mul_mat_vec_repacked_nc`), the draft graph has one layer plus the LM head.

(tb-pp)=

## Prefill (PP)

### Main table (critical path, ms per 1K prompt tokens, profiled)

Each cell is `busy + idle-before (share of the column total)` in ms per 1K prompt tokens (sum over the measured ubatches x 1000 / n_prompt). The column total equals the profiled llama-bench time of the measured run. In PP, kernels and copies on one GPU run on several streams at once, so time is attributed by union: at each moment the GPU's time goes to one step, compute kernels first, then AllReduce kernels and copies, then other copies; time with nothing active is idle, charged to the step that starts after it.

<!-- tb:pp-main -->
| Row | 27B@0 2026-11-10 | 27B@32K 2026-11-10 | FN@0 2026-11-10 | G4@0 2026-11-10 | FN@32K 2026-11-10 |
| --- | --- | --- | --- | --- | --- |
| Measured | 2026-11-10 19:28 PST | 2026-11-10 19:33 PST | 2026-11-10 19:29 PST | 2026-11-10 19:30 PST | 2026-11-10 19:36 PST |
| Source commit | 0f7ce1b04 | 0f7ce1b04 | 0f7ce1b04 | 0f7ce1b04 | 0f7ce1b04 |
| Build (sha256 llama-bench / libggml-hip.so) | `build-exp-spec` (`e4e83771fc55 / 1ac1232870d5`) | `build-exp-spec` (`e4e83771fc55 / 1ac1232870d5`) | `build-exp-spec` (`e4e83771fc55 / 1ac1232870d5`) | `build-exp-spec` (`e4e83771fc55 / 1ac1232870d5`) | `build-exp-spec` (`e4e83771fc55 / 1ac1232870d5`) |
| Model, quant, file revision | Qwen3.8-27B Q8_0, rev 4ca720788d1e | Qwen3.8-27B Q8_0, rev 4ca720788d1e | Qwen3.8 Flash Next UD-IQ3_XXS rev d96201194f10 + Q8_0 n-gram table | Gemma 4 31B-it Q8_0, rev c1ac76e99d55 | Qwen3.8 Flash Next UD-IQ3_XXS rev d96201194f10 + Q8_0 n-gram table |
| Settings | -ts 1/1 -b 2048 -ub 2048 -fa on, depth 0, pp2048 | -ts 1/1 -b 2048 -ub 2048 -fa on, depth 32768, pp2048 (one ubatch at depth 32K) | -ts 1/1 -b 1024 -ub 512 -fa on, depth 0, pp2048 | -ts 1/1 -b 2048 -ub 2048 -fa on, depth 0, pp2048 | -ts 1/1 -b 1024 -ub 512 -fa on, depth 32768, pp2048 (four ubatches at depth 32K-34K) |
| Unprofiled PP t/s (ms per 1K tokens) | 705.3 +- 1.4 (1417.8) | 457.8 +- 4.2 (2184.4) | 591.4 +- 11.9 (1690.9) | 602.5 +- 1.4 (1659.9) | 441.6 +- 4.3 (2264.6) |
| Profiled ms per 1K: all; traced ubatches; llama-bench | 1421.4; 1406.5; 1421.4 | 2280.9; 2194.2; 2280.9 | 1660.3; 1608.9; 1660.3 | 1668.1; 1646.5; 1668.1 | 2315.7; 2256.9; 2315.7 |
| Measured ubatches; kernels / AllReduces per ubatch | 1; 3510 / 128 | 1; 3414 / 128 | 4; 4314 / 96 | 1; 2963 / 120 | 4; 4454 / 96 |
| Run start: bench timer start to the first compute kernel (host prep, input upload) | 0.0 + 14.9 (1.1%) | 0.0 + 86.8 (3.8%) | 0.0 + 51.4 (3.1%) | 0.0 + 21.6 (1.3%) | 0.0 + 58.8 (2.5%) |
| Ubatch start: host gap before the first compute kernel | - | - | 0.0 + 114.1 (6.9%) | - | 0.0 + 132.2 (5.7%) |
| Ubatch start: input upload, embedding | - | - | 1.4 + 20.7 (1.3%) | 0.1 + 0.0 (0.0%) | 1.4 + 20.1 (0.9%) |
| Residual add + norm | 16.6 + 1.2 (1.2%) | 17.2 + 1.2 (0.8%) | - | 24.6 + 1.1 (1.5%) | - |
| Hyper-connection pre/post with its small GEMMs (Flash Next) | - | - | 253.0 + 3.4 (15.4%) | - | 262.1 + 3.4 (11.5%) |
| Per-layer n-gram embedding (Flash Next) | - | - | 8.7 + 0.0 (0.5%) | - | 8.8 + 0.0 (0.4%) |
| Mixer GEMM: attention input projections (q+gate, k, v) | 53.5 + 0.0 (3.8%) | 51.4 + 0.0 (2.3%) | 35.5 + 0.0 (2.1%) | 241.1 + 0.0 (14.5%) | 35.8 + 0.0 (1.5%) |
| Mixer GEMM: GDN input projections (qkv, z, beta/alpha) | 183.5 + 0.0 (12.9%) | 193.0 + 0.0 (8.5%) | 124.1 + 0.0 (7.5%) | - | 126.2 + 0.0 (5.5%) |
| Attention: q/k norm, rope, KV write, gate | 7.2 + 0.0 (0.5%) | 7.9 + 0.0 (0.3%) | 6.4 + 0.0 (0.4%) | 40.6 + 0.0 (2.4%) | 6.6 + 0.0 (0.3%) |
| Attention: flash attention (batched) | 27.1 + 0.0 (1.9%) | 725.5 + 0.0 (31.8%) | 20.1 + 0.0 (1.2%) | 204.3 + 0.0 (12.2%) | 532.9 + 0.0 (23.0%) |
| Attention: QSA indexer and key pooling (Flash Next) | - | - | 22.3 + 0.0 (1.3%) | - | 73.3 + 0.1 (3.2%) |
| Recurrent: chunked gated delta net | 111.7 + 0.0 (7.9%) | 118.8 + 0.0 (5.2%) | 79.9 + 0.0 (4.8%) | - | 80.4 + 0.0 (3.5%) |
| Recurrent: conv, l2 norms, gates, state copies | 30.3 + 0.0 (2.1%) | 30.8 + 0.0 (1.4%) | 21.5 + 0.1 (1.3%) | - | 21.3 + 0.2 (0.9%) |
| Mixer GEMM: output projection (slabs) | 105.6 + 0.0 (7.4%) | 115.0 + 0.0 (5.0%) | 75.7 + 0.0 (4.6%) | 136.7 + 0.0 (8.2%) | 77.1 + 0.0 (3.3%) |
| AllReduce after mixer: exposed | 41.3 + 3.0 (3.1%) | 62.5 + 4.0 (2.9%) | 24.8 + 9.0 (2.0%) | 29.2 + 3.4 (2.0%) | 27.1 + 8.3 (1.5%) |
| FFN dense GEMM: gate/up | 476.1 + 0.0 (33.5%) | 496.8 + 0.0 (21.8%) | - | 580.0 + 0.0 (34.8%) | - |
| FFN dense: activation | 10.3 + 0.0 (0.7%) | 10.5 + 0.0 (0.5%) | - | 13.7 + 0.0 (0.8%) | - |
| FFN dense GEMM: down (slabs) | 269.3 + 0.0 (18.9%) | 272.6 + 0.0 (12.0%) | - | 307.9 + 0.0 (18.5%) | - |
| MoE: router + top-k | - | - | 22.0 + 0.0 (1.3%) | - | 25.4 + 0.0 (1.1%) |
| MoE: ids helper, tile list (repack_tile_off) | - | - | 30.1 + 0.0 (1.8%) | - | 31.6 + 0.0 (1.4%) |
| MoE: routed expert GEMM + activation | - | - | 559.8 + 0.0 (33.7%) | - | 592.2 + 0.0 (25.6%) |
| MoE: shared expert + gate | - | - | 49.6 + 0.1 (3.0%) | - | 52.1 + 0.0 (2.2%) |
| MoE: weighted reduce + combine | - | - | 20.8 + 0.0 (1.3%) | - | 21.6 + 0.0 (0.9%) |
| AllReduce after FFN: exposed | 29.4 + 3.5 (2.3%) | 34.8 + 2.7 (1.6%) | 40.9 + 10.0 (3.1%) | 29.0 + 3.4 (1.9%) | 41.2 + 9.7 (2.2%) |
| Output: last-token rows, final norm (+ softcap) | - | - | 0.3 + 0.0 (0.0%) | - | 0.3 + 0.0 (0.0%) |
| Output: LM head (last token) | 0.4 + 0.0 (0.0%) | 0.4 + 0.0 (0.0%) | 0.5 + 0.0 (0.0%) | 0.5 + 0.0 (0.0%) | 0.5 + 0.0 (0.0%) |
| q8_1 quantize of GEMM inputs | 35.6 + 0.7 (2.6%) | 37.1 + 1.1 (1.7%) | 36.2 + 0.7 (2.2%) | 29.8 + 1.1 (1.8%) | 38.2 + 0.8 (1.7%) |
| Copies H2D/D2H outside AllReduce, exposed | 0.0 + 0.0 (0.0%) | - | 1.3 + 1.3 (0.2%) | 0.0 + 0.0 (0.0%) | 3.1 + 9.2 (0.5%) |
| Glue: crops, copies, other | - | - | 0.0 + 0.5 (0.0%) | - | 0.0 + 0.5 (0.0%) |
| Critical-path rounding (GPU switch at AllReduce ends) | 0.0 + 0.2 (0.0%) | 0.0 + 10.8 (0.5%) | 0.0 + 14.1 (0.8%) | 0.0 + 0.2 (0.0%) | 0.0 + 13.1 (0.6%) |
| AllReduce overlapped with compute (info, not in total) | 276.0 | 275.7 | 84.3 | 287.4 | 82.4 |
| Total, profiled: ms per 1K; ms/token; t/s | 1421.4; 1.421; 703.5 | 2280.9; 2.281; 438.4 | 1660.3; 1.660; 602.3 | 1668.1; 1.668; 599.5 | 2315.7; 2.316; 431.8 |
| Total, scaled to unprofiled: ms per 1K; ms/token; t/s | 1417.8; 1.418; 705.3 | 2184.4; 2.184; 457.8 | 1690.9; 1.691; 591.4 | 1659.9; 1.660; 602.5 | 2264.6; 2.265; 441.6 |
<!-- /tb:pp-main -->

:::{dropdown} Per-GPU view (ms per 1K prompt tokens)

<!-- tb:pp-gpu -->
| Datapoint | GPU | GEMM ms (calls/ubatch) | flash_attn* | AR exposed | AR overlapped | Start gaps (ubatch + run) | Other idle | Busy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 27B@0 | ROCm0 | 899.6 (877) | 23.9 | 200.6 | 157.1 | 15.0 | 94.3 | 1312.2 |
| 27B@0 | ROCm1 | 1088.4 (877) | 27.1 | 70.9 | 276.0 | 14.9 | 8.4 | 1398.1 |
| 27B@32K | ROCm0 | 1043.8 (877) | 716.6 | 148.9 | 207.4 | 87.2 | 71.0 | 2122.8 |
| 27B@32K | ROCm1 | 1111.1 (877) | 713.7 | 137.9 | 251.2 | 86.8 | 18.1 | 2176.1 |
| FN@0 | ROCm0 | 915.2 (1135) | 18.5 | 119.3 | 36.7 | 165.3 | 181.6 | 1313.4 |
| FN@0 | ROCm1 | 997.5 (1135) | 20.1 | 83.3 | 66.4 | 165.6 | 104.2 | 1390.5 |
| G4@0 | ROCm0 | 1083.9 (767) | 170.3 | 178.5 | 195.9 | 21.6 | 116.9 | 1529.6 |
| G4@0 | ROCm1 | 1266.1 (767) | 204.3 | 58.3 | 287.4 | 21.6 | 9.0 | 1637.5 |
| FN@32K | ROCm0 | 987.9 (1135) | 437.9 | 114.5 | 40.8 | 190.5 | 256.5 | 1868.7 |
| FN@32K | ROCm1 | 1020.0 (1135) | 532.9 | 87.6 | 62.5 | 191.0 | 133.1 | 1991.7 |
<!-- /tb:pp-gpu -->

"Busy" includes exposed AllReduce time, which on the early GPU is mostly waiting for the other GPU's copies. "AR overlapped" is AllReduce work (bf16 converts, D2H and H2D copies) hidden under compute; it is not in Busy. The raw script outputs are `pp-*.txt` and `pp-*.json` in [old-ignore/2026-11-10/token-budget/](old-ignore/2026-11-10/token-budget/).

:::

(tb-pp-rows)=

### What the PP rows mean

- Ubatch cut: llama-bench runs a warm-up prompt (same size), then per rep the depth prefill (`-d`, plus a state save) and the measured `-p`. Only the last `ceil(n_prompt / ub)` ubatches are used. Ubatch boundaries are the largest host gaps (no kernel, no copy) between two AllReduce kernels; the script prints the smallest boundary gap against the largest other gap as a check.
- Run start: from the bench timer start (the end of the measured run minus the bench time) to the first compute kernel. It holds the host work before the first graph (batch split, graph build, KQ mask fill) and the input upload of the first ubatch.
- Ubatch start: the host gap and input copies before the first compute kernel of every later ubatch (Flash Next only, here: 4 ubatches per run).
- AllReduce exposed: AllReduce work (markers, bf16 converts, copies) with no compute kernel on that GPU at the same time. On the critical path this is the real exchange tail; on the other GPU it is mostly waiting. AllReduce overlapped: the same work under compute (info row, not in the total).
- GEMM rows include all slabs. The slab GEMMs of the output and down projections are the GEMMs after the last other compute kernel of an AllReduce segment.
- Critical path: per AllReduce segment, the GPU whose last compute kernel before the AllReduce ended later (the one the other waits for); rounding goes to the last row.
- Depth: a PP2048 column from empty context averages attention over depth 0..2048. The @32K columns are the same prompt after a 32K-token prefill, so attention reads 32K-34K keys per query.

### Per-model notes (prefill)

- 27B@0: 705 t/s unprofiled. GEMM is 76% of the critical path (`mmq_gemm_repacked`, 877 calls per ubatch): FFN gate/up 33.5%, down 18.9%, GDN input projections 12.9%, output projections 7.4%, attention input 3.8%. Chunked GDN is 7.9%. The critical GPU is ROCm1 (MI50) in almost every segment: its GEMM time is 1088 ms per 1K against 900 on ROCm0 (+21%), so ROCm0 spends about 290 ms per 1K exposed in AllReduce or idle waiting. AllReduce is mostly hidden: 77 ms per 1K exposed on the critical path (5.4%) and 276 overlapped. Host time is 15 ms per 1K (run start, 1.1%).
- 27B@32K: 458 t/s unprofiled (one ubatch at depth 32K). Flash attention grows from 27 to 726 ms per 1K (1.9% -> 31.8%); GEMM rows are about flat. Run start grows to 87 ms per 1K (178 ms per ubatch): host mask build and upload for 34K keys.
- FN@0: 591 t/s unprofiled (602 profiled; noise between runs). Routed expert GEMM + activation is the largest row (34%): `mul_mat_q` gate/up is 365-391 ms per 1K (94 calls per ubatch, about 2 ms each) and the repacked down `mmq_gemm_repacked_w32` 75-78. Hyper-connection pre/post is 15%. Host time is large: about 78 ms between ubatches (ubatch start, 6.9%) plus run start (3.1%) and input upload (1.3%), about 11% in total. `repack_tile_off` is 5.7 ms per 1K (0.34%), the ids helper 24.5 ms per 1K. AllReduce exposed is 85 ms per 1K on the critical path (5.1%). The two GPUs launch `mul_mat_q` with different tile grids per layer (192x3328 on one, 128x3328 on the other, alternating), so each layer has a slow GPU; the critical routed row (560) is above either GPU's own sum (452 / 492).
- G4@0: 602 t/s unprofiled. GEMM is 76% (FFN gate/up 34.8%, down 18.5%, attention input 14.5%, output 8.2%); flash attention is already 12.2% at depth 0 (head 256 and 512, 60 calls per ubatch at about 6 ms on ROCm1). ROCm1 is again the slow GPU (GEMM 1266 against 1084 ms per 1K).
- FN@32K: 442 t/s unprofiled (four ubatches at depth 32K-34K). Flash attention grows 20 -> 533 ms per 1K (23%) and the QSA indexer 22 -> 73; MoE, GDN and GEMM rows stay flat. The host gaps grow a little (ubatch start 114 -> 132 ms per 1K).

(tb-method)=

## Method and limits

:::{dropdown} Method details (data, token and ubatch cuts, idle, critical path, scaling)

- Data: rocprofv3 `--kernel-trace` of `llama-bench -p 0 -n N -d DEPTH -r 1` on the merged tree `0f7ce1b04` (upstream `aa75ff323`), 2026-11-10, TheRock ROCm, two gfx906 (ROCm0 = MI60, 64 CU, bus 03:00; ROCm1 = MI50, 60 CU). `Agent 1` is ROCm0.
- Tokens: a token runs from the end of one LM head mat-vec (largest mat-vec grid) to the end of the next. The last N heads are the decode tokens; N-1 intervals are used. Intervals above 1.2x the median are warm-up (eager launches before HIP graph capture, 1-3 per run); they are kept out of the step rows and added as one row of excess time, so the column sums to the all-token mean.
- Steps: kernels are split at the AllReduce kernels. A segment with `flash_attn` is attention, with `gated_delta_net` is recurrent, the last one is output, the rest are FFN. Inside a segment, the last mat-vec before the AllReduce is the output (or down) projection, other mat-vecs are input (or gate/up) projections; kernel names, grid sizes and order give the rest (rules in `classify_segment`). q8_1 quantize kernels have their own row in all models.
- Idle: a GPU gap is charged to the step of the kernel that follows it. Gaps before the first compute kernel of a token are "token start". The profiler inflates gaps: one `hipGraphLaunch` takes about 1 ms under rocprofv3 against about 20 us without it ([P3](old-ignore/2026-11-10/exp-p3/README.md)), so token-start gaps are mostly profiler cost; about 40% of the ROCm0 AllReduce wait is this start lag and the rest is ROCm1's slower small kernels (see the per-GPU view; corrected 2026-11-10).
- AllReduce wait: the part of a GPU's AR kernel before the other GPU started the same AR. Exchange is the rest.
- Critical path: for each AllReduce segment, the GPU that started that AR later; for the output segment, the GPU whose LM head ended later. The few us left when switching GPUs go to the rounding row.
- Scaled total: profiled total x (unprofiled ms / profiled ms). The profiler slows TG by 7-24%, mostly in small kernels and host gaps, so scaled step rows are rough.
- llama-bench does not sample: host sampling is not in any plain column. In llama-server and llama-cli it adds per-token host time (see the speculative boxes).
- Not attributable: nothing large. The "glue" row (crops before the last layer, stray copies) is 0.01 ms in every column.
- PP: `rocprofv3 --kernel-trace --memory-copy-trace` of `llama-bench -p 2048 -n 0 -d DEPTH -r 1`, same tree and GPUs. Rules are in `pp_main` and `classify_pp` and summarized in [What the PP rows mean](#tb-pp-rows). Steps are split at the AllReduce marker kernels on the compute stream; inside a segment, the GEMMs after the last other compute kernel are the output (or down) projection slabs, other GEMMs are input (or gate/up) projections. The profiler barely changes PP (profiled within 2-4% of unprofiled, both directions), so scaled PP rows are close to real. Each column total equals the profiled llama-bench time of the measured run by construction (the run start row closes the gap).

:::

(tb-read)=

## How to read and use it

- Busy is kernel time; idle-before is time the GPU had no kernel before that step started. A large idle-before means the step waited for the host or for the other GPU.
- The two GPUs meet at every AllReduce, so the slower rank sets the pace. The main table follows that rank per segment. ROCm0 AR time is mostly waiting for ROCm1 (per-GPU view).
- Only critical-path savings speed up TG. A busy saving on the rank that already waits only turns into more AR wait. A saving on one rank counts fully only if that rank is late in that segment.
- Back to speed: `TG t/s = 1000 / ms_per_token`. Example, 27B@0 unprofiled: 26.70 ms = 37.45 t/s; saving 1.0 ms on the critical path gives 1000 / 25.70 = 38.9 t/s. Use unprofiled (scaled) ms for this, not profiled.
- Speculative: `t/s = tokens_per_step * 1000 / ms_per_step`, with `tokens_per_step = 1 + accepted drafts per step` (about `1 + n_draft * acceptance`). Example, 27B MTP2: 2.55 * 1000 / 40.6 = 62.8 t/s.
- Context length: only the attention rows (flash attention, and the QSA indexer on Flash Next) should grow with depth (KV read); GDN/recurrent rows and AllReduces stay flat. The 27B@32K run also has 1.1 ms more critical mat-vec time than 27B@0 (ROCm0 only, different build), so the whole-token slope below may overstate the depth cost by up to about 0.03 ms per 1K (note added 2026-11-10). For 27B, flash attention is `0.45 + 0.088 * n_ctx_K` ms (profiled, n_ctx_K = context in 1K tokens, from the @0 and @32K columns). The whole unprofiled token fits `26.7 + 0.099 * n_ctx_K` ms, which gives about 36.3 t/s at 8K, 35.3 at 16K and 30.3 at 64K. Limits: two points only; it assumes the F16 KV cache and the same flash-attention kernel choice at all depths (kernel switch points and other KV types change the slope); confirm with a measured column before you quote a number.
- Prompt processing: the decode columns do not convert to PP; use the [Prefill (PP)](#tb-pp) table. PP is GEMM-bound per ubatch (other kernels than decode), and the AllReduce cost per token is much smaller and mostly hidden. (Updated 2026-11-10: the first text said this doc has no PP data.)
- PP back to speed: `PP t/s = 1,000,000 / ms_per_1K`. Example, 27B@0 unprofiled: 1417.8 ms per 1K = 705 t/s; saving 100 ms per 1K on the critical path gives 1e6 / 1317.8 = 759 t/s. The ms/token cell (ms per 1K / 1000, 1.42 ms for 27B) is the prompt cost per token; compare it with the TG ms/token to see how many prompt tokens cost as much as one decoded token (27B: about 19).
- PP and depth: flash attention (and the QSA indexer on Flash Next) grows with the depth already in the cache; run start also grows (host KQ mask for all keys). GEMM, GDN, MoE and AllReduce rows stay flat. A long prompt from empty context averages the depths it passes through; to estimate, interpolate the attention row between the @0 and @32K columns at the mean depth.
- PP critical path: as in TG, only critical-path savings count. ROCm1 (MI50) is the late GPU in most PP segments (slower GEMM), so a GEMM speed-up on ROCm0 alone turns into more AllReduce wait.

(tb-update)=

## How to update (and how this doc is made)

- Tools: `rocprofv3 --kernel-trace` on llama-bench (one run, `-r 1`; PP also `--memory-copy-trace`) for the split, plus an unprofiled llama-bench `-r 3` of the same test for the real t/s. [token_budget_trace.sh](scripts/token_budget_trace.sh) runs both (`N_GEN` for TG, `pN_PROMPT` such as `p2048` for PP); [token_budget.py](scripts/token_budget.py) (stdlib only, `python3 -I`) makes the split (`--phase pp` for PP), prints it and appends it to the CSVs.
- Source of truth: [token-budget-columns.csv](token-budget-columns.csv) (column_id, phase, model, quant, depth, ubatch, batch, n_tokens, n_gpus, provenance, plain and profiled totals, unit, scale, notes, record_link) and [token-budget-steps.csv](token-budget-steps.csv) (column_id, step_key, step_label, gpu = ROCm0 / ROCm1 / critical, busy_ms, idle_ms, scaled_ms on critical rows; `info_*` rows hold per-GPU kernel totals and call counts in busy_ms). Values are in the column's unit (ms/token for TG, ms per 1K prompt tokens for PP); `kernels_per_unit` and `ar_per_unit` are per token (TG) or per ubatch (PP).
- Rule: the CSVs are append-only. Never edit or delete a row; a new measurement is a new column_id (`<model>@<depth> <TG|PP> <date>`). `--append-csv` and `append-json` refuse an id that exists. A correction goes in a new column plus a dated note.
- `python3 -I vr/scripts/token_budget.py render` rewrites the tables between the `<!-- tb:tg-main -->`, `tb:tg-gpu`, `tb:pp-main` and `tb:pp-gpu` markers from the CSVs (columns in CSV order). Prose outside the markers is hand-written; keep it in step with the numbers.
- Provenance cells: Measured (trace file time), Source commit (tree that was built), Build (directory plus sha256 prefixes of `llama-bench` and `libggml-hip.so`, which holds the kernels), model and file revision, settings, unprofiled t/s, profiled totals, kernels and AllReduces per token or ubatch.
- Keep raw outputs (`*.txt`, `*.json` of the script; not the 5-210 MB traces) in a dated `old-ignore/<date>/token-budget/` folder; the CSV `record_link` points to the `.txt`.

:::{dropdown} Commands: one TG and one PP column end to end

One column end to end (build with the [catch-up flags](how-to-catchup.md#one-final-validation-pass); GPU work goes through the lock with a timeout, the script does both):

```sh
source /home/raistlin/amd-clean/env/therock.sh
B=build-ilp; O=/tmp/tb-27b-d0
GPU_LOCK=/tmp/gpu.lock vr/scripts/token_budget_trace.sh $O $B 64 0 -- \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline \
  -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa on -b 2048 -ub 2048 -t 20
python3 -I vr/scripts/token_budget.py $O/trace/t_kernel_trace.csv qwen35 64 \
  --plain $O/plain.json --prof $O/prof.json \
  --id "27B@0 YYYY-MM-DD" --commit $(git rev-parse --short HEAD) --build $B --bin $B/bin/llama-bench \
  --modeldesc "Qwen3.8-27B Q8_0, rev 4ca720788d1e" --settings "-ts 1/1 -b 2048 -ub 2048 -fa on, depth 0, tg64" \
  --json $O/budget.json > $O/budget.txt
sed -n '/^COLUMN/,$p' $O/budget.txt   # check the column, then add --append-csv vr --record <link to the kept .txt>
python3 -I vr/scripts/token_budget.py render   # regenerate the tables
```

A PP column (memory copies are traced too; Flash Next uses `-b 1024 -ub 512`):

```sh
GPU_LOCK=/tmp/gpu.lock vr/scripts/token_budget_trace.sh $O $B p2048 0 -- \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline \
  -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa on -b 2048 -ub 2048 -t 20
R=old-ignore/YYYY-MM-DD/token-budget
python3 -I vr/scripts/token_budget.py $O/trace/t_kernel_trace.csv qwen35 2048 --phase pp \
  --plain $O/plain.json --prof $O/prof.json \
  --id "27B@0 PP YYYY-MM-DD" --commit $(git rev-parse --short HEAD) --build $B --bin $B/bin/llama-bench \
  --modeldesc "Qwen3.8-27B Q8_0, rev 4ca720788d1e" --settings "-ts 1/1 -b 2048 -ub 2048 -fa on, depth 0, pp2048" \
  --json vr/$R/pp-27b-d0.json --append-csv vr --record $R/pp-27b-d0.txt > vr/$R/pp-27b-d0.txt
python3 -I vr/scripts/token_budget.py render
```

Check the `ubatch cut` line in the output: the smallest boundary gap should be far above the largest other gap (a WARNING is printed otherwise; pass `--ubatches-total` if llama-bench ran with `--no-warmup` or several reps).

:::

Model kinds: `qwen35` (27B), `qwen4exp` (Flash Next; use `-b 1024 -ub 512`, depth 8192, `-m` the local UD-IQ3_XXS shard 1), `gemma4` (Gemma 4 31B, `-m` the cached Q8_0 file). A new model needs its own rules in `classify_segment` (TG) and `classify_pp` (PP) if its kernels do not fit the generic ones; check that no large time lands in the glue row. Each run takes 1-3 minutes plus the depth prefill; runs need the user's OK like any benchmark ([PERFORMANCE.md](PERFORMANCE.md#how-to-test-and-what-the-status-words-mean)).
