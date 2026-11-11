# Token budget: where one decode token goes

One decode token, split into model steps, side by side for the target models on two gfx906 GPUs (`-sm tensor`, host-staged AllReduce). Each column comes from one `rocprofv3 --kernel-trace` of llama-bench TG plus an unprofiled llama-bench run, cut into steps by [token_budget.py](scripts/token_budget.py). Use it to see which step is worth attacking before you start an experiment. For model limits and ideas see [PERFORMANCE.md](PERFORMANCE.md#what-limits-each-model).

This is a growing record. Never edit the numbers of an old column; add a new column (see [How to update](#how-to-update-and-how-this-doc-is-made)).

## Main table (critical path, ms per token, profiled)

Each cell is `busy + idle-before (share of the profiled token)`. Busy is kernel time on the GPU that is on the critical path for that part of the token. Idle-before is GPU time with no kernel, charged to the step that waited to start after it. Step rows are means over steady tokens; the warm-up row adds the extra time of the first few tokens, so each column sums to the all-token profiled mean. The last row scales the total to the unprofiled llama-bench speed; scale step rows by the same factor if you need estimates, but the profiler inflates small kernels and host gaps more than mat-vecs (see [limits](#method-and-limits)).

| Row | 27B@0 2026-11-10 | 27B@32K 2026-11-10 | FN@8K 2026-11-10 | G4@0 2026-11-10 |
| --- | --- | --- | --- | --- |
| Measured | 2026-11-10 16:46 PST | 2026-11-10 17:50 PST | 2026-11-10 17:48 PST | 2026-11-10 18:28 PST |
| Source commit | 0f7ce1b04 | 0f7ce1b04 | 0f7ce1b04 | 0f7ce1b04 |
| Build (sha256 llama-bench / libggml-hip.so) | `build-exp-q3` (`246887e7bebf / d67316b02bbe`) | `build-exp-spec` (`e4e83771fc55 / 1ac1232870d5`) | `build-exp-spec` (`e4e83771fc55 / 1ac1232870d5`) | `build-exp-spec` (`e4e83771fc55 / 1ac1232870d5`) |
| Model, quant, file revision | Qwen3.8-27B Q8_0, rev 4ca720788d1e | Qwen3.8-27B Q8_0, rev 4ca720788d1e | Qwen3.8 Flash Next UD-IQ3_XXS rev d96201194f10 + Q8_0 n-gram table | Gemma 4 31B-it Q8_0, rev c1ac76e99d55 |
| Settings | -ts 1/1 -b 2048 -ub 2048 -fa on, depth 0, tg64 | -ts 1/1 -b 2048 -ub 2048 -fa on, depth 32768, tg32 | -ts 1/1 -b 1024 -ub 512 -fa on, depth 8192, tg64 | -ts 1/1 -b 2048 -ub 2048 -fa on, depth 0, tg64 |
| Unprofiled TG t/s (ms/token) | 37.45 +- 0.21 (26.70) (TG128 r3, exp-q3 record, same build) | 33.48 +- 0.97 (29.87) | 43.68 +- 0.60 (22.90) | 32.68 +- 0.15 (30.60) |
| Profiled ms/token: all tokens; steady; llama-bench | 29.44; 28.58; - | 34.29; 32.88; 35.12 | 27.18; 25.63; 28.39 | 33.08; 32.84; 33.09 |
| Kernels / AllReduces per token | 1595 / 128 | 1594 / 128 | 2908 / 96 | 1463 / 120 |
| Token start: host gap, no kernel | 0.00 + 1.02 (3%) | 0.00 + 1.20 (3%) | 0.00 + 2.09 (8%) | 0.00 + 1.07 (3%) |
| Token start: input copies, embedding | 0.04 + 0.00 (0%) | 0.03 + 0.00 (0%) | 0.08 + 0.00 (0%) | 0.06 + 0.00 (0%) |
| Warm-up tokens: excess over steady tokens, spread over all | 0.00 + 0.86 (3%) | 0.00 + 1.41 (4%) | 0.00 + 1.55 (6%) | 0.00 + 0.24 (1%) |
| Residual add + norm (Flash Next: hyper-connection combine + mix) | 1.39 + 0.00 (5%) | 1.47 + 0.00 (4%) | 3.88 + 0.00 (14%) | 1.66 + 0.00 (5%) |
| Per-layer n-gram embedding (Flash Next) | - | - | 0.25 + 0.00 (1%) | - |
| Mixer: input projections (mat-vec) | 4.60 + 0.00 (16%) | 4.74 + 0.00 (14%) | 2.69 + 0.00 (10%) | 4.51 + 0.00 (14%) |
| Attention: q/k norm, rope, KV write, gate | 0.66 + 0.00 (2%) | 0.67 + 0.00 (2%) | 0.50 + 0.00 (2%) | 1.67 + 0.00 (5%) |
| Attention: flash attention | 0.45 + 0.00 (2%) | 3.34 + 0.00 (10%) | 0.65 + 0.00 (2%) | 1.96 + 0.00 (6%) |
| Attention: QSA indexer (Flash Next) | - | - | 2.45 + 0.00 (9%) | - |
| Recurrent: conv, gated delta net, gates, state | 3.23 + 0.00 (11%) | 3.24 + 0.00 (9%) | 2.32 + 0.00 (9%) | - |
| Mixer: output projection (mat-vec) | 1.57 + 0.00 (5%) | 1.62 + 0.00 (5%) | 0.76 + 0.00 (3%) | 2.40 + 0.00 (7%) |
| AllReduce after mixer: exchange | 0.83 + 0.03 (3%) | 0.84 + 0.07 (3%) | 0.50 + 0.01 (2%) | 0.69 + 0.02 (2%) |
| AllReduce after mixer: wait for other GPU | - | - | - | - |
| FFN dense: gate/up mat-vec | 8.19 + 0.00 (28%) | 8.78 + 0.00 (26%) | - | 10.11 + 0.00 (31%) |
| FFN dense: activation | - | - | - | 0.04 + 0.00 (0%) |
| FFN dense: down mat-vec | 4.28 + 0.00 (15%) | 4.57 + 0.00 (13%) | - | 5.24 + 0.00 (16%) |
| MoE: router + top-k | - | - | 1.82 + 0.00 (7%) | - |
| MoE: routed experts | - | - | 2.58 + 0.00 (9%) | - |
| MoE: shared expert + gate | - | - | 1.41 + 0.00 (5%) | - |
| MoE: weighted reduce + combine | - | - | 1.03 + 0.00 (4%) | - |
| AllReduce after FFN: exchange | 0.73 + 0.02 (3%) | 0.72 + 0.05 (2%) | 0.49 + 0.01 (2%) | 1.16 + 0.02 (4%) |
| AllReduce after FFN: wait for other GPU | - | - | - | - |
| Output: final norm (+ softcap) | 0.01 + 0.00 (0%) | 0.01 + 0.00 (0%) | 0.08 + 0.00 (0%) | 0.03 + 0.00 (0%) |
| Output: LM head mat-vec | 0.88 + 0.00 (3%) | 0.95 + 0.00 (3%) | 0.52 + 0.00 (2%) | 1.00 + 0.00 (3%) |
| q8_1 quantize of mat-vec inputs | 0.63 + 0.00 (2%) | 0.62 + 0.00 (2%) | 1.50 + 0.00 (6%) | 1.18 + 0.00 (4%) |
| Glue: copies, crops, other | 0.01 + 0.00 (0%) | 0.01 + 0.00 (0%) | 0.01 + 0.00 (0%) | 0.01 + 0.00 (0%) |
| Critical-path rounding (GPU switch at AllReduce ends) | - | - | - | - |
| Total, profiled | 29.44 | 34.29 | 27.18 | 33.08 |
| Total, scaled to unprofiled | 26.70 | 29.87 | 22.90 | 30.60 |

Column notes:

- 27B@0: the trace is the Q3 trace ([exp-q3](old-ignore/2026-11-10/exp-q3/README.md)); no unprofiled TG64 JSON was kept, so the unprofiled figure is the Q3 TG128 r3 result of the same build (`GGML_RP_MMV_ROWS=8`). Its profiled llama-bench figure was 33.84 t/s (29.55 ms) in `trace-bench.txt`.
- 27B@32K and FN@8K: traces from the P3/Q1 follow-up profile runs of `build-exp-spec`; G4@0 was traced for this record with [token_budget_trace.sh](scripts/token_budget_trace.sh).
- The critical path has no "wait" rows by construction: the GPU that arrives last at an AllReduce does not wait. The waits are in the per-GPU view below.
- Times are PST (the host clock); the trace file time is the end of the profiled run.

## Per-GPU view (steady tokens, ms per token)

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

"Busy" includes AR wait (the AR kernel spins). ROCm1 starts each token about 0.8-1.2 ms after ROCm0 under the profiler (one slow `hipGraphLaunch` per lane, see [P3](old-ignore/2026-11-10/exp-p3/README.md)), so ROCm0 waits at the first AllReduces of the token. Without the profiler this lag is about 20 us, so most of the token-start gap and the ROCm0 AR wait are profiler cost. Inside the token there is almost no idle: the GPUs run back to back between AllReduces. The full per-GPU tables (busy + idle per step) are in [old-ignore/2026-11-10/token-budget/](old-ignore/2026-11-10/token-budget/) (`*.txt`, `*.json`).

Sanity checks against today's records: 27B@0 mat-vec 19.30 / 19.51 ms (Q3: 19.33 / 19.54); 27B@32K flash attention 3.08 / 3.34 ms (profile summary: 3.08 / 3.33); Flash Next `topk_moe_cuda` 1.35 ms (48 calls x 28 us; the earlier summary showed 1.26 because it divided 60 tokens of calls by 64).

## Per-model notes

- Qwen3.8-27B Q8_0 (`qwen35`): 64 layers, 48 gated delta net (GDN) + 16 full attention (every 4th), dense SwiGLU FFN with gate/up fused into one mat-vec and the activation fused into it (so "activation" is empty). 2 AllReduces per layer (128). Mat-vec is 68% of the token (19.3 ms); the post-AR kernel is the fused `add_rms_norm_mul` with the q8_1 epilogue ([Q1](old-ignore/2026-11-10/exp-q1/README.md)). The GDN "recurrent" row is about 25 small kernels per layer (state gather, conv, l2 norms, gates, GDN, gated norm). At 32K only flash attention grows (0.45 -> 3.34 ms); the rest is flat.
- Qwen3.8 Flash Next UD-IQ3_XXS (`qwen4exp`): 48 layers, 36 GDN + 12 attention with the QSA indexer, MoE with top-10 routed experts plus a shared expert, 4 hyper-connection streams in place of residual + norm, and one per-layer n-gram embedding layer. 2 AllReduces per layer (96). 2908 kernels per token at about 8 us each: mat-vec is only 31% (7.9 ms). The large rows are fixed-cost glue: hyper-connection mix 3.9 ms, QSA indexer 2.5 ms (lightning indexer, 10-kernel radix top-k, top-k mask), router + top-k 1.8 ms, q8_1 quantize 1.5 ms. The final hyper-connection mix runs unfused before the LM head (output norm row).
- Gemma 4 31B-it Q8_0 (`gemma4`): 60 layers, 50 sliding-window (head 256) + 10 global (head 512) attention, dense GELU FFN (51 layers fused gate/up + GELU, 9 with separate gate and up), attention and FFN post-norms, final logit softcap. 2 AllReduces per layer (120). It ran under `-sm tensor` without problems. Mat-vec is 70% (23.0 ms): gate/up alone is 10.1 ms, the largest single row of all models. Q/K norm and RoPE are fused (`rms_norm_mul_rope`). Global layers take 26.6 us per `flash_attn_tile<512>` against 20.4 us for SWA layers at depth 0.

## Speculative modes (per verify step)

No speculative traces exist. Measured: ms per step, tokens per step and acceptance from the llama-server 5-prompt workload ([spec_accept.sh](scripts/spec_accept.sh), 256 greedy tokens, about 2K-token and short prompts). The split inside a step is an estimate, marked `est`, except where marked `meas`; `derived` = measured step minus the estimates.

```
27B Q8_0, MTP2 (2026-11-10, build-exp-spec, exp-spec record, r1/r2)
  step 40.6 ms (meas, 40.81 / 40.34), 2.55 tokens/step (meas, server mean len), accept 0.777
  -> 15.9 ms/token (62.1 t/s) vs 26.7 plain
  +-----------------------------------------------+----------+
  | verify batch, 3 tokens (1 + 2 drafts), TP     | ~31  est |  plain 26.7 x 1.15-1.19 (3-token/1-token kernel time at 32K: 1.19, meas)
  | MTP draft decode x2 (1 dense layer + LM head) | ~4.5 est |  ~0.45 layer + 0.88 head + ~0.9 launch per draft
  | host: CPU sampling of 5 rows x 248K logits,   | ~5 derived |  backend sampling falls back to CPU under llama-server
  |   accept, GDN state rollback, batch build     |          |
  +-----------------------------------------------+----------+

27B Q8_0, DFlash5 (2026-11-10, build-exp-spec, exp-spec record, r1/r2)
  step 54.9 ms (meas, 55.03 / 54.79), 3.88 tokens/step (meas), accept 0.581
  -> 14.2 ms/token (69.3 t/s)
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

## Method and limits

- Data: rocprofv3 `--kernel-trace` of `llama-bench -p 0 -n N -d DEPTH -r 1` on the merged tree `0f7ce1b04` (upstream `aa75ff323`), 2026-11-10, TheRock ROCm, two gfx906 (ROCm0 = MI60, 64 CU, bus 03:00; ROCm1 = MI50, 60 CU). `Agent 1` is ROCm0.
- Tokens: a token runs from the end of one LM head mat-vec (largest mat-vec grid) to the end of the next. The last N heads are the decode tokens; N-1 intervals are used. Intervals above 1.2x the median are warm-up (eager launches before HIP graph capture, 1-3 per run); they are kept out of the step rows and added as one row of excess time, so the column sums to the all-token mean.
- Steps: kernels are split at the AllReduce kernels. A segment with `flash_attn` is attention, with `gated_delta_net` is recurrent, the last one is output, the rest are FFN. Inside a segment, the last mat-vec before the AllReduce is the output (or down) projection, other mat-vecs are input (or gate/up) projections; kernel names, grid sizes and order give the rest (rules in `classify_segment`). q8_1 quantize kernels have their own row in all models.
- Idle: a GPU gap is charged to the step of the kernel that follows it. Gaps before the first compute kernel of a token are "token start". The profiler inflates gaps: one `hipGraphLaunch` takes about 1 ms under rocprofv3 against about 20 us without it ([P3](old-ignore/2026-11-10/exp-p3/README.md)), so token-start gaps and ROCm0 AllReduce waits are mostly profiler cost.
- AllReduce wait: the part of a GPU's AR kernel before the other GPU started the same AR. Exchange is the rest.
- Critical path: for each AllReduce segment, the GPU that started that AR later; for the output segment, the GPU whose LM head ended later. The few us left when switching GPUs go to the rounding row.
- Scaled total: profiled total x (unprofiled ms / profiled ms). The profiler slows TG by 7-24%, mostly in small kernels and host gaps, so scaled step rows are rough.
- llama-bench does not sample: host sampling is not in any plain column. In llama-server and llama-cli it adds per-token host time (see the speculative boxes).
- Not attributable: nothing large. The "glue" row (crops before the last layer, stray copies) is 0.01 ms in every column.

## How to read and use it

- Busy is kernel time; idle-before is time the GPU had no kernel before that step started. A large idle-before means the step waited for the host or for the other GPU.
- The two GPUs meet at every AllReduce, so the slower rank sets the pace. The main table follows that rank per segment. ROCm0 AR time is mostly waiting for ROCm1 (per-GPU view).
- Only critical-path savings speed up TG. A busy saving on the rank that already waits only turns into more AR wait. A saving on one rank counts fully only if that rank is late in that segment.
- Back to speed: `TG t/s = 1000 / ms_per_token`. Example, 27B@0 unprofiled: 26.70 ms = 37.45 t/s; saving 1.0 ms on the critical path gives 1000 / 25.70 = 38.9 t/s. Use unprofiled (scaled) ms for this, not profiled.
- Speculative: `t/s = tokens_per_step * 1000 / ms_per_step`, with `tokens_per_step = 1 + accepted drafts per step` (about `1 + n_draft * acceptance`). Example, 27B MTP2: 2.55 * 1000 / 40.6 = 62.8 t/s.
- Context length: only the attention rows grow with depth (KV read); GDN/recurrent rows, mat-vecs and AllReduces stay flat. For 27B, flash attention is `0.45 + 0.088 * n_ctx_K` ms (profiled, n_ctx_K = context in 1K tokens, from the @0 and @32K columns). The whole unprofiled token fits `26.7 + 0.099 * n_ctx_K` ms, which gives about 36.3 t/s at 8K, 35.3 at 16K and 30.3 at 64K. Limits: two points only; it assumes the F16 KV cache and the same flash-attention kernel choice at all depths (kernel switch points and other KV types change the slope); confirm with a measured column before you quote a number.
- Prompt processing: these are decode budgets and do not convert to PP. PP is GEMM-bound per ubatch with other kernels (MMQ/rocBLAS, batched attention), and the AllReduce cost per token is much smaller. This doc does not tell you PP limits or PP speed. A PP column can be made the same way: trace `llama-bench -p 2048 -n 0`, cut per ubatch instead of per token (needs new rules in the script), and report ms per 1K prompt tokens. `PP t/s = n_prompt * 1000 / total_ms`. For long-prompt prefill, the attention cost per ubatch grows with the depth already in the cache, so measure at the depths you care about.

## How to update (and how this doc is made)

- Tools: `rocprofv3 --kernel-trace` on llama-bench TG (one run, `-r 1`) for the split, plus an unprofiled llama-bench `-r 3` of the same test for the real t/s. [token_budget_trace.sh](scripts/token_budget_trace.sh) runs both; [token_budget.py](scripts/token_budget.py) (stdlib only, `python3 -I`) makes the split and prints a ready column.
- Provenance rows: Column id (short model@depth plus date, unique), Measured (trace file time), Source commit (tree that was built), Build (directory plus sha256 prefixes of `llama-bench` and `libggml-hip.so`, which holds the kernels), model and file revision, settings, unprofiled t/s, profiled ms/token (all tokens; steady tokens; llama-bench), kernels and AllReduces per token.
- Rule: never edit an old column's numbers. Add a new column to the right; keep the row order. If the table gets too wide, start a new table section with the same rows (for example one per model) and leave the old one as is.
- Keep raw outputs (`*.txt`, `*.json` of the script; not the 70-180 MB traces) in a dated `old-ignore/<date>/token-budget/` folder and link it from the column notes.

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
sed -n '/^COLUMN/,$p' $O/budget.txt   # append each cell to the matching row of the main table
```

Model kinds: `qwen35` (27B), `qwen4exp` (Flash Next; use `-b 1024 -ub 512`, depth 8192, `-m` the local UD-IQ3_XXS shard 1), `gemma4` (Gemma 4 31B, `-m` the cached Q8_0 file). A new model needs its own rules in `classify_segment` if its kernels do not fit the generic ones; check that no large time lands in the glue row. Each run takes 1-3 minutes plus the depth prefill; runs need the user's OK like any benchmark ([PERFORMANCE.md](PERFORMANCE.md#how-to-test-and-what-the-status-words-mean)).
