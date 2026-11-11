# 2026-11-10 exp-spec: merge check and speculative device-row paths

Check share-view correctness and 27B MTP2/DFlash5 after the aa75ff323 merge, then measure the upstream device-row paths (`LLAMA_MTP_DEV_ROWS=1`, `LLAMA_DFLASH_DEV=1`, idea M1). Kind: regression screen + A/B experiment.

- Source: `0f7ce1b04` unmodified, build `build-exp-spec` (catch-up flags); baseline `build-catchup-20261105-213622` (pre-merge fork).
- Model: 27B Q8_0 rev `4ca720788d1e`; DFlash draft `Qwen3.8-27B-DFlash2-Q8_0` rev `2d9571f8ce46`.
- Settings: standard 27B. Speculative A/B uses the `spec_accept.sh` workload (llama-server, 5 prompts, 256 greedy tokens), plus prompt ms and reply sha per prompt.
- Noise: the host was shared with two other agents (builds, `-j 6` each). Plain TG measured 16.7-17.4 t/s instead of about 37, so absolute t/s are low and noisy; compare ms/step within this session only.

## Part 1: merge validation

Share-view (`-sm layer -ngl 56`, 8 layers on CPU, 48 greedy tokens, llama-completion): baseline and candidate, `GGML_SCHED_SHARE_VIEW_INPUTS=1` and `=0`, all four texts `3ce39e8a822d`. Pass.

Plain greedy, `-sm tensor`, flake.nix prompt, 192 tokens: baseline and candidate identical (`1ecfab2fe93d`). Pass.

| Build | Mode | TG t/s (r1 / r2) | ms/step (r1 / r2) | Change ms/step | Accept | Prompt ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | MTP2 | 57.45 / 59.70 | 45.23 / 43.56 | - | 0.816 | 1789 / 1732 |
| candidate | MTP2 | 61.74 / 62.43 | 40.81 / 40.34 | -8.6% (means) | 0.777 | 1690 / 1707 |
| baseline | DFlash5 | 65.81 / 65.62 | 58.90 / 59.06 | - | 0.595 | 1682 / 1694 |
| candidate | DFlash5 | 69.15 / 69.36 | 55.03 / 54.79 | -6.8% | 0.581 | 1654 / 1661 |

No speed regression: the candidate is faster per verify step in both modes (baseline MTP2 reps differ by 1.7 ms/step, so the MTP2 size is uncertain; the DFlash5 reps agree within 0.3 ms). Speculative text differs from the baseline on 2 of 5 prompts (p2, p3) in both modes, while plain greedy is identical; the merge changed near-tie results of batched verify, and acceptance moved with the text (MTP2 p3 0.809 -> 0.641). llama-cli single runs (flake.nix, 512 tokens) also diverge in the first 200 reply bytes. Treat the new texts as the new reference. llama-cli single runs: baseline/candidate DFlash5 60.6/57.7, MTP2 61.9/57.3 t/s; per PERFORMANCE.md rule 4 these do not count for A/B and the host load was high.

Side note: `--spec-draft-backend-sampling` with MTP under llama-server logs `backend offload failed for seq_id=0; using CPU sampler` on both builds (not new).

## Part 2: device-row paths

| Variant | TG t/s (r1 / r2) | ms/step (r1 / r2) | Change ms/step | Accept | Prompt ms (p1, about 1750 tokens) | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MTP2 default | 61.74 / 62.43 | 40.81 / 40.34 | - | 0.777 | 2791 / 2852 | - |
| MTP2 `LLAMA_MTP_DEV_ROWS=1` | 61.80 / 62.37 | 40.76 / 40.39 | +0.0% | 0.777 | 3343 / 3331 (+18%) | same |
| DFlash5 default | 69.15 / 69.36 | 55.03 / 54.79 | - | 0.581 | 2918 | - |
| DFlash5 `LLAMA_DFLASH_DEV=1` | 29.77 | 128.07 | +133% | 0.581 | 3059 | same |

MTP dev rows engage under `-sm tensor` (`h_nextn_dev_alloc: Meta(ROCm0,ROCm1) h_nextn buffer size = 40.02 MiB`, `target h rows on the device: yes`, visible only with `-lv 4`). They do not change step time, and they turn off the deferred prefill, so long-prompt TTFT rises 18%. Combining (deferred prefill for prompts, dev rows for decode) can at best match the default, so it was not built.

DFlash dev features engage but are 2.3x slower. `layer_inp_dev` is allocated on `model.dev_output()`, which is the meta device for a TP target, so the buffer is a mirrored meta tensor. The draft runs on ROCm0 alone, cannot read a meta buffer, and the scheduler copies the whole 200 MiB buffer (n_batch 2048 x 5 taps x 5120 x F32) to ROCm0 for every draft decode. A fix needs the ROCm0 simple tensor of the mirrored meta tensor as `layer_inp_src` (today `ggml_backend_meta_buffer_simple_tensor` is static in ggml-backend-meta.cpp), and the draft would still host-wait on the target's written event because it has no meta backend. The fork host path saves only the small D2H/H2D tap copy, so the expected gain is below 1-2 ms of a 55 ms step. Not built.

Verdict: merge passes (correct output, no speed regression). `LLAMA_MTP_DEV_ROWS` validated wrong for 27B TP (no step gain, +18% TTFT). M1 as-is is validated wrong under TP (2.3x slower); a meta simple-tensor fix is possible but small upside. No code changes kept. Not covered: Flash Next, 32K depth, KLD.

Scripts used: `build-exp-spec/out/scripts/` (spec.sh = spec_accept.sh with DFlash and reply sha; p1a.sh; plain.sh; cli.sh). Raw: `build-exp-spec/out/`.
