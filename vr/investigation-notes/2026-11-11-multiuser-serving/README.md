# Multi-user serving settings for Qwen 27B and Flash Next

2026-11-11. Build `build-play-b1` (dc9c4e367 runtime). Two gfx906 GPUs, tensor parallel, F16 KV. Results feed the presets in [vr/server/models.ini](../../server/models.ini).

In short: Qwen 27B can serve 4 users with MTP2 and a 262K shared context with no speed loss. Flash Next cannot use its MTP draft with more than one user yet: our fork's fast rollback only supports one slot, and the fallback halves speed. Flash Next therefore serves 4 users without a draft, with a 128K shared context. (The rollback limit was fixed later the same day; see the update below.)

## Diagnosis: Flash Next with several slots

The earlier unified-KV report (see [the unverified notes](../2026-11-11-unified-kv-unverified/verification/RESULTS.md)) saw Flash Next drop from about 64 to 31 t/s. Unified KV is not the cause. `llama_context` clamps `n_rs_seq` to 0 for `LLM_ARCH_QWEN4EXP` whenever `n_seq_max > 1` (src/llama-context.cpp, "qwen4exp recurrent rollback currently requires n_seq_max=1"). The server then uses full-state checkpoints for speculative rollback, which costs about 35 ms per verify step. Qwen 27B (QWEN35) keeps rollback with several slots, so it is unaffected.

Isolating run: same placement (draft ROCm1, split 0.52/0.48), 32K, 1024/512, only the slot count changed.

| Flash Next | n_rs_seq | 1 user TG | ms/step | accept |
| --- | ---: | ---: | ---: | ---: |
| 1 slot, MTP3 | 3 (ring) | 63.8 | 48.1 | 0.71 |
| 2 slots unified, MTP3 | 0 (checkpoints) | 32.3 | 83.4 | 0.75 |
| 2 slots unified, no draft | - | 49.9 | 20.0 | - |

Update, same day: fixed in [the multi-slot rollback notes](../2026-11-11-multislot-rollback/README.md). The clamp is gone; with several slots qwen4exp uses the plane-indexed snapshot path that Qwen 27B already uses (it does write conv rollback slots there, through `build_conv_state_store`), and `seq_cp` / `find_slot` got two small fixes. Flash Next with 2 slots and MTP3 now gives 67-69 t/s for one user (44-46 ms/step). The rows below are the pre-fix measurements.

## Results

Workload: [spec_accept.py](../../scripts/spec_accept.py) 5 prompts x 256 greedy tokens, one at a time ("1 user"), then `conc.py`: N simultaneous requests of 256 tokens on 1.2-1.7K-token prompts ("aggregate" = generated tokens / wall time, prefill included). Long fill: one request of the stated size, then a short question about it (`longfill.py`; `longfill2.py` also sets ignore_eos). One run each.

| Config | 1 user TG | ms/step | N users aggregate | Long fill: PP / TG / output | Peak used GPU0 / GPU1 (GB of 34.3) |
| --- | ---: | ---: | --- | --- | --- |
| Qwen MTP2, 4 slots, 262144 unified | 65.6 (61.7 rerun) | 38.4 | 4: 41.2 | 200K: 219 / 28.3 / correct | 33.0 / 33.0 |
| Qwen no draft, 4 slots, 262144 | 34.7 | 28.8 | 4: 43.9 | - | 28.4 / 28.5 |
| FN MTP3, 1 slot, 32K (old preset placement) | 63.8 | 48.1 | 2 (queued): 34.6 | - | 30.6 / 33.0 |
| FN MTP3, 1 slot, 100352 (old preset) | - | - | - | 90K: 327 / 2.4 (first-token delay) / correct | - |
| FN no draft, 4 slots, 100352, ts 0.52 | 50.2 | 19.8 | 4: 41.2 | - | 33.0 / 30.9 |
| FN no draft, 4 slots, 131072, ts 0.47 (preset) | 48.3 | 20.7 | 4: 40.7 | 120K: 290 / 28.6 / correct | 31.7 / 33.7 |
| FN no draft, 4 slots, 163840, ts 0.49 | 48.4 | 20.6 | 4: 41.2 | 150K: 258 / 27.5 / correct | 31.7 / 34.2 |

- Qwen MTP2 nearly doubles a lone user's speed and loses about 6% aggregate when all 4 users are busy, so the preset keeps it. MTP also lowers prompt speed about 10% (584-602 vs 619-657 t/s).
- The old Flash Next MTP preset with 90K of prompt spent about 24 s before its 64 output tokens, while the 20 verify steps need about 1 s. This looks like the draft catching up on the prompt in 64-token draft ubatches (PERFORMANCE SP4). Not instrumented.
- Flash Next with a raw (no chat template) 120-150K prompt sometimes stops at once (first token EOG). With ignore_eos it answers correctly, so the long-context state is fine. Long chat-template requests were not tested.
- Router check with [vr/server/serve.sh](../../server/serve.sh): both presets load with `n_slots = 4`, `kv_unified = 'true'`, and answer a chat request.

## Approaches that did not work

These limits are worth retrying after the memory or rollback fixes:

- Flash Next no-draft 262144, split 0.5/0.5: compute buffer allocation (2164 MiB on ROCm0) failed at load.
- Flash Next no-draft 196608, split 0.5/0.5: loaded, then a 50 MiB runtime allocation on ROCm0 failed at the first request (abort).
- Flash Next no-draft 163840: works, including a 150K fill, but only about 0.2 GiB stays free on ROCm1, and it needs the "retrying without pipeline parallelism" fallback. Too tight for 4 users prefilling at once. Moving the split from 0.47 to 0.49 did not change peak use.
- Flash Next multi-slot with MTP at 100352 (from the earlier notes): the draft's compute buffer grows from 72 to 269 MiB with 4 slots and does not fit with the sidecar on either GPU.

Hint for next time: the full-pool attention fallback ("Flash Attention over K/V rows" disabled) likely makes compute buffers scale with the unified pool size (2164 MiB at 262K); that, not the KV itself, appears to limit the pool. Unverified for Flash Next.

## Flash Next 2-slot MTP3 context sweep (after the multi-slot rollback fix)

Build `build-mslot-ring` (fbc067862, see [the rollback record](../2026-11-11-multislot-rollback/README.md)). Flash Next MTP3, Q8_0 sidecar on ROCm1, `-np 2 --kv-unified`, 1024/512. Each pool: 5-prompt serial run, 2 concurrent requests, then one prompt at 90% of the pool. ROCm1 total is 34.34 GB; it holds the draft and runs out first.

| Pool | Split | 1 user TG (ms/step) | 2 users aggregate | Long fill: tokens, PP / TG | ROCm1 free at peak |
| --- | --- | ---: | ---: | --- | ---: |
| 131072 | 0.52 | - | - | load failed (draft weights 3317 MiB on ROCm1) | - |
| 114688 | 0.52 | - | - | load failed (224 MiB on ROCm1) | - |
| 98304 | 0.52 | - | - | load failed (135 MiB on ROCm1) | - |
| 98304 | 0.56 | - | - | loaded; 50 MiB runtime allocation on ROCm1 failed at the first request (abort) | - |
| 81920 | 0.52 / 0.55 | 67.6 / 68.9 (45.3 / 44.4) | 36.1 / 36.4 | 73757, 342 / 36.6 | 0.03 GB |
| 73728 | 0.52 | 65.6 (46.8) | 35.4 | 66384, 347 / 42.4 | 0.16 GB |
| 65536 | 0.52 | 67.6 (45.3) | 36.1 | 59011, 359 / 44.1 | 0.37 GB |

The `flashnext` preset uses 65536, the largest pool with a margin for runtime scratch allocations (the 98K crash was a 50 MiB one). Split 0.55 or 0.56 left memory use unchanged; tensor split moves in coarse steps here. Long-fill TG includes the draft's prompt catch-up (SP4), so it is lower than steady state.

Final presets (vr/server/models.ini, router check on build-mslot-ring: all load and answer):

| Preset | Slots / pool | 1 user TG | N users aggregate | PP (server) | Longest verified prompt |
| --- | --- | ---: | ---: | ---: | --- |
| `qwen27b` MTP2 | 4 / 262144 | about 65 | 4: about 41 | about 590 | 200K |
| `flashnext` MTP3 | 2 / 65536 | about 67 | 2: about 36 | about 480 | 59K |
| `flashnext-long` no draft | 4 / 131072 | about 48 | 4: about 41 | about 500 | 120K |

Retry ideas for a larger MTP pool: move the draft to ROCm0 with a split that frees ROCm0 (earlier 4-slot attempt at 64K left only 10 MiB there), shrink `spec-draft-n-ubatch` or target `-ub` to cut compute buffers (costs PP), or load the sidecar weights at lower precision.

## Files

`raw/`: harness (`run.sh`, `run2.sh`, `conc.py`, `longfill*.py`, `all*.sh`, `sweep*.sh`, `routercheck.sh`), result text and server logs (prompt-progress lines removed).
