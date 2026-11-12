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

## Files

`raw/`: harness (`run.sh`, `conc.py`, `longfill*.py`, `all*.sh`, `routercheck.sh`), result text and server logs (prompt-progress lines removed).
