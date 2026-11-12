# Flash Next MTP rollback with several server slots

2026-11-11. New build `build-mslot-ring` (commit fbc067862) vs old `build-play-b1`. Two gfx906 GPUs, tensor parallel, F16 KV.

In short: Flash Next now keeps its fast MTP rollback when the server has 2 or 4 slots, so one active user gets about 67 t/s instead of 32 t/s. Qwen 27B is unchanged.

## What changed

- `src/llama-context.cpp`: removed the clamp that set `n_rs_seq` to 0 for `LLM_ARCH_QWEN4EXP` when `n_seq_max > 1`.
- With several sequences, `llama_memory_recurrent` keeps the snapshot ring off and uses the plane-indexed snapshot path, the same path Qwen 27B (QWEN35) already uses with several slots. Qwen4exp already writes its conv and PLE rollback slots on that path (`build_conv_state_at` falls back to `build_conv_state_store`), and the delta-net state uses the plane-indexed fallback in `delta-net-base.cpp`. The ring stays the path for one slot.
- `src/llama-memory-recurrent.cpp`, two small fixes for several sequences (they also apply to QWEN35):
  - `seq_cp` copies a pending rollback (`rs_idx`, `rs_target`) to the destination sequence. The destination shares the source cell, so it must read the same snapshot plane.
  - `find_slot` sets `rs_max = 0` on a cell whose sequence was moved by the gather swap. Only the current state row follows a moved sequence, not its rollback planes, so a later rollback of that sequence is refused instead of reading another sequence's snapshots.

The ring with several sequences (the brief's first option) was not needed: the plane-indexed path is as fast here (44-46 ms per verify step vs 45.6 with the ring).

## Correctness

- Startup at `-np 2` (`-lv 4`): `llama_context: n_rs_seq = 3`, recurrent memory `2 seqs 3 rs_seq`; no clamp warning. Speed and accept rates show rollback is used, not checkpoints.
- Greedy MTP3 output is not token-identical to no-draft output on any build, including the old 1-slot ring path: verify batches of 4 tokens use different kernels than 1-token decodes, so near-ties flip. `tiecheck.py` asks the no-draft server for the top tokens at each first difference: every difference between new np2 MTP3 and np2 no-draft, and between new np2 MTP3 and old np2 MTP3 (checkpoints), is a near-tie between the top two tokens (for example 0.362 vs 0.314, 0.510 vs 0.472), except one case below. See `raw/out/tiecheck_fn.txt`.
- Exception, found here and present on the old build too: on prompt `p2.txt` (the head of `src/llama-context.cpp`; cases named `p1` in `tokcheck.py`, which counts from 0), every MTP3 configuration (old 1-slot ring, old 2-slot checkpoints, new) emits token 262 as the 3rd output token, which no-draft decoding ranks outside its top 5. This is not caused by this change. It is unexplained and worth a separate look (MTP prompt prefill or the first verify step).
- Token-identical results (`tokdiff.py`, 256 tokens each, ignore_eos):
  - New np2 MTP3, serial cases (slot 0, slot 1, slot reuse, swapped slot) = old np1 MTP3 ring path, all 4 cases.
  - New np2 MTP3 = new np1 MTP3 for all serial cases; new np4 = new np2 for all cases except one n_cmpl branch (a 0.51/0.47 near-tie at token 17, different slot assignment).
  - Two identical prompts on both slots at once: both outputs equal the serial and no-draft outputs.
  - Repeat runs of np2 MTP3 are identical across all 11 cases, including concurrent, prompt-cache reuse and `n_cmpl = 2` (seq_cp) cases.
  - Prompt files are `vr/investigation-notes/2026-11-11-unified-kv-unverified/verification/prompts/p1-p3.txt` (in the main checkout); case names count from 0.
  - Prompt-cache reuse on np2 differs from np1 at token 0 both with and without a draft (multi-slot prompt-cache behavior, not rollback); old and new np2 MTP3 agree there.
- Qwen 27B MTP2: new = old token for token on all cases, np1 and np4.
- `test-llama-context` (`--test 'rollback|replay|shared'`, `-c 4096`): `shared` passes on both models. `replay` (2 sequences) passes on 27B and fails on Flash Next with nmse 0.0033 (bound 1e-4). The same test edited to 1 sequence, which uses the old ring path, gives the same nmse (0.0034), so this is the 4-token vs 1-token kernel difference on this quantized MoE, not the multi-sequence path. `rollback` fails on both models with "a rollback past the snapshots of single-token ubatches was accepted": that is the 1-slot ring keeping `valid_depth` across ubatches by design (see the merge traps in IMPROVEMENTS), unchanged here.

## Results

Server: `spec_accept.py` 5 prompts x 256 greedy tokens one at a time ("1 user"; PP is the mean prompt speed of the three 1.2-1.7K-token prompts), then `conc.py` with N simultaneous requests ("aggregate", prefill included). Flash Next: 32K context, unified KV, ts 0.52/0.48, b 1024 / ub 512, MTP3 draft on ROCm1. Qwen 27B: MTP2, b/ub 2048, 32K (np1) or 262144 (np4) unified. One run each unless noted.

| Model | Slots | Build | 1 user TG | ms/step | accept | PP (server) | N users aggregate |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Flash Next MTP3 | 1 | old | 67.2 | 45.6 | 0.709 | 481 | - |
| Flash Next MTP3 | 1 | new | 66.6 | 45.9 | 0.709 | 478 | - |
| Flash Next MTP3 | 2 | old (checkpoints) | 32.5 | 82.9 | 0.749 | 486 | 2: 22.0 |
| Flash Next MTP3 | 2 | new | 69.1, 67.2 (2 runs) | 44.3, 45.7 | 0.709 | 485, 478 | 2: 36.5 |
| Flash Next MTP3 | 4 | old | not measured (checkpoint path, as np2) | | | | |
| Flash Next MTP3 | 4 | new | 66.7 | 46.0 | 0.709 | 476 | 4: 31.0 |
| Qwen 27B MTP2 | 1 | old | 57.7 | 43.7 | 0.777 | 539 | - |
| Qwen 27B MTP2 | 1 | new | 57.3 | 44.0 | 0.777 | 523 | - |
| Qwen 27B MTP2 | 2 | old / new | not measured | | | | |
| Qwen 27B MTP2 | 4 | old | 58.9 | 42.8 | 0.777 | 528 | 4: 36.1 |
| Qwen 27B MTP2 | 4 | new | 58.8 | 42.9 | 0.777 | 515 | 4: 35.9 |

Server PP is one run and noisy (single prompts vary by 10%); the change does not touch prefill. llama-bench (no draft, one sequence, 3 repetitions):

| Model | Build | PP2048 | TG128 |
| --- | --- | ---: | ---: |
| Qwen 27B Q8_0 (ts 1/1, b/ub 2048) | old | 702.1 | 36.0 |
| Qwen 27B Q8_0 | new | 705.5 | 37.7 |
| Flash Next UD-IQ3_XXS (ts 0.52/0.48, b 1024, ub 512) | old | 606.9 | 49.1 |
| Flash Next UD-IQ3_XXS | new | 609.1 | 50.5 |

Notes:

- Flash Next with 4 slots and MTP fits at 32K (31.6 / 33.9 GB used). With 4 busy users MTP3 gives 31.0 t/s aggregate, below the earlier no-draft 4-slot figure (41.2 t/s at 100K, other run). For mostly single-user traffic, 2 slots with MTP3 is the better trade; for 4 busy users, no draft still wins. The presets in `vr/server/models.ini` are unchanged.
- Peak memory with 2 slots and MTP at 32K: 31.0 / 33.3 GB. Larger contexts with MTP and several slots were not tried.

## Files

`raw/`: `build.sh`, `srv.sh` (start server, run client, stop), `fn.sh` / `q27.sh` (per-model runs), `all.sh`, `bench.sh`, `tokcheck.py` (greedy token ids for the slot schedule), `tokdiff.py`, `tiecheck.py`; `results-server.txt`, `results-bench.txt`; `out/`: token JSON, server logs, tie check.
