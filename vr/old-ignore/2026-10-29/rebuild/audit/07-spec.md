# Audit 07: speculative decoding and app layer

Scope: `common/speculative.cpp` (+848/-70 per brief), `common/speculative.h`, `common/common.cpp`, `common/common.h`, `common/arg.cpp`, `examples/speculative/speculative.cpp`, `tools/server/server-context.cpp`.
Diff base: `418f905a4` to `HEAD`. Line numbers below are HEAD unless marked otherwise. Raw diffs: `.../audit/spec.diff` and `.../audit/app.diff` in the scratchpad.

Approximate LOC are counted from hunks and are not exact.

## 1. Feature table

| Feature | Files | ~LOC +/- | Class | Upstream equiv | Gating | Recommendation | Confidence |
|---|---|---|---|---|---|---|---|
| DFlash/DSpark deferred prompt encode (record chunks, replay at first draft, position-indexed tap accum) | common/speculative.cpp: `process()`, `flush_pending_upto()`, `replay_chunks()`, `flush_pending()`, `pending_chunk` | ~+250 / -40 | B (perf) | None. Base DFlash encodes each prompt ubatch eagerly with a target sync per chunk | DFlash/DSpark only. `LLAMA_DFLASH_DEFER` (default 1). Needs `llama_set_embeddings_layer_inp_accum` (src API) | Keep if prefill PP matters. Biggest single block; upstream separately with benchmarks | med |
| Lag-one async prompt replay on a worker thread | speculative.cpp: `replay_thread`, `replay_join()`, `async_replay()`, `replay_busy` | ~+60 | B (perf) | None | `LLAMA_DFLASH_ASYNC_REPLAY` (default 1) | Candidate to drop: threading adds risk (worker touches `ctx_dft` while the main thread drives `ctx_tgt`). Keep sync path only if the PP gain is small | med |
| Flush scheduling knobs (`FLUSH_ROWS`, `OPPORTUNISTIC`) | speculative.cpp ~1591-1600 (HEAD) | ~+20 | E/B (tuning knob) | None | `LLAMA_DSPARK_FLUSH_ROWS` (2048), `LLAMA_DSPARK_OPPORTUNISTIC` (1) | Hardcode after tuning; remove env reads | med |
| DSpark draft window (sliding draft KV, keep newest N rows) | speculative.cpp `flush_pending_upto()` ~1359 | ~+35 | C/B, behavior change by default | None | `LLAMA_DSPARK_DRAFT_WINDOW` (default 4096, on) | Quality-affecting default. Keep only as explicit opt-in, or document the acceptance tradeoff. vr evidence is 8-12k prompts on Qwen3.8 DFlash2 only | med |
| Defer short prompt tails | speculative.cpp ~1548 | ~+5 | B | None | `LLAMA_DFLASH_DEFER_TAIL` (default 1) | Keep, or hardcode 1 | med |
| Lazy prefill (replay runs at first draft, after the first token is sampled) | speculative.cpp: `common_speculative_lazy_prefill()` ~1022; `lazy_begun` in DFlash and MTP | ~+30 | B | None | `LLAMA_SPEC_LAZY_PREFILL` (default 1) | Keep; it is the default path. Remove env read once stable | med |
| MTP deferred prefill (stage whole prompt pre-norm hidden, one draft replay after prefill) plus `KV_ONLY` replay | speculative.cpp: MTP `deferred_prefill`, `finish_deferred_prefill()`, `prefill_chunks`, `finish_lazy_prefill()` | ~+170 / -10 | B (perf) | None | n_seq==1, !is_mem_shared, !chain_heads, `LLAMA_MTP_DEFERRED_PREFILL` (default on), `LLAMA_MTP_NO_KV_ONLY` (default off, meaning KV-only ON) | Keep if the MTP prefill win holds. Consider dropping the escape-hatch env read | med |
| MTP `process_decode()` refactor (takes tokens/pos/seq arrays, not `batch_in`) | speculative.cpp MTP `process()`, `process_decode()` | ~+80 / -40 | A/C (needed by deferred prefill) | None | Always on | Keep. Tied to deferred prefill | med |
| MTP `pending_h` reset on new sequence (`positions[i_first] == 0` zeroes carried hidden) | speculative.cpp `process_decode()` | ~+4 | A (correctness) | None. Base carries `pending_h` across requests | Always on | Upstream as bugfix candidate | med |
| MTP adaptive disable (per-seq disable on non-finite draft probability, all-disabled turns off pre-norm extraction) | speculative.cpp: `disabled`, `disable_seq()`, `all_disabled()`, `pre_norm_disabled`; `LLAMA_MTP_ADAPTIVE_DISABLE` | ~+60 | A (robustness) + C | None | Always on for MTP. Env `LLAMA_MTP_ADAPTIVE_DISABLE` (default 1) | Keep the non-finite guard (speculative.cpp ~2535). The rest is complexity; env knob can go | med |
| DFlash per-seq row mapping (`i_batch_rows[]` replaces begin/end range) | speculative.cpp DFlash `process()` ~1502-1540 | ~+30 / -15 | A (correctness for combined server batches) | None. Base assumes each seq is contiguous in the batch | Always on | Upstream as bugfix | high |
| DFlash null anchor guard (`dp.id_last < 0`) | speculative.cpp ~1760 | ~+5 | A | None | Always on | Upstream as bugfix | high |
| DFlash mask token fallback to target vocab | speculative.cpp ~1190-1200 (HEAD) | ~+8 | A (correctness for no_vocab sidecar, DSpark) | None | Always on, only when draft lacks mask | Upstream as bugfix | high |
| DFlash draft on first target device (single-GPU draft, `load_private_head`) | speculative.cpp `common_speculative_init_result` ~3440-3457 | ~+20 | C/B | None (`-devd` exists) | DFlash only, split_mode tensor, no `-devd`. Env `LLAMA_DFLASH_DRAFT_TP` (default 0 = single GPU). DSpark is NOT covered | Keep. Changes default for DFlash on TP targets; verify in PR notes | med |
| Draft repack knob | speculative.cpp ~3433 | ~+6 | E (knob, default = base) | Base already loads draft with extra bufts | `LLAMA_DRAFT_REPACK` (default 1) | Remove env read, or keep as knob. Comment at 3429-3431 contradicts code | high |
| DSpark target repack off by default | common/common.cpp ~1765-1769 | ~+6 | B/F (behavior change for DSpark) | None | Spec type DSpark. Env `LLAMA_DSPARK_TARGET_REPACK` (default 0 = repack OFF) | Keep if acceptance evidence holds (vr/old-ignore FEATURES.md:285 says validated). Changes default | med |
| DFlash mirror-output flag, passed through a KV override key `mxxm.tensor_mirror_output` | common/common.cpp ~1746-1790 | ~+45 | E for the DFlash branch, C for DSpark | None | DSpark: always mirrored. DFlash: `LLAMA_DFLASH_MIRROR_OUTPUT` (default 0, "slower" per QUICKSTART) | Remove the DFlash env branch (dead in default config). Keep DSpark mirror only. Consider replacing the KV-override hack with a plain params field. Needs src check | med |
| Stats breakdown and timers (`t_process`, `t_tgtsync/gather/encode/readnx/inject`, `t_pp_*`, `n_pp_rows`) | speculative.cpp: struct fields ~185-235; `common_speculative_process()` ~3710; `print_stats()` ~3895-3940 | ~+90 | E (debug) | None | Always on, because `gen_perf = true` is hardcoded. `print_stats` log level changed from SPC_TRC to LOG_INF, so it prints at default verbosity | Remove the dead counters. Keep one or two timers, or gate on LLAMA_TRACE. Restore TRC level or keep INF deliberately | high |
| `common_speculative_reset()` plus server call | speculative.h:98-99; speculative.cpp ~3847; server-context.cpp:679 | ~+30 | A (slot reuse) | None | Always on | Keep the call. The bool return is ignored by the server (`speculative_prompt_invalid` is unused). Simplify to void, or actually use it | high |
| eagle3 and DFlash and MTP `llama_sched_reserve` before prompt | speculative.cpp ~645, ~1258, ~2080 | ~+8 | B | None in common (base has no call here) | Always on for these drafts | Keep | med |
| Server: `slot.sampled = LLAMA_TOKEN_NULL` and no drafting before first sample | server-context.cpp ~400, ~3899 | ~+6 | A (correctness: draft anchor -1) | None | Always on | Upstream as bugfix | high |
| Server: `prompt_checkpoint_storage_lossless` (DSV4 `get_can_checkpoint()==false`) | server-context.cpp ~1121, ~1451, ~1635, ~2754 | ~+25 | A/C (correctness: no lossy snapshots) | None. `get_can_checkpoint()` exists in base, only DSV4 overrides it | Always, but only changes behavior for DSV4 | Upstream as bugfix | med |
| Server: `LLAMA_NO_MIDPROMPT_CKPT` | server-context.cpp ~4596-4612 | ~+14 | B experiment knob | None | Env (default off) | Remove unless used. No vr win recorded | med |
| Server: `reserve_prompt_graph()` (DSpark prompt reserve, explicit synchronize) | server-context.cpp ~4650-4672, call ~4779 | ~+40 | B | None | DSpark only, n_seq 1, 2+ devices, layer split. Env `LLAMA_DSPARK_PROMPT_RESERVE` (default 1) | Keep if DSpark is kept. Narrow gate. Note: adds a host sync per prompt | med |
| Server: `spec_i_batch_last` (vector, moved from, never written) | server-context.cpp:309, :5092 | ~+2 | E (dead) | None | n/a | Delete both lines. Verified: only one write-free move site | high |
| Server: slot per-seq draft-sampled guard, `result.prob` comment edit | server-context.cpp ~3899 and ~5128 | ~+3 | A / cosmetic | None | Always on | Keep the guard. Comment-only edit is noise | high |
| examples/speculative: prompt decode split by `n_batch` | examples/speculative/speculative.cpp ~197-222 | ~+25 / -12 | A (fix: prompt longer than n_batch asserts) | Base single decode | Example binary only | Upstream as bugfix | high |
| `-tps` / `--tensor-parallel-size` (CLI) | common/arg.cpp ~2852-2863; common.h ~500; common.cpp ~1737 | ~+18 | C (functional) | None | Only meaningful with `-sm tensor`. Default 0 = one TP group. Env `LLAMA_ARG_TENSOR_PARALLEL_SIZE` | Keep only if pipeline-stage topologies are used. Not needed for the dual-GPU target | high |
| `-tpsd` / `--spec-draft-tensor-parallel-size` (CLI) | common/arg.cpp ~2863-2876; common.h ~358; speculative.cpp ~3360 | ~+15 | C (functional) | None | Only with `-sm tensor`. Default 0 = single TP group over draft devices. Env `LLAMA_ARG_SPEC_DRAFT_TENSOR_PARALLEL_SIZE` | Keep if a draft width other than the target's is needed. Dual-GPU: not needed | high |
| `load_mtp` nextn and sched reserve before warmup | common/common.cpp ~1428-1431 | ~+5 | B | None | MTP-loaded default-type contexts only | Keep | med |
| Cosmetic: duplicate `#include <cmath>`, double blank lines, `n_taps` alias, `int32_t rc` relocation, struct realignment | speculative.cpp: 26, ~1074, ~1908, 1252, 1438, 1686, ~185-190 | ~+15 | E (noise) | n/a | n/a | Drop from any upstream PR | high |

Totals in this slice: about +1060 / -80 across 7 files, matching the diff stat (speculative.cpp 918 lines changed, server 97, common.cpp 60, arg.cpp 23, examples 34, common.h 7, speculative.h 3).

## 2. Per-feature notes

### 2.1 Speculative prompt replay (the main theme)

Three independent mechanisms share the name "prompt replay". All three move draft-KV construction for the prompt out of the target's prefill loop.

- Lazy prefill (`LLAMA_SPEC_LAZY_PREFILL`, default 1). `begin()` no longer replays the prompt. The first token is sampled first, then `draft()` or `process()` runs `flush_pending()`. Affects DFlash and MTP. Evidence: vr/old-ignore/2026-10-29/prior-guides/QUICKSTART.md:98.
- DFlash deferred encode (`LLAMA_DFLASH_DEFER`, default 1). `process()` records each prompt chunk (`pending_chunk {seq, p0, n, epoch}`) and skips the target sync. Position-indexed accum on the target is required. Replay happens in `flush_pending_upto()` with per-ubatch readiness events. Flushes are gated by `LLAMA_DSPARK_FLUSH_ROWS` and `LLAMA_DSPARK_OPPORTUNISTIC`. Evidence in code comments: a 98 ms per-call sync cost, and 58.2 s of 63.0 s in flush wait over 8 flushes at 16k. Also vr/old-ignore/2026-10-12/notes/2026-10-12-roofline.md:140.
- MTP deferred prefill (`LLAMA_MTP_DEFERRED_PREFILL`, default on for single sequence, non-shared-KV, single-head). Stages the whole prompt pre-norm hidden in a pinned buffer (`llama_set_embeddings_pre_norm_accum`), then builds draft KV in one pass (`finish_deferred_prefill()`), optionally KV-only (`llama_set_mtp_prefill_kv_only`). Depends on src/ changes (not in this slice).

Correctness note: the DFlash replay is correct only if chunk boundaries and order match the eager path. The code preserves both. Lag-one replay runs on a thread (`LLAMA_DFLASH_ASYNC_REPLAY`). That is the riskiest part of this slice.

### 2.2 Repacked draft weights

- `LLAMA_DRAFT_REPACK` (default 1): draft keeps extra buffer types, which matches base. The env var only exists to turn repack OFF. The stale comment at speculative.cpp:3429-3431 says drafts load canonical, but the code keeps the repack. Fix the comment or drop the knob.
- `LLAMA_DSPARK_TARGET_REPACK` (default 0): this is the material default change. For DSpark, the target loads without extra buffer types, because "target-side repacking changes the verification rounding enough to lower DSpark acceptance on gfx906" (common.cpp ~1765). Evidence: vr/old-ignore/2026-10-29/prior-guides/FEATURES.md:285.

### 2.3 Single-GPU DFlash draft

`common_speculative_init_result` (speculative.cpp ~3440-3457). For DFlash with `-sm tensor` and no `-devd`, the draft is loaded on `devices[0]` with `split_mode = NONE`. This removes an AllReduce per draft layer. It needs `llama_context::load_private_head` (src). Gated by `LLAMA_DFLASH_DRAFT_TP` (default 0 = single GPU). The gate checks `COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH` only, so DSpark drafts stay tensor-parallel. Evidence: vr/IMPROVEMENTS.md:13, QUICKSTART.md:102.

### 2.4 New CLI arguments (only two)

| Arg | Env | Needed? |
|---|---|---|
| `-tps, --tensor-parallel-size T` | `LLAMA_ARG_TENSOR_PARALLEL_SIZE` | Functional. Needed only for TP groups narrower than all GPUs (pipeline stages). Dual-GPU target does not need it. Default 0 |
| `-tpsd, --spec-draft-tensor-parallel-size T` | `LLAMA_ARG_SPEC_DRAFT_TENSOR_PARALLEL_SIZE` | Functional. Needed only when the draft width should differ from the target's. Default 0 |

Everything else in this slice is an env var, not a CLI arg. The CLI args have no collision in base (checked `-tps`, `-tpsd`, `-tp`).

### 2.5 Server changes

These are the per-sequence and concurrent fixes.

- Draft anchor (`slot.sampled = LLAMA_TOKEN_NULL` until the first token is sampled, guard at ~3899). Prevents `get_rows(-1)` in the draft graph. Correctness fix.
- Slot release calls `common_speculative_reset(spec, id)`. Discards deferred chunks and sampler state. Return value is unused (see table).
- Checkpoint storage gating. Only DSV4 returns `get_can_checkpoint() == false` (src/llama-kv-cache-dsv4.cpp:1672). Previously snapshots were taken anyway. Now boundaries are kept and snapshot storage is skipped. Correctness fix.
- DSpark prompt reserve (`LLAMA_DSPARK_PROMPT_RESERVE`, default 1). Narrow gate, explicit `llama_synchronize` plus reserve on the first prompt batch.
- `LLAMA_NO_MIDPROMPT_CKPT` (default off). Experiment knob. Removes mid-prompt checkpoints. The comment cites 0.7-2.1 s per checkpoint on 8-GPU TP. No vr win recorded.

Concurrency: DFlash accum is partitioned by sequence, so multi-slot is supported. MTP deferred prefill and the DSpark window are gated to n_seq == 1. DFlash's `rows[]` change is the concurrent fix: combined server batches need not keep a sequence contiguous.

### 2.6 Correctness fixes that could go upstream as separate bugfix PRs

1. DFlash `rows[]` per-seq mapping (non-contiguous batch rows). speculative.cpp `process()`.
2. DFlash null-anchor guard (`dp.id_last < 0`). speculative.cpp ~1760.
3. DSpark mask-token fallback to the target vocab. speculative.cpp ~1190.
4. MTP `pending_h` zeroing when a new sequence starts (`positions[i_first] == 0`). speculative.cpp `process_decode()`.
5. MTP non-finite draft probability guard. speculative.cpp ~2535.
6. MTP empty-batch guard (`batch.size() == 0`). speculative.cpp ~2490.
7. Server: `slot.sampled = LLAMA_TOKEN_NULL` and draft gate. server-context.cpp ~400, ~3899.
8. Server: checkpoint storage gating for `!get_can_checkpoint()`. server-context.cpp ~1451, ~2754.
9. examples/speculative: prompt decode split by `n_batch`. examples/speculative/speculative.cpp ~197-222.

### 2.7 Experiment knobs and dead code

Dead counters, verified by grep at HEAD:
- `t_encode_us` (speculative.cpp:215) and `t_readnx_us` (:216) are never written. The comment at ~1680 says so. They are still summed and printed (~3902-3906).
- `t_pp_encode_us` (:225) is never written. It is printed at ~3911.

Dead variables:
- `spec_i_batch_last` (server-context.cpp:309, :5092). Written nowhere. The moved-to local `i_batch` is unused.
- `speculative_prompt_invalid` (server-context.cpp:679). Unused.

Print-level change: `common_speculative_print_stats` changed `SPC_TRC` to `LOG_INF` (speculative.cpp ~3938). Stats now print at default verbosity. This is an unrequested behavior change.

Debug and stale text:
- Duplicate `#include <cmath>` (speculative.cpp:19 and :26; the first one is base).
- Stale comment at speculative.cpp:3429-3431 (contradicts code, see 2.2).
- Long multi-line comments in DFlash `process()` (~1540-1600) exceed the AGENTS 1-2 line guidance.
- Double blank lines around the DFlash `pending_chunk` struct and before `common_sampler_accept` (~1074, ~1908). Cosmetic.

Per-DFlash-mirror branch: the `LLAMA_DFLASH_MIRROR_OUTPUT` branch in common.cpp is default-off, and QUICKSTART calls it "slower". The DSpark path always uses mirror output, so the KV-override plumbing stays needed either way. Consider a direct field instead of the string-key override (needs a src check, not done here).

## 3. Quick wins (HEAD line numbers)

Each is safe and removes noise or dead code. None changes default behavior.

1. `common/speculative.cpp:26`: delete duplicate `#include <cmath>`. Already included at :19.
2. `common/speculative.cpp:215-216, :225`: delete `t_encode_us`, `t_readnx_us`, `t_pp_encode_us`, plus their uses in the print block (~3902-3912). Never written.
3. `tools/server/server-context.cpp:309` and `:5092`: delete `spec_i_batch_last` and `const auto i_batch = std::move(...)`. Never written, unused.
4. `tools/server/server-context.cpp:679`: change to a plain call `common_speculative_reset(spec, id);`, or make the function void. Return value unused.
5. `common/speculative.cpp:3429-3431`: fix the comment to match the code (repack is kept unless `LLAMA_DRAFT_REPACK=0`). Or drop the env read and keep the repack path.
6. `common/speculative.cpp:~3938`: restore `SPC_TRC` for the statistics line, or keep `LOG_INF` on purpose. Pick one and say so in the commit.
7. `common/speculative.cpp:~1252`: inline `n_taps` back to `target_layer_ids_n`. Pure rename.
8. `common/common.cpp:~1752-1756`: remove the `LLAMA_DFLASH_MIRROR_OUTPUT` branch. Default-off, slower, not in the default config. DSpark mirroring stays.
9. `tools/server/server-context.cpp:~4596-4612`: remove `LLAMA_NO_MIDPROMPT_CKPT`. Experiment knob with no recorded win.
10. Whitespace: double blank lines near speculative.cpp ~1074 and ~1908. Also struct realignment at ~185-190 is tied to the new field, so leave it or drop the new fields together.

Not quick wins (need validation or design call): DFlash async replay thread, DFlash deferred encode, MTP deferred prefill, DSpark window default, DSpark target repack default, DFlash single-GPU default.

## 4. Env var inventory (getenv in this slice)

Only `getenv` or `std::getenv` calls. Defaults as coded. "on" means the feature is enabled unless set to 0.

| Env var | File:line (HEAD) | Default | Type |
|---|---|---|---|
| `LLAMA_TRACE` (base) | speculative.cpp:2770 (base 2061) | unset | base |
| `LLAMA_SPEC_LAZY_PREFILL` | speculative.cpp:1022 | on (1) | B |
| `LLAMA_DFLASH_ASYNC_REPLAY` | speculative.cpp:1105 | on (1) | B, threading |
| `LLAMA_DFLASH_DEFER` | speculative.cpp:1272 | on (1) | B |
| `LLAMA_DSPARK_DRAFT_WINDOW` | speculative.cpp:1359 | 4096 | C/B, behavior change |
| `LLAMA_DFLASH_DEFER_TAIL` | speculative.cpp:1548 | on (1) | B |
| `LLAMA_DSPARK_FLUSH_ROWS` | speculative.cpp:1591 | 2048 | B tuning |
| `LLAMA_DSPARK_OPPORTUNISTIC` | speculative.cpp:1595 | on (1) | B tuning |
| `LLAMA_MTP_ADAPTIVE_DISABLE` | speculative.cpp:2110 | on (1) | A/C |
| `LLAMA_MTP_DEFERRED_PREFILL` | speculative.cpp:2119 | on (1) | B |
| `LLAMA_MTP_NO_KV_ONLY` | speculative.cpp:2325 | off (KV-only is on) | B escape hatch |
| `LLAMA_DRAFT_REPACK` | speculative.cpp:3433 | on (1) | E knob, default = base |
| `LLAMA_DFLASH_DRAFT_TP` | speculative.cpp:3445 | off (single GPU) | C/B |
| `LLAMA_DFLASH_MIRROR_OUTPUT` | common.cpp:1749-1755 | off (0) | E for DFlash, default-off |
| `LLAMA_DSPARK_TARGET_REPACK` | common.cpp:1765 | off (0) | B/F, default changed |
| `LLAMA_NO_MIDPROMPT_CKPT` | server-context.cpp:4609 | off | B experiment |
| `LLAMA_DSPARK_PROMPT_RESERVE` | server-context.cpp:4652 | on (1) | B |
| `LLAMA_SERVER_SLOTS_DEBUG`, `LLAMA_SERVER_SLOTS_N_DIFF`, `LLAMA_TRACE` | server-context.cpp:1580-1598 | base | base, not new |

CLI-linked env vars (new): `LLAMA_ARG_TENSOR_PARALLEL_SIZE`, `LLAMA_ARG_SPEC_DRAFT_TENSOR_PARALLEL_SIZE`.

New getenv reads added in this slice: 16 (speculative.cpp 12, common.cpp 2, server 2). The base `LLAMA_TRACE` reads are not counted. Removing the experimental and default-off knobs (MIRROR_OUTPUT, NO_MIDPROMPT_CKPT, MTP_NO_KV_ONLY, DRAFT_REPACK, MTP_ADAPTIVE_DISABLE, DSPARK_OPPORTUNISTIC, DSPARK_FLUSH_ROWS) would drop 7 of the 16.

## 5. Evidence and limits

- Read only. No build, no runs, no git state changes.
- vr docs consulted: vr/IMPROVEMENTS.md:13, vr/PERFORMANCE.md:14, vr/old-ignore/2026-10-29/prior-guides/QUICKSTART.md:97-109, vr/old-ignore/2026-10-29/prior-guides/FEATURES.md:285, vr/old-ignore/2026-10-12/notes/2026-10-12-roofline.md:140. Also grepped the vr patch files for each env var.
- No recorded win found in vr for: `LLAMA_NO_MIDPROMPT_CKPT`, `LLAMA_DFLASH_MIRROR_OUTPUT`, `LLAMA_DSPARK_OPPORTUNISTIC`, `LLAMA_MTP_NO_KV_ONLY`.
- Not verified here: whether `llama_set_embeddings_layer_inp_accum`, `llama_set_embeddings_pre_norm_accum`, `llama_set_mtp_prefill_kv_only`, `load_private_head`, and the `mxxm.tensor_mirror_output` key consumer exist and behave as used. Those live in src/ and belong to other slices.
- The threading in DFlash async replay was reviewed by reading only. No race test was run.
