> Historical record. Commands and recommendations may be obsolete; see [the current guide](../../../README.md).

# Three-slot context and speculative decoding check

Historical record. Earlier recommendations and task deadlines describe the recorded tree; use the [current vr guide](../../../README.md) for operating instructions. Exact pre-cleanup bytes are preserved in `before-cleanup.tar.gz`.

Date: 2026-10-12. Source: `a39e959d9c362d66c3702aa582e5f51febdab757`, clean tracked source. The existing `build-ilp/bin/llama-server` was dated October 11, older than the speculative changes, so it was rebuilt before validation. Rebuilt server SHA256: `cfb6ab4b3f40150f13dccd06dd028392d406babd8e7253aa17e2154d7ffa2232`.

Scope: the Qwen3.8-27B Q8_0 target in the current vr notes, plain decoding, its embedded MTP head at depth 2, and the cached Qwen3.8-27B DFlash2 Q8_0 draft at depth 5. These are three decoding modes of the same target, not separate target models. Flash-Next, other quants, vision and quantized KV are outside this check.

Both GPUs report 32,752 MiB VRAM. All runs use full GPU offload, tensor parallelism, F16 K/V, flash attention, three slots, separate per-slot KV streams, greedy sampling, seed 42, warmup enabled and text only (`--no-mmproj`). `GGML_CUDA_TP_OVERLAP` is unset. No runtime source was changed.

Raw commands, logs, responses, VRAM peaks and slot observations are in [results/2026-10-12-three-slots](../three-slots). The local harness [check.py](../three-slots/check.py) starts one server at a time, sends requests concurrently to explicit slots 0/1/2 and shuts down its own server afterward.

## Capacity method

`-c` is total capacity across the three slots. The per-slot capacity is `-c / 3`. Context includes the prompt and generated tokens.

The target has 16 attention layers, four KV heads, 256 dimensions per K/V head and two bytes per F16 element: 65,536 bytes per total context token across both GPUs. With an equal split, each GPU spends 32,768 bytes per total context token on KV. Compute reservations, recurrent snapshots, MTP caches and DFlash private weights must also fit.

Plain decoding at `-c 196608` used 21,814/21,808 MiB idle; at `-c 491520` it used 32,571/32,564 MiB, leaving only about 180 MiB before request allocations. The latter is too aggressive for a 500 MiB reserve. The measured slope including the reserved compute arena is about 38.2 kB per total context token per GPU.

MTP dynamically expands its draft compute arena during concurrent verification. Its peak at `-c 196608` was 25,822/25,814 MiB, about 1,750 MiB above startup. Sizing only from startup would overestimate capacity.

DFlash puts its draft, private embedding/output weights and draft compute arena on GPU0. An equal target split wastes substantial spare memory on GPU1. An uneven split can recover capacity, but KV head and recurrent-state granularity make its memory use discontinuous; a proportional-weight estimate is insufficient. The balanced settings below were measured directly. DFlash also allocates five target-layer tap regions in pinned host RAM: 100 KiB per total context token, about 32 GiB at the largest setting here.

These are aggressive allocation limits validated with the requests below. They are not proof that three fully populated maximum-length prompts fit all future runtime allocations or retain acceptable speed. The longest populated prompt in this check is 11,728 tokens.

## Concurrent checks

Wave 0 sends three different prompts concurrently: build-document summary (11,728 tokens), CUDA review (4,617 tokens), and Python code generation (19 tokens), each requesting 256 generated tokens. This mixes ongoing prefill with generation and exercises prompt tails, speculative rejection and recurrent rollback. Wave 1 reuses all slots for new 19/20/19-token prompts, each requesting 128 generated tokens. Prompt reuse is disabled in requests; the server's default RAM prompt cache remains enabled. Slot polling confirms that all three are processing at once.

Logprob checks compare every returned token against the reported target top candidates, excluding the model's EOG IDs because `ignore_eos=true`. No non-finite or non-argmax returned token was observed in the completed checks. Plain responses expose only 1,150 probability entries for 1,152 generated tokens; conclusions about logprobs cover the returned entries. This does not establish numerical identity with serial decoding or eliminate cache bugs.

At equal `-c 196608`, the first wave took 38.37 s plain, 44.06 s MTP and 44.16 s DFlash with `LLAMA_TAP_STATIC=0`. The second wave took 7.75 s, 9.89 s and 8.55 s respectively. These are whole-wave wall times, including prompt work and scheduling, not isolated decode throughput. The earlier single-slot speculative speedups do not carry over to this workload.

MTP accepted 447/629 draft tokens in wave 0 and 226/308 in wave 1. Every slot accepted drafts in both waves. Backend draft sampling falls back to CPU on all three sequences with this tensor-parallel configuration; the requested sampling flag does not establish GPU sampling.

## Findings

1. **DFlash default startup fails for three slots and ubatch 2048.** Persistent target-layer tap tensors have 2048 rows (`src/llama-context.cpp:875`), while `graph_reserve` rounds the reservation to a multiple of three, 2049 (`src/llama-context.cpp:3355`). The resulting graph trips `GGML_ASSERT(t->ne[1] <= dev->ne[1])` at `src/llama-graph.cpp:1578`. `-b 2046 -ub 2046` preserves persistent taps and avoids this assertion. `LLAMA_TAP_STATIC=0` also avoids it. A source fix should size the persistent tap tensors for the rounded reservation.
2. **DFlash has recoverable draft-cache position errors during mixed prefill/generation.** This occurs with both the static-tap workaround and persistent taps plus batch 2046. For example, sequence 2 has draft cache maximum 24 but the next draft starts at 20; sequence 0 has maximum 11735 but starts at 11732. The M-RoPE input check rejects the batch and `draft()` returns without proposals (`common/speculative.cpp:1737`). Requests finish and acceptance resumes, but this is not a clean concurrency pass. The source-level cause is shared prompt-tail state; see the follow-up below.
3. **Large DFlash context causes a long prompt-cache save delay in one run.** At `-c 335616 -ts 49,51`, the second wave took 54.00 s despite roughly 5-6 s of reported generation time. Logs show a roughly 46 s gap while saving the previous slot to the RAM prompt cache. It completed; this was not a persistent deadlock. The final balanced run did not reproduce this delay (8.67 s second wave). Cause is unproven.
4. **MTP passed the tested three-slot execution and reuse cases.** No draft-cache position errors, non-finite draft fallback or assertion was observed. Both speculative modes can submit more than eight target verification rows with three active slots. The tuned repacked narrow matvec admits at most eight rows (`ggml/src/ggml-cuda/q8_repack/repack-common.cuh:19`); nine rows for MTP2 and up to eighteen for DFlash5 can leave that path. This is a plausible contributor to the concurrent slowdown, not a measured attribution.

The vr quickstart's speculative commands and speed claims are single-slot results. They should not be read as three-slot guidance.

## Final capacity results

| Mode | Total `-c` | Context per slot | Target `-ts` | Batch / ubatch | Peak free GPU0 / GPU1, MiB | Result |
| --- | ---: | ---: | --- | --- | --- | --- |
| Plain | 479232 | 159744 | equal | 2048 / 2048 | 515.6 / 522.9 | Passed both waves |
| MTP2 | 354816 | 118272 | equal | 2048 / 2048 | 567.9 / 554.9 | Passed both waves, accepted drafts in every slot |
| DFlash5 | 338688 | 112896 | 48.25 / 51.75 | 2046 / 2046 | 437.9 / 598.9 | Completed both waves, two recoverable draft-cache errors |

DFlash's limiting GPU is slightly below the nominal 500 MiB reserve. This table is a measured aggressive configuration, not a production recommendation for DFlash while its concurrency bugs remain. MTP and plain are the clean tested choices; plain is faster on these concurrent prompts and provides more capacity.

Final wave wall times (seconds), with the different capacity/split settings above:

| Mode | Mixed long/short prompts, 3 x 256 outputs | New short prompts, 3 x 128 outputs |
| --- | ---: | ---: |
| Plain | 38.43 | 7.76 |
| MTP2 | 44.10 | 10.00 |
| DFlash5 | 44.45 | 8.67 |

All final requests returned their requested token counts. Three active slots were observed in both speculative modes, and all slots were idle afterward. Final MTP accepted 447/629 and 226/308 draft tokens in the two waves; final DFlash accepted 538/1096 and 284/472. There were zero non-argmax tokens among the 1150/1152/1152 returned probability entries for plain/MTP/DFlash respectively. Verification-row kernel rounding means exact text identity between modes is not required by this check.

Copyable commands from the repository root (run one server at a time):

```sh
# Plain, 159744 tokens per slot
./build-ilp/bin/llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline --no-mmproj \
  -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -ctk f16 -ctv f16 \
  -np 3 -c 479232 -b 2048 -ub 2048 --spec-type none

# MTP2, 118272 tokens per slot
./build-ilp/bin/llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline --no-mmproj \
  -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -ctk f16 -ctv f16 \
  -np 3 -c 354816 -b 2048 -ub 2048 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling

# DFlash5 test configuration; concurrency errors remain
./build-ilp/bin/llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline --no-mmproj \
  -ngl 999 -dev ROCm0,ROCm1 -sm tensor -ts 48.25,51.75 -fa on -ctk f16 -ctv f16 \
  -np 3 -c 338688 -b 2046 -ub 2046 \
  --spec-type draft-dflash --spec-draft-n-max 5 \
  --spec-draft-hf z-lab/Qwen3.8-27B-DFlash2-GGUF:Q8_0 \
  --spec-draft-model Qwen3.8-27B-DFlash2-Q8_0.gguf
```

The commands retain the tested MTP backend sampling request, which falls back to CPU here. The model selector resolves the cached revision listed in the raw startup logs; no models were downloaded. The server may attempt automatic fitting at startup, but its logs explicitly say fitting is unimplemented for tensor split; these capacities were chosen manually.

## Source review follow-up: DFlash position errors

`common_speculative_impl_draft_dflash::in_prompt` is one boolean shared across all sequences (`common/speculative.cpp:1111`). `begin(seq_id, ...)` clears it globally (line 1281); processing a large chunk for any sequence sets it globally (line 1522). A short chunk is then treated as prefill whenever that global flag is set (line 1521), even when it contains verification tokens from a different slot that is already generating.

The resulting failure crosses the rollback boundary:

1. One slot's long prompt sets `in_prompt=true`.
2. Another slot's six verification rows are classified as a prompt tail and queued in `pending` at line 1541, rather than injected immediately.
3. Target verification rejects some rows. The server trims the target and draft backend caches through `slot.mem.seq_rm(...)` (`tools/server/server-context.cpp:4600`, `common/common.cpp:1644`). That operation cannot remove deferred chunks from the speculator's `pending` vector. DFlash's `accept()` is a no-op (line 1846).
4. The next `draft()` flushes the deferred rows before proposing tokens (line 1690), inserting the rejected future positions back into the draft cache.
5. The next token-input draft starts inside those restored future positions. The M-RoPE check in `src/llama-batch.cpp:369` correctly rejects it. The observed cache maximum 24 versus next start 20 matches this sequence.

The fix belongs in prompt/decode classification and deferred-chunk lifetime. Track prompt state per sequence, change only that sequence in `begin()`/`reset()`, and ensure generation/verification rows are never deferred as prompt tails. Pending rows must not survive rollback of the positions they represent. Removing the M-RoPE check or blindly trimming KV immediately before drafting would hide the invalid replay.

The existing `LLAMA_DFLASH_DEFER_TAIL=0` switch bypasses this misclassification while retaining large-chunk prompt deferral. A controlled comparison used batch/ubatch 2046, total context 196608, equal target split and identical concurrent prompts. Both runs completed all six requests and returned zero non-argmax tokens:

| Prompt-tail deferral | Draft-cache position errors | Wave 0 / wave 1 wall time, seconds |
| --- | ---: | --- |
| Default, enabled | 2 | 43.44 / 8.60 |
| `LLAMA_DFLASH_DEFER_TAIL=0` | 0 | 43.73 / 9.29 |

Raw files use suffixes `-tail1-control` and `-tail0-cause`. This isolates prompt-tail deferral as the trigger and supports the source trace above. It validates a workaround on these prompts; it does not fix the shared state in source or establish a new maximum-context memory limit for the workaround.

The startup assertion is independent: the static tap's row allocation is smaller than the graph's rounded reservation. Fix that allocation independently of the prompt-state bug.

The isolated 46 s prompt-cache delay remains unassigned. Source review confirms that `prompt_save()` allocates and serializes target and draft state synchronously while launching requests (`tools/server/server-context.cpp:317`, line 2822), so slow cache saving blocks subsequent slot launches. The logs do not distinguish time in allocation, checkpoint copying and backend readback; attributing the full delay to one of those needs profiling.
