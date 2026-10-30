> Historical record. Commands and recommendations may be obsolete; see [the current guide](../../README.md).

# Three-slot validation and context limits

Updated 2026-10-12. Target: cached Qwen3.8-27B Q8_0, embedded MTP2 or cached Qwen3.8-27B DFlash2 Q8_0 with five draft tokens. Two gfx906 GPUs, 32,752 MiB each. The fixes are local changes on `a39e959d9c362d66c3702aa582e5f51febdab757`.

Runtime identity: the server executable is dynamically linked, so its unchanged hash alone does not identify the fix. Fixed `libllama.so` SHA256: `50b0872a57b31659926f13c233824b94f9475a7114b709b905dc43bcdab3dda8`; fixed `libllama-common.so`: `8a4945294a1ad1a31dfb39d6b871b6d8b8e42abeb4e1df8cce94d76e88009bd3`. The runtime tree is pinned by the [patch kit](patches/llama-head-08ffba21e/README.md).

## Fixes

- Persistent layer-input taps now allocate enough rows for the rounded graph reservation. Three slots with ubatch 2048 reserve 2049 rows; allocating only 2048 caused the startup assertion.
- DFlash prompt state is per sequence. Construction/reset marks that sequence as processing a prompt; `begin(seq_id)` ends its prompt phase. Verification rows are processed eagerly even at large batch widths. One slot's prompt can no longer defer another slot's rejected verification rows past rollback and replay them into the draft cache.

Default persistent taps, prompt-tail deferral and batch/ubatch 2048 are enabled in the fixed-build tests. The old `-ub 2046`, `LLAMA_TAP_STATIC=0` and `LLAMA_DFLASH_DEFER_TAIL=0` workarounds are unnecessary for these tested cases. The [pre-fix report](notes/2026-10-12-three-slots-before-fix.md) retains the controlled reproduction and source trace.

## Capacity

Keep the target split at 50/50. `-c` is total capacity; with three slots each gets `-c / 3`, including prompt and output. All checks use text only, F16 K/V, flash attention, full GPU offload, warmup enabled and `GGML_CUDA_TP_OVERLAP` unset.

| Mode | Total context | Per slot | Peak free GPU0 / GPU1, MiB | Evidence |
| --- | ---: | ---: | --- | --- |
| Plain | 479232 | 159744 | 515.6 / 522.9 | Pre-fix baseline, two waves passed |
| MTP2 | 354816 | 118272 | 567.6 / 554.5 | Fixed build, two waves passed |
| DFlash5 | 242688 | 80896 | 566.2 / 7582.8 | Fixed build, two waves passed |

DFlash capacity is limited by GPU0, which also holds the single-GPU draft and private head. Total context 242688 leaves about 566 MiB on the limiting GPU under this test. Spare memory on GPU1 is expected with a fixed equal split.

The target has 16 full attention layers, four KV heads, 256 dimensions per head and F16 K/V: 65,536 KV bytes per total context token across both GPUs. Compute reservations, recurrent snapshots and speculative caches add to that. Measured plain allocation growth is about 38.2 kB per total token per GPU. MTP expands compute buffers under concurrent verification, so startup memory alone overestimates capacity. DFlash also uses substantial pinned host tap storage, about 100 KiB per total token.

These are aggressive allocation checks, not proof that three fully populated maximum contexts fit every runtime allocation. The longest real input was 11,728 tokens. Vision, quantized KV, Flash-Next and other target quants/models were not checked.

## Concurrent regression method

The [harness](three-slots/check.py) starts one server, requests explicit slots 0/1/2 concurrently, polls slot activity and VRAM, retains raw responses/logs, and shuts down its server.

- Wave 0: build-document summary (11728 input tokens), code review (4617) and code generation (19), each with 256 outputs. This mixes prefill, generation and speculative rollback.
- Wave 1: reuse all slots for new short prompts (19/20/19 inputs), each with 128 outputs.
- Greedy, seed 42, `ignore_eos=true`, no request prompt reuse; server RAM prompt caching remains enabled.
- Compare returned tokens against reported target top candidates, excluding EOG IDs. This checks greedy selection against each run's logits, not identity with serial decode.

Fixed DFlash at total context 242688 passed all six requests, with three slots observed active, accepted drafts in every slot, zero decode-initialization errors/assertions and zero non-argmax returned tokens. Wave times were 42.83 / 8.41 seconds. A lower-capacity check at 230400 also passed (43.21 / 8.44 seconds; 1080.5 / 8033.4 MiB peak free). Logs confirm five persistent tap tensors with 2049 rows.

Fixed MTP2 at total context 354816 passed all six requests, with three active slots observed, drafts accepted in every slot, zero decode-initialization errors/assertions and zero non-argmax tokens among all 1152 returned probability entries. Wave times were 44.18 / 9.95 seconds; acceptance totals were 447/629 and 226/308. No non-finite draft fallback appeared.

The fixed-build run already underway with the historical uneven split (48.25/51.75, context 338688) also passed: wave times 45.16 / 8.64 seconds, zero decode errors. It is retained as supplemental evidence; current commands and capacity work hold split at 50/50.

At equal context 196608, the earlier baseline waves took plain 38.37/7.75 seconds, MTP 44.06/9.89 and pre-fix DFlash 44.16/8.55 with static taps disabled. These are whole-wave times, not isolated TG. Plain was fastest on this concurrent workload. Single-request speculative speedups do not predict three-slot throughput. Verification can exceed the optimized eight-row matvec path; attribution needs profiling.

MTP's `--spec-draft-backend-sampling` falls back to CPU for this TP setup. Earlier plain responses returned 1150 probability entries for 1152 outputs; speculative checks returned all 1152. No non-argmax returned token was observed. A separate large-context pre-fix run had a roughly 46-second synchronous RAM prompt-cache save delay; its allocation/readback cause remains unassigned.

## Commands

Run from the repository root, one server at a time. Rebuild the fixed source first using the [quickstart](../2026-10-29/prior-guides/QUICKSTART.md).

```sh
# Plain: 159744 tokens per slot
./build-ilp/bin/llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline --no-mmproj \
  -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -ctk f16 -ctv f16 \
  -np 3 -c 479232 -b 2048 -ub 2048 --spec-type none

# MTP2: 118272 tokens per slot
./build-ilp/bin/llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline --no-mmproj \
  -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -ctk f16 -ctv f16 \
  -np 3 -c 354816 -b 2048 -ub 2048 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling

# DFlash5: validated equal-split setting, 80896 tokens per slot
./build-ilp/bin/llama-server \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline --no-mmproj \
  -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -ctk f16 -ctv f16 \
  -np 3 -c 242688 -b 2048 -ub 2048 \
  --spec-type draft-dflash --spec-draft-n-max 5 \
  --spec-draft-hf z-lab/Qwen3.8-27B-DFlash2-GGUF:Q8_0 \
  --spec-draft-model Qwen3.8-27B-DFlash2-Q8_0.gguf
```

Raw commands, model revisions, logs, probability checks and VRAM peaks: [results directory](three-slots). No model downloads were needed. Tensor-split fitting is unimplemented; these capacities were chosen manually.
