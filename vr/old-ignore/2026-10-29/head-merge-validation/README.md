# Official-head merge validation, 2026-10-29

Official head: `418f905a4efe910d2a75636bbe65d48eca38790a`. Candidate branch `experiment/catchup-418f905a4`, runtime tree `12229543ad2d128b4663a2f9839401dfb7bec6cb`. Parent baseline build reports `84932540a`; its exact accepted-grouped source snapshot is `fbdb024841f459a285357de1163c9b34f4c29c4e` (before the dead HIP branch removal).

The cleaned kit applied with source/API conflicts, which were reconciled before building. The HIP Release build passed for CLI, bench, server, backend-op test and focused TP harness. Canonical grouped IQ2_S/IQ4_NL cases passed 18/18 on ROCm0 in 1.82 seconds. No broad suite was run.

## Measurements

| Workload | Baseline | Candidate | Change |
| --- | --- | --- | --- |
| Qwen3.8-27B Q8_0, PP2048/TG128 | 690.256 / 36.802 tok/s | Not run | Supplemental baseline only |
| Flash Next IQ3_XXS, grouped MTP2, 128 outputs | 345.09 / 37.68 tok/s | 247.31 / 40.22 tok/s | PP -28.3%; TG +6.7% |

The Flash Next comparison used one run per binary, with explicit `-lm dio` on both. Each run exited 0 in 41.54 seconds (baseline) and 38.04 seconds (candidate). Both processed 2,097 prompt tokens and generated 128 tokens. Draft acceptance matched at 73/107 (0.68224), mean accepted length 2.35. The cleaned generated text matched exactly (1,066 characters; SHA256 `ea8578dc340498afa10fbfe1a63995df7c5ffffc3125f4b4e6aa9bec8d13af6c`). This is a single pair and the candidate is not promoted. The earlier 40-second pre-DIO attempt remains archived but is superseded.

The candidate has a substantial PP regression in this single comparison despite matching output and acceptance. Release/HIP build and compiler settings match, as do CPU thread count, MTP preparation and tap extraction paths, and slab overlap source. Draft preparation was 40.158 ms candidate vs 44.746 ms baseline, so it does not account for the prompt delta. No cause is established. Further profiling needs user approval; keep the candidate experimental.

Measured GPU-check wall time: 68.575 seconds for the initial Q8/failed Next checks plus 79.58 seconds for the matched Next pair. The additional grouped op check took 1.82 seconds. No broad suite was run.

## Reproduction and identity

- [Q8 baseline command](baseline-command.txt), [environment](baseline-env.txt), [result](baseline-result.txt), [raw JSON](baseline.stdout).
- [Matched Flash Next commands](flashnext-dio-commands.txt), [environment](flashnext-dio-env.txt), [run status](flashnext-dio-status.txt), [prompt hash](flashnext-dio-prompt.sha256), [normalized generated text](flashnext-dio-baseline.generated.txt) and [candidate text](flashnext-dio-candidate.generated.txt).
- Raw [baseline stdout](flashnext-dio-baseline.stdout), [baseline stderr](flashnext-dio-baseline.stderr), [candidate stdout](flashnext-dio-candidate.stdout), [candidate stderr](flashnext-dio-candidate.stderr), and per-run [timings](flashnext-dio-baseline.time), [candidate timing](flashnext-dio-candidate.time).
- [CLI and library hashes](flashnext-dio-binaries.sha256), [baseline loaded libraries](flashnext-dio-baseline-libraries.sha256), [candidate loaded libraries](flashnext-dio-candidate-libraries.sha256), and corresponding [ldd records](flashnext-dio-baseline-ldd.txt), [candidate ldd record](flashnext-dio-candidate-ldd.txt).
- [Source identities](source-identities.txt), [prompt hash](flashnext-prompt.sha256), [binary hashes](flashnext-binaries.sha256), and per-binary library hash files alongside this report.

The Next setup uses the cached Unsloth target revision `c8b5954a88c2775c546b92593eda40ea041d3176`, three IQ3_XXS shards, and the self-contained Q8_0 MTP sidecar. Fixed 50/50 TP, Q8 K/V, context capacity 100096, batch/ubatch 1024/512, single-GPU draft on ROCm1, grouped expert reuse on, GPU feedback off, greedy seed 42, and `-lm dio` on both binaries. The exact expanded commands and all GGML/LLAMA/HIP environment overrides are recorded above. The prompt is the repository's `flake.nix`; its hash identifies the actual bytes.

The target shards total about 82 GB on disk, not all GPU-resident. Known host PLE placement is part of fitting the two 32 GiB GPUs and 96 GB RAM; this is not a maximum-capacity test. Full target-weight hashing was interrupted to avoid disk contention; an incomplete `.sha256` file is not an identity check for all shards. Revision paths and shard sizes are recorded instead.
