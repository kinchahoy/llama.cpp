# Exploratory verification, outside repository records

2026-11-11. No deployment config changes. All test servers stopped. Current build-play-b1 benchmark identifies dc9c4e367. Exact model paths, command arguments, raw responses, logs and runtime hashes are retained in this directory or its parent.

## Performance

Qwen used batch/ubatch 2048/2048 throughout these verification runs. Flash Next used its recorded 1024/512. Greedy sampling was supplied only to reproduce the recorded test methodology; deployment sampling was not changed.

The corrected llama-bench run used ROCm0/ROCm1 and split 1/1 (slashes are required by llama-bench), PP2048/TG128, r3, F16 KV, full offload and flash attention. The JSON metadata was checked for both devices and batch/ubatch 2048. It reports PP 706.904 +/- 0.116 t/s and TG 37.779 +/- 0.159 t/s. This is the plain benchmark, not a measurement of unified server PP. The logged plain reference is approximately PP 697-705 and TG 37.5-37.9. The invalid earlier comma-syntax single-GPU sweep is explicitly named INVALID and excluded.

Server comparisons use the existing spec_accept.py five-prompt workload, 256 greedy output tokens per prompt, requests sequential, same current build and same prompt files for both settings. One run each: exploratory regression screening, not alternating statistical A/B.

| Configuration | Context / slots | Mean TG t/s | Mean ms/verify step | Mean acceptance |
| --- | --- | ---: | ---: | ---: |
| Qwen MTP2 reference, non-unified | 16384 / 1 | 64.20 | 39.23 | 0.777 |
| Qwen MTP2 unified | 274432 / 4 | 63.44 | 39.81 | 0.777 |
| Qwen MTP2 non-unified control | 274432 / 4 | 66.48 | 38.48 | 0.795 |
| Flash Next MTP3 reference, draft ROCm1, split 52/48 | 100352 / 1 | 63.72 | 48.15 | 0.709 |
| Flash Next MTP3 unified, draft ROCm0, split 46/54 | 32768 / 4 | 31.25 | 84.82 | 0.742 |

Recorded Qwen MTP2: about 62 t/s and 40-41 ms/step. Recorded Flash Next MTP3: 62.7 t/s and 49.5 ms/step. The reference configurations reproduce those approximately. Qwen unified retains approximate generation speed. Flash Next unified does not: multiple settings differ in this comparison, and checkpoint replay is a likely contributor, not an isolated proven sole cause.

Long-prompt server PP, p1/p2/p3 respectively: Qwen reference 652/620/582 t/s; Qwen unified 593/598/577. Those individual runs are about 9.0/3.5/0.9 percent lower. The plain 706.9 benchmark does not prove unified PP is unchanged. Flash Next reference 469/488/454; unified 448/500/469 t/s. Tiny prompts p4/p5 are excluded from the long-prompt PP comparison, not averaged into a misleading PP number.

## Focused unified-KV correctness checks

Two different prompts (Nix build choices and server API documentation) were run serially, concurrently, and concurrently with slots 0/1 swapped, with no request prompt reuse, greedy sampling and 128 output tokens each. Model batch sizes stayed as above. All six responses completed for each unified model, with 128 probability rows and 128 token IDs per response. Health stayed OK. These checks do not populate the whole allocated pool.

Qwen: 768/768 returned tokens match their run's reported top token. Flash Next: 758/768 match the reported top token; in the other 10 rows the reported top token is model EOG, explicitly suppressed by ignore_eos=true. Those 10 rows cannot be fully checked with top-1 alone and are not sampling violations. No unexplained top-token mismatch was found. Raw summary non_argmax_rows includes these suppressed-EOG cases; do not treat that raw list as correctness failures.

Serial and concurrent text is not always identical. Qwen unified prompt 0 diverges at token 91 in both concurrent schedules; prompt 1 diverges at token 1 in the first concurrent wave but matches serial when slots are swapped. The non-unified control also diverges at token 91 for prompt 0 in one concurrent wave; its other three comparisons match serial. This demonstrates that serial/concurrent byte equality is not a universal invariant even without unified KV. It does not prove every unified difference harmless.

Flash Next also differs between serial and concurrent output. Its logs and source show that unsupported recurrent partial rollback is probed at startup and the server selects checkpoint restore/replay. The warning alone is not evidence of failed rollback during generation. Performance remains substantially worse than the recorded single-slot setup. No Flash Next non-unified concurrent control was run here.

Conclusion: focused greedy selection and slot-reuse checks found no unexplained greedy violation, but exact output equivalence did not pass. These are limited checks, not a general correctness or deployment signoff. DFlash was not included in this new serial/concurrent correctness and performance comparison.

## Output similarity and build audit

Qwen's Nix responses matched through token 91, then changed wording (pkgs.callPackage versus callPackage and a different description). Its server summaries covered the same topic but differed in organization; the swapped-slot server summary matched serial exactly. Flash Next was less consistent: the serial Nix response continued Nix code, while concurrent responses explained the configuration in prose. Its server summaries covered broadly similar capabilities but differed in wording and structure. These were raw completion prompts, so code continuation alone does not establish corruption; exact output equivalence was not demonstrated.

The server reports version 0.9.0-dev, build 12328, commit dc9c4e367, built with Clang 24.0.0. ldd under the TheRock environment resolved llama and ggml libraries from build-play-b1/bin. This is the newer optimized build, not the exact historical F2/P4 build. The historical single-slot Flash Next settings reproduced about 63.7 t/s versus recorded 62.7 t/s on this runtime. The slower four-slot comparison changed context (100352 to 32768), draft placement (ROCm1 to ROCm0), target split (52/48 to 46/54), slot count and unified KV. Batch/ubatch stayed 1024/512. Unified KV alone was not isolated as the cause of the slowdown. Checkpoint replay is a likely contributor, not a proven sole cause.

## Artifact location and stopping point

At the user's request, this complete exploratory artifact directory was moved from /tmp/vr-unverified-2026-11-11 to vr/investigation-notes/2026-11-11-unified-kv-unverified. Historical absolute paths inside reproduction scripts refer to the original temporary location or the earlier archive location; inspect and update output paths before rerunning. Nothing in this directory is an official performance record or deployment recommendation. No deployment configuration was modified. Tests are finished and all test subprocesses have exited. No further work is scheduled.
