# Local gfx906 harness

To build and run the current version, see [QUICKSTART.md](QUICKSTART.md). To merge a new llama.cpp head, apply the [active patch kit](patches/llama-head-08ffba21e/README.md); older kits are in `OLD-IGNORE/ports/`. Start with the [benchmark workflow](#benchmark-workflow) for current CLI comparisons. Use [MERGE-GFX906.md](MERGE-GFX906.md) for porting and merge preservation, and the [optimization shortlist](2026-10-11-gfx906-optimization-ideas-q8-q4-q6.md) for unmeasured experiments. The [2026-10-12 TP profile](2026-10-12-tp-profile-findings.md) measures where PP and TG time goes on this GPU pair, and the [roofline note](2026-10-12-roofline.md) estimates the remaining headroom; its [work status log](2026-10-12-work-status.md) tracks the follow-up changes.

`CMakeLists.txt` and `test-gfx906-tp.cpp` are the active tensor-parallel test harness. Enable them with `-DLLAMA_BUILD_VR_EXPERIMENTS=ON` when configuring the root build.

The current working tree includes the [2026-10-11 upstream merge](MERGE-GFX906.md#2026-10-11-reconciliation), its build flags, validation limits, and DFlash vision follow-up. The pinned runtime patch kit and its older benchmark command are in [ports/llama-head-e358d-to-mx-20261001](../ports/llama-head-e358d-to-mx-20261001/README.md). Follow [MERGE-GFX906.md](MERGE-GFX906.md) for typical conflict resolutions that preserve the MI50/MI60 performance paths. The October 1 kit has not been regenerated for the October 11 tree. The [older pinned kit](../ports/llama-head-0bc845-to-mx-38b4/README.md) remains available. Historical experiments, discussion, results, rejected patches, and generated build files are in [OLD-IGNORE](OLD-IGNORE/ARCHIVE.md).

For future llama.cpp versions, start with the [injection commands](MERGE-GFX906.md#inject-into-a-future-upstream-head). The guide includes conflict/resume instructions, a source map of the performance paths, commands to export the next kit, and a [copyable agent handoff](MERGE-GFX906.md#copyable-agent-handoff). Apply all five patch groups and finish source reconciliation before building and measuring.

## Benchmark workflow

Run from the repository root. The trust order for this machine is:

1. A representative CLI prompt with long generation, such as `-n 4000`, is the strongest performance evidence. Use the same output limit for every arm and record the actual generated count if EOS ends a run early.
2. The CLI command below with `-n 256` is the trusted quick check.
3. `llama-bench` is faster and useful for screening and merge regression checks, but carries less confidence for the real workload. Confirm promising results with the CLI.

```sh
./build-ilp/bin/llama-cli \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 \
  -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on \
  -f flake.nix -st -b 2048 -ub 2048 -c 150000 \
  --spec-type none --spec-draft-n-max 2 \
  --temp 0.0 --top-k 1 -s 42 -n 256 --simple-io
```

`--spec-type none` disables speculation; `--spec-draft-n-max 2` has no effect in this baseline. `-c 150000` allocates context capacity; it does not populate 150000 positions. Record the actual prompt token count. A deep-context claim needs a genuinely long input.

Explore `-b 2048 -ub 2048` and `-b 4096 -ub 4096` as separate paired comparisons. Keep the prompt bytes, model, context capacity, sampling, generation limit, warmup policy and environment identical within each comparison. CLI warmup remains enabled in these commands. Keep `GGML_CUDA_TP_OVERLAP` unset when checking the normal HIP host-staged AllReduce path.

Record source/binary identity, model revision, prompt hash (`sha256sum flake.nix`), exact command and relevant environment. Retain generated text and stderr timings separately. Report PP and TG tokens/s, actual generated tokens, time to first generated token and load time separately; draft prefill can move work into the first response. For speculative runs also retain drafted/accepted counts. Repeat and alternate baseline/candidate order before calling a small difference a gain. A text match is useful but does not replace rollback or numerical correctness tests.

## MTP and DFlash2 CLI checks

MTP uses `draft-mtp`; 2 and 3 below are maximum draft-token counts, not guaranteed acceptance lengths. The model must contain compatible MTP weights or have a matching MTP sidecar. Check startup logs to confirm the draft loaded and the requested depth was not clamped.

Observed on the current Qwen3.8 Q8_0 CLI workload:

| MTP depth | Prompt | Generation |
| ---: | ---: | ---: |
| 2 | 413.2 t/s | 45.0 t/s |
| 3 | 411.4 t/s | 45.6 t/s |

These are single recorded results; rerun the paired baseline and speculative arms before treating the small depth difference as meaningful.

2026-10-12 depth sweep, same command with `-c 100000`, `-n 256` and `-devd ROCM0`. `-devd` has no effect for MTP: the MTP context is built on the target model and only a separate draft model (`--spec-draft-model`) uses the draft device list, so these runs still use the target's tensor split. Acceptance was not recorded.

| MTP depth | Prompt | Generation |
| ---: | ---: | ---: |
| 2 | 412.2 t/s | 44.3 t/s |
| 3 | 420.2 t/s | 46.4 t/s |
| 4 | 412.7 t/s | 38.6 t/s |
| 5 | 415.9 t/s | 35.0 t/s |

Depth 3 is the best single result here. Depth 4 and 5 lose TG, which suggests the extra draft steps cost more than they gain from acceptance. Record `draft acceptance ... mean len` to confirm.

```sh
for depth in 2 3; do
  ./build-ilp/bin/llama-cli \
    -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 \
    -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on \
    -f flake.nix -st -b 2048 -ub 2048 -c 150000 \
    --spec-type draft-mtp --spec-draft-n-max "$depth" \
    --temp 0.0 --top-k 1 -s 42 -n 256 --simple-io
done
```

DFlash2 uses `draft-dflash`; selector metadata in the GGUF selects the DFlash2 path. Use the [z-lab GGUF repository](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/tree/main), which provides `Qwen3.8-27B-DFlash2-Q8_0.gguf`. The [original model card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) evaluates block size 8 with seven draft tokens; start at 7, then tune against the same baseline. This is a test setting, not a measured optimum on gfx906.

```sh
./build-ilp/bin/llama-cli \
  -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 \
  -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on \
  -f flake.nix -st -b 2048 -ub 2048 -c 150000 \
  --spec-type draft-dflash \
  --spec-draft-hf z-lab/Qwen3.8-27B-DFlash2-GGUF:Q8_0 \
  --spec-draft-model Qwen3.8-27B-DFlash2-Q8_0.gguf \
  --spec-draft-n-max 7 \
  --temp 0.0 --top-k 1 -s 42 -n 256 --simple-io
```

The first DFlash2 run downloads the draft if it is not cached. Check that startup names the selected draft and enables `draft-dflash`, and compare acceptance and total response time as well as TG. Repeat the baseline, MTP 2, MTP 3 and DFlash2 with `-n 4000` for stronger evidence. These are text-only checks; [DFlash2 vision remains unvalidated](MERGE-GFX906.md#dflash2-vision-follow-up).

Observed DFlash2 result with `-c 100000`, `--spec-draft-n-max 4`, and the command above:

`[ Prompt: 372.2 t/s | Generation: 40.1 t/s ]`
