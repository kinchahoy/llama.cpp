# Benchmark scripts

Scripts used for the [2026-10-29 rebuild](../old-ignore/2026-10-29/rebuild/README.md) comparisons. Model paths point at the local Hugging Face cache. Each run needs explicit OK first, as described in [PERFORMANCE.md](../PERFORMANCE.md).

| Script | Use |
| --- | --- |
| `configure_stage.sh SRC_DIR` | Configure and build `SRC_DIR/build-rb` with the build-ilp flags (`-mllvm -amdgpu-sched-strategy=max-ilp`, CXX `-O3`). Use it for every build that is compared |
| `bench_stage.sh LABEL BUILD_DIR OUT_DIR` | llama-bench PP2048/TG128, r2, on 27B Q8_0 and Flash Next IQ3_XXS; writes JSON and the binary hashes |
| `fn_reps.sh LABEL BUILD_DIR OUT_DIR` | Flash Next only, r5, for its noisier PP |
| `output_check.sh LABEL BUILD_DIR OUT_DIR` | Greedy 96-token `llama-completion` on both models; compare the text files across builds |
| `mtp_check.sh LABEL BUILD_DIR OUT_DIR` | One `llama-cli` run each of plain, MTP2 and MTP3 on both models; speeds are in the `[ Prompt ... Generation ... ]` line of each output |

`llama-bench` takes `-dev ROCm0/ROCm1`, while `llama-cli` and `llama-completion` take `-dev ROCm0,ROCm1`.
