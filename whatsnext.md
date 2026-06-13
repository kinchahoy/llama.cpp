# MI50/gfx906 validation handoff

Date: 2026-06-13

This branch contains two independent gfx906 optimizations. The Q8_0 change has
an operator-level measured win. The Q4_K change has strong compiler evidence but
still needs runtime acceptance testing.

## What changed

### Q8_0 large prefill

`ggml/src/ggml-cuda/mmq.cu` keeps dense Q8_0 batches through 256 on MMQ and uses
the existing rocBLAS path above 256 on exact Vega 20. This preserves the small
batch path and avoids the Q4_K regression caused by globally forcing rocBLAS.

At batch 512, the operator gain was 27.2% on ROCm0 and 23.9% on ROCm1.

### Q4_K prefill

`ggml/src/ggml-cuda/mmq.cuh` and `ggml/src/ggml-hip/CMakeLists.txt` use eight
wave64 waves only for the Q4_K MMQ translation unit on exact gfx906. The output
tile remains `128x64`; per-thread accumulators fall from 32 to 16 and compiled
scratch falls from 44 bytes to zero.

This should remove spill traffic while preserving tile reuse, but it is not
accepted until correctness and end-to-end benchmarks pass.

## Progressive test plan

Use `scripts/bench-gfx906-qwen36.sh` with an unchanged `origin/master` build and
a build from this branch. The script uses exact model files from these repos:

- `unsloth/Qwen3.6-27B-MTP-GGUF`
- `unsloth/Qwen3.6-27B-GGUF`

The tested quant files are Q4_0, Q4_K_M, and Q8_0. In this document, Q4_K means
the repository's Q4_K_M file.

The defaults follow `configs/server-models.ini`: all GPU layers, flash
attention, batch and microbatch 2048, and device order `ROCm1/ROCm0`. The slash
separator is required by `llama-bench`; the server config uses commas.

Build both revisions with identical settings:

```bash
scripts/build-gfx906-comparison.sh
```

The script creates a detached, persistent mainline worktree at
`build/.mainline-src`, checked out at `origin/master`. It configures mainline in
`build/mainline` and the current working tree in `build/gfx906-2026-06`, using
the same compilers and CMake options. It builds `llama-bench`,
`test-backend-ops`, `llama-cli`, and `llama-server`. The worktree and build
directories are retained for incremental rebuilds. Set `MAINLINE_REF` to compare
against another commit.

Run the quick gate. Each test has one measured repetition:

```bash
scripts/bench-gfx906-qwen36.sh quick
```

This loads each model once per binary and measures `pp32`, `pp512`, `pp2048`,
and `tg128`. The measured `pp32` case is a cheap kernel warmup; the script
disables llama-bench's full-size warmup to avoid duplicating every test. Control
and candidate runs are interleaved per model. Results are written under
`bench-results/gfx906-qwen36/quick`.

Review or rerun the gate without benchmarking:

```bash
scripts/compare-gfx906-qwen36.py \
  bench-results/gfx906-qwen36/quick/control \
  bench-results/gfx906-qwen36/quick/candidate
```

The default gate requires:

- at least 10% Q8_0 prefill gain at 512 and 2048 tokens;
- at least 5% Q4_K_M prefill gain at 512 and 2048 tokens;
- no Q4_0 prefill regression greater than 3%;
- no generation regression greater than 3% for any quant.

Override thresholds with `MIN_Q8_GAIN`, `MIN_Q4K_GAIN`, and `MAX_REGRESSION`.
Single-run data is intentionally a coarse gate, not publication-quality
statistics.

Only after the quick gate passes, run the long stage:

```bash
scripts/bench-gfx906-qwen36.sh long
```

`long` rechecks the quick gate and exits without running if it fails. It then
runs a cheap `pp32` warmup followed by one combined `pp20000+tg5000` test for
every repo and quant. A combined test avoids separate long prefill passes, but
its reported throughput is aggregate prompt-plus-generation throughput. Use the
quick results to attribute gains to prefill versus generation.

To run both stages in sequence:

```bash
scripts/bench-gfx906-qwen36.sh all
```

Model files are reused from the Hugging Face cache across binaries and stages.
Set `HF_TOKEN` if required and `RESULTS_DIR` to move the output directory. When
`rocm-sdk` is available, the script also adds the TheRock runtime library
directories to `LD_LIBRARY_PATH`. `CONTROL_BIN` and `CANDIDATE_BIN` can still
override the default comparison binaries.

## Important limitation

`llama-bench` measures the loaded base model path and does not configure MTP
speculative decoding. The MTP repository is still useful for checking model
loading and base-model kernel performance, but speculative acceptance rate and
end-to-end MTP speed require a separate `llama-server` workload with
`spec-type=draft-mtp`.

## Acceptance and fallback

Keep the Q8_0 dispatch if correctness passes and Q8_0 prefill improves without
generation regressions.

Keep the Q4_K eight-wave change only if backend correctness passes and both
Q4_K model-level prefill tests improve. If it is correct but slower, remove the
eight-wave specialization and test the measured four-wave X=40 fallback.

After an eight-wave win, the next controlled experiment is vectorized aligned
Q4_K LDS reads. Keep zero scratch as a hard constraint.

## Files

- `improvements-mi50-ideas.md`: technical rationale and rejected experiments
- `benchmark-mi50-rtx3090.md`: isolated MI50 and RTX 3090 comparison
- `gfx906-q8-rocblas-dispatch.patch`: portable Q8_0 source patch
- `gfx906-q4k-8wave.patch`: portable Q4_K source patch
- `gfx906-q8-rocblas-dispatch.README.md`: Q8_0 patch application notes
