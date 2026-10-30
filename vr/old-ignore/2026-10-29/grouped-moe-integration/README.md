# Grouped-MoE integration, 2026-10-29

Accepted subset: grouped IQ2_S/IQ4_NL expert weight reuse and focused tests only. GPU-feedback code is excluded. Source is checkpoint `84932540a` plus seven runtime/test files; exact trees are in [source-snapshot.json](source-snapshot.json). No commit was created.

- HIP Release build passed for `llama-cli`, `llama-bench`, `test-backend-ops`, `vr-test-gfx906-tp`, and `vr-test-flashnext-moe` with the existing TheRock ILP configuration.
- Patch replay verified 121 manifest files with zero mismatches. Patch checksums passed, and the runtime files match the exported target tree.
- With `GGML_CUDA_FLASHNEXT_MOE=1`, the 18 focused canonical IQ2_S/IQ4_NL CPU-reference cases passed on ROCm0. The repacked harness then did not finish within the shared 110-second timeout (exit 124); TP checks later in that command were not reached. This does not establish whether the repacked kernel or test setup dominated the time. No extended retry was run.
- No new performance or model-output comparison was run. Earlier passing repacked cases and CLI evidence remain in the [original experiment](../../2026-10-25/flashnext-fast/README.md).

Command after sourcing the TheRock environment:

```sh
GGML_CUDA_FLASHNEXT_MOE=1 timeout --kill-after=5s 110s bash -eu -c '
  unset GGML_CUDA_TP_OVERLAP
  ./build-ilp/bin/test-backend-ops test -b ROCm0 -o MUL_MAT_ID -p "type_a=(iq2_s|iq4_nl).*n_mats=16,n_used=10"
  ./build-ilp/bin/vr-test-flashnext-moe
  ./build-ilp/bin/vr-test-gfx906-tp --n 1 --check
  ./build-ilp/bin/vr-test-gfx906-tp --n 2048 --check
'
```

The sandbox could not see ROCm devices, so the actual GPU checks ran with device access outside it. The build initially hit a read-only default compiler cache and succeeded with `CCACHE_DIR=/tmp/mx-grouped-ccache` and `CCACHE_TEMPDIR=/tmp/mx-grouped-ccache-tmp`. Pre-build binaries/libraries and CMake cache were preserved at `/tmp/mx-before-grouped-moe/`; that temporary directory is not a durable archive. See the [current guide](../../../README.md).
