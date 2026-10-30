# gfx906 MMQ tile traversal experiment

The staged port is based on upstream 418f905a4efe910d2a75636bbe65d48eca38790a with staged runtime tree 12229543ad2d128b4663a2f9839401dfb7bec6cb. The experiment is the separate unstaged `source.diff` in the catchup worktree; the index and patch kit were not changed.

The HIP gfx906 non-stream-K MMQ kernel uses row tiles as the inner grid dimension, restoring the accepted tree's traversal. Host grid dimensions, device block indices, and the gridDim.y limit are swapped consistently. Other GPU architectures keep upstream's column-first traversal. Stream-K work partitioning is unchanged. This is a performance experiment, not a claim of broad backend validation.

Build: `source /home/raistlin/amd-clean/env/therock.sh && cmake --build build-catchup --target llama-cli -j 8` from the catchup worktree. See `build.log` and `source.diff`. The build passed. `run.sh` records both commands, environment, binary hashes, dynamic library paths and hashes, raw output, status and time. Its two GPU runs were capped at 55 seconds each. The earlier `sandbox-attempt/` failed before inference because ROCm devices were not visible; the subsequent run had GPU access.

| Metric | Accepted baseline | Traversal candidate |
| --- | ---: | ---: |
| Prompt tokens/s | 347.97 | 351.30 |
| Generation tokens/s | 36.77 | 39.51 |
| Wall seconds including loading | 34.29 | 35.43 |
| Draft acceptance | 73/107 | 73/107 |

Both exited 0 with 2097 prompt and 128 generated tokens. Normalized generated text matched exactly, SHA-256 `7c9cebe84420a20e67a9bca2bcca42e04e9b0498a3fa2c705fff16290cdfed8f`. Both commands used the same model, draft sidecar, input, environment, and options. The earlier separate comparison reported 247.31 prompt tokens/s for the unmodified port; it was not an unmodified-versus-fixed pair in this session. This single matched pair shows restored throughput for this workload and strongly implicates traversal, but does not establish performance or correctness across other workloads.
