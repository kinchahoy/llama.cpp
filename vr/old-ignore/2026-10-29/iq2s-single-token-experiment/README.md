# IQ2_S one-token gfx906 experiment

The unstaged `source.diff` adds a two-row MMVQ kernel only for gfx906, IQ2_S, expert IDs, one output column, and even output rows. The existing kernel reads both rows unconditionally, so odd rows retain the old path. The separate MMQ traversal fix remains in the worktree. The staged overlay and patch kit were not changed. This gate cannot select dense Q8_0 operations.

The candidate build passed. Three one-token IQ2_S ROCm0 tests passed against CPU reference: ordinary MUL_MAT_ID and two fused gate variants. The test took 1.17 seconds. No broader correctness claim is made.

The intended baseline was copied before the build, but I failed to verify its RPATH before starting the comparison. It loaded the rebuilt candidate's shared libraries. I stopped that invalid run after 29.54 seconds including model loading, before a candidate run started. No performance result is claimed. Total GPU-check wall time used was 30.71 seconds.

`run-double-pair.sh` prefixes each binary's own library directory and caps each run at 60 seconds plus 1 second kill grace. Read-only `baseline.corrected.ldd` and `candidate.corrected.ldd` verify the resolved paths; saved baseline HIP library SHA-256 is a4666a0593317e9c6c56a3ded3638d57b4933c59d8431fcfb3a3401af6a7b5d2 and candidate is fe34c2bf8c7e6c13f2410408d7984511a527e902321cecac0b0c50002a87e4d8. The failed baseline logs are in `invalid-baseline/`.

The user requested a doubled prompt and 256 generated tokens after the corrected shorter baseline had started. That run was stopped after 31.50 seconds; no short candidate was started. Its partial records are in `aborted-short/`. The doubled prompt concatenates `flake.nix` twice (14,492 bytes, SHA-256 `d121f276660ecead20225aea5e164b8c3a105273c6453833ac17178320241fa1`). Both longer runs used `-t 20 -tb 20`, CPU affinity 0-19, identical model and inference settings, and no MTP.

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Prompt tokens/s | 396.35 | 398.80 |
| Generation tokens/s | 37.20 | 38.08 |
| Wall time including load | 39.21 s | 41.44 s |
| Process CPU, whole run | 99% | 104% |

Both exited 0 and processed 4,143 prompt tokens and generated 256 tokens. Normalized generated text matched; the saved text files have SHA-256 `ac812967017729185d9d368e33e3105b55785a44e88d6b75fe8df8a5c11772cd`. The candidate measured 0.62% higher PP and 2.37% higher TG in a single pair; this is not enough to establish a reproducible speedup. `baseline.ldd`, `candidate.ldd`, and the library hash files record the actual resolved binaries. `baseline.time` and `candidate.time` report process CPU averaged over the entire run, including load, rather than per-core utilization. No additional GPU runs were made.

The saved `baseline-bin/` executables and libraries are local ignored build artifacts; hashes and resolved library paths are retained in Git.
