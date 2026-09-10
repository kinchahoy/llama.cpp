# Result records

`2026-08-13-gfx906-q8.csv` is the compact performance ledger for the exact Q8_0 tensor-parallel overlap experiment on two MI60 GPUs. PP is PP2048 and TG is TG1024, both measured after a depth-16384 fill with batch/ubatch 2048/2048, one repetition, and no warmup.

`score` is `TG + PP/10`. `scenario_seconds` is `16384/PP + 1024/TG`. Blank cells mean that arm did not produce the corresponding full-model measurement.

Commands, precision caveats, model hashes, build identities, correctness checks, and raw artifact locations are in [../RESULTS.md](../RESULTS.md).
