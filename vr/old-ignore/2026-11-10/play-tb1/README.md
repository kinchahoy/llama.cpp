# play-tb1: single-row bitonic top-k on HIP (DROPPED)

`test-backend-ops perf -o TOP_K -b ROCm0`, 1 row, k=512, gfx906. Radix is the baseline (threshold 0), bitonic is the 276834fdb path (threshold 16384). TOP_K correctness: 677/677 pass.

| ncols | radix us | bitonic us |
|---|---|---|
| 2112 | 71.3 | 130.9 |
| 4096 | 78.7 | 168.1 |
| 8256 | 71.1 | 624.9 |
| 8192 | 74.3 | 396.0 |
| 12288 | 87.5 | 743.4 |
| 16384 | 87.3 | 882.6 |

Bitonic loses 1.8x to 12x at every real QSA width, so there is no crossover. The 12 radix launches cost about 70-90 us per call. Verdict: drop the code change. The model A/B was not run. The QSA-shaped test cases are kept.

Archived code: [bitonic-one-row.patch](bitonic-one-row.patch) (the tried change against `0ca731b29`; reverted on the branch). Next approach worth trying: a fused single-block radix top-k for one row (keeps the radix work, removes the 12 launches).
