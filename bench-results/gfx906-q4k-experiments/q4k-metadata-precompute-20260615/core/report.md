# gfx906 Q4_K experiment: core

Delta: Q4_K gfx906: precompute scale/min metadata in the DP4A LDS tile

- Run ID: `q4k-metadata-precompute-20260615`
- Device: `ROCm0`
- Test: `pp8192`
- Alternating runs: 3
- Warmup: disabled
- Gate: at least `+5%`

```text
test          control  candidate    change   samples  gate
pp8192         198.19     220.03   +11.02% 3/  3  pass
```
