# gfx906 Q4_K experiment: core

Delta: Q4_K gfx906: keep mmq_y=128/mmq_x=64/four waves, but change launch-bounds minimum resident blocks 2->1 to trade occupancy for zero spill

- Run ID: `q4k-min-blocks-1`
- Device: `ROCm0`
- Test: `pp8192`
- Repetitions: 1
- Warmup: disabled
- Gate: at least `+5%`

```text
test          control  candidate    change  gate
pp8192         243.88     161.60   -33.74%  FAIL
```
