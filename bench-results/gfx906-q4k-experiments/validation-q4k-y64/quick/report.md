# gfx906 Q4_K experiment: quick

Delta: Q4_K gfx906: mmq_y 128->64; keep mmq_x=64 and four waves to halve per-thread accumulators and reduce scratch/LDS pressure

- Run ID: `validation-q4k-y64`
- Device: `ROCm0`
- Test: `pp512`
- Repetitions: 1
- Warmup: disabled
- Gate: at least `+5%`

```text
test          control  candidate    change  gate
pp512          191.80     187.47    -2.26%  FAIL
```
