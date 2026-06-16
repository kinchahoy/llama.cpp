# gfx906 Q4_K experiment: short

Delta: Q4_K gfx906: precompute scale/min metadata in the DP4A LDS tile

- Alternating runs: 3
- Maximum allowed regression: 3%

```text
test          control  candidate    change   samples  gate
pp512          176.00     191.36    +8.73% 3/  3  pass
pp2048         215.01     235.14    +9.36% 3/  3  pass
tg128           18.32      19.42    +6.03% 3/  3  pass
```
