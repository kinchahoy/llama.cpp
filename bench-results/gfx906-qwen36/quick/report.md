# gfx906 Qwen3.6 quick comparison

Generated: 2026-06-13T12:35:04-07:00

## Test setup

- Control: `/home/raistlin/infer/llama.cpp/build/mainline/bin/llama-bench`
- Candidate: `/home/raistlin/infer/llama.cpp/build/gfx906-2026-06/bin/llama-bench`
- Devices: `ROCm1/ROCm0`
- GPU layers: 99
- Flash attention: on
- Batch / microbatch: `2048 / 2048`
- Split mode: layer
- Repetitions: 1
- Built-in warmup: disabled
- Tests: `pp32`, `pp512`, `pp2048`, `tg128`
- Models: Qwen3.6-27B and Qwen3.6-27B-MTP
- Quant files: Q4_0, Q4_K_M, Q8_0
- Gate thresholds: Q8_0 +10%, Q4_K_M +5%, max regression 3%

## Comparison

```text
model                                      test                control  candidate    change  gate
base__q4_0                                 tg128                 26.07      25.99    -0.31%  pass
base__q4_0                                 pp32                  26.18      29.43   +12.42%  pass
base__q4_0                                 pp512                278.24     282.03    +1.36%  pass
base__q4_0                                 pp2048               292.85     293.52    +0.23%  pass
base__q4_k_m                               tg128                 22.95      23.12    +0.74%  pass
base__q4_k_m                               pp32                  14.83      26.59   +79.25%  info
base__q4_k_m                               pp512                225.61     203.80    -9.67%  FAIL
base__q4_k_m                               pp2048               237.24     211.09   -11.02%  FAIL
base__q8_0                                 tg128                 19.82      19.75    -0.37%  pass
base__q8_0                                 pp32                  71.92      82.45   +14.63%  info
base__q8_0                                 pp512                141.46     171.33   +21.12%  pass
base__q8_0                                 pp2048               152.26     239.25   +57.13%  pass
mtp__q4_0                                  tg128                 25.82      25.74    -0.31%  pass
mtp__q4_0                                  pp32                  24.73      29.49   +19.26%  pass
mtp__q4_0                                  pp512                283.82     282.99    -0.29%  pass
mtp__q4_0                                  pp2048               296.63     296.10    -0.18%  pass
mtp__q4_k_m                                tg128                 23.06      23.36    +1.27%  pass
mtp__q4_k_m                                pp32                  25.93      28.06    +8.19%  info
mtp__q4_k_m                                pp512                227.84     205.80    -9.67%  FAIL
mtp__q4_k_m                                pp2048               237.20     211.91   -10.66%  FAIL
mtp__q8_0                                  tg128                 19.92      20.08    +0.79%  pass
mtp__q8_0                                  pp32                  56.62      71.74   +26.70%  info
mtp__q8_0                                  pp512                131.61     165.10   +25.44%  pass
mtp__q8_0                                  pp2048               155.17     238.49   +53.70%  pass
FAIL: skip the long benchmark
```
