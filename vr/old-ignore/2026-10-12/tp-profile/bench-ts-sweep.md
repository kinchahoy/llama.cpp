> Historical record. Commands and recommendations may be obsolete; see [the current guide](../../../README.md).

| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          | ts           |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       | 999 |     2048 | tensor |   1 | ROCm0/ROCm1  | 50.00/50.00  |          pp2048 |        546.67 ± 0.34 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       | 999 |     2048 | tensor |   1 | ROCm0/ROCm1  | 50.00/50.00  |           tg128 |         30.22 ± 0.13 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       | 999 |     2048 | tensor |   1 | ROCm0/ROCm1  | 53.00/47.00  |          pp2048 |        579.89 ± 2.64 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       | 999 |     2048 | tensor |   1 | ROCm0/ROCm1  | 53.00/47.00  |           tg128 |         30.42 ± 0.24 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       | 999 |     2048 | tensor |   1 | ROCm0/ROCm1  | 56.00/44.00  |          pp2048 |        591.33 ± 0.71 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       | 999 |     2048 | tensor |   1 | ROCm0/ROCm1  | 56.00/44.00  |           tg128 |         30.17 ± 0.19 |

build: 5d6e2c412 (11547)
