> Historical record. Commands and recommendations may be obsolete; see [the current guide](../../../README.md).

| model                          |       size |     params | backend    | ngl | n_ubatch |     sm |  fa | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -------: | -----: | --: | ------------ | --------------: | -------------------: |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       | 999 |     2048 | tensor |   1 | ROCm0/ROCm1  |          pp2048 |        545.24 ± 2.60 |
| qwen35 27B Q8_0                |  27.04 GiB |    27.32 B | ROCm       | 999 |     2048 | tensor |   1 | ROCm0/ROCm1  |           tg128 |         30.17 ± 0.12 |

build: 5d6e2c412 (11547)
