# P3: late rank before each AllReduce on 27B TG (2026-11-10)

Kind: profile, config sweep and one small code experiment. Result: no change kept, `-ts 1/1` stays the TG default.

- Source: `0f7ce1b04`, branch `exp-p3-work`. A: `build-catchup-20261110-151625`. B: `build-exp-p3` (parallel token-graph launch, reverted).
- Model: Qwen3.8-27B Q8_0 rev `4ca720788d1e`, `-ngl 999 -dev ROCm0/ROCm1 -sm tensor -fa on -b 2048 -ub 2048 -t 20`.
- Trace: the Q1 `rocprofv3 --kernel-trace` of A (TG64, [exp-q1](../exp-q1/README.md)), plus new traces of B and of A at `-ts 1.05/1` ([prof_ts.sh](prof_ts.sh)). Per-AR analysis: [seg.py](seg.py), [tok.py](tok.py), [tstart.py](tstart.py).

## Attribution (profiled, per token)

ROCm0 spends 3.90 ms in `ggml_cuda_ar_kernel`, ROCm1 1.64 ms. The AR start skew (ROCm1 - ROCm0) per AR call has two parts:

| Part | ms/token | Cause |
| --- | ---: | --- |
| Steady skew: median 14 us before each attention/SSM AR, 3 us before each FFN AR | 1.1 | ROCm1 runs each small kernel about 0.6 us longer (4.8 vs 4.2 us for `quantize_q8_1`, `rms_norm`, `cpy`, `get_rows`, ...): 0.74 ms/token over all non-mat-vec kernels. Mat-vec is only 1.1% slower on ROCm1 (19.90 vs 19.67 ms) |
| Tail: about 2 ARs per token with skew above 100 us | 1.2 | ROCm1 starts each token about 1 ms after ROCm0 (median 993 us) under the profiler |

The tail is a profiler artifact. The token graph replay launches lane 0 and then lane 1 on one thread (`ggml-backend-meta.cpp`, replay loop). Under rocprofv3 one `hipGraphLaunch` takes about 1 ms, so ROCm1 starts late. Without the profiler, timing that call gives 19 us (lane 0) and 18 us (lane 1).

So the real ROCm1 lag is the steady part, about 1.1 ms/token: mostly per-kernel time on the MI50 (lower power cap, 170 W vs 210 W; bus 87:00 vs 03:00), and little mat-vec bandwidth. An uneven split does not fix per-kernel time.

## Tried: parallel token-graph launch

Lanes 1..n-1 launch their token graph on the existing dispatcher worker (new `launch`/`exec` job fields), lane 0 on the caller. Under the profiler the token-start skew fell from 993 to 177 us and ms/token from 31.49 to 30.19. Without the profiler there is nothing to gain (18 us per launch): A/B r3 TG128 37.10 +- 0.25 vs 36.97 +- 0.04, PP2048 704.3 vs 700.5. Reverted.

## -ts sweep (A build, llama-bench r3, GPUs cooled below 55 C edge before each run)

| `-ts` | PP2048 t/s | TG128 t/s | TG vs 1/1 | PP vs 1/1 |
| --- | ---: | ---: | ---: | ---: |
| 1/1 (3 runs) | 701.8 / 699.6 / 699.9 (+- 0.5-1.7) | 37.20 / 37.25 / 37.32 (+- 0.06-0.13) | - | - |
| 1.03/1 | 709.6 +- 0.9 | 36.84 +- 0.08 | -1.2% | +1.3% |
| 1.05/1 | 709.5 +- 1.4 | 36.72 +- 0.20 | -1.4% | +1.3% |
| 1.1/1 | 714.0 +- 1.7 | 36.36 +- 0.11 | -2.4% | +2.0% |
| 1.15/1 | 716.9 +- 0.7 | 36.03 +- 0.17 | -3.3% | +2.4% |
| 1/1.03 | 675.7 +- 1.0 | 36.01 +- 0.04 | -3.4% | -3.5% |

The first sweep without cool-down ran back to back and fell to 31.7 TG by the fifth config (GPU heat, or load from other agents). Its numbers are not valid. Cool the GPUs between configs in long sweeps.

Rows are split by `tensor_split` in `llama_model` split state and floored to the granularity (whole heads for attention and SSM, 128 for FFN). The last lane gets the remainder, and `rotation` alternates that lane per layer. The ratio is honored. Under the profiler, 1.05/1 balances the AR (21.1 vs 20.7 us per call) and gives 30.98 ms/token against 31.49. Without the profiler, every non-1/1 ratio loses TG in both directions. The cause is not found: AR balance is not the limit.

Greedy (64 tokens, two prompts): 1.05/1 gives the same text as 1/1 on prompt 1. On prompt 2 the text forks after about 40 tokens, and both texts are coherent (normal for a different reduction split).

## Verdict

P3 for 27B is validated wrong as a TG lever. The real late-rank cost is about 1.1 ms/token of per-kernel time on ROCm1. An uneven split cannot remove it, and the AR wait in profiles is inflated by the profiler's slow graph launch. Keep `-ts 1/1`. `-ts 1.1/1` gives +2% PP for -2.4% TG, so use it only for prompt-heavy work. A ratio from per-device bandwidth would not help TG. The remaining routes are fewer kernels per attention/SSM segment (fusion) and AR-LL.
