# TG host gap diagnosis (2026-10-29, baseline mx-final/build-rb, rocprofv3 kernel+HIP+memcpy trace, n=32, last 20 tokens)

Profiled TG: 27B 25.3 t/s (unprofiled 36.6), Flash Next 21.4 t/s (unprofiled 39.1). Profiler inflates host API cost, so absolute gap values are upper bounds.

## Both models: token graph IS replayed
Each token = 2 hipGraphLaunch (one per lane), no per-kernel launches. Host work per token is only: set_inputs (hipMemcpyAsync + hipStreamSynchronize rounds), 2 launches, 1 long hipStreamSynchronize.

## 27B Q8_0
- 29.1-29.4 ms/token profiled. Both-GPU-idle: 1.04 ms/token (87 gaps / 20 tokens). One gap ~0.45-0.6 ms/token right after the long sync, 3 small ones 0.03-0.08 ms.
- Idle window is hipStreamSynchronize (822 us/tok), memcpy async 50 us/tok. Sequence between the long sync end and the graph launch (~430 us): D2H logits/sampled token readback + 2 sync, then ~4 rounds of (WaitEvent, 2-4 small H2D memcpy, sync 55-77 us), then 10 small H2D memcpy (3-4 us each) and event records.
- hipGraphLaunch is host-blocking: ~1.0 ms (lane 0) then ~0.9 ms (lane 1), issued serially from one thread, so lane 1 starts ~1 ms after lane 0.

## Flash Next IQ3_XXS
- 33.6 ms/token profiled; both-GPU-idle 4.48 ms/token (250 gaps / 20 tokens), consistent with the earlier 4.3 ms.
- Per token ~70 small hipMemcpyAsync (5-8 us each) plus ~10 hipStreamSynchronize (10-140 us) before the launch, one memcpy that stalled 648 us, ~1.7 ms host time between previous token end and first graph launch. The big idle gaps seen are 5.6 ms and 2 ms (every few tokens) and 0.3 ms.
- hipGraphLaunch blocks ~2.0 ms per lane, serial: lane 1 starts ~2 ms after lane 0.
- Top idle causes (us/token inside idle windows): hipStreamSynchronize 864, hipMemcpyAsync 165, hipGraphLaunch 142.

## Experiment: parallel replay launch (not kept)
Launched lanes 1.. on the existing lane dispatcher worker threads (ggml-backend-meta.cpp replay path). Diff saved in parallel-launch-attempt.diff. Output byte-identical (cmp, both models). llama-bench TG128 (r2, one pair, BASE1/NEW1):
- 27B: 36.63 +- 0.31 -> 36.67 +- 0.17
- Flash Next: 39.14 +- 0.07 -> 38.78 +- 0.07
No gain (FN -0.4, noise level or slightly worse). In a NEW profile the launches overlap but each slows (1.2 / 2.5 ms, runtime lock contention), period 29.4 -> 29.15 ms profiled only. Reverted. So the serial launch stagger is not on the critical path unprofiled (the 1-2 ms launch cost is mostly profiler overhead, or kernels start while the call is still running).

## Remaining (not fixed - broad)
The pre-launch input upload path (about 70 small H2D copies and about 10 sync round trips per token on Flash Next, ~14 copies and ~5 syncs on 27B) is the real host work. Fixing it needs one of: batching the per-lane input tensors into one staging buffer and one upload per lane, dropping the per-step stream syncs in the meta set_tensor path, or async decode pipelining (set inputs for token n+1 while token n runs; needs sampled token on-device). Estimated saving 0.4 ms/token (27B, ~1.5%) and 1.5-2.5 ms/token (FN, 4-6%) unprofiled upper bound; none implemented (design change in llama-context/meta set_tensor, with the sampled-token dependency).
