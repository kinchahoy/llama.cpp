# IQ3_XXS layer-split check, 2026-10-29

One run using the same candidate binary and settings as the IQ3_XXS GPU-experts control, replacing `-sm tensor` with `-sm layer`. It completed in 23.37s with exit 0: 353.76 PP / 30.99 TG tok/s, 2,097 prompt tokens and 128 generated tokens. The tensor control was 377.50 / 38.94, so this one run was 6.3% lower PP and 20.4% lower TG. No retry or further run was made.

The runtime reported the same model header (Qwen3.8 Flash Next, qwen4exp, A3B, 176.94B) and lazy-read PLE buffer (27,465.95 MiB, CPU mapped). Model buffers were reported per device: ROCm0 13,162.35 MiB plus 12,831.13 MiB repacked; ROCm1 11,949.13 plus 12,269.08 MiB repacked; ROCm_Host 497.31 MiB. Final memory breakdown: ROCm0 total/free/model/context/compute/unaccounted = 32,752/4,852/25,993/838/648/419 MiB; ROCm1 = 32,752/6,569/24,218/831/723/408 MiB; Host = 28,123 MiB (27,963 mapped + 160). Allocation fit at context 100096.

CPU affinity and counts remained 0-7 and 8/8 threads. `time -v` reported 74% CPU and 3,217,560 KiB max RSS. The normalized output differed from the tensor run (hashes in the `*.generated.txt` files); this comparison makes no quality claim.

Exact [command](commands.txt), [environment](../environment.txt), [prompt hash](prompt.sha256), [binary hashes](binaries.sha256), [loaded library hashes](libraries.sha256), [raw output](stdout.txt), [runtime log](stderr.txt), [status](status.txt), and [time/CPU/RSS](run.time) are archived. The command is the only split-mode variation; the run changed prompt numerical results as well as performance, so it does not justify a general split-mode recommendation.
