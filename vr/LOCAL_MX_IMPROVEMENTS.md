# Local MX improvements

This branch adds an exact-shape Q8_0 tensor-parallel overlap path beyond public mx-llama.cpp. It targets the Qwen3.6 and Qwen3.8 27B down projection on two gfx906 GPUs at PP2048.

The committed implementation in `d78f9d6f4` overlaps MMQ with internal peer transport. It divides the 2048 prompt columns between two fixed 1024-column ownership halves and optionally transfers partial sums as BF16. The path is opt-in and restricted to HIP gfx906, two devices, Q8_0, stream 0, and the exact local matrix shape `8704 x 5120` by `8704 x 2048`.

An uncommitted extension added `GGML_CUDA_TP_OVERLAP_SLAB_COLS` to test 256, 512, 1024, and 2048-column MMQ compute slabs. Post-merge tests rejected it: 512 columns was slower than 1024, 256 produced incorrect output, and 2048 caused an illegal GPU memory access. The committed implementation therefore keeps the proven fixed 1024-column slabs.

Historical best results from the committed experiment were PP2048 319.48 t/s, TG1024 18.40 t/s, and about 29.7 t/s generation with MTP depth 2. See `README.md` and `RESULTS.md` for the exact controls, precision caveat, commands, and model provenance.

`patches/local-compute-slab-tuning.patch` records the rejected extension as it existed immediately before merging official llama.cpp HEAD.
