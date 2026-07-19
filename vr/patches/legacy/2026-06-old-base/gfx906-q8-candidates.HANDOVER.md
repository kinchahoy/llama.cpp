# Handover: gfx906 Q8_0 / Q8_1 candidate patches

Four candidate patches: `dpp-warp-reductions`, `q81-quantize-dpp-reduce`, `q80-mmvq-wide-vdr`,
`q80-mmq-threshold-env`. **None has been compiled or benchmarked.** They are written, they apply
cleanly, and that is all that is known about them.

They ship as part of the full stack in `vr/patches/all-gfx906.series` (nine patches: the four
accepted, plus `q4k-mmvq-paired-fusion`, plus these four), and as the squashed
`gfx906-all-in-one.patch`. Both target **upstream 0c4fa7a98**, not the branch HEAD.

Do not promote anything here into `optimal-gfx906.series` until it passes Step 3 below.

## If you are doing the big-bang run

Applying all nine at once is fine for a first "does it build, does it get faster" signal, and
`gfx906-all-in-one.patch` exists for exactly that. Be clear-eyed about what it buys you: a single
number with **no attribution**. Five untested patches are in that stack, two of them
(`paired-fusion`, `wide-vdr`) are explicit coin flips their own notes say to revert if neutral,
and one (`q81-quantize-dpp-reduce`) can regress non-Q8_0 models. A wash could easily be one
patch winning 5% and another losing 5%.

So: big-bang to find out whether the direction is alive. If the number moves either way, fall
back to `all-gfx906.series` and A/B the candidates individually per Step 3 before keeping any of
them. Do not promote a patch to the accepted series on the strength of a big-bang result alone.

## Why these exist

The accepted series does no Q8 kernel work at all: `gfx906-q8-rocblas-dispatch.patch` only
moves the MMQ/rocBLAS crossover. The Q8_0 and Q8_1 kernels themselves are stock upstream, so
that is where the remaining headroom is for a Q8_0 model.

Also confirmed while writing these, so nobody re-derives it: `ggml_cuda_dp4a` already lowers to
the native `v_dot4_i32_i8` via `__builtin_amdgcn_sdot4` under an explicit `__gfx906__` guard in
`common.cuh`. The integer dot product is not an optimization target.

## Guarding

Every patch is inert outside single-target gfx906 HIP builds:

- `gfx906-dpp-warp-reductions.patch` adds helpers to `common.cuh` whose DPP bodies are behind
  `#if defined(GGML_USE_HIP) && defined(__gfx906__)`, with a `__shfl_xor_sync` fallback that is
  line-for-line the loop it replaces. The patch adds **no call sites**, so on its own it cannot
  change behavior anywhere. The fallback exists so the HIP host compilation pass, which parses
  device function bodies, still resolves the helpers.
- The other three gate their call sites on compile definitions that
  `ggml/src/ggml-hip/CMakeLists.txt` only sets when `CMAKE_HIP_ARCHITECTURES` is exactly
  `gfx906`, reusing the existing `GGML_HIP_COMPILE_TARGET_COUNT EQUAL 1` block.
- `gfx906-q80-mmq-threshold-env.patch` is host code already behind a runtime
  `cc == GGML_CUDA_CC_VEGA20` check, so it is a no-op on every other device at runtime.

Net effect on a non-gfx906 build: some unused inline template functions in a header. Nothing else.

## The patches

### 1. `gfx906-dpp-warp-reductions.patch` (shared, no quant)

Adds `gfx906_dpp_reduce_max<width>` and `gfx906_dpp_reduce_sum<width>` to `common.cuh` for
`width` in {4, 8, 16}.

`__shfl_xor_sync` lowers to `ds_bpermute_b32` on GCN, which round-trips the value through the
LDS crossbar. DPP permutes the operand inside the VALU instead, so a reduction step becomes one
VALU instruction and costs no LDS bandwidth.

DPP has no lane-XOR mode, so the XOR butterfly is replaced by permutations that are equivalent
*for an associative and commutative reduction*, meaning every participating lane still ends up
holding the reduced value:

- two `quad_perm` swaps (0xb1, 0x4e) fold each quad;
- the quad is then uniform, so `row_half_mirror` (lane i reads 7-i within each group of 8)
  reaches the other quad and folds the two quads;
- likewise `row_mirror` (lane i reads 15-i) folds the two halves of a 16-lane row.

**The correctness argument to check first if something looks wrong:** `bound_ctrl=true` makes a
read from an inactive lane return 0, so callers must have 0 as the identity of their reduction.
That holds for a sum, and for an amax only because it reduces absolute values. Any future caller
that reduces a possibly-negative quantity with `gfx906_dpp_reduce_max` is a bug.

This patch is pure infrastructure. Benchmarking it alone is pointless; it exists so patch 2 and
any future DPP work share one implementation.

### 2. `gfx906-q81-quantize-dpp-reduce.patch` (Q8_1)

Uses the helpers for the amax and sum reductions in `quantize_mmq_q8_1`. This kernel quantizes
the activations feeding **every** MMQ call regardless of the weight quant, so it is on the
prompt-processing critical path for all models, not just Q8_0 ones. That also means it is the
one patch here that can regress a non-Q8_0 model, so check Q4_K too.

The reduction widths are derived, not hardcoded: the existing loops run over `vals_per_scale/4`
and `vals_per_sum/4` lanes, which is 8 or 16 for the amax and 4 or 8 for the sum. All fit in one
DPP row, which is why no `row_bcast` is needed.

The reference fork (`github.com/iacopPBK/llama.cpp-gfx906`) ships this idea as a "DPP-based Q8_1
epilogue", which is the reason to expect it pays.

### 3. `gfx906-q80-mmvq-wide-vdr.patch` (Q8_0)

`VDR_Q8_0_Q8_1_MMVQ` 2 -> 4, so each thread loads four consecutive ints, which coalesce into one
16 byte `global_load_dwordx4` — the widest load the Vega20 memory path issues. Token generation
is bandwidth bound, so this targets tg directly.

**This is not a free constant change.** `blocks_per_iter = vdr * nwarps*warp_size / qi`, so
doubling vdr halves the threads cooperating per Q8_0 block from 4 to 2 and reshapes the MMVQ
dispatch. It is structurally valid (vdr=4 divides QI8_0=8) but it trades parallelism for load
width, and which side wins is exactly what the benchmark decides. If tg is neutral or worse,
drop it — do not try to rescue it.

### 4. `gfx906-q80-mmq-threshold-env.patch` (Q8_0)

The `ne11 <= 256` crossover in the accepted rocBLAS dispatch patch is a hand-picked number. This
makes it readable from `GGML_CUDA_GFX906_Q8_MMQ_MAX_NE11` so it can be swept without a rebuild.
Default is unchanged at 256, so with the env var unset this patch is behaviorally identical to
what is already accepted.

This is a tuning instrument, not an optimization. Its purpose is to find the right number; once
found, consider hardcoding the winner and dropping the env lookup.

## Step 1 — build a candidate tree

Keep the current accepted build as the control. Do not build these in-place. The stack applies to
**upstream 0c4fa7a98**, not to the branch HEAD, which already carries the four accepted patches:

```bash
cd /home/raistlin/infer/llama.cpp
git worktree add --detach ../llama-gfx906-cand 0c4fa7a98
cd ../llama-gfx906-cand

# big bang:
git apply /home/raistlin/infer/llama.cpp/vr/patches/gfx906-all-in-one.patch

# or, for per-patch attribution:
P=/home/raistlin/infer/llama.cpp/vr/patches
for p in $(grep -v '^#' "$P/all-gfx906.series" | awk 'NF{print $1}'); do
    git apply "$P/$p" || { echo "FAILED: $p"; break; }
done

AMDGPU_ARCH=gfx906 ./vr/scripts/build-gfx906-optimal.sh
```

Expect compile errors on first attempt — none of this has seen a compiler. The likely failure
points, in order: `__builtin_amdgcn_update_dpp` argument types (it takes `int`, and the masks
and `bound_ctrl` must be compile-time constants); `__float_as_int`/`__int_as_float` availability;
and the CMake `COMPILE_DEFINITIONS` list syntax on `mmvq.cu`, which now carries two definitions
and must stay semicolon-separated inside quotes or the second one is silently dropped. If a
definition goes missing the code still compiles — it just quietly takes the upstream path — so
confirm the flags actually reached the compiler:

```bash
grep -o 'GGML_CUDA_[A-Z0-9_]*GFX906[A-Z0-9_]*' build/gfx906-optimal/compile_commands.json | sort -u
```

You should see four distinct definitions across the accepted and candidate sets. If
`GGML_CUDA_MMVQ_Q4K_GFX906_BRANCHLESS_SCALES` is absent, patch 3's CMake edit clobbered an
accepted optimization and every later number is invalid.

## Step 2 — correctness, per patch

Perplexity against the control build, same file, same chunk count. Numerics will differ in the
low bits — kernel reordering guarantees that — so the bar is "within noise", not "identical":

```bash
./build/gfx906-optimal/bin/llama-perplexity -hf <model> -ngl 99 -fa on \
  --device rocm0,rocm1 -sm tensor --mlock -f README.md --chunks 8
```

Control PPL for `unsloth/Qwen3.6-27B-MTP-GGUF:Q8_0` over 8 chunks of `README.md` is
**2.4135 +/- 0.1067** (stock upstream gives 2.4037, i.e. the accepted series is already within
noise). A candidate that lands inside roughly +/- 0.02 of 2.4135 is fine; a jump of 0.1 or more
means a real bug, not rounding.

**Patch 2 must also be checked against a Q4_K model**, because it changes the activation
quantizer shared by every quant type. Note this is an outstanding gap in the accepted series
too: the Q4_K and Q6_K patches in `optimal-gfx906.series` have never been perplexity-checked
against a Q4_K model, only against Q8_0, which does not exercise them at all.

## Step 3 — performance, per patch

A/B each patch separately against the control. Do not evaluate the stack as a lump; patch 3 in
particular is a coin flip and could hide a win from patch 2.

- Patch 2 (Q8_1 DPP): affects **prompt processing**. Watch pp t/s. Control on the current
  hardware is ~255 t/s pp.
- Patch 3 (wide vdr): affects **token generation**. Watch tg t/s. Control is ~43 t/s tg.
- Patch 4 (threshold): sweep, do not A/B.
  ```bash
  for n in 128 192 256 384 512; do
      GGML_CUDA_GFX906_Q8_MMQ_MAX_NE11=$n ./build/gfx906-optimal/bin/llama-bench -m <model> -p 512 -n 128
  done
  ```

Keep a patch only if it is a clear win on its own metric — call it >2%, matching the bar the
paired-fusion handover set. Anything neutral gets reverted, not kept "because it should help".

## Not written as patches, deliberately

These came out of the same review and are real, but they are rewrites rather than edits, and
shipping an untested 300-line kernel would defeat the point of this directory. Sketched here so
the reasoning is not lost:

- **Q8_1 cross-op activation cache.** Q, K and V projections all consume the same input tensor
  and each re-quantizes it to Q8_1 from scratch. Cache once, reuse three times. Same insight as
  the accepted paired-fusion patch, applied across ops instead of within one. Needs plumbing and
  an invalidation story in `ggml-cuda.cu`, which is why it is not a patch yet.
- **Q8_0 MMQ software pipelining.** `load_tiles_q8_0` does a plain synchronous global -> LDS
  load. Prefetching the next k-tile into registers while the current tile's dp4a work runs would
  hide HBM latency. The reference fork does exactly this, plus a `need_check` path restructured
  to avoid LDS bank conflicts.
- **Flash attention.** The branch contains **zero** FA changes, and every run uses `-fa on`, so
  FA is on the hot path for every token at 200k context and is entirely untuned. The fork does
  gfx906 tile-kernel selection, GCN thread counts, and an optimized RoPE. On expected payoff this
  is probably worth more than any of the four patches above.
- **MoE half-warp dispatch.** Fork uses 32-thread dispatch for the small per-expert matrices that
  poorly utilize wave64. Not applicable to the current dense model and therefore not validatable
  here; stage it separately if a MoE model ever enters the benchmark set.

## Rejected while writing these

**Storing `x_df` as half in the Q8_0 MMQ tile.** Proposed, then dropped after doing the
arithmetic: the scale region is `mmq_y*8.25` ints against a `mmq_y*65` int quant region, so
halving it saves about 5% of the tile's LDS. That will not move the occupancy tier, and it would
require changing `tile_x_sizes` semantics and the shared-memory size calculation, which is real
bug surface. Bad trade. Do not re-propose it without a measured occupancy limit to point at.
