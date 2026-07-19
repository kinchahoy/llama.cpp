# gfx906 worktree benchmark notes

Last updated: 2026-07-13

This note records the current head-versus-patched worktree setup for the
private gfx906 branch. The comparison is for local MI50/MI60 testing only.

## Built worktrees

Both worktrees were created from upstream commit `99f3dc32296f825fec94f202da1e9fede1e78cf9`.

| Role | Source tree | Build dir | Binaries |
| --- | --- | --- | --- |
| Clean head | `/tmp/llama-gfx906-head` | `/tmp/llama-gfx906-head/build/gfx906-vanilla` | `bin/llama-bench`, `bin/test-backend-ops`, `bin/llama-cli` |
| Patched candidate | `/tmp/llama-gfx906-patched` | `/tmp/llama-gfx906-patched/build/gfx906-optimal` | `bin/llama-bench`, `bin/test-backend-ops`, `bin/llama-cli` |

The patched tree has `vr/patches/current/gfx906-current-bigbang.patch` applied.
The patch also appears through `vr/patches/optimal-gfx906.series` and
`vr/patches/all-gfx906.series`.

The build commands used the `vr/scripts` gfx906 helpers with:

```bash
TARGETS="llama-bench test-backend-ops llama-cli"
BUILD_JOBS=24
GGML_VULKAN=OFF
CMAKE_EXTRA_ARGS="-DGGML_CCACHE=OFF"
```

`GGML_CCACHE=OFF` was required because ccache tried to write through a
read-only cache path. `GGML_VULKAN=OFF` avoids unrelated Vulkan compilation
and keeps this comparison focused on ROCm/HIP.

The optional embedded UI asset download failed in the sandbox due DNS/network
restriction. The builds continued with no embedded UI; that does not affect the
benchmark binaries listed above.

## Shell setup

```bash
HEAD=/tmp/llama-gfx906-head/build/gfx906-vanilla/bin
PATCH=/tmp/llama-gfx906-patched/build/gfx906-optimal/bin

MODEL_Q4K=/path/to/your/Q4_K_M.gguf
MODEL_Q8=/path/to/your/Q8_0.gguf
```

Replace the model paths with the actual benchmark models. A local search under
`/home/raistlin` found only vocab GGUFs, not full model files.

## Operator smoke and perf checks

Run correctness first on the patched tree:

```bash
"$PATCH/test-backend-ops" test -b ROCm0 -o MUL_MAT -p 'q4_K|q8_0' -j 1
```

The CSV commands below are useful as support/path probes, but in the current
`test-backend-ops` output they do not include timing columns:

```bash
"$HEAD/test-backend-ops"  perf -b ROCm0 -o MUL_MAT -p 'q4_K|q8_0' --output csv > head-mulmat.csv
"$PATCH/test-backend-ops" perf -b ROCm0 -o MUL_MAT -p 'q4_K|q8_0' --output csv > patch-mulmat.csv
```

Observed output from the first run:

- The captured
  `vr/bench-results/worktree-comparison-2026-07-13/head-mulmat.csv` and
  `patch-mulmat.csv` files are identical.
- Each file has 15 lines: one header plus 14 `MUL_MAT` support records.
- Every row is `supported=1` with an empty `error_message`.
- Covered shapes are `type_a=q8_0` and `type_a=q4_K`, `type_b=f32`,
  `m=4096`, `k=14336`, and `n=1,2,3,4,5,8,512`.
- No latency or throughput fields are present, so this pair does not measure a
  head-versus-patch speed delta.

Review validation on 2026-07-18 rebuilt the modified gfx906 HIP library and ran
the focused Q4_K/Q8_0 ROCm0 `MUL_MAT` correctness filter. All 90 supported
cases passed. This is pre-merge evidence and does not replace the complete
post-merge gate.

Use `llama-bench` for the primary performance comparison. If operator-level
timing is needed, use console perf output or a profiling tool that exposes
durations, then record the exact command and output format beside the result.

## Primary model benchmarks

Use `-fa on` for these runs because the current patch series includes
FlashAttention vector changes and the target workload uses that path.

Single GPU:

```bash
"$HEAD/llama-bench"  -m "$MODEL_Q4K" -ngl 99 -fa on -dev ROCm0 -sm none -p 8192,16384 -n 0,256 -b 8192 -ub 512 -r 5 -o jsonl > head-q4k-single.jsonl
"$PATCH/llama-bench" -m "$MODEL_Q4K" -ngl 99 -fa on -dev ROCm0 -sm none -p 8192,16384 -n 0,256 -b 8192 -ub 512 -r 5 -o jsonl > patch-q4k-single.jsonl

"$HEAD/llama-bench"  -m "$MODEL_Q8" -ngl 99 -fa on -dev ROCm0 -sm none -p 8192,16384 -n 0,256 -b 8192 -ub 512 -r 5 -o jsonl > head-q8-single.jsonl
"$PATCH/llama-bench" -m "$MODEL_Q8" -ngl 99 -fa on -dev ROCm0 -sm none -p 8192,16384 -n 0,256 -b 8192 -ub 512 -r 5 -o jsonl > patch-q8-single.jsonl
```

Dual GPU:

```bash
"$HEAD/llama-bench"  -m "$MODEL_Q4K" -ngl 99 -fa on -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -p 8192,16384 -n 0,256 -b 8192 -ub 512 -r 5 -o jsonl > head-q4k-dual.jsonl
"$PATCH/llama-bench" -m "$MODEL_Q4K" -ngl 99 -fa on -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -p 8192,16384 -n 0,256 -b 8192 -ub 512 -r 5 -o jsonl > patch-q4k-dual.jsonl

"$HEAD/llama-bench"  -m "$MODEL_Q8" -ngl 99 -fa on -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -p 8192,16384 -n 0,256 -b 8192 -ub 512 -r 5 -o jsonl > head-q8-dual.jsonl
"$PATCH/llama-bench" -m "$MODEL_Q8" -ngl 99 -fa on -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -p 8192,16384 -n 0,256 -b 8192 -ub 512 -r 5 -o jsonl > patch-q8-dual.jsonl
```

Short-context baseline:

```bash
"$HEAD/llama-bench"  -m "$MODEL_Q4K" -ngl 99 -fa on -dev ROCm0 -sm none -p 128 -n 256 -r 5 -o jsonl > head-q4k-short.jsonl
"$PATCH/llama-bench" -m "$MODEL_Q4K" -ngl 99 -fa on -dev ROCm0 -sm none -p 128 -n 256 -r 5 -o jsonl > patch-q4k-short.jsonl
```

Interpret `-n 0` rows as prompt-processing evidence and `-n 256` rows as
generation evidence after the requested prompt length.

## Ad hoc CLI observation

One manual Q8_0 dual-GPU `llama-cli` run used:

```bash
BIN -hf unsloth/Qwen3.6-27B-MTP-GGUF:Q8_0 -ngl 99 -fa on -np 3 -b 2048 \
  --ubatch-size 2048 --device rocm0,rocm1 -f flake.nix -st -c 200000 \
  --mlock -dio -sm tensor --spec-type draft-mtp --spec-draft-n-max 4 --temp 0
```

with `BIN` replaced by each worktree's `llama-cli`.

| Build | Prompt | Generation | Classification |
| --- | ---: | ---: | --- |
| Patched | 244.8 t/s | 44.7 t/s | Triage-only |
| Head | 192.6 t/s | 44.6 t/s | Triage-only |

This is a strong Q8_0 prompt-processing signal and a neutral generation signal.
Do not treat it as an accepted result because it is one manual ordered pair and
uses speculative MTP, three parallel sequences, tensor split, and a real CLI
prompt rather than the controlled `llama-bench` matrix above.

## Result hygiene

- Keep head and patched commands identical except for the binary path.
- Run alternating invocations when a delta is near the noise floor.
- Record `rocm-smi` clocks, power limits, temperatures, and which physical card
  maps to `ROCm0` and `ROCm1` beside promoted results.
- Do not classify the `test-backend-ops --output csv` files above as
  performance evidence unless a timing-bearing output is also captured.

## Status after upstream merge

The private branch was merged with upstream commit `571d0d540` on 2026-07-18.
Upstream's MMQ refactor moved the Q4_K precompute into
`mmq-load-tiles.cuh` and `mmq-vec-dot.cuh`; the Q4_K/Q6_K launch bounds now use
the RDNA2/GCN configuration table.

Completed after the merge:

- Fresh exact-gfx906 ROCm build with every private compile gate verified.
- Complete ROCm0 `MUL_MAT` correctness gate: 1134 of 1134 passed.
- Current bigbang patch regenerated against `571d0d540`.
- Both series files resolve to the current patch, which applies cleanly to a
  fresh archive of `571d0d540`.

Remaining next steps:

1. Verify the fused Q4_K and Q8_0 paths directly; the operator gate does not
   exercise value-plus-gate fusion.
2. Rebuild a clean-head comparison worktree at `571d0d540`.
3. Run the controlled Q4_K and Q8_0 `llama-bench` matrix above in alternating
   order and record telemetry with each result.
