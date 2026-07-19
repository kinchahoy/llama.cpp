# Handover: test gfx906 Q4_K paired fused MMVQ

Patch: `vr/patches/gfx906-q4k-mmvq-paired-fusion.patch`

You are validating **one** experimental patch. Follow these steps exactly. Do
not redesign anything. If a step fails, stop and report the failing step.

## What this patch does (one paragraph)

Q4_K token generation on gfx906 spends 40.68% of GPU kernel time in the fused
Q4_K MMVQ kernel (value matrix + gate matrix). That fused loop calls the same
Q4_K dot-product helper twice with the same Q8_1 activation data but two
different weight matrices. This patch adds a paired helper that loads/decodes
the shared Q8_1 activation **once** and evaluates both weight matrices from it.
The math is bit-for-bit identical to the two separate calls; only the duplicate
activation load is removed. It is gated to exact single-`gfx906` HIP builds and
only affects the `type == GGML_TYPE_Q4_K`, `ncols_dst == 1`, fused-gate path.

This is a **hypothesis, not a guarantee**. The README's handoff notes the
compiler may already eliminate the duplicate activation work; if so this patch
is neutral. **If the TG screen below is neutral (< ~2%) or negative, revert the
patch — do not keep it.** Only prompt-processing is unaffected either way; this
touches TG only.

## Step 1 — apply to a separate candidate build tree

Keep the current accepted source as the control. Apply the patch on top for the
candidate only:

```bash
cd /home/raistlin/infer/llama.cpp
git apply --check vr/patches/gfx906-q4k-mmvq-paired-fusion.patch   # must print nothing
git apply         vr/patches/gfx906-q4k-mmvq-paired-fusion.patch
```

## Step 2 — build control and candidate (exact gfx906)

Control = current accepted tree (no new patch). Candidate = with the patch.

```bash
# control (patch NOT applied): build first, then apply patch and build candidate
BUILD_DIR=build/q4k-fused-control vr/scripts/build-gfx906-optimal.sh
# now apply the patch (Step 1) and:
BUILD_DIR=build/q4k-fused-pair    vr/scripts/build-gfx906-optimal.sh
```

Both must target exactly `gfx906`. Verify the gate compiled into the candidate:

```bash
rg 'GGML_CUDA_MMVQ_Q4K_GFX906_PAIRED_FUSION' build/q4k-fused-pair/compile_commands.json
```

If that prints nothing, the specialization did **not** compile (multi-arch or
wrong target). Fix the target before continuing — a silent generic build makes
the benchmark meaningless.

## Step 3 — depth-0 TG screen (the decision measurement)

Model:

```text
/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-Q4_K_M.gguf
```

Run one screen from each build on ROCm0:

```bash
MODEL=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-Q4_K_M.gguf

build/q4k-fused-control/bin/llama-bench -m "$MODEL" \
  -ngl 99 -fa on -sm none -dev ROCm0 -p 0 -n 64 -d 0 -r 1 -o jsonl

build/q4k-fused-pair/bin/llama-bench    -m "$MODEL" \
  -ngl 99 -fa on -sm none -dev ROCm0 -p 0 -n 64 -d 0 -r 1 -o jsonl
```

One sample per build is the default screen. Repeat (alternating control /
candidate) only if the result is promising (> ~2%), contradictory, or being
promoted. Do **not** run depth 8192 — it is known-noisy for this work and not
decision-relevant here.

Decision rule:
- Candidate TG > control by ~2% or more, repeated → promising, go to Step 4.
- Within ~2% (neutral) or worse → **revert the patch** (`git apply -R ...`) and
  record it as neutral/rejected in `RESULTS.md`. Stop.

## Step 4 — correctness (required before keeping the patch)

Two things. Both must pass on the **candidate** build.

1. Full ROCm0 MUL_MAT gate, must be **1103 of 1103**:

```bash
build/q4k-fused-pair/bin/test-backend-ops -b ROCm0 -o MUL_MAT
```

2. **Important:** the ordinary MUL_MAT test does *not* exercise the fused
   value+gate path this patch changes. You must additionally verify the fused
   path directly. Find an existing fusion graph test in `tests/` (look for a
   test that builds a fused mul_mat + gate/GLU graph), or export/run a
   representative fused graph, and confirm the candidate matches the control
   (or CPU reference) on it. Do not accept the patch on the plain MUL_MAT gate
   alone.

## Step 5 — record

Write the raw JSONL numbers and a classification (confirmed / triage-only /
rejected / neutral) into `vr/RESULTS.md` under a new dated Q4_K TG subsection,
following the existing table format. Note the control and candidate build dirs
and the model. If neutral/rejected, keep the patch file for reference but leave
the source tree on the generic path (patch reverted).

## Quick reference — what "pass" means

| Check | Requirement |
| --- | --- |
| `git apply --check` | prints nothing |
| paired define compiled | rg finds it in candidate `compile_commands.json` |
| TG depth-0 screen | candidate faster by ~2%+, repeated, to keep |
| MUL_MAT gate | 1103 / 1103 on candidate |
| fused-path check | candidate matches reference on a fused gate graph |
