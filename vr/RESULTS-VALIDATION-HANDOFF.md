# RESULTS.md validation handoff - minimal trust-but-verify test set

Date: 2026-07-02. Purpose: cheapest set of measurements that either confirms or
kills the load-bearing claims in `vr/RESULTS.md`, which are currently based on
single-sample, warmup-disabled, no-telemetry runs and show repeated
order-sensitivity. Read `vr/RESULTS.md` first. This doc does not change code.

Model under test (dense, hybrid attention+SSM, verified by gguf header dump):

```text
/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-Q4_K_M.gguf
```

Verified model facts (gguf header): arch qwen35, dense (0 expert tensors, no
expert_count key), 65 blocks, 48 of which carry BOTH attn_qkv and a full ssm_*
mixer (conv1d/dt/a/alpha/beta/out), nextn_predict_layers=1. Tensor dtype counts
294 Q4_K / 67 Q6_K / 48 Q5_K / 1 Q8_0 - matches RESULTS.md line 151.

## Why the ledger is not yet trustworthy (the confounds to remove)

1. DVFS / thermal drift. MI50-0 is 225 W capped. If clocks are not pinned, the
   first run of an A/B sees a cold/boost clock and the second sees a throttled
   clock. This alone produces the "order-sensitive" results the ledger keeps
   flagging (Q5_K, group4pad1). Every small % gain in the ledger is suspect
   until clocks are pinned or captured.
2. Single sample, warmup disabled. No spread, no median, no way to tell a 5%
   "signal" from run-to-run noise.
3. No provenance/telemetry manifest on the June 20 baseline (RESULTS.md says so
   itself). The 189 t/s single-GPU pp8192 denominator for the whole "400"
   target comes from that triage-only run.

## Measurement protocol (apply to every test below)

- Pin clocks before any run (removes confound 1). Needs sudo; run it yourself
  with the `!` prefix in the session:
  ```bash
  sudo rocm-smi -d 0 --setperflevel high     # deterministic high clocks
  # restore afterwards: sudo rocm-smi -d 0 --setperflevel auto
  ```
- Capture telemetry during each run (proves a "gain" is not a clock artifact).
  Note: `--showsclkclk`/`--showmclkclk` do not exist in this rocm-smi; use
  `-c` (clocks) `-P` (power) `-t` (temp):
  ```bash
  ( while true; do printf '%s,' "$(date +%s)"; \
      rocm-smi -d 0 -c -P -t --csv 2>/dev/null | tail -1; sleep 1; done ) > telem.csv &
  SMIPID=$!; trap "kill $SMIPID" EXIT
  ```
- Use llama-bench built-in warmup (do NOT pass `--no-warmup`) and `-r 5` for
  spread (5 is the default; keep it explicit). Report median and min/max, not
  the mean of one sample.
- Fixed geometry: `-ngl 99 -fa on -sm none -dev ROCm0`. Single device removes
  split-topology variance.
- Order check: run each A/B BOTH as control-first and candidate-first. If the
  two orderings disagree by more than the within-config spread, the result is
  confounded - do not record a delta, fix the setup.

## Prerequisites

- Tooling env: `source vr/scripts/setup-therock-env.sh` puts the therock venv
  (rocm-smi, rocprofv3, compilers) on PATH. There is no `/opt/rocm` on this
  box; the ROCm SDK lives in the venv.
- Build: `build/gfx906-optimal/bin/llama-bench` exists but is STALE - built
  Jun 28 23:32, before the Jul 2 rebase commit (0a8fe8255). Rebuild with
  `vr/scripts/build-gfx906-optimal.sh` before measuring anything; the output
  dir stays `build/gfx906-optimal/`, which is what the commands below use.
- Profiler for Test 1: `rocprof`/`rocprofv2` do not exist here; use
  `rocprofv3` (in the therock venv). Its CLI differs: `--hip-trace` no longer
  implies kernel tracing, and `--stats` only works combined with a tracing
  option. The Test 1 command below is already in rocprofv3 form.

## The minimal test set (ranked by information per minute)

### Test 1 - Kernel-time split at pp8192 (HIGHEST value, single run)  [DONE 2026-07-02]

RESULT: matmul 80% (Q4_K 54 / Q6_K 20 / Q5_K 6), DeltaNet 8%, FA 4%. Matmul
dominates; SSM/attention do not. Counter follow-up shows the matmul is
register-occupancy bound, not LDS/bandwidth bound. Full reading:
`vr/bench-results/gfx906-pp8192-kernel-profile-2026-07-02/FINDINGS.md`. The
command below is what produced it; kept for reproduction.

This is the one measurement the whole optimization premise needs and the ledger
never captured. It tells you whether Q4_K matmul, Flash Attention, or the SSM
scan dominates prefill. For a hybrid model this is decisive: if SSM/attention
dominate, no amount of Q4_K repack reaches 400.

```bash
MODEL=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-Q4_K_M.gguf
rocprofv3 --kernel-trace --stats -S -f csv -d rocprof-out -- \
  build/gfx906-optimal/bin/llama-bench -m "$MODEL" -ngl 99 -fa on -sm none \
  -dev ROCm0 -p 8192 -n 0 -r 1
# -S prints a per-kernel duration summary to stderr at exit; the same data
# lands in rocprof-out/<hostname>/<pid>_kernel_stats.csv. Aggregate by kernel:
#   Q4_K mmq/mmvq | flash-attn | ssm_scan + ssm_conv | rope/norm/other
```
Deliverable: a percentage breakdown. Pass/interpret: if Q4_K mmq is <40% of
kernel time, the §5.1 repack cannot be "the single biggest mover" and the
189->320 projection is unsupported. Cost: ~1-2 min GPU + parsing.

### Test 2 - Trustworthy single-GPU prefill baseline (validates the denominator)
Re-measure the 189 t/s figure under the clean protocol.

```bash
build/gfx906-optimal/bin/llama-bench -m "$MODEL" -ngl 99 -fa on -sm none \
  -dev ROCm0 -p 512,2048,8192 -n 0 -r 5 -o jsonl
```
Deliverable: median +/- range for pp512/2048/8192. Pass: pp8192 median within a
few % of 189 t/s confirms the denominator. If it lands materially higher/lower
with pinned clocks, the entire "2.1x to 400" framing is rescaled. Cost: ~3-5 min.

### Test 3 - Re-confirm the biggest accepted claim: Q8_0 rocBLAS dispatch
The ledger's largest gain (+21 to +57% PP). If the strongest claim survives the
clean protocol, the ledger's method is probably sound and smaller claims can be
trusted by extension; if it shrinks, treat every "confirmed historical signal"
as triage-only. A Q8_0 quant of the SAME model sits in the same snapshot dir;
A/B the dispatch threshold via the gate in `ggml/src/ggml-cuda/mmq.cu`
(`ggml_cuda_should_use_mmq`, mmq.cu:267), both orderings.

```bash
Q8MODEL=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-Q8_0.gguf
# control = current threshold, candidate = swept threshold; pp512,2048
build/<A>/bin/llama-bench -m "$Q8MODEL" -ngl 99 -fa on -sm none -dev ROCm0 \
  -p 512,2048 -n 0 -r 5 -o jsonl     # repeat for build B, then reverse order
```
Pass: gain reproduces at >2x the within-config spread in BOTH orderings.
Cost: ~5-8 min.

### Test 4 (optional, lower value) - Operator TFLOPS spot-check
Re-measure the historical operator numbers (Q4_K 15.9 / Q8_0 8.52 TFLOPS) on
the CURRENT build, since the ledger's came from commit d8a24ccee at a shape
(m=4096,k=14336,n=512) that does not match this model (embedding 5120, ffn
17408). Synthetic, not end-to-end - only do it if Test 1 says matmul matters.

```bash
build/gfx906-optimal/bin/test-backend-ops perf -b ROCm0 -o MUL_MAT
```

## What the ledger already gets right (do not re-litigate)

- Every number cited in ARCH-REVAMP-NOTES.md matches RESULTS.md exactly (spot
  checked: 390.38 dual pp8192, 189.48 single, 15.91/8.52 operator TFLOPS,
  -33.74% min_blocks, -9.27% y64, -2.4% 2-accumulator, +0.19% VDR-4, 40.68% TG
  MMVQ share). The citations are faithful; the issue is measurement quality of
  the underlying runs, not fabrication.
- The dtype histogram and Q5_K tensor count are correct.

## Time budget

Tests 1-3 under pinned clocks: roughly 15-25 min of GPU time total plus one
model load each. That is the minimal set that converts the ledger from
"triage-only single samples" to "decision-grade" for the claims that the 400
target and the §5.1 repack actually depend on.
