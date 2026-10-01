# User Preferences & Working Style Guide

Reference notes for LLMs collaborating with kinchahoy.

---

## 1. Communication & Output Style

- **Brevity first:** Keep explanations short, dense, and ASD-STE100 technical. Avoid boilerplate, corporate fluff, and preamble.
- **Punchy attribution:** When crediting or listing features, stick strictly to 2–3 words per item (e.g., `[repo](url) - pipeline tensor parallelism`).
- **Direct actionable steps:** Provide exact, verified commands ready to copy-paste.
- **Do not over-explain:** Answer the direct question immediately; dive into underlying mechanics only when explicitly asked.

---

## 2. Git & Release Workflow

- **Never push automatically:** Never run `git push` or `gh pr create` autonomously. Always provide the verified commands for manual execution.
- **Timestamp sensitivity:** 
  - Be mindful of commit and committer dates regarding public visibility (e.g., distinguishing personal evening hacking from daytime work hours).
  - Quickly check if work should be timeshifted via `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE` outside work hours (e.g., late evenings PDT).
  - Understand GitHub tracking: green squares rely on Author Date, but the public `/events` API logs raw push request arrival times.
- **Stack discipline:**
  - Keep upstream branches strictly pristine and read-only.
  - Isolate PR staging and experiments in dedicated worktrees (`.worktrees/cleanup`).
  - Maintain clean, rebaseable, bisectable linear stacks where each commit has a single well-scoped purpose.
---

## 3. Hardware & Architecture Context

- **Platform:** Dual AMD Radeon Instinct MI60 (`gfx906`, 32GB HBM2 per GPU, 64GB total VRAM, connected via PCIe).
- **Environment:** Ubuntu Linux, AMD ROCm 6.x / 7.x, Clang / hipcc, Nix / Flakes (`flake.nix`).
- **CMake flags:**
  ```sh
  -DAMDGPU_TARGETS=gfx906 \
  -DGGML_HIP=ON \
  -DGGML_HIP_RCCL=OFF \
  -DGGML_CCACHE=OFF \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release
  ```
- **No RCCL:** `GGML_HIP_RCCL=OFF` is deliberate. Use host-staged PCIe AllReduce (`Stastez` PR #27825) which consistently outperforms RCCL rings over PCIe on dual GPUs.
- **Precision focus:** Optimizing Q8_0, Q5_1, and MXFP4 using hardware-specific weight repacking to saturate dual-issue compute units on `gfx906`.

---

## 4. Benchmarking Standard

- **Canonical model:** `unsloth/Qwen3.8-27B-GGUF:Q8_0` via `-hf` (auto-resolved from Hugging Face).
- **Standard screening command:**
  ```sh
  HIP_VISIBLE_DEVICES=0,1 \
  ./build/bin/llama-bench \
    -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 \
    -ngl 999 -fa on -sm tensor -dev ROCm0/ROCm1 \
    -b 4096 -ub 4096 -p 0 -n 0 \
    -pg 2048,0 -pg 0,256 -d 16384 \
    -r 1 -o jsonl --progress --no-warmup
  ```
- **Target metrics on 2x MI60:**
  - Upstream baseline: ~199 t/s PP2048 / ~19.7 t/s TG256
  - Optimized stack (PR 1-3): ~380-420 t/s PP2048 / ~25-27 t/s TG256

---

## 5. Provenance & Ecosystem Map

- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp):** Upstream base engine.
- **[mixa3607/mx-llama.cpp](https://github.com/mixa3607/mx-llama.cpp):** Pipeline tensor parallelism (`-tps`).
- **[Stastez PR #27825](https://github.com/ggml-org/llama.cpp/pull/27825):** ROCm host-staged AllReduce over PCIe.
- **[iacopPBK/llama.cpp-gfx906](https://github.com/iacopPBK/llama.cpp-gfx906):** gfx906 Q8 weight repacking.
