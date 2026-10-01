# gfx906 runtime port: llama.cpp e358d591 to resolved MX tree

This kit contains the resolved MI50/MI60 runtime overlay for upstream `e358d59178377be4c58ba567925e05faadbccb57`, merged with MX pre-merge head `8535054eb819d4a4a3faf267a238df792ebc2f4f`. The source resolution is pinned by Git tree `6f1050df17a6d4442c0067df642efc003a9dbc4e`, retained as the merge commit's tree and locally as `refs/port-snapshots/gfx906-20261001`. A following documentation commit adds this kit and its merge guidance.

The five file-disjoint patches export 114 runtime files against the new upstream head. Apply all of them in filename order. The original [0bc845-to-38b4 kit](../llama-head-0bc845-to-mx-38b4/README.md) remains unchanged as a runtime export. [MERGE-GFX906.md](../../vr/MERGE-GFX906.md) explains typical conflict decisions, quant priorities, and the AllReduce preprocessor trap. See also the implementation details in [q8_repack/README.md](../../ggml/src/ggml-cuda/q8_repack/README.md).

| Patch | Contents |
| --- | --- |
| `0001-core-ggml.patch` | Core ggml and meta backend, lane dispatch and split buffers |
| `0002-gpu-backends.patch` | gfx906 kernels, Q8 repack, AllReduce, HIP/CUDA integration |
| `0003-model-runtime.patch` | Loading, placement, graphs, conversion, GGUF compatibility |
| `0004-common-tools.patch` | Common options, applications, benchmark and existing test infrastructure |
| `0005-build-harness.patch` | Build integration and existing gfx906 harness |

`MANIFEST.tsv` lists exported source paths. `EXCLUDED.txt` lists differences outside runtime scope: documentation, private CI, historical results, and archived experiments/build files. No new tests were added. The replay verifier applied the five patches to a clean upstream archive and compared all 114 files byte for byte with the resolved snapshot. Its two trailing-blank-line warnings originate in retained runtime files.

## Apply to the pinned head

Use a clean checkout at the exact upstream commit. Set `KIT` to this directory before running:

```sh
KIT=/home/raistlin/infer/mx-llama.cpp/ports/llama-head-e358d-to-mx-20261001
(
    set -e
    cd /path/to/clean/llama.cpp
    test "$(git rev-parse HEAD)" = e358d59178377be4c58ba567925e05faadbccb57
    test -z "$(git status --porcelain)"
    (cd "$KIT" && sha256sum -c SHA256SUMS)
    for patch in "$KIT"/patches/*.patch; do
        git apply --check "$patch"
        git apply "$patch"
    done
)
```

For future versions, follow the [injection workflow](../../vr/MERGE-GFX906.md#inject-into-a-future-upstream-head): fetch the official head, create an isolated worktree, ensure the base blobs exist, apply with `git apply --3way`, and resume after each resolved conflict. Full blob IDs support three-way application when the base objects are present. Apply all five groups. Keep the repack buffer and its consumers together; preserve Q4_0/Q4_K canonical gfx906 paths. The guide also provides [next-kit export commands](../../vr/MERGE-GFX906.md#create-the-next-reusable-kit) and a [copyable agent handoff](../../vr/MERGE-GFX906.md#copyable-agent-handoff).

## Single standard PP/TG check

Source reconciliation must finish before these commands. Build only the benchmark target for the requested simple check:

```sh
cmake -S . -B build -DAMDGPU_TARGETS=gfx906 -DGGML_HIP=ON -DGGML_HIP_RCCL=OFF -DGGML_CCACHE=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_VR_EXPERIMENTS=ON
cmake --build build -j 16 --target llama-bench
MODEL=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/fe1e2a23d973adb629709749dc4f6756df66ef10/Qwen3.8-27B-Q8_0.gguf
HIP_VISIBLE_DEVICES=0,1 ./build/bin/llama-bench -m "$MODEL" -ngl 999 -fa on -sm tensor -dev ROCm0/ROCm1 -b 4096 -ub 4096 -p 0 -n 0 -pg 2048,0 -pg 0,256 -d 16384 -r 1 -o jsonl --progress --no-warmup
```

Use the normal environment: no repack, lane-dispatch, shared-expert, or AllReduce opt-outs; no TP overlap flag. The documented target is 400+ PP and approximately 30 TG tokens/s. The recorded default pre-merge run was 431.13 PP / 27.81 TG. A separate custom-AR-disabled run averaged 435.14 PP / 28.43 TG. The single new sample is compared with the default historical run; no fresh baseline or Q4 sweep is requested.

The gfx906 Release build passed. The single run measured **424.57 PP2048 / 28.86 TG256 tokens/s** at depth 16384. PP exceeds 400; TG remains **1.42 tokens/s (4.7%) below** the older September 10 reference of 414.24 PP / 30.28 TG. Against the recorded default checkpoint immediately before this merge, PP is -1.52% and TG is +3.77%. These are historical comparisons, not fresh paired measurements; the remaining TG deficit predates this merge and its cause is unproven. Retain both references in future port reports.

[benchmark.jsonl](benchmark.jsonl) records the two results and all benchmark settings; [benchmark.log](benchmark.log) records both gfx906 devices and progress. Repack is enabled. No custom-AR, lane-dispatch, shared-expert, or overlap environment overrides were set. The build's `build_commit` field remains `8535054eb` because the benchmark ran before the merge was committed; the measured source is the resolved tree identified above.

Q4_0/Q4_K performance and optional Qwen4Exp/MTP execution are not measured by the Q8 workload.

## Reproduce or refresh the export

`python3 regenerate.py` regenerates the five patches and manifests. `python3 verify.py` replays them on a clean archive. These scripts accept commit or tree refs through `--base` and `--target`; defaults pin the head and source tree above. The target tree must be available in the local Git object database to run the byte comparison; applying the patches does not require it. The merge commit retains this tree in normal full-history clones of the fork; the local snapshot ref is an additional retention mechanism. The merge commit's ID can also be supplied as the target.

The source tree predates the new port-kit documentation and benchmark artifacts, so the export contains no self-reference. Re-export after any runtime correction; documentation-only result updates do not change its runtime content.
