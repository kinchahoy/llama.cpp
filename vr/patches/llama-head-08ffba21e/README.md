# MX gfx906 overlay against llama.cpp 08ffba21e

Upstream base: `08ffba21e5c2f1728b6b405c05ff3913f65706d4` (ggml-org/llama.cpp, 2026-10-11). Target: snapshot tree `refs/port-snapshots/gfx906-20261012` (`4bbd50f7`), i.e. MX overlay `8d73dc74e` plus the uncommitted 2026-10-12 TP work.

| Patch | Files | Content |
| --- | ---: | --- |
| `0001-ggml-core-and-tp-meta-backend.patch` | 24 | ggml core, tensor-parallel meta backend, small backend interface changes |
| `0002-ggml-hip-gfx906-kernels.patch` | 39 | CUDA/HIP backend: Q8 repack, gfx906 MMQ/MMVQ tuning, host-staged AllReduce, GDN kernels |
| `0003-llama-model-runtime.patch` | 38 | `src/`, `include/`, `gguf-py/`: models, split KV, recurrent snapshots, MTP/DFlash |
| `0004-common-tools-tests-harness.patch` | 13 | `common/`, `tools/`, `tests/`, root `CMakeLists.txt`, `vr/` TP harness |
| `0005-tp-token-graph-fusions-ar-overlap.patch` | 10 | 2026-10-12 work: capture-safe AllReduce + whole-token graph, add+norm+Q8_1 fusion, mat-vec rows, duplex and slab-overlapped prefill AllReduce, transposed concat |

0001-0004 touch disjoint files. 0005 applies on top of them. Apply in numeric order. `EXCLUDED.txt` lists non-runtime paths (docs, results, scratch files) that are not in any patch; copy `vr/` separately if wanted.

Measured on two gfx906 (MI60 + MI50): Qwen3.8-27B Q8_0 llama-bench 551.8 -> 682.0 PP2048, 30.2 -> 36.8 TG128 versus the overlay without 0005. Details: [findings](../../2026-10-12-tp-profile-findings.md).

## Merge onto a new upstream head

```sh
KIT=/home/raistlin/infer/mx-llama.cpp/vr/patches/llama-head-08ffba21e
REPO=/home/raistlin/infer/mx-llama.cpp          # any full-history clone
PORT=/path/to/new-port                            # must not exist
(cd "$KIT" && sha256sum -c SHA256SUMS)
git -C "$REPO" fetch https://github.com/ggml-org/llama.cpp.git master
HEAD_NEW=$(git -C "$REPO" rev-parse FETCH_HEAD)
git -C "$REPO" worktree add --detach "$PORT" "$HEAD_NEW"
for p in "$KIT"/patches/*.patch; do echo "== $p"; git -C "$PORT" apply --3way "$p" || break; done
```

On a stop: `git -C "$PORT" status --short` and `git -C "$PORT" ls-files -u` show the conflicts. With `apply --3way`, stage 2 is the new upstream and stage 3 is base + MX. Resolve each file as a unit (keep upstream changes, keep the MX behavior), `git add` it, then continue with the remaining patches only. Do not replay a patch that already landed.

Preserve while resolving (the fast paths; a clean build does not prove they run):

- HIP must use the first `#if !defined(GGML_USE_MUSA)` branch of `ggml-cuda/allreduce.cu` (host-staged pipeline). Normal runs must not need `GGML_CUDA_TP_OVERLAP`.
- `q8_repack/` layout, upload, dispatch and fused single-token FFN; gfx906 selectors in `mmq.cuh` / `mmq-config-gfx906.cuh`.
- Meta backend lane dispatch, token-graph capture (`comm_ar_prepare` / `launch_rank`) and the `ar_slab_mark` hook.

## Check after building

Build with the command in [QUICKSTART.md](../../QUICKSTART.md), then run:

```sh
GGML_META_PROFILE=1 build-ilp/bin/llama-bench -hf unsloth/Qwen3.8-27B-GGUF:Q8_0 --offline \
  -ngl 999 -fa on -sm tensor -dev ROCm0/ROCm1 -b 2048 -ub 2048 -p 2048 -n 128 -r 2
```

Pass: `[meta-prof] replayed tokens=` is printed, and PP/TG are within a few percent of 682 / 36.8. Then one CLI run with `-f flake.nix --temp 0 --top-k 1 -s 42 -n 256` and compare the text with the previous build.

## Export the next kit

After the port builds and passes the check, stage all source edits in `$PORT`, then:

```sh
NEW=$(git -C "$PORT" rev-parse HEAD)                     # the upstream commit
T=$(git -C "$PORT" write-tree); git -C "$PORT" update-ref refs/port-snapshots/gfx906-$(date +%Y%m%d) "$T"
NEXT="$PORT/vr/patches/llama-head-$(git -C "$PORT" rev-parse --short=9 "$NEW")"
mkdir -p "$NEXT" && cp "$KIT"/{regenerate.py,verify.py} "$NEXT"/
```

In `$NEXT/regenerate.py` set `DEFAULT_BASE=$NEW`, `DEFAULT_MID` and `DEFAULT_TARGET` to the new snapshot ref, and fold 0005 into 0001-0004 (set `DEFAULT_MID` = target and drop `PATCH_TOP`) or keep a new top patch for new work. Run `regenerate.py`, then `verify.py` (replays the kit on a clean base and compares every file), and move the old kit to `vr/OLD-IGNORE/`.
