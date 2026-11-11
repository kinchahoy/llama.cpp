#!/usr/bin/env bash
# usage: plain.sh BUILD... ; greedy plain llama-completion, tensor split, flake prompt, 192 tokens
source /home/raistlin/amd-clean/env/therock.sh
set +e +u
unset GGML_CUDA_PEER_AR GGML_CUDA_TP_OVERLAP
OUT=/home/raistlin/infer/vrversion-llama.cpp/build-exp-spec/out/plain; mkdir -p $OUT
M27=/home/raistlin/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-Q8_0.gguf
i=0
for B in "$@"; do
  i=$((i+1))
  timeout --kill-after=5s 200s "$B/bin/llama-completion" -m "$M27" --offline \
    -ngl 999 -dev ROCm0,ROCm1 -sm tensor -ts 1,1 -fa on -b 2048 -ub 2048 -c 4096 -t 20 -tb 20 \
    --temp 0 --top-k 1 -s 42 -n 192 -no-cnv --simple-io --no-display-prompt \
    -f /home/raistlin/infer/vrversion-llama.cpp/flake.nix < /dev/null > $OUT/b$i.txt 2> $OUT/b$i.log
  echo "b$i rc=$? sha=$(sha256sum < $OUT/b$i.txt | cut -c1-12) $(grep -E 'eval time' $OUT/b$i.log | grep -oE '[0-9.]+ tokens per second' | tr '\n' ' ')"
done
cmp $OUT/b1.txt $OUT/b2.txt && echo IDENTICAL
