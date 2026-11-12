while pgrep -f all.sh >/dev/null; do sleep 10; done
./run.sh fnE_np4_nospec_262k_ts50 4 -- -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -t 20 --fit off -ctk f16 -ctv f16 -m /home/raistlin/models/flashnext/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf -b 1024 -ub 512 -np 4 --kv-unified -ts 0.5,0.5 -c 262144
./run.sh fnF_np4_nospec_196k_ts50 4 -- -ngl 999 -dev ROCm0,ROCm1 -sm tensor -fa on -t 20 --fit off -ctk f16 -ctv f16 -m /home/raistlin/models/flashnext/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf -b 1024 -ub 512 -np 4 --kv-unified -ts 0.5,0.5 -c 196608
source /home/raistlin/amd-clean/env/therock.sh
echo "== b1 llama-bench FN plain @8K"
/home/raistlin/infer/vrversion-llama.cpp/build-play-b1/bin/llama-bench -m /home/raistlin/models/flashnext/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf -ngl 999 -dev ROCm0/ROCm1 -sm tensor -ts 1/1 -fa 1 -b 1024 -ub 512 -t 20 -p 2048 -n 128 -d 8192 -r 3 2>&1 | grep -E "^\|"
