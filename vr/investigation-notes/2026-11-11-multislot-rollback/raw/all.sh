#!/usr/bin/env bash
# remaining server runs: Flash Next new np1/np2/np4, Qwen 27B old/new np1/np4
D=$(cd "$(dirname "$0")" && pwd)
NEW=/home/raistlin/infer/vrversion-llama.cpp/build-mslot-ring; OLD=/home/raistlin/infer/vrversion-llama.cpp/build-play-b1
bash $D/fn.sh $NEW fn_new_np2_spec_b 2 1
bash $D/fn.sh $NEW fn_new_np1_spec 1 1
bash $D/fn.sh $NEW fn_new_np4_spec 4 1
bash $D/q27.sh $OLD q_old_np1_spec 1 1
bash $D/q27.sh $NEW q_new_np1_spec 1 1
bash $D/q27.sh $OLD q_old_np4_spec 4 1 262144
bash $D/q27.sh $NEW q_new_np4_spec 4 1 262144
