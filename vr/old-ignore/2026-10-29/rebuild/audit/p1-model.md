# Pass 1: model / graph cleanup (p1-model)

Tree: /home/raistlin/infer/mx-rebuild. Not built, not run. Diffs below are vs the staged overlay (git diff, unstaged part of this pass), approximate lines.

## Items

| Item | Status | Files | Approx lines (-/+) | Notes |
|---|---|---|---|---|
| Q70 | done | src/llama-graph.cpp | -20 / +2 | Deleted static ggml_mul_mat_aux (body identical to llama_mul_mat_hadamard, llama-impl.h:57). Both call sites in build_attn_store_kv now call llama_mul_mat_hadamard. |
| Q71 | done | src/llama-model-loader.cpp | -14 | Removed `case GGML_OP_CPY` in weight_buft_supported (no llm_tensor_info entry uses CPY, so default path unchanged) and the LLAMA_DSV41_ENGRAM_CAST_DEVICE block. tn_tensor is still used elsewhere. |
| Q72 | done | src/llama-model.cpp | -10 | Removed ENGRAM_EMBED CPU-pin branch and its 4-line comment. |
| Q73 | done | src/llama-arch.h, src/llama-arch.cpp, src/llama-hparams.h, src/llama-model.h | arch.h -18, arch.cpp -22, hparams.h -31, model.h -5 | Removed fork-only: LLM_KV_ENGRAM_{LAYER_IDS,HEAD_DIM,N_HEADS,MAX_NGRAM_SIZE,COMPRESSED_VOCAB_SIZE,NUM_EMBEDDINGS,PRIMES,OFFSETS,MULTIPLIERS} and their name rows; LLM_TENSOR_ENGRAM_{EMBED,WKV,Q,K} enums, name rows, info rows; LLAMA_MAX_ENGRAM_* defines; hparams engram_{head_dim,n_heads,max_ngram_size,pad_token_id,compressed_vocab,n_layers,layer_ids,num_embeddings,multipliers,primes,offsets}, engram_layer_index(), engram_n_hash_cols(); llama_layer engram_{embed,wkv,q,k}. PAD_TOKEN_ID and TOKEN_MAP enums/rows moved back to upstream position (upstream order kept). Kept upstream: ENGRAM_KV/GATE, engram_pad_id, engram_token_map, engram.* keys used by deepseek41.cpp. gguf-py: no fork-only engram lines, nothing to remove. |
| Q74 | done | src/models/models.h, src/models/deepseek4.cpp | -3 / +0 | Removed head_hc_flags member and its comment; the 3 hc_head create_tensor calls pass 0 (same value). |
| Q75 | skipped | - | - | Per brief. |
| Q76 | done | src/llama-model.cpp | -4 | Removed orphan comment above ctxs_bufs/cpu_buft_list. |
| Q77 | done | src/llama-model.cpp, src/models/models.h, src/models/deepseek4.cpp, src/llama-model-loader.cpp | model.cpp +2 blank, models.h +1 blank, deepseek4.cpp -1 blank, loader -1 blank | Blank lines now match upstream/master at these spots. |
| Q78 | done | src/llama-model.cpp | -14 | Removed debug LLAMA_LOG_ERROR in split-state lookup (kept the GGML_ASSERT). Removed both unconditional fprintf "[shexp-split]" calls; getenv gate kept, lambda returns the bool directly. Default behavior unchanged. |
| Q79 | done | src/llama-model.cpp | -4 / +1 | Replaced the contradictory NextN/MTP comment on gpu_buft_list_plain with a one-line description (plain list used when LLAMA_MTP_REPACK=0). |
| Q80 | not done | - | - | Per brief (later pass). DSV4.1 plumbing untouched: dsv41_* hparams, deepseek4.cpp v41 branches, LLAMA_DSV41_QNORM / NO_COMPRESS all remain. |
| Q81 | not done | - | - | Per brief (product decision). LLAMA_PLE_PREFAULT remains. |

## Symbols left in place because another file uses them
- None removed while still referenced. Kept (used by upstream code): LLM_KV_ENGRAM_PAD_TOKEN_ID, LLM_KV_ENGRAM_TOKEN_MAP, LLM_TENSOR_ENGRAM_KV/GATE, hparams.engram_pad_id, llama_model_deepseek4::engram_token_map.
- Kept (fork, out of this pass): LLM_TENSOR_NEXTN_FC_HIDDEN, dsv41_* / dsv4_ratio_* hparams and LLM_KV_ATTENTION_*_SOURCE_* enums, ATTN_OUT_A RESHAPE op change, MINIMAX_M2 unsupported-list removal, llm_arch_sm_tensor_replicates_attention.

## Side effects to know
- Q78: GGML_ASSERT(!suffix_fallback.empty()) now aborts without the log line that printed tensor and layer details.
- Q71: weight_buft_supported no longer has a CPY case; only reachable via the removed env probe, so no default-path change.
- Non-ASCII found in owned files (llama-graph.cpp:551, 565 and llama-model-loader.cpp:1434) predates this pass; not touched.

## Not verified
- No build, no test run (per brief). Compile risk is limited to removed symbols, all grep-checked: no remaining references in src, common, ggml, tools, gguf-py.
