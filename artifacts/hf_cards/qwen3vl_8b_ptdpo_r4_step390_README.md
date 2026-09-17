---
license: apache-2.0
base_model: Qwen/Qwen3-VL-8B-Instruct
tags:
- vision-language-model
- reinforcement-learning
- ptd-po
- dual-track-opd
library_name: transformers
---

# Qwen3-VL-8B PTD-PO r4 step 390

This checkpoint is the RL-only PTD-PO run described in the Dual-Track-OPD
reproduction. It starts from Qwen3-VL-8B-Instruct and uses the self-generated
ViRL39K hint recipe recorded in the repository runbook.

## Checkpoint

- Run: `qwen3vl_virl39k_ptd_base_4gpu_20260906_r4`
- Step: `390`
- Base model: `Qwen3-VL-8B-Instruct`
- Format: Hugging Face `model.safetensors`
- Exported from the verl FSDP checkpoint using
  `scripts/sft_rl/export_hf_from_grpo_ckpt.sh`

## Training summary

- Dataset: ViRL39K, 38,348 kept rows
- Teacher hints: local Qwen3.6-35B-A3B, hint ratio 0.8009
- Response length: 12,288 tokens
- PTD coefficient: 5e-2
- PTD top-K: 100
- PTD threshold: 1.0

Full provenance and data manifest are in the GitHub repository
`LukaDD7/Dual-Track-OPD`, branch `eval-shared`.

## Intended use

Research checkpoint for reproducing PTD-PO evaluation. It is not a general
production model.

