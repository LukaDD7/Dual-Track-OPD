---
license: apache-2.0
base_model: Qwen/Qwen3-VL-8B-Instruct
tags:
- vision-language-model
- on-policy-distillation
- va-opd
- reinforcement-learning
- virl39k
- dual-track-opd
library_name: transformers
---

# Qwen3-VL-8B VA-OPD ViRL39K K=4 step 900

This is the best-validation checkpoint of the paper-contract VA-OPD
project-domain run. It starts from `Qwen3-VL-8B-Instruct`, distills from a
local `Qwen3-VL-32B-Instruct` teacher, and trains on the ViRL39K multimodal
reasoning pool with K=4 rollouts per prompt.

## Checkpoint identity

- Run: `qwen3vl_32b_teacher_8b_student_virl39k_va_opd_paper_k4_full_perfC_v1`
- Checkpoint: `global_step_900`
- Selection metric: `val-core/ViRL39K/reward/mean@1`
- Validation score: **0.644**
- Base student: `Qwen/Qwen3-VL-8B-Instruct`
- Teacher: local `Qwen3-VL-32B-Instruct`
- Training date: 2026-09-26 to 2026-09-27
- Export format: Hugging Face Transformers safetensors

This score is the operational 500-example ViRL39K validation monitor used for
checkpoint selection. It is not an external benchmark result and should not be
compared directly with Project15 or public benchmark scores.

## Training contract

| Item | Value |
|---|---:|
| Dataset | ViRL39K |
| Raw train rows | 38,348 |
| Kept train rows after exact prompt filtering | 38,346 |
| Validation monitor rows | 500 |
| Rollouts per prompt (K) | 4 |
| Prompt batch size | 16 |
| Epoch contract | 5 |
| Optimizer | AdamW |
| Learning rate | 1e-6 |
| Maximum prompt length | 6144 |
| Maximum response length | 2048 |

The two filtered train rows were over the 6144 multimodal token budget:

```text
ViRL39K:MMMath-3435  11887 tokens
ViRL39K:MMMath-3669   8082 tokens
```

## VA-OPD configuration

```text
loss_mode = va_opd_k1
top_fraction = 0.20
lambda_high = 0.50
tau = 1.0
use_policy_gradient = true
use_task_rewards = false
```

The execution setting is **perfC**:

```text
ppo_max_token_len_per_gpu = 20480
enable_gradient_checkpointing = true
```

This changes dynamic microbatching and floating-point accumulation order, but
not the VA-OPD objective.

## Validation context

At step 900, the operational ViRL39K validation score reached 0.644, the
highest recorded score in the run through the 2026-09-29 update. The baseline
was 0.584 and the mean over the first 44 validation points was 0.6163. The
curve is noisy: nearby points include 0.602 at step 875, 0.610 at step 925,
and 0.640 at step 1000.

The full curve and provenance are maintained in
`docs/va_opd_32b_8b_virl39k_paper_k4_and_perf_20260920.md`.

## Provenance

- Repository: `LukaDD7/Dual-Track-OPD`
- Branch at export: `exp/nll-tailopd-v1`
- Repository commit at export: `f8007eb`
- VERL backend commit: `e003163181731412595257a72ec173071efb125f`
- Backend patch: `patches/verl/va_opd_native_e0031631.patch`
- Native VA hook: `src/dual_track_opd/va_opd/native_verl.py`

The protected FSDP source checkpoint is stored outside Git under:

```text
fc-opd-storage/checkpoints/va_opd_native/protected_runs/qwen3vl_32b_teacher_8b_student_virl39k_va_opd_paper_k4_full_perfC_v1/global_step_900
```

## Intended use

This is a research checkpoint for reproducing the Dual-Track-OPD VA-OPD
paper-contract experiment. It is not a general production model.
