# GKD 200-Step Qwen3-VL Geometry3K — Results & Diagnosis

Run date: 2026-07-10. Environment: `fc-opd-verl071-cu128` (CUDA 12.8, torch 2.8, vLLM 0.11).

## Setup

| Item | Value |
|---|---|
| Student | Qwen3-VL-4B-Instruct |
| Teacher | Qwen3-VL-32B-Instruct (GPU 0, transformers, top-k=32) |
| Train GPUs | 1,2,3,4 (FSDP) |
| Batch × rollout | 4 × 4 = 16 samples/step |
| Max prompt/response | 6144 / 1024 |
| LR | 2e-6 |
| Objective | forward KL GKD (pure distillation, no GRPO) |
| Dataset | Geometry3K train (1901 rows) / val (200 rows) |
| Run dir | `runs/gkd_geometry3k/qwen3vl_geometry3k_gkd_20260710_154542` |
| Checkpoint | `checkpoints/gkd_geometry3k/qwen3vl_geometry3k_gkd_20260710_154542/global_step_200` |

## Result

| Metric | Step 10 | Step 200 |
|---|---|---|
| `val-core/geometry3k/reward/mean@N` | 0.265 | **0.010** |
| `val-aux/geometry3k/score/mean@N` | 0.265 | 0.010 |

**200 steps completed successfully, but the model collapsed to near-zero reward.**

## Mode Collapse Trajectory

| Step | fc_opd_loss | grad_norm | entropy | resp_len_mean | clip_ratio | teacher_valid | critic/score |
|---|---|---|---|---|---|---|---|
| 1 | 13.52 | 26.3 | 0.49 | 757 | 0.63 | 0.83 | 0.25 |
| 3 | 13.33 | 30.8 | 0.48 | 925 | 0.75 | 0.98 | 0.19 |
| 10 | 8.63 | 34.3 | 2.76 | 447 | 0.25 | 0.74 | 0.44 |
| 50 | 0.72 | 5.2 | 3.42 | 1024 | 1.0 | 1.0 | 0.0 |
| 71 | 0.64 | 8.9 | 3.56 | 987 | 0.94 | 1.0 | 0.0 |
| 75 | 0.69 | 3.5 | 3.28 | 1024 | 1.0 | 1.0 | 0.0 |

## Root Cause: Forward-KL Mode Collapse

Four convergent symptoms:

1. **Length explosion**: `response_length` climbs from ~757 tokens (step 1) to 1024 (saturating max) by step 50. Every response becomes the maximum allowed length.

2. **Entropy explosion**: `actor/entropy` from 0.49 (step 1) to 3.28 (step 75) — the student distribution flattens, spreading probability mass evenly across the vocabulary.

3. **Gradient collapse**: `grad_norm` from 26.3 (step 1) to 3.5 (step 75) — the loss gradient signal decays as the distribution becomes uniform.

4. **Reward collapse**: Teacher score goes from 0.25 (step 1, some correct answers) to 0.0 (step 50+, all wrong). The critic/score hits zero long before the fc_opd_loss converges.

**Mechanism**: Forward KL (`KL(p_teacher || q_student)`) is mean-seeking — the student optimizes by covering all modes of the teacher distribution, including low-probability tokens. The teacher is a 32B model producing a soft distribution over the full vocabulary; the 4B student collapses to a uniform distribution over all tokens to minimize the forward KL, generating maximal-length nonsense. This is a known pathology of forward KL for distillation (arXiv 2605.21924 §3.2).

## Recommended Fix

**Switch from forward KL to reverse KL** (`KL(q_student || p_teacher)`) or JSD. Reverse KL is mode-seeking — the student focuses on the teacher's high-probability tokens, which correspond to geometrically meaningful generations. JSD provides a balanced compromise.

The `src/dual_track_opd/fc_opd/` already contains JSD support (`distill_js_divergence` in `verl_sparse_kd.py`) and the `vision_opd_2c_offline_builder.py` with `visual_grounding_gap_loss.py` for the VA-OPD pathway.

## Timing

- ~21.7s/step, 220 tokens/sec throughput
- 200 steps in ~72 minutes
- ~37 GB GPU memory allocated, ~43 GB reserved
