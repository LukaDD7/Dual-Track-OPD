# NLL-TailOPD v1 RL and B6-mixed results — 2026-09-18

## Objective

Validate NLL-TailOPD as an initialization stage before an identical GRPO
continuation. The comparison is against a base student that receives the same
GRPO protocol without TailOPD initialization.

## Arms

| Arm | Initialization checkpoint |
|---|---|
| Base + GRPO | `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-2B-Instruct` |
| TailOPD + GRPO | `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/sft_rl/hf/tailopd_step264_grpo_k8_step131` |

The TailOPD initialization checkpoint was selected as `global_step_264` from
the formal TailOPD run after validation accuracy peaked at `0.2835`.

## Shared RL protocol

- Dataset: full Geometry3K training split
- Algorithm: GRPO
- Rollouts per prompt: K=8
- Optimizer steps: 131
- Training epochs: 1
- Train batch size: 16 prompts
- PPO mini batch size: 8 prompts
- Max response length: 1024
- GPUs: 4

## RL validation

| Arm | Geometry3K official-test validation |
|---|---:|
| Base + GRPO | 0.39767 |
| TailOPD + GRPO | 0.44925 |
| Δ | +0.05158 |

## B6-mixed results

MMBench is reported on a 0–100 scale. All other metrics are 0–1.

| Benchmark | TailOPD + GRPO | Base + GRPO | Δ |
|---|---:|---:|---:|
| GQA v2 | 0.5952 | 0.5934 | +0.0018 |
| DynaMath v2 | 0.5212 | 0.5100 | +0.0112 |
| ViewSpatial v2 | 0.3717 | 0.3720 | -0.0003 |
| MMMU-Pro v2 | 0.3370 | 0.2942 | +0.0428 |
| ReMI strict exact | 0.27154 | 0.26192 | +0.00962 |
| MMBench | 76.5464 | 75.7732 | +0.7732 |

For a common 0–1 macro average, MMBench is divided by 100:

| Arm | Six-benchmark macro |
|---|---:|
| TailOPD + GRPO | 0.47701 |
| Base + GRPO | 0.46487 |
| Δ | +0.01214 |

## ReMI strict scoring

ReMI uses the pinned strict exact protocol with the full 2,600-row denominator:

```bash
python scripts/sft_rl/remi_reeval.py \
  --mode exact \
  --jsonl <run>/replay/remi.jsonl \
  --label-jsonl assets/remi_replay/raw_responses/qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl
```

| Arm | Correct | Extraction rate |
|---|---:|---:|
| TailOPD + GRPO | 706 / 2600 | 0.96808 |
| Base + GRPO | 681 / 2600 | 0.95962 |

## Result directories

Raw outputs remain outside Git:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/nll_tailopd_v1_rl_compare
```

Primary directories:

```text
tailopd_step264_grpo_k8_b6_full
tailopd_step264_grpo_k8_b6_full_nojudge
tailopd_step264_grpo_k8_b6_full_judged
base2b_grpo_k8_b6_full
base2b_grpo_k8_b6_full_nojudge
base2b_grpo_k8_b6_full_judged
```

## Interpretation

The V1 small-scale signal is positive. TailOPD initialization followed by
identical GRPO improves the six-benchmark macro average by about 1.21 points
over base initialization plus GRPO. The largest gains are MMMU-Pro (+4.28
points), DynaMath (+1.12 points), ReMI (+0.96 points), and MMBench (+0.77
points). GQA and ViewSpatial are essentially unchanged.

This is a method-validation result on Geometry3K, not a final-scale claim.
The next sensitivity steps are K=8 versus K=4 and temperature/tau ablations;
larger multimodal data and DAPO robustness remain gated on the V1 go/no-go.
