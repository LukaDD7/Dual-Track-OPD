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

## Pre-GRPO reference points

These rows are context only. The 2B base reference uses the older v1
Target-6 protocol; the TailOPD-only row uses v2 but was only partially run.
They are therefore not a directly comparable pre-GRPO B6 pair.

| Arm / stage | Protocol | Geometry3K | GQA | DynaMath | ViewSpatial | MMMU-Pro | ReMI | MMBench/100 | Six-benchmark macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B base, pre-GRPO | v1 Target-6 reference | 0.36439 | 0.59262 | 0.49202 | 0.36800 | 0.27399 | 0.00500 | 0.76117 | 0.41547 |
| TailOPD step264, pre-GRPO | v2 partial | 0.28952 | 0.27127 | 0.00000 | 0.00000 | 0.12139 | not run | not run | not computed |

The pre-GRPO Geometry3K values are the step-0 scores of the two GRPO runs.
The selected TailOPD checkpoint itself was selected at `0.28350` on the formal
TailOPD validation set, which is a different 200-prompt selection protocol.

## Shared RL protocol

- Dataset: full Geometry3K training split
- Algorithm: GRPO
- Rollouts per prompt: K=8
- Optimizer steps: 131
- Training epochs: 1
- Train batch size: 16 prompts
- Rollout batch: 128 responses per optimizer step
- PPO mini batch size: 8 prompts
- Max response length: 1024
- Max prompt length: 1024
- Optimizer: AdamW, lr `1e-6`, constant schedule, betas `[0.9, 0.999]`,
  weight decay `0.01`, gradient clipping `1.0`
- Rollout sampling: temperature `1.0`, top-p `1.0`, top-k `-1`, seed 42
- Policy loss: token-mean aggregation, PPO clip `0.2`
- GRPO advantage normalization by group standard deviation: enabled
- KL: `low_var_kl`, `kl_loss_coef=0.01`, no KL term in reward
- Validation: every 10 steps, greedy decoding with `n=1`
- Checkpoint frequency: every 25 steps
- GPUs: 4
- Distillation and TailOPD: disabled during the GRPO continuation

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

## GRPO training truncation

Training truncation uses the logged `response_length/clip_ratio`, i.e. the
fraction of sampled training rollouts that reached the 1,024-token response
limit. Each arm has 131 steps with 128 rollouts per step.

| Arm | Truncated rollouts | Mean rate | Per-step range | Final-step rate | Mean response tokens |
|---|---:|---:|---:|---:|---:|
| TailOPD + GRPO | 5,515 / 16,768 | 0.32890 | 0.09375–0.69531 | 0.16406 | 650.21 |
| Base + GRPO | 7,578 / 16,768 | 0.45193 | 0.15625–0.71875 | 0.29688 | 690.93 |

TailOPD initialization reduced mean training truncation by 12.30 percentage
points and final-step truncation by 13.28 points.

## B6 evaluation truncation

Evaluation truncation is measured from the raw sample logs. For lmms-eval
benchmarks, a row is truncated when `token_counts[0].output_tokens` reaches
the task budget. ReMI is truncated when its OpenAI-compatible replay records
`finish_reason == "length"`; this agrees with `completion_tokens == 2048`.

| Benchmark | Budget | Base + GRPO n | Base truncated | Base rate | TailOPD n | TailOPD truncated | TailOPD rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| GQA v2 | 4,096 | 12,578 | 0 | 0.00000 | 12,578 | 0 | 0.00000 |
| DynaMath v2 | 4,096 | 5,010 | 503 | 0.10040 | 5,010 | 560 | 0.11178 |
| ViewSpatial v2 | 4,096 | 5,712 | 0 | 0.00000 | 5,712 | 0 | 0.00000 |
| MMMU-Pro v2 | 4,096 | 1,730 | 66 | 0.03815 | 1,730 | 426 | 0.24624 |
| ReMI | 2,048 | 2,600 | 43 | 0.01654 | 2,600 | 73 | 0.02808 |
| MMBench | 1,024 | 4,329 | 14 | 0.00323 | 4,329 | 50 | 0.01155 |
| Total | — | 31,959 | 626 | 0.01959 | 31,959 | 1,109 | 0.03470 |

The evaluation-side truncation pattern differs from training: TailOPD + GRPO
has more 4,096-token CoT saturation on MMMU-Pro, DynaMath, ReMI, and MMBench,
while GQA and ViewSpatial remain untruncated for both arms. MMMU-Pro is the
main contributor and should be reported alongside its +4.28-point gain.

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

The MMMU-Pro gain should be interpreted with its higher eval-side truncation
rate (24.62% versus 3.82%): the TailOPD arm uses substantially more of the
4,096-token reasoning budget on that benchmark.

This is a method-validation result on Geometry3K, not a final-scale claim.
The next sensitivity steps are K=8 versus K=4 and temperature/tau ablations;
larger multimodal data and DAPO robustness remain gated on the V1 go/no-go.
