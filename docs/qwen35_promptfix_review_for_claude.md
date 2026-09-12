# Qwen3.5 Geometry3K prompt-fix review and execution instructions

> Audience: Claude Code running on the GPU server.
>
> Repository branch: `codex/va-opd`
>
> Starting point reviewed: `cdbbedc` (`fix: task rewards unreachable without boxed prompt; add USE_FCOP_DATASET + VALIDATION_DATA_DIR`)
>
> This document is an execution handoff. Read it completely before changing code or launching a GPU run.

## 1. Executive decision

The diagnosis in `cdbbedc` is directionally correct: the original Geometry3K parquet prompt did not request a boxed final answer, while the active `geo3k` scorer extracts answers from `\boxed{...}`. Wiring `FCOPDDataset` through `data.custom_cls` is a valid way to inject the shared prompt contract into both student rollout and teacher forced-forward scoring.

Do **not**, however, launch experiment #2 with the command currently shown in `docs/qwen35_geo3k_distill_handoff.md`. There is one experiment-contamination bug that must be fixed first, and the reward criteria in the handoff need to be made more precise.

Required order:

1. fix run isolation and expose diagnostic controls in `scripts/run_qwen35_formal.sh`;
2. run a fresh, unique, 20-step prompt-reachability smoke with task rewards disabled;
3. inspect reward components and response truncation;
4. launch task-reward integration only if accuracy reward is demonstrably reachable;
5. keep any reward-function relaxation as a separately named ablation, not a silent global change.

Do not touch unrelated diagnostic-experiment changes in the worktree.

## 2. What was verified in the review

The following parts of `cdbbedc` are feasible:

- verl at backend commit `334d9f8b` supports `pkg://...` in `load_extern_object`;
- `data.custom_cls.path=pkg://dual_track_opd.fc_opd.verl_dataset` plus `data.custom_cls.name=FCOPDDataset` is compatible with verl's dataset factory;
- exporting the project `src/` directory through `PYTHONPATH` is suitable for the current single-node Ray run;
- `FCOPDDataset` replaces `prompt` using `geometry3k_training_prompt`, which adds `Put the final answer in \boxed{}.`;
- `trainer.validation_data_dir` dumps one JSONL file per validation step;
- dump records are named `input`, `output`, `gts`, `score`, and `step` plus any reward extras. The ground-truth field is `gts`, not `ground_truth`;
- the updated formal shell script passes `bash -n` syntax validation.

The prompt fix is therefore worth testing. It must be treated as a prompt-distribution intervention, not as a no-op logging change: it can change student rollout behavior and the conditional distributions scored by both student and teacher. The distillation-loss magnitude is not required to be identical to experiment #1.

## 3. Blocking bug: experiment #2 currently resumes experiment #1

### 3.1 Why this happens

The default experiment name currently encodes teacher, student, loss mode, and `USE_TASK_REWARDS`, but not `USE_FCOP_DATASET`:

```text
qwen3_6_27b_to_qwen3_5_4b_k1_false
```

That is the same checkpoint directory used by formal experiment #1. The backend config defaults to:

```yaml
trainer.resume_mode: auto
```

Experiment #1 already has a final checkpoint around step 79. If experiment #2 uses the same name, verl loads that checkpoint instead of starting from the original 4B model. Because the configured total is also about 79 steps, the trainer may execute only one additional step and stop. This invalidates the comparison and can mix or overwrite artifacts.

### 3.2 Required script changes before running

Update `scripts/run_qwen35_formal.sh` with the following behavior:

1. Add environment controls:

```bash
RESUME_MODE=${RESUME_MODE:-disable}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}
ROLLOUT_N=${ROLLOUT_N:-1}
```

2. Pass these controls to Hydra through `EXTRA_ARGS`:

```bash
EXTRA_ARGS+=(trainer.resume_mode="${RESUME_MODE}")
EXTRA_ARGS+=(trainer.val_before_train="${VAL_BEFORE_TRAIN}")
EXTRA_ARGS+=(actor_rollout_ref.rollout.n="${ROLLOUT_N}")
if [ -n "${TOTAL_TRAINING_STEPS}" ]; then
  EXTRA_ARGS+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi
```

3. Include dataset/task/rollout semantics in the default experiment name, or require an explicit unique `EXPERIMENT_NAME`. A suitable default pattern is:

```text
<teacher>_to_<student>_<loss>_task<true|false>_<raw|fcop>_n<rollout_n>
```

4. Forward wrapper CLI overrides after generated arguments:

```bash
bash run_qwen3_5_4b_fsdp.sh "${EXTRA_ARGS[@]}" "$@"
```

5. Before a `resume_mode=disable` run, resolve the target checkpoint directory. If it already contains checkpoints, fail with a clear message unless an explicitly named override such as `ALLOW_EXISTING_RUN_DIR=1` is supplied. Do not delete old checkpoints automatically.

6. Print all of the following before preflight:

- experiment name and checkpoint directory;
- dataset class selection;
- resume mode;
- rollout `n`;
- total training steps or derived epoch mode;
- validation-before-training setting;
- validation dump directory.

Keep default behavior changes explicit in the handoff documentation. If changing the default resume mode is considered too broad, it is acceptable to retain `auto` globally, but both experiment commands in this document must still use a unique name and `RESUME_MODE=disable`.

### 3.3 Minimal tests for the script patch

Before launching GPUs:

```bash
bash -n scripts/run_qwen35_formal.sh
```

Also perform a dry inspection of the composed command or Hydra config and confirm all of these exact effective values:

```text
Using dataset class: FCOPDDataset
trainer.resume_mode=disable
trainer.val_before_train=True
trainer.total_training_steps=20
actor_rollout_ref.rollout.n=1
distillation.distillation_loss.use_task_rewards=False
```

Do not claim success based only on shell variable echoing; confirm the final Hydra-resolved config or trainer startup logs.

## 4. Correct interpretation of the Geometry3K reward

The active backend scorer computes:

```text
total_score = 0.9 * accuracy_reward + 0.1 * format_reward
```

`accuracy_reward` extracts a boxed answer. `format_reward` requires a full-string match containing both literal reasoning tags and a boxed answer:

```text
<think>...</think>...\boxed{...}
```

Consequently, “any boxed output receives at least 0.1” is false. The relevant cases are:

| Output condition | Accuracy | Format | Total score |
|---|---:|---:|---:|
| wrong, boxed, no literal think tags | 0 | 0 | 0.0 |
| correct, boxed, no literal think tags | 1 | 0 | 0.9 |
| wrong, boxed, valid literal think tags | 0 | 1 | 0.1 |
| correct, boxed, valid literal think tags | 1 | 1 | 1.0 |

This matters because the project prompt contract intentionally does not demand legacy literal `<think>` markup; it delegates reasoning behavior to the native model/chat template. A model may therefore satisfy the intended answer contract and receive the 0.9 accuracy component while never receiving the 0.1 format component.

For diagnostics, compute and report these separately:

- `boxed_rate`;
- `think_boxed_format_rate` using the actual scorer;
- `accuracy_reward_rate` using the actual scorer;
- total reward mean, standard deviation, min, and max;
- counts for scores `0.0`, `0.1`, `0.9`, and `1.0` where applicable.

Do not silently change the shared `geo3k.format_reward`. If native Qwen output is incompatible with the legacy tag requirement and a boxed-only format reward is scientifically desirable, implement it under a new explicit reward name/config and label it as an ablation. Preserve the original scorer for comparability.

## 5. Experiment #2 should be a diagnostic smoke, not a formal result

### 5.1 Purpose

Experiment #2 answers only these questions:

1. Did `FCOPDDataset` put the boxed instruction into the prompt actually seen by rollout?
2. Can the unmodified student produce a boxed and correct answer under this prompt?
3. Does the current reward pipeline score those outputs correctly?
4. Are final answers being lost because responses hit the 2048-token cap?

Task rewards remain disabled in the optimization loss. Reward need not increase monotonically during this smoke, and distillation loss need not match experiment #1 exactly.

### 5.2 Recommended command after implementing Section 3

```bash
FORMAL_GPUS=0,1,2,3 \
NGPUS_PER_NODE=3 \
TRAIN_BATCH_SIZE=24 \
PPO_MINI_BATCH_SIZE=24 \
USE_FCOP_DATASET=1 \
USE_TASK_REWARDS=False \
ROLLOUT_N=1 \
RESUME_MODE=disable \
VAL_BEFORE_TRAIN=True \
TOTAL_TRAINING_STEPS=20 \
SAVE_FREQ=-1 \
TEST_FREQ=5 \
EXPERIMENT_NAME=qwen3_6_27b_to_qwen3_5_4b_k1_taskfalse_fcop_n1_promptfix_smoke \
VALIDATION_DATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/val_dump_k1_promptfix_smoke \
bash /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/scripts/run_qwen35_formal.sh
```

`SAVE_FREQ=-1` is intentional: this diagnostic does not need a 51 GB checkpoint. Validation should run at step 0, every five steps, and the final step. Confirm actual behavior in the logs.

Do not use `CLEAN_START=1` on a node where another legitimate Ray experiment may be running; it can stop the shared Ray cluster. Clean only processes belonging to this run and only when safe.

### 5.3 Required inspection

At step 0 and subsequent validation dumps, verify:

- startup says `Using dataset class: FCOPDDataset`;
- decoded `input` contains exactly one boxed instruction and the expected image/question contract;
- `gts` is populated and matches the parquet ground truth;
- `output` contains or does not contain literal `<think>` tags as an observed fact;
- boxed, format, and accuracy rates are reported separately;
- `response_length/clip_ratio` is reported;
- boxed/accuracy rates are stratified by truncated vs non-truncated responses if possible;
- reward-manager routing remains `hiyouga/geometry3k` and uses the intended scorer.

### 5.4 Go/no-go criteria

The minimum evidence that the prompt/reward path is wired is:

```text
prompt contains boxed instruction
AND boxed_rate > 0
AND total reward is nonzero for at least one sample
```

The stronger condition required before calling task accuracy reward “reachable” is:

```text
accuracy_reward_rate > 0
AND reward has nonzero variance
```

Only the stronger condition justifies proceeding directly to the task-reward integration run. A constant 0.1 format reward proves formatting reachability but provides no correctness discrimination.

## 6. If experiment #2 still has zero reward

Do not immediately relax `geo3k.format_reward`. Diagnose in this order:

1. **Dataset routing:** confirm the runtime class is `FCOPDDataset`, not `RLHFDataset`.
2. **Rendered input:** confirm the validation input includes the injected instruction.
3. **Response truncation:** formal experiment #1 had `response_length/clip_ratio` around `0.958`. The model may still reach the 2048-token limit before emitting its final boxed answer.
4. **Output syntax:** distinguish no box, malformed box, correct box without think tags, and valid think-plus-box format.
5. **Ground truth:** confirm `gts` is populated and the scorer receives the expected string form.
6. **Reward routing:** run the exact dumped output and ground truth through the server environment's active `geo3k.compute_score` and its two component functions.

If most outputs are truncated, the next experiment should change exactly one of:

- increase `MAX_RESPONSE_LENGTH`, with corresponding vLLM/token-memory checks; or
- use a versioned, more concise final-answer instruction that asks the model to conclude before the limit.

Do not combine a longer response limit, a new prompt, and a changed reward scorer in one run.

## 7. Experiment #3: task-reward integration smoke

### 7.1 Preconditions

Proceed only after experiment #2 establishes:

- correct custom prompt routing;
- nonzero boxed rate;
- at least some accuracy reward, not only format reward;
- nonconstant reward;
- acceptable truncation or a separately validated truncation fix.

### 7.2 Command

Use the exact validated experiment #2 configuration and change task rewards plus run identity:

```bash
FORMAL_GPUS=0,1,2,3 \
NGPUS_PER_NODE=3 \
TRAIN_BATCH_SIZE=24 \
PPO_MINI_BATCH_SIZE=24 \
USE_FCOP_DATASET=1 \
USE_TASK_REWARDS=True \
ROLLOUT_N=1 \
RESUME_MODE=disable \
VAL_BEFORE_TRAIN=True \
TOTAL_TRAINING_STEPS=20 \
SAVE_FREQ=-1 \
TEST_FREQ=5 \
EXPERIMENT_NAME=qwen3_6_27b_to_qwen3_5_4b_k1_tasktrue_fcop_n1_reward_smoke \
VALIDATION_DATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/val_dump_k1_reward_smoke \
bash /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/scripts/run_qwen35_formal.sh
```

### 7.3 Correct metric expectations

Primary integration evidence:

- `critic/score/*` and `critic/rewards/*` reflect the nonzero task score;
- `critic/advantages/max` is nonzero;
- `actor/pg_loss` becomes nonzero when task advantages are nonzero;
- distillation-specific metrics remain finite;
- total grad norm is finite and stable;
- format and accuracy components are tracked independently.

Do **not** require `actor/pg_clipfrac` to become nonzero. Clip fraction measures whether the PPO importance ratio crosses the clipping range; valid nonzero task gradients can exist while clip fraction remains zero.

Do not require reward to rise monotonically in a 20-step smoke. The purpose is to prove that enabling task rewards changes the actual optimization term without numerical failure. A longer, seeded run is required for a learning claim.

### 7.4 Meaning of `ROLLOUT_N=1`

The backend script currently fixes `actor_rollout_ref.rollout.n=1`. In verl `334d9f8b`, the GRPO implementation special-cases singleton UID groups with mean `0` and standard deviation `1`, so the computed advantage equals the raw reward. It does not become zero.

Therefore experiment #3 can generate a task gradient, but its semantics are effectively an uncentered, no-baseline REINFORCE-style update rather than standard group-relative GRPO. Document it as a wiring/integration smoke.

For a research-quality task-RL comparison, add a separate experiment with `ROLLOUT_N=4` or `8`, recompute effective sequence batch size and memory usage, and keep prompt batch vs generated sequence batch explicit. Do not conflate `ROLLOUT_NUM_WORKERS` with `ROLLOUT_N`: the former controls concurrency; the latter controls samples per prompt.

## 8. Stabilization guidance

The combined actor objective is conceptually:

```text
task policy loss + distillation_loss_coef * distillation policy loss
```

`loss_max_clamp` clips per-token distillation estimator values. It is not a generic task-reward stabilizer.

If task reward improves while teacher alignment worsens:

1. inspect task `actor/pg_loss` vs distillation-specific policy-loss magnitude;
2. inspect reward accuracy vs format components;
3. consider increasing `distillation_loss_coef` or explicitly rescaling task reward;
4. consider actor LR and rollout `n`/variance;
5. lower `loss_max_clamp` only when evidence shows rare extreme distillation estimates or gradient spikes.

Blindly lowering the clamp may weaken the distillation anchor and make task domination worse.

## 9. Reproducibility artifacts required for both runs

Before launching, write a run manifest outside checkpoint-heavy Git paths, preferably beside the validation dump. It must contain:

- project repo commit and `git status --short`;
- backend path, backend commit, and backend dirty status;
- complete resolved Hydra config;
- SHA256 of train and validation parquet files, or the canonical dataset manifest hash;
- student and teacher checkpoint paths;
- CUDA, torch, vLLM, verl, transformers, and flash-attn versions;
- `CUDA_VISIBLE_DEVICES` and GPU model/driver snapshot;
- exact shell command;
- validation dump path;
- checkpoint path, or an explicit note that `SAVE_FREQ=-1` disables checkpoints;
- summary metrics, decision, and notes after completion.

Do not commit raw validation JSONL, model checkpoints, weights, or large logs to this repository.

## 10. Deliverables expected from Claude Code

Complete the work in this order and report each item explicitly:

1. patch `scripts/run_qwen35_formal.sh` for safe run isolation and diagnostic controls;
2. update `docs/qwen35_geo3k_distill_handoff.md` so it no longer claims any boxed output earns 0.1, no longer reuses experiment #1's run identity, and no longer treats nonzero clip fraction as mandatory;
3. add or update lightweight tests for argument composition/default experiment naming where practical;
4. run shell syntax/config-composition checks;
5. commit and push the script/doc/test fix before launching the GPU run;
6. run experiment #2 as the 20-step diagnostic smoke;
7. analyze step-0 and later dumps into boxed/format/accuracy/truncation components;
8. decide go/no-go for experiment #3 using Section 5.4;
9. if go, run experiment #3 as a separate fresh 20-step smoke;
10. return exact commits, commands, run directories, resolved configs, component metrics, and any deviations from this plan.

Do not run experiment #3 merely because total reward is nonzero. Confirm that accuracy reward is reachable and that the reward distribution is not constant.
