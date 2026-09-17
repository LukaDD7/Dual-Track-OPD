# Qwen3.5 reverse-KL environment: v1 alignment and truncation next steps

> Audience: Claude Code on the GPU server.
>
> Read after pulling the commit that adds `TRAINER_USE_V1`, portable `DRY_RUN`, output guards, and `qwen35_run_manifest.py`.
>
> Do not touch the parallel diagnostic-experiment worktree changes. Do not use `CLEAN_START=1` while another legitimate Ray job is active.

## 1. What the new patch guarantees

`scripts/run_qwen35_formal.sh` now:

- performs `DRY_RUN=1` before conda activation, model/data checks, or GPU access;
- defaults explicitly to `TRAINER_USE_V1=True` and includes `_v1`/`_v0` in generated experiment names;
- rejects CLI overrides of identity-critical Hydra fields; use the documented environment variables instead;
- rejects nonempty checkpoint, validation, and metadata directories for cold-start runs unless explicitly overridden;
- requires raw validation and metadata output directories to live outside the Git checkout;
- writes `run_manifest.json`, `resolved_launch_config.json`, the Hydra `.hydra/config.yaml`/`overrides.yaml`, and a complete outer `train.log` under `RUN_METADATA_DIR`;
- records full dataset/model/environment hashes, repo/backend commit and dirty state, backend diff hash, package versions, hardware, exact backend argv, and final return code.

Do not commit raw dumps, train logs, manifests containing machine paths, checkpoints, or model artifacts.

## 2. Pull and CPU-only verification

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
git pull --ff-only origin codex/va-opd

pytest -q \
  tests/fc_opd/test_run_qwen35_formal_compose.py \
  tests/fc_opd/test_qwen35_run_manifest.py

bash -n scripts/run_qwen35_formal.sh
```

Then compose the target v1 command without touching conda/GPU:

```bash
DRY_RUN=1 \
FORMAL_GPUS=0,1,2,3 \
NGPUS_PER_NODE=3 \
TRAIN_BATCH_SIZE=24 \
PPO_MINI_BATCH_SIZE=24 \
USE_FCOP_DATASET=1 \
USE_TASK_REWARDS=True \
ROLLOUT_N=1 \
TRAINER_USE_V1=True \
RESUME_MODE=disable \
VAL_BEFORE_TRAIN=True \
TOTAL_TRAINING_STEPS=20 \
SAVE_FREQ=-1 \
TEST_FREQ=5 \
EXPERIMENT_NAME=qwen3_6_27b_to_qwen3_5_4b_k1_tasktrue_fcop_n1_v1_reward_smoke \
VALIDATION_DATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/val_dump_k1_reward_v1_smoke \
RUN_METADATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/qwen35_runs/k1_reward_v1_smoke \
bash scripts/run_qwen35_formal.sh
```

Confirm the composed output contains all of:

```text
trainer.use_v1=True
trainer.resume_mode=disable
trainer.val_before_train=True
trainer.total_training_steps=20
actor_rollout_ref.rollout.n=1
data.custom_cls.name=FCOPDDataset
hydra.run.dir=.../qwen35_runs/k1_reward_v1_smoke/hydra
```

## 3. Step A: target-trainer integration smoke

Purpose: repeat experiment #3 through the v1 trainer used by formal experiment #1. This changes trainer implementation only; it is still an `n=1` integration smoke, not a GRPO learning result.

After checking that GPUs 0-3 are free, run the same command as Section 2 without `DRY_RUN=1`.

Required evidence:

1. startup selects `TaskRunnerV1` / v1 trainer;
2. `run_manifest.json` says `trainer_use_v1=True`, records full 64-character train/val SHA256 values, and records backend dirty/diff state;
3. Hydra config exists under `RUN_METADATA_DIR/hydra/.hydra/config.yaml` and contains `trainer.use_v1: true`;
4. at least one batch has nonzero accuracy reward, `critic/advantages/max > 0`, and nonzero `actor/pg_loss`;
5. distillation loss, entropy, and grad norm stay finite;
6. validation dumps exist at step 0/5/10/15/20;
7. final manifest status is `completed` with return code 0.

If no correct sample occurs in 20 steps, that is a sparse-sampling inconclusive result rather than a wiring failure. Confirm reward routing from validation and report it; do not silently extend the run.

Stop and diagnose if v1 produces an exception, missing reward fields, nonfinite metrics, or a manifest/config mismatch. Do not proceed to a long run.

## 4. Step B: 4096-token validation-only truncation ablation

Purpose: determine whether the 2048-token cap, rather than the reward function, is the main limiter. Start from the prompt-fix step-0 configuration and change only `MAX_RESPONSE_LENGTH` from 2048 to 4096. No optimizer step is needed.

```bash
FORMAL_GPUS=0,1,2,3 \
NGPUS_PER_NODE=3 \
TRAIN_BATCH_SIZE=24 \
PPO_MINI_BATCH_SIZE=24 \
USE_FCOP_DATASET=1 \
USE_TASK_REWARDS=False \
ROLLOUT_N=1 \
TRAINER_USE_V1=True \
RESUME_MODE=disable \
VAL_BEFORE_TRAIN=True \
TOTAL_TRAINING_STEPS=1 \
MAX_RESPONSE_LENGTH=4096 \
PPO_MAX_TOKEN_LEN_PER_GPU=32768 \
SAVE_FREQ=-1 \
TEST_FREQ=5 \
EXPERIMENT_NAME=qwen3_6_27b_to_qwen3_5_4b_k1_taskfalse_fcop_n1_v1_r4096_valonly \
VALIDATION_DATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/val_dump_k1_promptfix_r4096_valonly \
RUN_METADATA_DIR=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/logs/qwen35_runs/k1_promptfix_r4096_valonly \
bash scripts/run_qwen35_formal.sh trainer.val_only=True
```

Before the real run, repeat once with `DRY_RUN=1`. The teacher preflight max length should resolve to `1024 + 4096 + 1 = 5121`.

Analyze all 200 validation samples with the active tokenizer/scorer and report:

- response token length distribution and exact fraction reaching 4096;
- boxed rate;
- accuracy reward rate;
- format reward rate;
- total reward mean/std and score counts;
- wall time, peak GPU memory if available, and any vLLM/teacher errors;
- direct comparison with experiment #2 step 0 at 2048 tokens.

Decision:

- if clip rate falls below 30%, no OOM occurs, and boxed/accuracy improve or stay healthy, select 4096 for the next training smoke;
- if clip rate remains above 70%, do not increase length again blindly—prepare a separately versioned concise-prompt ablation at 2048;
- if clip rate is 30-70%, report length quantiles, accuracy by truncated/non-truncated group, and cost before choosing.

Do not change the reward scorer in this step.

## 5. Stop point and report

After Steps A and B, stop before `ROLLOUT_N=4` or any run longer than 20 steps. Push only code/documentation fixes; keep raw artifacts outside Git.

Report:

1. exact repo commit used;
2. Step A/B commands and return codes;
3. `run_manifest.json` paths and SHA256 values;
4. resolved Hydra config paths;
5. all Step A integration metrics;
6. the full Step B length/boxed/accuracy analysis;
7. GPU memory/time cost;
8. recommendation for response length;
9. whether the next `ROLLOUT_N=4` smoke should keep 24 effective generated sequences by reducing prompt batch size.

The next research stage will be designed only after these two gates pass: v1 task-reward integration and a defensible response-length contract.
