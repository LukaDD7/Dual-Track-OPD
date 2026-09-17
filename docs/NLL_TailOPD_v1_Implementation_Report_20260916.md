# NLL-TailOPD v1 Implementation Report

Date: 2026-09-16

Branch: `exp/nll-tailopd-v1`

## Result

NLL-TailOPD v1 is implemented as a default-off, rollout-level reweighting layer.
It computes a length-normalized old-policy NLL score for each student rollout,
normalizes scores within the prompt's sibling group, converts them to softmax
weights, and multiplies the existing OPD signal by `K * w_k`.

The core algorithm is isolated in `src/dual_track_opd/tail_opd/`. The external
verl backend only contains a thin config and dataflow hook, captured in
`patches/verl/nll_tailopd_v1.patch`. Existing VA-OPD and FC-OPD code is not
modified.

## Mathematical mapping

For rollout `k` in prompt group `b`:

```text
s_bk = -sum_t(response_mask * old_log_prob) / sum_t(response_mask)
z_bk = (s_bk - mean_b(s)) / (std_b(s) + eps)
w_bk = softmax(z_bk / tau)
tail_scale_bk = K * w_bk
```

The backend applies:

```text
opd_signal_bkt = existing_opd_signal_bkt * tail_scale_bk
```

The score and scale are detached. If a group has zero standard deviation, all
weights are `1/K`, so `tail_scale=1` and the vanilla OPD gradient scale is
recovered. Complete groups satisfy `mean_k(tail_scale)=1`.

Both Vanilla OPD and TailOPD experiment runs use:

```text
loss_agg_mode = seq-mean-token-mean
```

The global backend default remains `token-mean`; the explicit override is
passed only by the TailOPD experiment launcher.

## Changed repository files

- `src/dual_track_opd/tail_opd/weights.py`
  - NLL score, group z-score, softmax weights, `K*w` scale, diagnostics.
- `src/dual_track_opd/tail_opd/__init__.py`
  - Public API.
- `tests/tail_opd/test_weights.py`
  - Uniform recovery, monotone ordering, group normalization, padding
    invariance, group isolation, no-gradient, and diagnostics tests.
- `patches/verl/nll_tailopd_v1.patch`
  - Minimal external backend integration.
- `scripts/hpc/apply_nll_tailopd_v1_patch.sh`
  - Idempotent backend patch application and syntax checks.
- `scripts/hpc/launch_nll_tailopd_v1.py`
  - Config-driven smoke/formal launcher and provenance manifest recorder.
- `configs/experiment/nll_tailopd_geometry3k_v1.yaml`
  - Geometry3K V1 run matrix, defaults, and future scale path.

## External backend patch

The pinned backend is:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/backends/verl-qwen35-v090-cu132
commit 483b8a009ba3a97563edee3a19887e4862b8094a
```

The patch changes:

- `verl/trainer/config/algorithm.py`
  - Adds default-off `algorithm.tail_opd` and validates V1-only options.
- `verl/trainer/config/ppo_trainer.yaml`
  - Adds the default-off Hydra config block so command-line overrides are
    accepted in struct mode.
- `verl/trainer/ppo/ray_trainer.py`
  - Computes weights once on the complete global batch after old log-probs are
    available and before actor update.
- `verl/trainer/distillation/losses.py`
  - Applies detached `K*w` to the existing per-token OPD estimator.
- `examples/on_policy_distillation_trainer/run_qwen3_5_4b_fsdp.sh`
  - Adds configurable `ROLLOUT_N`, loss aggregation, and TailOPD flags.

When `algorithm.tail_opd.enabled=false`, no `tail_opd_scales` field is created
and the existing OPD path is unchanged.

## Validation completed

- TailOPD unit tests: 6 passed.
- Backend Python syntax checks: passed.
- Backend launcher `bash -n`: passed.
- Patch reverse-check: passed, proving the applied backend state matches the
  stored patch.
- Hydra `--cfg job` parse check: passed with `algorithm.tail_opd.enabled=true`.
- TailOPD and Vanilla OPD dry-runs: passed.
- Provenance manifests record repo/backend Git state, patch hash, dataset
  hashes, model paths, resolved config, command, and output paths.

Current limitation: this Codex container exposes no CUDA device
(`torch.cuda.is_available() == False`, device count 0). Therefore the required
3-step GPU smoke has not been launched here.

## Launch commands

Apply or verify the backend patch:

```bash
bash scripts/hpc/apply_nll_tailopd_v1_patch.sh
```

3-step TailOPD smoke:

```bash
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/dtopd-dev/bin/python \
  scripts/hpc/launch_nll_tailopd_v1.py \
  --run tail_opd --stage smoke
```

Smoke hardware:

```text
4 total GPUs = 2 student/rollout GPUs + 2 teacher GPUs
student = Qwen3-VL-2B-Instruct
teacher = Qwen3-VL-8B-Instruct
```

3-step Vanilla OPD control:

```bash
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/dtopd-dev/bin/python \
  scripts/hpc/launch_nll_tailopd_v1.py \
  --run vanilla_opd --stage smoke
```

Formal Geometry3K A/B:

```bash
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/dtopd-dev/bin/python \
  scripts/hpc/launch_nll_tailopd_v1.py \
  --run vanilla_opd --stage formal

/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/dtopd-dev/bin/python \
  scripts/hpc/launch_nll_tailopd_v1.py \
  --run tail_opd --stage formal
```

The default models in the experiment config are:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-2B-Instruct
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-8B-Instruct
```

Override it in the YAML before formal runs if a different pinned teacher is
required.

## Validation gates

1. Run 3 optimizer steps and confirm no NaN/Inf, correct group normalization,
   `tail_opd/scale_mean ~= 1`, and checkpoint save/load.
2. Run 20-30 steps and confirm stable loss, KL, response length, and weight
   concentration.
3. Run the 5-epoch Geometry3K A/B with identical data, models, optimizer, K=4,
   and aggregation.
4. Select the best validation checkpoint, not epoch 5 by default.
5. Only after a positive V1 signal, test K=8 or tau sensitivity, then identical
   GRPO continuation.

Do not add correctness routing, teacher compatibility, token-level weighting, or
joint OPD+RL while validating V1.

## Code map and chain checks

Main entry:

```text
scripts/hpc/launch_nll_tailopd_v1.py
```

Layered path:

```text
YAML config
  -> scripts/hpc/launch_nll_tailopd_v1.py
  -> external verl launcher
  -> RayPPOTrainer
  -> distillation loss
```

Key code locations:

- `src/dual_track_opd/tail_opd/weights.py`
  - Core NLL, z-score, softmax, and scale math.
- `scripts/hpc/launch_nll_tailopd_v1.py`
  - Main entry, environment assembly, provenance, and stage switching.
- `verl/trainer/ppo/ray_trainer.py`
  - Computes weights after old log-probs and before actor update.
- `verl/trainer/distillation/losses.py`
  - Applies `K*w` to the existing OPD estimator.

Chain checks:

```bash
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/dtopd-dev/bin/python \
  -m pytest -q tests/tail_opd/test_weights.py

bash scripts/hpc/apply_nll_tailopd_v1_patch.sh

/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/dtopd-dev/bin/python \
  scripts/hpc/launch_nll_tailopd_v1.py \
  --run tail_opd --stage smoke --dry-run

/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/dtopd-dev/bin/python \
  scripts/hpc/launch_nll_tailopd_v1.py \
  --run tail_opd --stage smoke
```
