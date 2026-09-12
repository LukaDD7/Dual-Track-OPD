# Qwen3-VL Geometry3K Online GKD（Legacy / Ablation）

> This launcher is retained for historical reproduction and forward-GKD
> ablation. It is not the canonical VA-OPD implementation. For the recovered
> mainline, read `va_opd_root_cause_and_recovery_20260719.md` and execute
> `va_opd_hpc_runbook.md`.

This is the Route-1 baseline before VA-OPD reproduction. It is intentionally
separate from the text-only Megatron GKD environment gate and from offline
FC-OPD score diagnostics.

## Training contract

- Student: local `Qwen3-VL-4B-Instruct`.
- Teacher: local `Qwen3-VL-32B-Instruct`.
- Data: real Geometry3K questions and images.
- Rollouts: freshly sampled from the current student after weight sync
  (1 rollout per prompt, temperature 1.0, top-p 0.99).
- Teacher target: immediate forced-scoring of those exact rollout token IDs
  under the full image and canonical question.
- Objective: sparse forward KL(P_teacher || Q_student), teacher top-32
  renormalized so the 32-token support sums to probability 1. The remaining
  vocabulary tail is excluded from the loss. It replaces the GRPO policy loss;
  reward/advantages are transport-only and do not contribute to the actor
  gradient.
- Learning rate: 1e-6.
- Validation: a deterministic 200-row holdout, disjoint from training.

The default config reference is
`configs/experiment/qwen3vl_geometry3k_gkd.yaml`. Every run records repo and
backend commits/dirty state, dependency versions, model paths, data fingerprints,
resolved runtime parameters, and config hashes in `run_manifest.json`.

## First GPU run

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
git pull

bash scripts/hpc/run_gkd_geometry3k_qwen3vl.sh \
  --objective gkd \
  --steps 10 \
  --teacher-gpu 0 \
  --train-gpus 1,2,3,4
```

The first invocation prepares the Geometry3K parquets and image assets. It
then applies the two minimal verl overlays idempotently, runs CPU-safe model and
data checks, rejects stale GPU processes, starts the teacher, performs a real
image end-to-end warmup, starts Ray, and trains.

Success requires all requested actor updates, `actor/fc_opd_loss`, and a finite
actor gradient. A zero process exit without those signals is rejected.

For a non-GPU check:

```bash
bash scripts/hpc/run_gkd_geometry3k_qwen3vl.sh --preflight-only
```

After the 10-step run passes, use a longer checkpointed baseline:

```bash
bash scripts/hpc/run_gkd_geometry3k_qwen3vl.sh \
  --objective gkd \
  --steps 200 \
  --save-freq 50 \
  --teacher-gpu 0 \
  --train-gpus 1,2,3,4 \
  --background
```

## VA-OPD transition

The launcher also exposes `--objective va_opd` and `--objective va_opd_jsd`.
Those modes use full/degraded teacher conditions and pure VA-weighted
distillation on the same online rollouts. They should be enabled only after the
vanilla GKD baseline passes and its manifest/logs are preserved.

Those switches belong to the older project HTTP-teacher path. New experiments
must use `scripts/hpc/run_va_opd_native.sh`: native sampled-token reverse KL is
the main objective, and JSD is an explicitly named stability ablation only.
