# VA-OPD 32B Teacher → 8B Student on ViRL39K

Date: 2026-09-19

## Purpose

This is the project-domain and model-scaling VA-OPD run. It deliberately
separates three variables from the completed Geometry3K control run:

- student: `Qwen3-VL-8B-Instruct`;
- teacher: `Qwen3-VL-32B-Instruct`;
- training corpus: ViRL39K.

The run must not be compared directly with the 8B teacher → 2B student
Geometry3K reproduction without explicitly listing those changed variables.

## Model identity

| Role | Path |
|---|---|
| Student | `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-8B-Instruct` |
| Teacher | `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-32B-Instruct` |

The dedicated launcher refuses to start unless:

1. the student basename is exactly `Qwen3-VL-8B-Instruct`;
2. the teacher basename is exactly `Qwen3-VL-32B-Instruct`;
3. both `config.json` files declare `model_type=qwen3_vl`;
4. both `tokenizer.json` files are byte-identical.

The current shared `tokenizer.json` SHA-256 is:

```text
a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7
```

## Data strategy

Source:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/dataset/ViRL39K/39Krelease.parquet
```

Prepared output:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/virl39k_va_opd
```

Conversion policy:

- 38,870 source rows are scanned.
- The established ViRL GT filter keeps letter, yes/no, pure-number, and
  has-number answers; 522 `other` rows are dropped.
- Train parquet: 38,348 rows.
- Validation monitor: 500 rows, stratified by source.
- The monitor overlaps training by default. It is an operational monitor, not
  a held-out paper validation set, and it does not reduce the train pool.
- The validation scorer must support every retained GT type. The current
  gold-in-boxed contract audit is 0 failures over all 38,348 train rows and
  all 500 validation rows.
- Multi-image samples are preserved. Every image receives its own prepared
  degraded counterpart.
- Full image paths reference the stable original ViRL image tree; only
  degraded images are materialized in the prepared asset directory.
- The current full audit covers 42,362 full/degraded image pairs.

The degraded operation is exactly the VA-OPD paper operation:

```text
low = resize(image, (round(0.1*W), round(0.1*H)), BILINEAR)
degraded = resize(low, (W, H), NEAREST)
```

Full and degraded images are stored with identical dimensions so Qwen-VL
visual tokenization and sequence alignment remain unchanged.

## Algorithm and training configuration

Canonical config:

```text
configs/experiment/qwen3vl_32b_8b_virl39k_va_opd.yaml
```

Settings:

- objective: VA-OPD;
- rollouts per prompt: 8;
- prompt batch size: 16;
- rollout temperature: 1.0, `top_p=0.99`;
- VA: positive rectified full-minus-degraded teacher log-probability;
- rollout normalization: population standard deviation;
- high-token fraction: 0.20;
- high-group weight: 0.50;
- VA temperature: 1.0;
- KL target: full-image teacher;
- loss: `va_opd_k1`, reverse KL in policy-gradient form;
- task rewards disabled;
- loss aggregation: `seq-mean-token-sum`;
- loss clamp: 10.0;
- optimizer: AdamW;
- learning rate: `1e-6` (operational default; the arXiv v1 text does not
  publish this value);
- max prompt length: 6,144;
- max response length: 2,048;
- actor: FSDP2, gradient checkpointing on, CPU/optimizer offload off;
- actor GPUs: 4;
- teacher GPUs: 4, organized as two TP2 replicas;
- checkpoint/validation frequency: every 25 steps;
- full checkpoint history is retained (`VA_OPD_MAX_ACTOR_CKPT_TO_KEEP=null`).

One epoch is approximately:

```text
ceil(38348 / 16) = 2397 optimizer steps
```

Five epochs are approximately 11,985 steps. At the observed long-run pace this
is a multi-week allocation, so the default launcher is a bounded 500-step
pilot. A full run must be requested explicitly.

## Commands

Prepare or validate data only:

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
bash scripts/hpc/run_va_opd_32b_teacher_8b_student_virl39k.sh --data-only
```

CPU-safe full preflight:

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
bash scripts/hpc/run_va_opd_32b_teacher_8b_student_virl39k.sh --preflight-only
```

Four-step GPU smoke (use only on an idle 8-GPU allocation):

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
bash scripts/hpc/run_va_opd_32b_teacher_8b_student_virl39k.sh \
  --steps 4 \
  --batch-size 4 \
  --run-id qwen3vl_32b_teacher_8b_student_virl39k_va_opd_smoke4_v1
```

Bounded 500-step pilot:

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
bash scripts/hpc/run_va_opd_32b_teacher_8b_student_virl39k.sh
```

Resume after GPU-instance reclamation:

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
bash scripts/hpc/run_va_opd_32b_teacher_8b_student_virl39k.sh --resume
```

Explicit full 5-epoch run:

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
bash scripts/hpc/run_va_opd_32b_teacher_8b_student_virl39k.sh --full
```

The same fixed run ID is used for the default pilot and resume. Do not start a
second Ray-based VA-OPD/OPD launcher on the same host while this run is active.

## Required gates

- exact model identity above;
- every sibling group has exactly 8 rollouts;
- teacher full and degraded response IDs equal the exact student response IDs;
- all full/degraded image dimensions match pairwise;
- rollout group weights sum to 1 within `1e-5`;
- teacher log-probabilities, losses, gradients, and entropy are finite;
- at least one update completes;
- VA is not identically zero.

Checkpoints must be selected from the validation curve, not merely by final
step. Because all checkpoints are retained, retain the best-val step and the
final step when pruning disk usage.

Before using any validation curve produced before commit `0507801`'s follow-up
scorer fix, regenerate it with the fixed scorer; older curves undercount
yes/no and numeric-expression answers and must not be used for checkpoint
selection.
