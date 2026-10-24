# SCOPED Expert and Skill Reproduction Runbook

## Objective

Reproduce the four capability experts and the paper's skill-extraction stage from
SCOPED (arXiv:2610.22217), using the upstream data and hyperparameters.  The
multi-teacher OPD stage is intentionally deferred until a benchmark-driven data
strategy is selected.

This runbook does **not** treat the released SCOPED student checkpoints as the
four expert teachers.  The upstream repository currently publishes
`xiongyq/SCOPED-8B` and `xiongyq/SCOPED-4B` students, not the four RL expert
checkpoints.

## Pinned contract

- Upstream: `NEUIR/SCOPED`
- Commit: `74e112e29e2be0657d1d521c1ecc3be53df99e44`
- Dataset: `xiongyq/SCOPED-Train`
- Dataset revision: `912185f76295f847c09bc44f379cf85513472ebb`
- Manifest SHA256: `bc3340721edddbd869259cb1db50ca54446f8379deb4078067eb83896e47ca96`

The local contract is
[configs/external/scoped_expert_reproduction.yaml](../configs/external/scoped_expert_reproduction.yaml).

## Expert sizes

| Capability | Train | Val | Epochs | Rollouts |
|---|---:|---:|---:|---:|
| capability1 perception/counting/grounding | 9,499 | 501 | 5 | 8 |
| capability2 OCR/chart/table | 9,500 | 500 | 3 | 8 |
| capability3 math/symbolic | 9,500 | 500 | 3 | 8 |
| capability4 multi-image/spatial | 9,500 | 500 | 3 | 8 |

The 8-GPU and 4-GPU resources can run two 4-GPU experts concurrently.  Pair A
is capability2+capability3; Pair B is capability1+capability4.

## Setup

Run from the repository root on the HPC node:

```bash
bash scripts/hpc/setup_scoped_reproduction.sh --install
```

The checkout, virtualenvs, dataset, checkpoints, logs, and records live under
`fc-opd-storage/external/scoped/` and are ignored by Git.

Override `SCOPED_REPO_ROOT` and `SCOPED_STORAGE_ROOT` if the HPC layout differs.
The setup script refuses to reset or overwrite a dirty upstream checkout.

## Environment status

Both offline-ready virtualenvs are installed in the external checkout:

- `.venv-rl`: `torch 2.8.0+cu128`, `vllm 0.11.0`, `transformers 4.57.1`,
  `flash-attn 2.8.3.post1`, `ray 2.51.1`.
- `.venv-scoped`: `torch 2.11.0+cu130`, `vllm 0.24.0`,
  `transformers 5.10.4`, `flash-attn 2.8.3`, `ray 2.56.0`.

The two environment-only compatibility changes are:

- `requirements-rl.txt` uses `packaging==25.0` instead of `24.2`; the pinned
  `tensordict`/`pyvers` combination requires it.
- `.venv-scoped` uses the available prebuilt `flash-attn 2.8.3` wheel for
  torch 2.11/cp312 rather than `2.8.3.post1`, and installs
  `nvidia-cuda-runtime-cu12==12.8.90` so that the cu12 flash-attention
  extension can find `libcudart.so.12`.  The pair runner exports that library
  path automatically for export and skill stages.

CPU import/version checks passed in both environments.  GPU kernel execution
and distributed training still require verification on the GPU instance.

## Data status

The four-capability dataset is prepared at
`fc-opd-storage/external/scoped/storage/data/hf_dataset` (about 14GB, 8 parquet
shards).  It was downloaded with the pinned revision and verified twice: during
download and with `--offline`.  The dataset-ready provenance is recorded at
`fc-opd-storage/external/scoped/storage/records/dataset_ready.json`.

The local base model is
`/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-8B-Instruct`.

## RL experts

Run the RL stage on the GPU instance.  The CPU control node may see zero
CUDA devices even when shared storage is visible.

If both four-GPU groups are visible on one node, run Pair A first:

```bash
SCOPED_GPU_GROUP_A=0,1,2,3 \
AUTO_DOWNLOAD_DATA=0 \
SCOPED_GPU_GROUP_B=4,5,6,7 \
bash scripts/hpc/run_scoped_expert_pair.sh --stage rl --pair a
```

After Pair A succeeds, run Pair B:

```bash
SCOPED_GPU_GROUP_A=0,1,2,3 \
AUTO_DOWNLOAD_DATA=0 \
SCOPED_GPU_GROUP_B=4,5,6,7 \
bash scripts/hpc/run_scoped_expert_pair.sh --stage rl --pair b
```

The runner uses separate Ray temporary directories and writes per-capability
logs under `fc-opd-storage/external/scoped/storage/logs/`.  The dataset is
prepared before GPU launch; set `AUTO_DOWNLOAD_DATA=0` because GPU instances
are expected to have no network access.

To fit the remaining shared-storage budget, the local wrappers default to
`trainer.save_model_only=true` and `trainer.save_limit=1`.  This preserves the
upstream training hyperparameters and the latest exportable actor model while
avoiding upstream's five full model-plus-optimizer checkpoints per expert.
Set `SCOPED_SAVE_MODEL_ONLY=false` and `SCOPED_SAVE_LIMIT=5` to reproduce the
upstream checkpoint retention policy, but only with sufficient disk capacity.

If the 4-GPU allocations are on separate nodes, launch one expert per node
with `scripts/hpc/run_scoped_expert.sh`.  For example, on the first node:

```bash
SCOPED_GPU_GROUP=0,1,2,3 \
AUTO_DOWNLOAD_DATA=0 \
bash scripts/hpc/run_scoped_expert.sh --capability capability2
```

On the second node:

```bash
SCOPED_GPU_GROUP=0,1,2,3 \
AUTO_DOWNLOAD_DATA=0 \
bash scripts/hpc/run_scoped_expert.sh --capability capability3
```

Both jobs must point to the same external storage layout if the filesystem is
shared.  Otherwise keep separate storage roots and copy only the completed
checkpoint directories into the common `outputs/rl/` tree before export.

## Export teachers and extract skills

After all four RL jobs complete:

```bash
bash scripts/hpc/run_scoped_expert_pair.sh --stage export --pair a
bash scripts/hpc/run_scoped_expert_pair.sh --stage skill --pair a
```

The export writes `storage/outputs/teachers.env`.  Skill extraction writes one
`SKILL.md` per capability under `storage/outputs/skill_extraction/skills/`.

## Verification status

`VERIFIED LOCALLY`

- The pinned upstream commit and dataset revision are encoded in a tested config.
- The launch scripts use upstream `scripts/train.sh` and the paper's four
  capability configurations without vendoring SCOPED.
- Shell syntax and targeted Python tests pass locally.
- Both virtualenvs are installed and key imports/version checks pass locally.
- The 14GB pinned dataset and 17GB local Qwen3-VL-8B base model are present.

`REQUIRES SERVER VERIFICATION`

- CUDA kernel and distributed execution in both installed environments.
- RL throughput and completion for all four experts.
- Teacher export and skill extraction.
