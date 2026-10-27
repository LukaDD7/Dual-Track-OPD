# SCOPED Four-Expert Reproduction Progress — 2026-10-27

## Objective and pinned contract

Reproduce SCOPED's four RL capability experts from the paper's released
training recipe, then use the completed experts for teacher export and skill
extraction.  MOPD is intentionally deferred until the benchmark-driven data
strategy is selected.

- Branch: `exp/scoped-expert-reproduction`
- Upstream repo: `https://github.com/NEUIR/SCOPED.git`
- Upstream commit: `74e112e29e2be0657d1d521c1ecc3be53df99e44`
- Dataset: `xiongyq/SCOPED-Train`
- Dataset revision: `912185f76295f847c09bc44f379cf85513472ebb`
- Dataset manifest SHA256:
  `bc3340721edddbd869259cb1db50ca54446f8379deb4078067eb83896e47ca96`
- Base model:
  `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-8B-Instruct`
- Contract config:
  `configs/external/scoped_expert_reproduction.yaml`

## Ready assets

The networkless GPU runtime is prepared on shared storage:

- Upstream checkout:
  `fc-opd-storage/external/scoped/repo`
- Dataset: 14GB, 8 verified parquet shards
- `.venv-rl`: torch 2.8.0, vLLM 0.11.0, transformers 4.57.1, flash-attn
  2.8.3.post1, Ray 2.51.1
- `.venv-scoped`: torch 2.11.0, vLLM 0.24.0, transformers 5.10.4,
  flash-attn 2.8.3, Ray 2.56.0
- Python 3.12 headers for Triton runtime compilation:
  `fc-opd-storage/external/scoped/storage/runtime/python3.12-include`

The local launcher forces the local model path, `HF_HUB_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1`.  It starts one shared Ray head for a two-expert pair
and uses separate CUDA device groups.

## Current training status

Latest completed Pair A launch: `2026-10-26T03:04:34Z`.

| Capability | Last logged step | Checkpoint | Status |
|---|---:|---|---|
| capability2 OCR/chart/table | 101/222 | `global_step_100` (33GB) | exited after upstream `DataProto.concat` dimension bug |
| capability3 math/symbolic | 73/222 | `global_step_50` (33GB) | stopped when GPU instance was reclaimed; no new traceback in log |

Checkpoint locations:

- `fc-opd-storage/external/scoped/storage/outputs/rl/capability2/global_step_100`
- `fc-opd-storage/external/scoped/storage/outputs/rl/capability3/global_step_50`

Last observed rewards:

- capability2 step 101: overall/F1 `0.7918`
- capability3 step 73: overall/accuracy `0.5779`, format `0.9678`

Checkpoints are model-only (`save_model_only=true`) with `save_limit=1`,
protecting the shared disk from the upstream five full model-plus-optimizer
checkpoint policy.

## Launch and restart implementation

The reproducible GPU commands are recorded in:

- `scripts/hpc/run_scoped_expert_pair.sh`
- `scripts/hpc/run_scoped_expert.sh`
- `/inspire/hdd/global_user/mengweicheng-240108120092/lzy/manuscript-sft-rl-gpu`

Pair A command:

```bash
AUTO_DOWNLOAD_DATA=0 \
SCOPED_GPU_GROUP_A=0,1,2,3 \
SCOPED_GPU_GROUP_B=4,5,6,7 \
bash scripts/hpc/run_scoped_expert_pair.sh --stage rl --pair a
```

Pair B command:

```bash
AUTO_DOWNLOAD_DATA=0 \
SCOPED_GPU_GROUP_A=0,1,2,3 \
SCOPED_GPU_GROUP_B=4,5,6,7 \
bash scripts/hpc/run_scoped_expert_pair.sh --stage rl --pair b
```

After all four experts complete:

```bash
bash scripts/hpc/run_scoped_expert_pair.sh --stage export --pair a
bash scripts/hpc/run_scoped_expert_pair.sh --stage skill --pair a
```

## Local fixes and verification

Key commits on the branch:

- `3cb26c1` — pinned four-expert reproduction harness
- `7278900` — separate 4-GPU single-expert allocation support
- `3dfefb9` — offline-ready expert launch
- `6a1d5c8` — complete offline `.venv-rl` and `.venv-scoped`
- `7502be9` — correct `wait` behavior, Python headers, model-only checkpoints,
  and dataset collate normalization
- `21b9436` — one shared Ray head for a two-expert pair
- `70ef105` — force local offline base-model loading
- `1f0e0c9` — normalize object arrays before `DataProto.concat`

Runtime patches live in repository code rather than modifying the pinned
upstream checkout:

- `src/dual_track_opd/scoped_runtime/collate.py`
- `src/dual_track_opd/scoped_runtime/sitecustomize.py`

The `DataProto.concat` patch normalizes object-valued non-tensor fields to a
one-dimensional sample axis before concatenation.  A direct 1D/2D mixed-array
check and 11 local tests pass.

`VERIFIED LOCALLY`

- pinned upstream checkout and dataset manifest
- local import/version checks for both virtualenvs
- 11 targeted tests pass
- dry-run command propagation for Pair A and single-expert launches
- mixed-dimension `DataProto.concat` normalization

`REQUIRES SERVER VERIFICATION`

- full RL completion after the latest concat patch
- teacher export
- skill extraction
- four-expert downstream MOPD

## Immediate next steps

1. Restart Pair A on a new GPU instance using the command above.  The latest
   runtime patch is loaded at process startup.
2. Preserve the existing step-100 and step-50 model checkpoints before any
   directory cleanup.
3. If a new run is preferred over resuming partial artifacts, archive the
   current `outputs/rl/capability*` directories under
   `outputs/rl_partial/` before relaunch.
4. Complete capability2 and capability3, then run Pair B.
5. Run teacher export and skill extraction only after all four experts exist.
