# Support-Aware K=32 Overnight Handoff — 2026-08-04

Status: executable handoff for Claude Code on the CPU/network instance and GPU
instance.  The goal is to confirm K=8 screening states before any bridge arm is
allocated.  This run does **not** train a model and must not be described as an
OPD/FKL result.

## Decision and estimand

The completed K=8 run contains 256 prompts and 2,304 exact-token scored
responses, but K=8 is too coarse to distinguish true low support from a missed
rare mode.  The overnight run independently samples K=32 stochastic responses
on a frozen 64-prompt cohort:

- 16 `c=0` prompts;
- 16 `c=1-2` prompts;
- 16 `c=3-7` prompts;
- 16 `c=8` prompts.

Within each stratum the cohort builder targets an even split between screening
median response length `<=2048` and `>2048`, then deterministically backfills if
one length bucket is too small.  The selection is deterministic under seed
`20260804` and is frozen in `cohort_manifest.json`.

The new K=32 responses use seed `20260804`, not the K=8 seed 42.  They are an
independent confirmation sample.  Do not concatenate K=8 rollout IDs 1-8 with
the new K=32 rollout IDs.

Primary morning outputs:

1. K=8 to K=32 observed-stratum reclassification rate;
2. K=32 posterior pass probability and expected `U_8(p)`;
3. number of prompts that remain RL-ready under K=32;
4. teacher/student/gap within-prompt ranking split by response length and
   truncation;
5. protocol quality: exact-token identity, prompt-token hash, complete rollout
   keys, malformed rate, truncation rate, duplicates, and non-finite scores.

## Why the implementation changes scoring chunks

K=32 produces 33 sequences per prompt including greedy.  The old code sent all
responses through one padded student forward and one synchronous teacher HTTP
request.  At 4,096 response tokens this can materialize very large vocabulary
logits on the student and can exceed the teacher client's 120-second timeout.

The K=32 config therefore uses:

```yaml
teacher_score_chunk_size: 4
student_score_chunk_size: 4
teacher.timeout_seconds: 1800
```

Generation remains immutable.  Each chunk consumes slices of the same raw token
ID list and results are appended in original order.  Chunking must not decode or
re-tokenize a response.

## Fixed paths on the current HPC

```bash
PROJECT_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
OUTPUT_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs
SCREENING_RUN=${OUTPUT_ROOT}/support_aware_opd/diag_full_20260802_256_merged.rescored-exact-v1
SOURCE_DATASET=${OUTPUT_ROOT}/fc_opd/geometry3k_full/train.parquet
K32_COHORT_DIR=${OUTPUT_ROOT}/support_aware_opd/k32_cohort_20260804
K32_RUN_PREFIX=diag_full_k32_20260804
```

If a named output directory already exists, do not delete or overwrite it.
Inspect whether it is a valid partial run and resume it, or choose a versioned
suffix such as `_v2`.

## Phase A — CPU/network instance: pull and freeze the cohort

Run from a clean checkout.  Do not use `.codex-tmp` or an old copied source
tree.

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
git fetch origin
git switch codex/va-opd
git pull --ff-only origin codex/va-opd
git status --short
git rev-parse HEAD
```

Build the cohort:

```bash
export DTOPD_PYTHON=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vision-opd-cu128/bin/python
export OUTPUT_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs
export SCREENING_RUN=${OUTPUT_ROOT}/support_aware_opd/diag_full_20260802_256_merged.rescored-exact-v1
export SOURCE_DATASET=${OUTPUT_ROOT}/fc_opd/geometry3k_full/train.parquet
export K32_COHORT_DIR=${OUTPUT_ROOT}/support_aware_opd/k32_cohort_20260804

bash scripts/hpc/prepare_support_aware_k32_cohort.sh \
  "${SCREENING_RUN}" "${SOURCE_DATASET}" "${K32_COHORT_DIR}"
```

Required cohort artifacts:

```text
cohort.parquet
cohort_manifest.json
cohort_prompts.csv
```

Before GPU launch, inspect `cohort_manifest.json` and confirm:

- `schema_version == support-aware-k32-cohort-v1`;
- `num_prompts == 64`;
- exactly 16 prompts per `observed_stratum`;
- `num_shards == 4`;
- source and cohort SHA256 fields are non-empty;
- `git_dirty == false` whenever possible.

## Phase B — GPU instance: pull the exact same commit

The strict merge rejects shards produced from different git commits or different
dirty states.  Pull once before the smoke/full run, then do not edit or pull the
checkout until all four shards finish.

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
git fetch origin
git switch codex/va-opd
git pull --ff-only origin codex/va-opd
git status --short
git rev-parse HEAD

export DTOPD_PYTHON=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vision-opd-cu128/bin/python
export DTOPD_MODEL_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models
export DTOPD_OUTPUT_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs
export K32_COHORT_DIR=${DTOPD_OUTPUT_ROOT}/support_aware_opd/k32_cohort_20260804
export K32_RUN_PREFIX=diag_full_k32_20260804
```

## Phase C — start the frozen teacher scorer

Use an available GPU that is not included in the student GPU list.  The
Transformers teacher implementation places the 32B model on the visible device;
do not expose several unrelated GPUs and assume tensor parallelism.

Example using physical GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 \
FC_OPD_TEACHER_MODEL=${DTOPD_MODEL_ROOT}/Qwen3-VL-32B-Instruct \
FC_OPD_TEACHER_PORT=18080 \
nohup bash scripts/hpc/start_fc_teacher.sh \
  >${DTOPD_OUTPUT_ROOT}/../logs/k32_teacher_20260804.log 2>&1 &
```

Check:

```bash
curl -s http://127.0.0.1:18080/health
curl -s http://127.0.0.1:18080/metadata
```

The metadata tokenizer hash must match the student during diagnostic preflight.

## Phase D — mandatory two-prompt K=32 smoke

This smoke exercises the 33-response chunked scoring path.  It is intentionally
run with `--mode full`; `--mode smoke` would silently override K=32 to K=2.

Example using physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 \
bash scripts/hpc/run_support_aware_diagnostic.sh \
  --config configs/experiment/support_aware_geometry3k_k32.yaml \
  --mode full \
  --teacher-url http://127.0.0.1:18080 \
  --student-model ${DTOPD_MODEL_ROOT}/Qwen3-VL-4B-Instruct \
  --dataset ${K32_COHORT_DIR}/cohort.parquet \
  --output-root ${DTOPD_OUTPUT_ROOT} \
  --num-prompts 64 \
  --prompt-start 0 \
  --prompt-end 2 \
  --run-id diag_full_k32_20260804_smoke2 \
  --device cuda \
  --dtype bfloat16 \
  --exit-zero-on-complete
```

Do not proceed merely because the shell exit code is zero.  In the smoke
`summary.json` and `run_manifest.json`, require:

- two prompt summaries;
- 64 stochastic rows plus two greedy rows, exactly
  `2 * (32 + 1) = 66` total rollout rows;
- each prompt has stochastic rollout IDs 1 through 32 and one greedy ID 0;
- exact-token identity 1.0;
- prompt-token hash availability 1.0;
- complete rollout coverage 1.0;
- zero missing images;
- zero non-finite scores;
- tokenizer alignment true.

Malformed/truncation/teacher-AUC gates may fail on two prompts and are not smoke
failures.  Any token/hash/count/non-finite failure is fatal.

## Phase E — launch four K=32 shards

The teacher server is synchronous, so scoring requests serialize.  Student
generation still runs in parallel; the 1,800-second timeout allows requests to
queue.  Start with one student GPU per shard.  Replace `1,2,3,4` with verified
free physical GPU IDs and do not include the teacher GPU.

```bash
mkdir -p ${DTOPD_OUTPUT_ROOT}/../logs/support_aware_k32

nohup env \
  DTOPD_PYTHON=${DTOPD_PYTHON} \
  DTOPD_MODEL_ROOT=${DTOPD_MODEL_ROOT} \
  DTOPD_OUTPUT_ROOT=${DTOPD_OUTPUT_ROOT} \
  K32_RUN_PREFIX=${K32_RUN_PREFIX} \
  K32_TEACHER_URL=http://127.0.0.1:18080 \
  bash scripts/hpc/launch_support_aware_k32_parallel.sh \
    ${K32_COHORT_DIR} 1,2,3,4 4 \
  >${DTOPD_OUTPUT_ROOT}/../logs/support_aware_k32/launcher_20260804.log 2>&1 &
```

Expected run IDs:

```text
diag_full_k32_20260804_s0_0_16
diag_full_k32_20260804_s1_16_32
diag_full_k32_20260804_s2_32_48
diag_full_k32_20260804_s3_48_64
```

The shard script is idempotent:

- complete `PASS`/`GATE_FAIL` runs are left untouched;
- partial runs with `prompt_support_summary.jsonl` are resumed;
- an ambiguous pre-existing directory is rejected rather than overwritten.

Monitor without editing the checkout:

```bash
tail -f ${DTOPD_OUTPUT_ROOT}/../logs/support_aware_k32/${K32_RUN_PREFIX}/shard_0.log
find ${DTOPD_OUTPUT_ROOT}/support_aware_opd -path "*${K32_RUN_PREFIX}*" -name _partial_summary.json -print
```

## Phase F — strict merge and validation

After all shards complete:

```bash
DTOPD_PYTHON=${DTOPD_PYTHON} \
DTOPD_OUTPUT_ROOT=${DTOPD_OUTPUT_ROOT} \
K32_RUN_PREFIX=${K32_RUN_PREFIX} \
bash scripts/hpc/merge_support_aware_k32.sh ${K32_COHORT_DIR} 4
```

Expected merged path:

```text
${DTOPD_OUTPUT_ROOT}/support_aware_opd/diag_full_k32_20260804_merged
```

The strict merge rejects:

- incomplete or overlapping shard intervals;
- different configs, commits, or dirty states;
- missing greedy/stochastic rollout keys;
- tokenizer/model mismatch;
- decoded/re-tokenized response drift;
- teacher/student token-hash mismatch;
- missing prompt-token hashes;
- non-exact response masks.

`k32_validation.json` must have `valid: true`.  Statistical gate failures remain
listed under `gate_failures_are_reported_not_suppressed`; the launcher never
converts them into PASS.

## Morning handoff

Claude Code should return only after providing:

1. the pulled git commit and dirty state;
2. teacher model ID and tokenizer hash;
3. cohort directory and all source/cohort hashes;
4. four shard directories and exit statuses;
5. merged directory and `k32_validation.json`;
6. prompt count, K, rollout count, exact-token identity, missing/non-finite count;
7. malformed/truncation/duplicate rates;
8. K=32 observed-stratum counts;
9. failures or restarts with the original logs preserved.

Do not start bridge training in the same overnight task.  K=32 confirmation is
the stopping point; bridge cohort freezing and teacher-proposal feasibility are
the next morning decisions.
