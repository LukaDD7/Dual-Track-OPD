# Next GPU Experiment: Verified Teacher-Proposal Feasibility

Status: executable GPU handoff.  This is the immediate post-K=32 experiment.
It creates a verified proposal cache and student-reachability measurements; it
does **not** update the policy and must not be reported as an FKL/OPD gain.

## Scientific decision

The K=32 confirmation identified 11 `no_correct_observed` and 17
`rare_success` prompts in the frozen 64-prompt cohort.  Before modifying a
trainer, test the prerequisite shared by TREK-like FKL and BRTS-like hybrid
bridges:

1. can the 32B teacher produce at least one verifier-passing trajectory in four
   attempts;
2. can the frozen 4B student assign finite exact-token likelihood to that
   teacher trajectory under the same prompt/image history;
3. among correct proposals, which two have the lowest two-sided trimmed,
   length-normalized student NLL?

This experiment is worth running even if the later bridge is redesigned.  Its
`retained_proposals.jsonl` is already the immutable raw-token target dataset for
the mandatory verified-FKL baseline.

## Fixed contract

| Item | Setting |
|---|---|
| Student | `Qwen3-VL-4B-Instruct` frozen |
| Teacher/proposal source | `Qwen3-VL-32B-Instruct` |
| Prompt states | K32-confirmed no-correct + rare-success (expected 28 prompts) |
| Proposals | 4 per prompt, temperature 0.7, top-p 0.95, max 4096 |
| Verification | existing conservative Geometry3K final-answer verifier |
| Reachability | exact teacher token IDs forced under student |
| Trim | discard floor(10%) lowest NLL and floor(2%) highest NLL tokens |
| Retention | lowest-NLL two verifier-passing proposals per prompt |
| Main seed | 20260805 |

The 10% low-loss trim prevents boilerplate/format tokens from making a proposal
look artificially close.  The 2% high-loss trim prevents one isolated rare
token from making it look artificially far.  The score is a ranking proxy, not
a training loss or a proof of semantic teachability.

## Required paths

```bash
export PROJECT_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
export DTOPD_PYTHON=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vision-opd-cu128/bin/python
export DTOPD_MODEL_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models
export DTOPD_OUTPUT_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs
export K32_RUN=${DTOPD_OUTPUT_ROOT}/support_aware_opd/diag_full_k32_20260804_merged
export K32_COHORT=${DTOPD_OUTPUT_ROOT}/support_aware_opd/k32_cohort_20260804
```

Do not reuse a running HTTP teacher service for this experiment.  Each shard
loads one local teacher on logical `cuda:0` and one local student scorer on
logical `cuda:1`, so teacher generation and exact-token student reachability
remain in one auditable process.

## CPU/network instance: pull and preflight

```bash
cd "${PROJECT_ROOT}"
git fetch origin
git switch codex/va-opd
git pull --ff-only origin codex/va-opd
git status --short
git rev-parse HEAD

test -f "${K32_RUN}/k32_validation.json"
test -f "${K32_RUN}/prompt_support_summary.jsonl"
test -f "${K32_COHORT}/cohort.parquet"

"${DTOPD_PYTHON}" -m pytest -q tests/test_support_proposal_feasibility.py
"${DTOPD_PYTHON}" -m dual_track_opd.support_aware.proposal_feasibility --help
```

Stop if the checkout is dirty.  All full shards and their merge must record one
identical clean commit.  Raw proposal outputs remain under NFS and are not
committed.

## GPU instance: mandatory smoke

Use one verified-free teacher/student GPU pair.  The example maps physical GPU
0 to logical `cuda:0` for the 32B teacher and physical GPU 1 to logical
`cuda:1` for the 4B student.

```bash
cd "${PROJECT_ROOT}"
git fetch origin
git switch codex/va-opd
git pull --ff-only origin codex/va-opd
git status --short

export DTOPD_PYTHON DTOPD_MODEL_ROOT DTOPD_OUTPUT_ROOT

CUDA_VISIBLE_DEVICES=0,1 \
"${DTOPD_PYTHON}" -u -m dual_track_opd.support_aware.proposal_feasibility run \
  --config configs/experiment/support_aware_teacher_proposal_feasibility.yaml \
  --output-dir "${DTOPD_OUTPUT_ROOT}/support_aware_opd/proposal_feasibility_20260805_smoke" \
  --max-prompts 2 \
  --proposals-per-prompt 2 \
  --max-new-tokens 512
```

The smoke must produce:

```text
run_manifest.json
summary.json
proposals.jsonl
retained_proposals.jsonl
prompt_results/*.json
```

Required smoke checks:

- `summary.complete == true`;
- `expected_prompt_count == completed_prompt_count == 2`;
- `proposal_count == 4`;
- every proposal has non-empty `response_token_ids` and matching
  `response_token_hash`;
- every correct proposal has finite
  `trimmed_length_normalized_nll`, non-empty student token log-probs, and
  `student_scored_token_hash == response_token_hash`;
- incorrect/malformed proposals are never marked `retained_for_fkl`.

The smoke may find zero correct proposals; that is a scientific result, not a
protocol failure.  Token mismatch, missing artifacts, OOM, non-finite score on
a correct proposal, or a dirty/mismatched checkout is a protocol failure.

## Full run: occupy 8 GPUs with four independent pairs

After the smoke passes, remove the smoke pair from any other scheduler and run:

```bash
cd "${PROJECT_ROOT}"
export DTOPD_PYTHON DTOPD_MODEL_ROOT DTOPD_OUTPUT_ROOT
mkdir -p "${DTOPD_OUTPUT_ROOT}/../logs"

nohup bash scripts/hpc/launch_support_aware_proposals_parallel.sh \
  0:1,2:3,4:5,6:7 proposal_feasibility_20260805 \
  >"${DTOPD_OUTPUT_ROOT}/../logs/proposal_feasibility_20260805_launcher.log" 2>&1 &
```

Mapping is `teacher_gpu:student_gpu`.  Four shards split the same deterministic
28-prompt state-filtered UID list into non-overlapping contiguous slices.  Each
shard is resumable at prompt granularity: a prompt artifact is written through
an atomic temporary file only after all proposals and correct-proposal student
scores are complete.

Monitor:

```bash
tail -f "${DTOPD_OUTPUT_ROOT}/../logs/support_aware_proposals/proposal_feasibility_20260805/shard_0.log"
find "${DTOPD_OUTPUT_ROOT}/support_aware_opd" \
  -path '*proposal_feasibility_20260805_s*/summary.json' -print
```

If an instance is reclaimed, rerun the identical launcher from the identical
clean commit.  Existing per-prompt JSON files are skipped only when the config
and input provenance match exactly.  Do not delete partial output directories.

## Strict CPU merge

After all four shards report `status=completed`:

```bash
cd "${PROJECT_ROOT}"
export DTOPD_PYTHON DTOPD_OUTPUT_ROOT
bash scripts/hpc/merge_support_aware_proposals.sh \
  4 proposal_feasibility_20260805 proposal_feasibility_20260805_merged
```

The merge rejects:

- incomplete shard manifests;
- different configs, commits, or dirty states;
- missing or overlapping shard indices;
- missing, overlapping, or incomplete UID coverage;
- a pre-existing merged output directory.

## Morning decision table

Claude Code must report these values separately for `no_correct_observed` and
`rare_success`:

| Metric | Interpretation |
|---|---|
| prompts with at least one correct teacher proposal / prompts | proposal availability |
| correct proposals / total teacher proposals | teacher sample efficiency |
| retained proposals / prompts | size of executable FKL cache |
| retained trimmed NLL median/p10/p90 | student reachability |
| proposal truncation and malformed rates | target quality/confounding |
| teacher-generated tokens and wall time | proposal compute budget |

Decision rules:

1. **No-correct proposal availability >=50% and at least 8 prompts retained:**
   implement the verified-FKL bridge immediately; compare against no bridge and
   exact-token OPD on the same prompt manifest.
2. **Availability is high but retained NLL is extreme or responses mostly
   truncate:** proposals exist but are not student-compatible; test BRTS/SGPO
   compatibility selection before broad FKL.
3. **Availability <25%:** the current 32B teacher is not a sufficient support
   creator; do not spend a night training FKL on a tiny biased cache.  Switch
   proposal source (answer hint, stronger teacher, or strategy guidance).
4. **Rare succeeds while no-correct fails:** proceed with OPD/BRTS on rare
   support, but treat true zero-support creation as unresolved.

These are feasibility gates, not claims that one operator improves downstream
RL.  The next causal step is a fixed-budget bridge followed by fresh K=32 and
matched G=8 early GRPO.

## Required handback

Return:

1. pulled repo commit and clean/dirty status;
2. physical GPU pair mapping and model paths;
3. smoke directory and exact summary;
4. four shard directories and manifests;
5. merged directory and SHA256 of `retained_proposals.jsonl`;
6. state-separated morning decision table;
7. every restart/error with original logs preserved;
8. no bridge training unless a decision rule above passes and the retained
   proposal cache has been inspected.
