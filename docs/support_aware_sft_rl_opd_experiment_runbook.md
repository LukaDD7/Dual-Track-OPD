# Support-Aware SFT-RL-OPD Experiment Runbook

> **Scope update (2026-08-03):** Step 1 remains the authoritative frozen-policy
> diagnostic and is already running.  The old Step-2 comparison of RL-only,
> uniform RKL, and support-gated RKL is superseded by
> `docs/frontier_operator_causal_experiment.md`, which adds mandatory
> TREK-like verified-FKL and FKL→OPD baselines.  Do not launch the old three-arm
> pilot as the main study; it cannot distinguish the current claim from PACED,
> TREK, SRPO, or the sparse-to-dense pipeline.
> The detailed TREK algorithm, FKL/RKL boundary analysis, and 22-paper core
> matrix are in `docs/trek_competitive_landscape_and_fkl_opd.md`.

Status: implementation plan

Date: 2026-07-29

Owner: 震扬

Primary question: can teacher-guided OPD improve the early RL learning curve by
promoting correct low-probability reasoning modes without collapsing rollout
diversity?

This document is the operational specification for the first support-aware
SFT-RL-OPD experiments. It is written for a coding agent working in this
repository. Follow the contracts below; do not substitute a different dataset,
silently train on a benchmark split, or move research logic into
`third_party/verl/`.

## 1. Research Claim and Scope

The project is not proposing another visual-token weighting rule. The working
claim is:

> SFT/pre-RL initialization, RL, and OPD should be coordinated according to the
> current policy's effective support. RL explores and promotes candidate
> reasoning modes; teacher-guided OPD should be applied when it can distinguish
> a useful low-probability mode, and should consolidate that mode without
> suppressing exploration.

The first experiment tests only two necessary premises:

1. On frozen student rollouts, does the teacher assign a useful ranking signal
   to correct trajectories that are not the student's dominant mode?
2. If yes, does support-gated reverse-KL improve the early RL learning curve
   relative to RL-only and uniform reverse-KL at the same student rollout and
   optimizer-step budget?

This pilot does **not** attempt to prove a complete pretraining/SFT/RL scaling
law. It also does not compare every possible ordering of SFT, RL, and OPD.
Pre-RL, interleaved, and post-RL OPD timing becomes a follow-up experiment only
after the two premises above pass.

## 2. Fixed Decisions

### 2.1 Model roles

- Pre-RL policy / student:
  `Qwen3-VL-4B-Instruct`, using the existing local checkpoint.
- Teacher:
  `Qwen3-VL-32B-Instruct`, using the existing Transformers teacher service.
- Terminology:
  call the student checkpoint a **pre-RL initialization**, not a clean
  SFT-only checkpoint. Qwen Instruct checkpoints may contain multiple
  post-training stages.
- First pilot:
  do not train a new SFT checkpoint.
- Later causal study:
  train and save a controlled SFT-only checkpoint from a Base model, with an
  explicit manifest of the SFT traces and composition coverage.

Default HPC model paths:

```text
$DTOPD_MODEL_ROOT/Qwen3-VL-4B-Instruct
$DTOPD_MODEL_ROOT/Qwen3-VL-32B-Instruct
```

### 2.2 Dataset roles

Use **Geometry3K official train data** for frozen-policy diagnostics and the
micro-training pilot. The repository already has:

- a Geometry3K adapter;
- clean image handling;
- a deterministic answer verifier;
- prepared Parquet and validation paths;
- online student/teacher forced-scoring infrastructure.

Use **Geometry3K val/test** for model selection and learning-curve measurement.
No val/test row may enter training or teacher-guided update construction.

Use **DynaMath only as a held-out OOD evaluation** after the pilot recipe has
been selected on Geometry3K. DynaMath is an evaluation benchmark consisting of
501 programmatic seed questions and released variants. Do not train on the
released 5,010 benchmark rows, and do not repeatedly tune hyperparameters on
DynaMath.

The intended data flow is:

```text
Geometry3K train
  -> frozen rollout/support diagnostic
  -> 50-step micro-training

Geometry3K val/test
  -> checkpoint selection and learning-curve metrics

DynaMath
  -> one-time OOD readout after the pilot configuration is frozen
```

Primary prepared Geometry3K path on HPC:

```text
/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs/fc_opd/geometry3k_full/train.parquet
```

Existing validation convention:

```text
<train-path-without-.parquet>_val200.parquet
```

Before running, verify that the validation rows are disjoint from training by
`sample_uid`, source path, and image hash. Do not rely only on row position.

### 2.3 Three operational steps

1. **Frozen-policy diagnostic**: sample student trajectories, verify outcomes,
   and teacher/student-force score the exact same token IDs.
2. **Micro-training pilot**: compare RL-only, uniform RKL, and support-gated RKL
   for 50 student optimizer steps.
3. **Conditional scale-out**: only if Steps 1 and 2 pass, repeat with seeds and
   evaluate once on DynaMath.

Step 3 is a decision gate, not part of the first implementation milestone.

## 3. Existing Components to Reuse

Do not reimplement these components:

| Purpose | Existing component |
| --- | --- |
| Student rollout generation | `dual_track_opd.fc_opd.student_rollout_signal_audit.HFQwenStudentRolloutGenerator` |
| Structured rollout prompt | `fc_opd_structured_v2` in `student_rollout_signal_audit.py` |
| Teacher HTTP client | `dual_track_opd.fc_opd.teacher_client.TeacherClient` |
| Exact teacher forced scoring | `teacher_transformers.py`; response field `sampled_token_log_probs` |
| Teacher score tensor | `signal_decomposer.TeacherTopK.sampled_log_probs` |
| Student forced scoring | `dual_track_opd.fc_opd.student_scorer.StudentScorer` or the already-loaded rollout model |
| Geometry3K normalization | `dual_track_opd.fc_opd.geometry3k_adapter` |
| Geometry3K outcome verifier | `dual_track_opd.fc_opd.verifier.verify_geometry3k_response` |
| Online OPD batch construction | `dual_track_opd.fc_opd.online_batch` |
| verl tensor bridge | `dual_track_opd.fc_opd.verl_integration` |
| Sparse KD/RKL tensor math | `dual_track_opd.fc_opd.verl_sparse_kd` and current VA-OPD objective code |
| Teacher launcher | `scripts/hpc/start_fc_teacher.sh` |

Important current gap:

`student_rollout_signal_audit.py` already samples responses and receives
`TeacherTopK.sampled_log_probs`, but its JSONL writer does not persist the
teacher sampled-token log-probabilities, student sampled-token
log-probabilities, or Geometry3K correctness. The first implementation task is
to add a separate support-diagnostic layer that preserves these fields without
breaking the existing FC-OPD audit.

## 4. Step 1 — Frozen-Policy Support Diagnostic

### 4.1 Purpose

This step performs generation and forward scoring only. It has no backward
pass and no parameter update.

For each prompt:

1. generate one greedy response;
2. sample `K=8` stochastic responses from the frozen pre-RL student;
3. verify every final answer;
4. teacher-force score the exact response token IDs;
5. student-force score the exact same token IDs;
6. classify the prompt's observed support state;
7. test whether teacher scores rank correct-tail responses above wrong modes.

### 4.2 Pilot size and sampling

Use a deterministic 128-prompt subset from Geometry3K train.

Selection rule:

```text
sort by SHA256(sample_uid)
take the first 128 rows after all data-validity filters pass
```

Do not use `head(128)` on filesystem order. Store the selected `sample_uid`
list and its SHA256 in the run manifest.

Smoke configuration:

```yaml
num_prompts: 8
rollouts_per_prompt: 2
temperature: 0.7
top_p: 0.9
max_new_tokens: 256
seed: 42
```

Full diagnostic configuration:

```yaml
num_prompts: 128
rollouts_per_prompt: 8
temperature: 0.7
top_p: 0.95
max_new_tokens: 512
seed: 42
response_format: fc_opd_structured_v2
teacher_condition: full
```

Each stochastic rollout seed must be:

```text
seed + source_index * 1000 + rollout_id
```

The greedy response uses `temperature=0` and a separately recorded seed field.

### 4.3 Prompts and leakage rules

Use the existing `fc_opd_structured_v2` output contract:

```text
<visible_evidence>...</visible_evidence>
<diagram_inference>...</diagram_inference>
<reasoning>...</reasoning>
<answer>...</answer>
```

The prompt may contain the clean question, answer choices, and clean image. It
must not contain:

- the gold answer;
- `reward_model.ground_truth`;
- `extra_info.answer`;
- solution traces;
- answer-bearing task evidence;
- generated red-box/crop annotations.

Gold metadata is used only after generation by the verifier.

### 4.4 Exact forward-scoring contract

For a rollout `y=(y_1,...,y_T)`, compute:

```text
student_mean_logp =
    (1/T) * sum_t log p_student(y_t | x, y_<t)

teacher_mean_logp =
    (1/T) * sum_t log p_teacher(y_t | x, y_<t)

teacher_gap =
    teacher_mean_logp - student_mean_logp
```

Requirements:

- use the exact sampled response token IDs;
- use the same clean image and canonical question for student and teacher;
- use direct sampled-token log-probabilities, not top-k tail-bucket
  approximations;
- length-normalize before ranking trajectories;
- retain the unnormalized sum and token count for auditing;
- exclude XML tag tokens from a secondary content-only score, but keep an
  all-response score as the primary reproducible metric;
- do not retokenize teacher text into different token IDs;
- fail on tokenizer-hash or response-position misalignment.

The teacher service already returns exact per-position
`sampled_token_log_probs`. The student path must gather the sampled response
token log-probabilities from its full-condition forced forward.

The primary pilot requires the student and teacher to share the same tokenizer
hash and vocabulary indexing. Check this before generating the 128-prompt
diagnostic. If the hashes differ:

1. do not subtract teacher and student token-level log-probabilities;
2. do not run token-aligned RKL;
3. optionally produce a separately labeled text-likelihood diagnostic by
   retokenizing the response under each model and normalizing sequence
   log-likelihood by UTF-8 byte length;
4. treat Step 2 as blocked until a tokenizer-compatible teacher/student pair or
   a formally specified cross-tokenizer distillation method is selected.

The optional byte-normalized diagnostic is not interchangeable with
`teacher_gap` and must not pass the Step 1 gate for token-aligned RKL.

### 4.5 Verifier and effective-support states

Let `c_i` be the number of correct stochastic rollouts among `K` for prompt
`i`. Let `greedy_correct_i` be the greedy response outcome.

Count correctness from the verifier's boolean `correct` field. Do not treat the
diagnostic learning-value reward (`0.25` for a format-valid wrong answer in some
offline paths) as success. RL reward for the pilot must use the binary
Geometry3K answer score implemented by
`dual_track_opd.fc_opd.smoke_reward.compute_score`.

Assign exactly one state:

```text
exposed:
    greedy_correct_i == true
    OR c_i / K >= 0.5

correct_tail:
    greedy_correct_i == false
    AND 0 < c_i < K / 2

no_correct_observed:
    greedy_correct_i == false
    AND c_i == 0

other:
    all remaining boundary cases
```

`no_correct_observed` does not mean mathematical zero support. It means no
correct response was observed under the stated `K`, temperature, top-p, and
seed budget.

The first pilot must not claim step-level or partial correctness. Geometry3K
provides a final-answer verifier, not ground-truth reasoning modules.

### 4.6 Required outputs

All raw artifacts stay outside Git:

```text
$DTOPD_OUTPUT_ROOT/support_aware_opd/<run_id>/
  run_manifest.json
  resolved_config.yaml
  selected_prompts.jsonl
  rollouts.jsonl
  prompt_support_summary.jsonl
  summary.json
  logs/
```

`rollouts.jsonl` must contain at least:

```text
run_id
sample_uid
source_index
split
rollout_id
is_greedy
generation_seed
temperature
top_p
max_new_tokens
response_text
response_token_ids
response_token_count
response_hash
answer_extracted
gold_answer
format_valid
correct
student_sampled_token_log_probs
teacher_sampled_token_log_probs
student_mean_logp
teacher_mean_logp
teacher_gap
student_model_path
teacher_model_id
prompt_hash
image_hash
tokenizer_hash
errors
```

`prompt_support_summary.jsonl` must contain:

```text
sample_uid
K
correct_count
greedy_correct
support_state
unique_response_count
duplicate_rollout_rate
teacher_rank_of_best_correct
student_rank_of_best_correct
teacher_gap_rank_of_best_correct
teacher_top1_correct
teacher_gap_top1_correct
```

`summary.json` must include:

- counts by support state;
- response length percentiles;
- malformed rate;
- duplicate rollout rate;
- fraction of prompts with at least one correct rollout;
- correct-vs-wrong AUC for `teacher_mean_logp`;
- correct-vs-wrong AUC for `teacher_gap`;
- correct-tail Rank@1 and MRR;
- greedy, empirical pass@1, and empirical pass@K diagnostics;
- tokenizer/alignment failures;
- all reproducibility fields required by `AGENTS.md`.

### 4.7 Step 1 acceptance gate

Protocol gate:

- zero tokenizer-alignment failures;
- zero missing-image rows;
- zero non-finite student/teacher scores;
- malformed response rate <= 10%;
- duplicate stochastic rollout rate <= 25%.

Signal gate:

- at least 20 `correct_tail` prompts in the 128-prompt diagnostic, or increase
  the diagnostic set once to 256 prompts without changing sampling
  hyperparameters;
- `teacher_gap` correct-vs-wrong AUC >= 0.60;
- correct-tail Rank@1 from `teacher_gap` is higher than random candidate
  ranking;
- manual audit of 20 rows finds no gold-answer leakage.

If the protocol gate fails, fix the pipeline and rerun the same smoke. If the
signal gate fails, stop before micro-training and report that teacher likelihood
does not provide a reliable tail-selection signal under this setup.

### 4.8 Target command interface to implement

Add:

```text
configs/experiment/support_aware_geometry3k_pilot.yaml
src/dual_track_opd/analysis/support_diagnostic.py
scripts/hpc/run_support_aware_diagnostic.py
scripts/hpc/run_support_aware_diagnostic.sh
tests/test_support_diagnostic.py
```

Target smoke command:

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD

export DTOPD_MODEL_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models
export DTOPD_OUTPUT_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs

FC_OPD_TEACHER_MODEL="$DTOPD_MODEL_ROOT/Qwen3-VL-32B-Instruct" \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/hpc/start_fc_teacher.sh
```

In a second terminal:

```bash
CUDA_VISIBLE_DEVICES=1 \
bash scripts/hpc/run_support_aware_diagnostic.sh \
  --config configs/experiment/support_aware_geometry3k_pilot.yaml \
  --mode smoke
```

Target full diagnostic:

```bash
CUDA_VISIBLE_DEVICES=1 \
bash scripts/hpc/run_support_aware_diagnostic.sh \
  --config configs/experiment/support_aware_geometry3k_pilot.yaml \
  --mode full
```

The runner must refuse to start if the teacher health check, tokenizer hash,
dataset split audit, output-directory uniqueness check, or GPU availability
check fails.

## 5. Step 2 — 50-Step Micro-Training Pilot

### 5.1 Purpose

Test whether the diagnostic signal changes RL learning dynamics. The experiment
compares three arms from the exact same pre-RL checkpoint:

| Arm | Actor objective |
| --- | --- |
| `rl_only` | GRPO only |
| `rl_uniform_rkl` | GRPO + fixed-weight reverse-KL on all current rollouts |
| `rl_support_gated_rkl` | GRPO + reverse-KL only for mixed-support groups, weighted by online teacher gap |

The uniform-RKL arm is necessary. Without it, an improvement cannot be
attributed to support-aware scheduling rather than adding any teacher loss.

### 5.2 Online-only requirement

Step 2 must be on-policy:

1. sync the latest actor to the rollout worker;
2. sample current student rollouts;
3. verify the current rollouts;
4. teacher-force score those exact current token IDs;
5. student-force score those exact current token IDs;
6. compute the support gate and RKL weights;
7. apply GRPO plus auxiliary OPD loss;
8. update the actor in the same step.

Do not train from the Step 1 offline JSONL. Those scores become stale after the
first optimizer update.

### 5.3 Pilot hyperparameters

Fixed for all arms:

```yaml
student: Qwen3-VL-4B-Instruct
teacher: Qwen3-VL-32B-Instruct
dataset: Geometry3K train
train_prompts: 256 deterministic rows
validation_prompts: Geometry3K val200
optimizer_steps: 50
eval_steps: [0, 10, 25, 50]
rollouts_per_prompt: 4
temperature: 0.7
top_p: 0.95
max_new_tokens: 512
train_batch_size: 8
learning_rate: 1.0e-6
seed: 42
```

Pilot-only OPD settings:

```yaml
rkl_lambda: 0.05
teacher_gap_temperature: 1.0
```

Do not tune these values on DynaMath. One seed is enough for the pilot but can
only justify the label `promising`, not a research conclusion.

### 5.4 Support-gated RKL definition

For an online rollout group `i` with `K` responses and `c_i` verified correct:

```text
group_gate_i = 1 if 0 < c_i < K else 0
```

For rollout `j`:

```text
gap_ij = mean_t(log p_teacher(y_ijt) - log p_student(y_ijt))

weight_ij =
    softmax_j(gap_ij / teacher_gap_temperature)
```

The support-gated auxiliary loss is:

```text
L_support_opd =
    sum_i group_gate_i *
    sum_j weight_ij * RKL(student || teacher; y_ij)
```

The full actor objective is:

```text
L_actor = L_GRPO + rkl_lambda * L_support_opd
```

For `rl_uniform_rkl`, set `group_gate_i=1` and `weight_ij=1/K`.

This pilot deliberately applies OPD to mixed-outcome groups: the correct mode
has appeared, but it is not yet uniformly dominant. It does not claim to solve
`no_correct_observed` groups. Teacher proposals or mixture rollouts are a later
support-expansion experiment.

### 5.5 Implementation constraints

Research logic belongs in:

```text
src/dual_track_opd/losses/support_gated_opd.py
src/dual_track_opd/analysis/support_diagnostic.py
configs/experiment/support_aware_geometry3k_pilot.yaml
```

The verl integration should be a thin transport/call-site patch stored under:

```text
patches/verl/
```

Do not directly vendor a changed `third_party/verl`.

Reuse the current online post-rollout hook and student scorer. Preserve the
online student scorer in the main path; do not replace it with a teacher-only
router. The support gate requires current student likelihoods and current
rollout outcomes.

The online student scorer must represent the same actor weights that produced
the current rollout. A separately loaded `StudentScorer` is acceptable only if
its weights are explicitly synchronized at every actor-to-rollout sync. Prefer
reusing live actor forced logits when practical. Record an actor/scorer version
or step ID in each batch and fail if the scorer is stale.

Add unit tests for:

- support-state classification boundaries;
- teacher-gap length normalization;
- group softmax weights summing to one;
- zero OPD loss for `c_i=0` and `c_i=K`;
- uniform arm weights;
- finite loss and gradients;
- no gradient through teacher scores;
- identical behavior at `rkl_lambda=0` and RL-only auxiliary path;
- padded response-mask handling.

### 5.6 Metrics

Primary:

- Geometry3K validation pass@1 at steps 0, 10, 25, and 50;
- area under the early pass@1-vs-step curve;
- empirical correct-tail promotion rate on a fixed validation prompt set.

Safety:

- pass@4;
- unique responses per prompt;
- duplicate rollout rate;
- response entropy;
- malformed rate;
- wrong-mode amplification proxy: initially dominant wrong answer receives
  higher empirical frequency after training while no correct candidate appears.

Systems:

- student rollout tokens;
- optimizer steps;
- teacher forward tokens;
- wall-clock time;
- peak GPU memory;
- NCCL/Ray failures.

Student rollout and optimizer-step budgets must match across all three arms.
Teacher compute is additional for the two OPD arms and must be reported
separately; do not claim total-compute efficiency from this pilot.

### 5.7 Step 2 decision rule

Call the result `promising` only if:

- `rl_support_gated_rkl` improves early pass@1 curve area or step-50 pass@1
  over `rl_only`;
- pass@4 decreases by no more than 2 percentage points;
- duplicate rollout rate increases by no more than 10 percentage points;
- support-gated RKL is better than, or clearly more stable than, uniform RKL;
- all arms complete 50 optimizer steps with finite loss and gradients.

If support-gated RKL only lowers RKL loss, or only improves training prompts,
the hypothesis is not supported.

### 5.8 Target command interface to implement

Add:

```text
scripts/hpc/run_support_aware_geometry3k_pilot.sh
tests/test_support_gated_opd.py
```

Target preflight:

```bash
bash scripts/hpc/run_support_aware_geometry3k_pilot.sh \
  --config configs/experiment/support_aware_geometry3k_pilot.yaml \
  --preflight-only
```

Target arms:

```bash
bash scripts/hpc/run_support_aware_geometry3k_pilot.sh \
  --config configs/experiment/support_aware_geometry3k_pilot.yaml \
  --arm rl_only \
  --steps 50 \
  --teacher-gpus 0 \
  --train-gpus 1,2,3,4

bash scripts/hpc/run_support_aware_geometry3k_pilot.sh \
  --config configs/experiment/support_aware_geometry3k_pilot.yaml \
  --arm rl_uniform_rkl \
  --steps 50 \
  --teacher-gpus 0 \
  --train-gpus 1,2,3,4

bash scripts/hpc/run_support_aware_geometry3k_pilot.sh \
  --config configs/experiment/support_aware_geometry3k_pilot.yaml \
  --arm rl_support_gated_rkl \
  --steps 50 \
  --teacher-gpus 0 \
  --train-gpus 1,2,3,4
```

The runner must create a unique run directory, record the resolved config and
commits, verify power-of-two training world size, and fail if requested
optimizer steps or validation checkpoints are missing.

## 6. Step 3 — Conditional Scale-Out

Do not implement Step 3 until the Step 2 report is reviewed.

If the pilot is promising:

1. rerun the three arms with three seeds and a larger Geometry3K training
   subset;
2. freeze the method and hyperparameters using Geometry3K only;
3. evaluate RL-only and support-gated OPD once on the full DynaMath protocol;
4. then study OPD timing (`before RL`, `interleaved`, `after RL`) and teacher
   proposal/mixture rollout for `no_correct_observed` states.

The intended larger contribution is a support-aware multi-stage post-training
framework: SFT/pre-RL initialization establishes reusable capabilities, RL
discovers and recombines candidate reasoning modes, and OPD is scheduled to
guide or consolidate modes according to their effective support. This differs
from prior work that analyzes SFT-RL roles or applies one fixed distillation
loss uniformly throughout training.

## 7. Reproducibility and Artifact Contract

Every diagnostic and training run must record:

- repository commit and dirty status;
- backend commit or package version;
- full resolved config;
- dataset source, split, selected `sample_uid` list, and manifest hash;
- model and teacher checkpoint paths/revisions;
- tokenizer hash;
- generation parameters and seeds;
- raw output paths outside Git;
- summary metrics;
- environment/package report;
- GPU topology;
- start/end timestamps and exit status;
- notes on any retries or manual intervention.

Never commit:

- raw rollout JSONL;
- images or datasets;
- checkpoints;
- teacher top-k tensors;
- model weights;
- tokens, API keys, or `.env` files.

Small aggregate summaries may be committed after manual inspection.

## 8. Coding-Agent Execution Order

A coding agent should execute the following sequence:

1. Read `AGENTS.md`, this runbook, and the existing files listed in Section 3.
2. Inspect `git status --short`; preserve unrelated user changes.
3. Implement only Step 1 and its tests first.
4. Run CPU synthetic tests, then the 8-prompt GPU smoke.
5. Produce and inspect the Step 1 summary.
6. Stop if the Step 1 acceptance gate fails.
7. If it passes, implement Step 2 behind explicit config flags.
8. Run preflight, then one 2-step smoke per arm.
9. Only after all three arms pass, run 50 steps per arm.
10. Write a compact result report; do not start Step 3 automatically.

The agent must not silently reinterpret `no_correct_observed` as zero model
support, must not use DynaMath for training, and must not claim a method gain
from unmatched rollout or optimizer budgets.

## 9. Minimum Validation Commands

Before any GPU run:

```bash
python -m pip install -e ".[dev]"
pytest -q tests/test_support_diagnostic.py
pytest -q tests/test_support_gated_opd.py
```

Before publishing implementation changes:

```bash
git status --short
git diff --stat
pytest -q
```

If the full test suite is blocked by an external backend or missing model
weights, record the exact failing command and still run all project-owned unit
tests that do not require those external assets.

## 10. External Dataset References

- DynaMath project: <https://dynamath.github.io/>
- DynaMath sample dataset:
  <https://huggingface.co/datasets/DynaMath/DynaMath_Sample>

DynaMath is treated as evaluation-only in this plan.
