# Causal Frontier Bridge: Executable Research Plan

Status: implementation and handoff for the support-aware research line.  This
document deliberately does not modify the Qwen3.5/verl environment line and
does not authorize a training run before the frozen intervention gate below.

## 1. Research boundary

The contribution is not another hard-prompt gate and not a reimplementation of
TREK.  The target causal chain is:

```text
answer-free prefix intervention rescue
  -> minimal causal bridge unit
  -> unaided support-state transition after bridge training
  -> change in expected mixed-group utility U_G
  -> matched-budget early-RL sample efficiency
```

The immediate experiments stop before weight updates.  They answer two
prerequisites:

1. Does robust/local teacher gap retain signal after controlling for length,
   position and truncation?
2. Which answer-free teacher prefix, if any, causally changes the frozen
   student's continuation success relative to fresh unaided and same-length
   wrong-prefix controls?

Only a positive intervention result justifies implementing the localized bridge
training arm.

## 2. TREK code-availability and reuse audit

Paper: <https://arxiv.org/abs/2607.05339>

Checked on 2026-08-04:

- the arXiv abstract/HTML contains no code or project URL;
- GitHub repository search for the full title and arXiv ID returns no result;
- GitHub code search for `Teacher-Routed Exploration via Forward KL` returns no
  result;
- third-party paper indexes show `Request Code`, not a repository;
- the paper states that the internal pipeline is built on HybridFlow/VERL, but
  it does not release that integration.

Conclusion: there is no identifiable official TREK repository to clone or
vendor at this time.  Recheck before the training implementation.  Do not use a
similarly named third-party repository without author/project evidence.

What is reused rather than rebuilt:

- the repository's existing exact-token Qwen3-VL generation and verifier;
- the existing verified teacher-proposal cache and TREK-compatible
  `M=4/r=2/(0.10,0.02)` reachability baseline;
- the repository's existing verl backend for future FKL/GRPO training;
- existing immutable K=32 rollout token IDs and per-token teacher/student
  log-probabilities.

What is new:

- paired robust-gap diagnostics that hold the evaluated rollout set fixed;
- answer-leakage-safe prefix interventions;
- a same-prompt, same-horizon wrong-student-prefix control;
- a minimal-rescue rule based on posterior lift against both controls;
- prediction of later unaided support transition from frozen intervention
  rescue.

## 3. Isolation from the concurrent training-environment line

This delivery may touch only:

```text
src/dual_track_opd/support_aware/
configs/experiment/support_aware_*.yaml
scripts/hpc/*support_aware*
tests/test_support_*.py
docs/causal_frontier_bridge_*.md
```

It must not edit:

```text
configs/environment/
scripts/env/
scripts/qwen35_*
scripts/run_qwen35_*
src/dual_track_opd/fc_opd/prompt_contracts.py
third_party/verl/
patches/verl/
```

Before every push, fetch `origin/codex/va-opd`, rebase the research commit on
top of it, rerun support-aware tests, and push only by fast-forward.  A conflict
in an environment/runtime file means stop and report it; do not resolve by
choosing one side wholesale.

## 4. Experiment A: robust teacher-gap analysis (CPU only)

### 4.1 Why this is not another length bin

The current gap is already length-normalized:

```text
mean_t [log p_teacher(y_t|x,y_<t) - log p_student(y_t|x,y_<t)]
```

Prompt-relative centering cannot change within-prompt ranking.  This analysis
therefore changes the transfer/localization statistic, not merely its scale.
Every alternative is compared with the existing mean gap on exactly the same
rollouts and eligible prompts.

Metrics:

| metric | question |
|---|---|
| `trek_trimmed_gap` | are student-NLL token outliers causing the failure? |
| `prefix_{128,256,512}_gap` | at what horizon does correct/wrong separation disappear? |
| `window_*_gap` | is teacher signal localized to a decision region? |
| `suffix_64_gap` | is all useful signal confined to the final answer region? |
| `length_residual_gap` | does signal survive cross-fitted log-length/finish correction? |

The length residual is fitted without correctness labels and out of fold by
prompt.  It is diagnostic only and must not become a training reward based on
this experiment.

### 4.2 CPU command

```bash
export PROJECT_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD
export DTOPD_PYTHON=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vision-opd-cu128/bin/python
export DTOPD_OUTPUT_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs

cd "${PROJECT_ROOT}"
git fetch origin
git switch codex/va-opd
git pull --ff-only origin codex/va-opd
git status --short

bash scripts/hpc/run_support_aware_gap_robustness.sh
```

Expected outputs:

```text
gap_robustness_20260806/
  gap_scores.jsonl
  gap_analysis.json
  gap_metric_summary.csv
```

Required report:

- eligible prompt count per metric;
- alternative AUC and baseline AUC on the same rows;
- paired delta AUC and its prompt-bootstrap 95% CI;
- the same paired comparison split by finish reason and response-length bin;
- the first prefix/window whose paired delta CI excludes zero, if one exists;
- whether any apparent gain disappears inside the stopped-only stratum.

Do not select a statistic only because it has the highest point estimate.  A
candidate localization statistic needs a non-negative paired CI and a coherent
position trend; it must later predict prefix rescue out of sample.

## 5. Experiment B: frozen answer-free prefix intervention (GPU)

### 5.1 Estimand

For prompt `x`, verified teacher proposal `y_T`, horizon `h`, and frozen student
continuation `z`:

```text
R_teacher(x,h) = P[V(y_T[:h] + z)=1]
R_wrong(x,h)   = P[V(y_wrong[:h] + z)=1]
R_unaided(x)   = P[V(z)=1]
```

The experiment asks whether `R_teacher` exceeds both controls.  It does not ask
whether teacher text is likely under the student; the proposal experiment has
already measured that TREK baseline.

### 5.2 Fixed protocol

| item | value |
|---|---|
| Student | frozen `Qwen3-VL-4B-Instruct` |
| Prompt set | prompts with a retained correct proposal and at least one wrong K32 rollout |
| Teacher proposal | reachability rank 1 retained proposal |
| Horizons | 64, 128, 256, 512 exact response tokens |
| Arms | fresh unaided; teacher prefix; same-prompt wrong-student prefix |
| Continuations | K=8 per valid arm/horizon |
| Decode | temperature 0.7, top-p 0.95, max continuation 2048 |
| Reward | conservative Geometry3K final-answer verifier |
| Main group size | G=8 |
| Seed | 20260806 |

The prefix is rejected before generation when:

- the source is shorter than the requested horizon;
- decoded prefix contains `Answer:` or `\boxed{`;
- the verifier can already extract an answer from the prefix.

Wrong-prefix control selection first chooses the highest-student-likelihood
wrong K32 trajectory that is long enough and answer-free at that horizon, then
breaks ties by rollout ID.  It uses the same prompt and exact horizon, making it
a stronger control than an arbitrary failure and ruling out the explanation
that success improves merely because the student receives more response
history.

### 5.3 Preregistered rescue rule

With independent Jeffreys posteriors for K=8 continuation outcomes, a horizon
is a rescue only if all four hold:

```text
E[p_teacher - p_wrong]   >= 0.20
P(p_teacher > p_wrong)   >= 0.90
E[p_teacher - p_unaided] >= 0.20
P(p_teacher > p_unaided) >= 0.90
```

The minimal causal bridge is the shortest horizon satisfying the rule.  This
is a pilot decision threshold, not a publication-level significance claim.

### 5.4 Mandatory one-prompt smoke

Run only after the verified proposal merge exists.

```bash
export DTOPD_MODEL_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models
export DTOPD_OUTPUT_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/fc-opd-storage/outputs
export DTOPD_PYTHON=/inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/vision-opd-cu128/bin/python

cd "${PROJECT_ROOT}"
git pull --ff-only origin codex/va-opd
git status --short

CUDA_VISIBLE_DEVICES=0 "${DTOPD_PYTHON}" -u \
  -m dual_track_opd.support_aware.prefix_intervention run \
  --config configs/experiment/support_aware_prefix_intervention.yaml \
  --output-dir "${DTOPD_OUTPUT_ROOT}/support_aware_opd/prefix_intervention_20260806_smoke" \
  --max-prompts 1 \
  --horizons 64 128 \
  --continuations-per-arm 2 \
  --max-continuation-tokens 256
```

Smoke acceptance:

- `summary.complete == true` and exactly one completed prompt;
- unaided unit has exactly 2 continuations;
- each non-skipped prefix unit has exactly 2 continuations;
- composite response IDs begin byte-for-byte with recorded prefix IDs;
- prefix and continuation hashes are non-empty;
- no teacher prefix with `skipped_reason` has a rollout;
- no accepted prefix contains a final-answer marker or extractable answer;
- verifier output is present even when all continuations are wrong;
- no OOM or non-deterministic resume mismatch.

### 5.5 Full 8-GPU launch

The experiment is student-only.  One physical GPU runs one shard, so all eight
GPUs generate continuations rather than reserving four cards for an intermittent
scorer.

```bash
mkdir -p "${DTOPD_OUTPUT_ROOT}/../logs"

nohup bash scripts/hpc/launch_support_aware_prefix_intervention_parallel.sh \
  0,1,2,3,4,5,6,7 prefix_intervention_20260806 \
  >"${DTOPD_OUTPUT_ROOT}/../logs/prefix_intervention_20260806_launcher.log" 2>&1 &
```

Monitor:

```bash
tail -f "${DTOPD_OUTPUT_ROOT}/../logs/support_aware_prefix/prefix_intervention_20260806/shard_0.log"
find "${DTOPD_OUTPUT_ROOT}/support_aware_opd" \
  -path '*prefix_intervention_20260806_s*/summary.json' -print
```

Each prompt is atomically committed only after all its valid intervention units
finish.  Rerunning the identical shard/config/commit skips completed prompts.

### 5.6 Strict merge

```bash
bash scripts/hpc/merge_support_aware_prefix_intervention.sh \
  8 prefix_intervention_20260806 prefix_intervention_20260806_merged
```

The merge rejects dirty/different commits, config differences, incomplete
shards, shard-index gaps, UID overlap, or incomplete selected-UID coverage.

Merged outputs:

```text
continuation_rollouts.jsonl
intervention_units.jsonl
rescue_comparisons.jsonl
minimal_rescue_prefixes.jsonl
summary.json
run_manifest.json
prompt_results/*.json
```

## 6. Morning scientific decision

### Gate A: proposal availability

If fewer than 8 low-support prompts have a retained verified proposal, the
teacher proposal source is too weak for a bridge comparison.  Do not train.

### Gate B: causal rescue

Proceed to localized bridge training only if:

- at least 8 prompts have all three valid controls for at least one horizon;
- at least 4 prompts meet the preregistered rescue rule;
- the teacher-prefix aggregate lift is positive against both controls;
- rescue is not confined to prefixes adjacent to final-answer markers;
- malformed/truncation does not explain the lift.

If teacher and wrong prefixes improve equally, the effect is generic context
length/continuation priming, not a causal teacher bridge.  If neither improves,
full-trajectory proposal likelihood cannot be treated as learnability.

### Gate C: localization coherence

Robust gap is useful only if its prefix/window trend predicts intervention
rescue across prompts.  Report Spearman correlation between the pre-registered
gap statistic and posterior rescue lift.  Selecting the statistic after seeing
rescue outcomes is exploratory and must be labeled as such.

## 7. Conditional training stage (design only; do not launch yet)

Once Gates A-C pass, create a new implementation commit with four matched
arms:

| arm | transfer/update path |
|---|---|
| `rl_only` | fresh unaided GRPO |
| `trek_full_fkl_rl` | full retained trajectory teacher-forced NLL, then GRPO |
| `minimal_bridge_rl` | NLL only on minimal answer-free bridge span, then GRPO |
| `minimal_bridge_local_opd_rl` | bridge span, fresh on-policy recovery, fork-local OPD + GRPO |

Add an equal-token random teacher span ablation if compute allows.  Full FKL is
the mandatory TREK baseline, not the proposed method.

All arms must match:

- student rollout tokens;
- optimizer steps;
- prompt manifest;
- validation checkpoints;
- GRPO group size and decoding;
- seed set.

Teacher/bridge compute is recorded separately.  Do not claim total-compute
efficiency when teacher compute differs.

### Primary causal outputs

1. fresh unaided K=32 support transition matrix;
2. change in posterior expected `U_8` per prompt;
3. fraction entering mixed-success frontier, not only pass@1;
4. early-RL pass@1-vs-step AUC and tokens-to-target;
5. whether frozen prefix rescue predicts post-training unaided support change;
6. diversity, duplicate, entropy, malformed and truncation safety metrics.

The contribution is supported only if minimal rescue predicts later unaided
support creation and the resulting frontier increase predicts early-RL gain.
Simply reducing NLL or matching TREK accuracy is insufficient.

## 8. Required CC handback

Return without reinterpretation:

1. pulled commit, branch, and clean/dirty status;
2. confirmation that no environment/backend file changed;
3. robust-gap output path and `gap_metric_summary.csv`;
4. proposal merged path/hash used by the intervention;
5. smoke command, GPU, output and exact summary;
6. full GPU-to-shard map, logs and restart history;
7. merged output path and hashes of the four merged JSONL files;
8. per-state/per-horizon rescue table against both controls;
9. every answer-leakage skip reason and count;
10. an explicit `GO` or `NO-GO` for localized bridge implementation under the
    preregistered gates; do not start training automatically.
