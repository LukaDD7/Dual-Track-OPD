# FC-OPD Student Rollout Signal Audit

This audit is the OPD-compatible pre-training signal check. It samples student
responses first, freezes those response token IDs, and teacher-force scores each
rollout under the requested conditions.

## Modes

- `answer_only`: mechanical smoke only. It validates generation and teacher
  scoring, but Vision-OPD prompts often produce 3-4 token option answers, which
  are too short for token-level FC-OPD training.
- `fc_opd_structured`: default. The student is instructed to output
  `<visual_evidence>`, `<reasoning>`, and `<answer>` spans without using gold
  answers.

## Smoke Commands

Mechanical smoke:

```bash
DATASET="$PROJECT_ROOT/third_party/Vision-OPD/data/train.parquet" \
LIMIT=4 \
ROLLOUTS_PER_PROMPT=2 \
CONDITIONS=full,blur \
ROLLOUT_RESPONSE_FORMAT=answer_only \
bash scripts/hpc/run_fc_opd_student_rollout_signal_audit.sh
```

OPD-compatible smoke:

```bash
DATASET="$PROJECT_ROOT/third_party/Vision-OPD/data/train.parquet" \
LIMIT=4 \
ROLLOUTS_PER_PROMPT=2 \
CONDITIONS=full,blur \
ROLLOUT_RESPONSE_FORMAT=fc_opd_structured \
bash scripts/hpc/run_fc_opd_student_rollout_signal_audit.sh
```

Then scale to `LIMIT=16`, `ROLLOUTS_PER_PROMPT=4`,
`ROLLOUT_RESPONSE_FORMAT=fc_opd_structured`.

## Diagnostics

Rows include image/degraded existence flags, crop/bbox metadata marked unused by
default, per-rollout seeds, prompt hashes, response hashes, and per-row
full-vs-blur gradient cosine when available.

Summaries report response length distribution, short response rate,
unique-response-per-prompt mean, duplicate rollout rate, identical-prompt count,
and warning flags. K is useful only if duplicate rollout rate is low.
