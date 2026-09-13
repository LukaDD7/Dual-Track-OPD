# ReMI and MV-MATH Scoring Protocol Audit

Date: 2026-09-13

## Conclusion

Both benchmarks should remain **internal diagnostics** in the Project15 v1 table.
They are useful signal, but they are not pinned `lmms-eval v0.7.1` community
tasks and must not use the old lenient `normalized_exact_diagnostic` number.

## MV-MATH

The official MV-MATH repository uses an LLM equivalence judge, not a pure exact
match:

* choice (1,109) and single-step (800): DeepSeek-Chat returns `true`/`false`;
* multi-step (100): DeepSeek-Chat returns `correct_steps/total_steps`;
* the headline score counts choice+single-step true rows plus multi-step rows
  where all steps are correct, divided by 2,009;
* multi-step also reports SAR (Step Accuracy Rate) and QCR (Question
  Completeness Rate).

We added `dual_track_opd.eval.score_mv_math`, which implements that protocol
against any OpenAI-compatible judge (default: the existing local
`Qwen3-VL-32B-Instruct` endpoint). It supports a resumable per-row sidecar, so
an interruption does not restart the judge drain.

Official-compatible command:

```bash
python -m dual_track_opd.eval.score_mv_math \
  --mode official \
  --replay-jsonl <project15_run>/replay/mv_math.jsonl \
  --metadata-json /inspire/hdd/global_user/mengweicheng-240108120092/lzy/dataset/MV-MATH/MV-MATH.json \
  --judge-url http://127.0.0.1:8801/v1 \
  --judge-model Qwen3-VL-32B-Instruct \
  --judge-sidecar <project15_run>/replay/mv_math_official_judge.jsonl \
  --resume \
  --output-json <project15_run>/replay/mv_math_official_summary.json
```

When no judge is online, the new `--mode choice` gives a deterministic
full-denominator diagnostic for the 1,109 choice rows only. It is useful for
sanity checks but is not the official 2,009-row MV-MATH metric.

Current choice-only diagnostics from the completed raw replays:

| Arm | Correct / choice rows | Accuracy | Choice rows truncated at 4,096 |
|---|---:|---:|---:|
| Base | 569 / 1,109 | 51.31% | 451 |
| PTD-PO r4 step390 | 551 / 1,109 | 49.68% | 523 |
| TailSFT | 442 / 1,109 | 39.86% | 784 |

The high truncation rate means these numbers are completion-biased diagnostics,
not capability-equivalent scores. TailSFT is especially affected.

## ReMI

The ReMI paper reports:

* exact match for textual outputs after lowercase/spacing and a small set of
  documented postprocessing rules;
* relaxed numeric accuracy with 1% tolerance generally;
* 3% tolerance for GeomShapes and GeomCost;
* 10-minute tolerance for Clocks.

The paper prompted for JSON containing `explanation` and `answer`, with 512
output tokens. Our historical replay prompts are answer-only, so we do not claim
paper-prompt equivalence.

Use the existing honest sidecar rather than `summary.json`:

```bash
python scripts/sft_rl/remi_reeval.py \
  --mode exact \
  --jsonl <project15_run>/replay/remi.jsonl \
  --label-jsonl /inspire/hdd/global_user/mengweicheng-240108120092/lzy/eval_runs/qwen3vl8b_baseline/raw_responses/qwen3vl8b_ReMI_test_len65536_maxtok1024_raw.jsonl
```

This uses a full 2,600-row denominator. Its task-aware extraction and numeric
tolerances are closer to the paper than the old `normalized_exact_diagnostic`,
which counted only an extractable subset and inflated the score. LLM-judge mode
remains a diagnostic drain and is not the primary metric.

## Formal reporting rule

1. Project15 v1's 13 native `lmms-eval` benchmarks remain the formal v1 rows.
2. Report ReMI and MV-MATH only as clearly labeled diagnostics.
3. MV-MATH's official-compatible metric requires the LLM judge sidecar; do not
   substitute the choice-only deterministic diagnostic.
4. ReMI must use `remi_reeval.py --mode exact`, not the old
   `normalized_exact_diagnostic`.
