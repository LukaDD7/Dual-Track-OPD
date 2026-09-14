# FC-OPD Experiment Log

## Validated Infrastructure Milestones

- Vision-OPD adapter and prompt audit passed.
- Fixed-response protocol audit passed.
- Structured student-rollout audit passed.
- Trainable Vision-OPD-6K FC-OPD-2C offline score builder passed at `limit16_k4`.
- Real Qwen3-VL-4B FC-OPD-2C optimizer-step smoke passed.

The Vision-OPD 2C builder output is structurally valid as offline training data:
one row per prompt-rollout pair, `response_source=student_rollout`, structured
XML response spans, raw `full`/`blur` teacher top-k scores, exact token/score
alignment, `top_k=32`, and no builder errors or leakage warnings in smoke runs.

## Vision-OPD Red-Box Contamination Finding

Vision-OPD is no longer main experimental evidence. Local images likely contain
explicit red bounding-box localization cues baked into the PNG/JPG files:

- HPC red-box audit over 128 samples found
  `full_image_red_box_suspected_rate = 0.9140625`,
  `bbox_image_red_box_suspected_rate = 1.0`, and
  `likely_conclusion = red_box_likely_baked_into_full_images`.
- `teacher_images` have very high red-pixel ratios in inspected samples.
- `data/images` also show nonzero strong-red ratios.
- Human inspection previously showed visible red boxes in full images.
- Prompts explicitly say to focus on objects inside the red bounding box.
- Current Gaussian-blur degraded images are derived from loaded full images, so
  any baked-in red box is also present in degraded images.

## Current Status

- Vision-OPD results are infrastructure validation and red-box-contaminated /
  localization-cued ablation only.
- Vision-OPD full-run 2C and 4C are frozen as main-data experiments.
- Clean-data 4C experiments move to Geometry3K first, then ViRL39K after schema
  inspection.

## Decision

- Do not run Vision-OPD full 2C as a main experiment.
- Do not run Vision-OPD full 4C as a main experiment.
- Use `--allow-red-box-contaminated-images` only to mark an explicit
  localization-cued ablation.
- Implement and validate clean-data Geometry3K 4C evidence cache, signal audit,
  trainable offline score builder, and real-student min-train smoke before any
  full clean-data run.
