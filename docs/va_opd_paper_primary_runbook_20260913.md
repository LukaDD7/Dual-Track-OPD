# VA-OPD paper-primary reproduction runbook (2026-09-13)

## Status

The repository already contains a paper-faithful native VA-OPD implementation and a 4-step real-image OPD canary from 2026-08-06. However, the historical mainline stopped before the VA-OPD gate: the canary used `--objective opd`, not the degraded-image VA pass.

**2026-09-13 update: CPU preparation is complete.** The 2B student was downloaded from ModelScope with per-file SHA-256 verification, the missing vLLM source wheel was rebuilt from the pinned commit, the environment manifest was repaired, and the full paper-primary preflight passed. The current CPU instance still cannot start training because it has no GPU access.

This runbook fixes the primary-setting deviations:

- model: Qwen3-VL-8B teacher and Qwen3-VL-2B student;
- data: full Geometry3K training set (2,101 prompts), not the older 1,901-prompt holdout split;
- optimization: AdamW, batch 16, 5 epochs, `K=4`; learning rate is not specified in arXiv v1, so this runbook uses the existing native-OPD operational default `1e-6` and records it explicitly;
- VA: positive rectified full-minus-degraded teacher log-probability, population standard deviation, softmax temperature 1.0, high-token fraction 0.20, and high-group weight 0.5;
- degradation: 10% bilinear spatial downsample, nearest-neighbor upsample back to the original dimensions;
- KL: sampled-token reverse KL on the full-image teacher, with no task reward.

The 200-row validation file in the paper data directory is an operational overlap monitor only; it is not a held-out validation set and must not be used for the paper comparison.

## Prepared artifacts

- Train: `fc-opd-storage/outputs/fc_opd/geometry3k_va_paper/train.parquet` — 2,101 rows.
- Monitor: `fc-opd-storage/outputs/fc_opd/geometry3k_va_paper/val_monitor.parquet` — 200 overlapping rows.
- Manifest: `fc-opd-storage/outputs/fc_opd/geometry3k_va_paper/train.parquet.manifest.json`, with `holdout_val=false`.
- All 2,301 full/degraded image pairs were checked on 2026-09-13; dimensions match and degradation mode is `lowres_10pct_nearest`.
- Config: `configs/experiment/qwen3vl_8b_2b_geometry3k_va_opd_paper.yaml`.
- Launcher profile: `scripts/hpc/run_va_opd_native.sh --profile paper`.
- The launcher retains at most two actor checkpoints by default (`VA_OPD_MAX_ACTOR_CKPT_TO_KEEP`); this preserves resume/latest evidence without filling the currently constrained shared disk.
- Student model: `models/Qwen3-VL-2B-Instruct`, 4.0 GiB, downloaded from ModelScope at revision `master`; all 13 files match the API-provided SHA-256 values.
- Student manifest: `models/Qwen3-VL-2B-Instruct/modelscope_download_manifest.json`.
- Rebuilt vLLM wheel: `fc-opd-storage/wheelhouse/va-opd-cu128-r595-v1/vllm-0.12.0-cp312-cp312-linux_x86_64.whl`, 744,492,366 bytes, SHA-256 `a735dc578ee34f513dabe03a68fefb6376fb16b93c4bb71070d88fcb216bc2e4`.
- Final preflight run: `fc-opd-storage/runs/va_opd_native/qwen3vl_geometry3k_native_va_opd_paper_paper_preflight_final_20260913_20260913_122803/run_manifest.json` (config SHA-256 `ddff55f970d082b3b3615f76d11415df68f9d590a63c5e72cc77b16518639861`).
- Disk after downloads: about 305 GiB free on the shared GPFS root; the temporary 9-GiB vLLM build tree under `/tmp` was removed.

## Current blockers

1. The paper source claims full hyperparameters are in `app:impl`, but `A_appendix` is commented out. The exact learning-rate schedule, weight decay, warmup, and other optimizer settings are not recoverable from the published v1 source; do not describe any run using `1e-6` as fully paper-exact.
2. A GPU allocation with six idle GPUs is required: four actor GPUs plus two teacher GPUs.
3. `pip check` reports the previously documented benign metadata conflict (`pyvers` wants `packaging<26`, while `packaging==26.2` is installed). The preflight and import gates still pass; do not downgrade packages immediately before a run.

## 2026-09-14 reward fix

The first paper-primary OPD smoke failed during validation because the prepared paper dataset uses `data_source=geometry3k`, while verl's built-in reward table recognizes `hiyouga/geometry3k`. The launcher had also only populated the legacy `custom_reward_function` override; the resolved `reward.custom_reward_function.path` remained `None`. The launcher now sets both the legacy and current reward paths, so `smoke_reward.py::compute_score` is used directly and the dataset source name no longer determines reward dispatch. The subsequent empty-key transfer-queue error was a downstream symptom, not a separate bug.

The first VA-OPD smoke then failed because the runtime image can be resized before reaching the degraded teacher pass (for example, a persisted 640×415 pair arrives as 640×416). The persisted full/degraded assets were correct and passed preflight. `build_degraded_multi_modal_data` now verifies the persisted pair against each other and, if the runtime full image has a different size, nearest-resizes the prepared degraded image to that runtime size. This preserves the visual-token alignment needed by the two teacher passes without silently trusting an upstream resize as the data protocol.

The next VA-OPD smoke reached actor update but raised `KeyError: va_opd_token_weights`. The first backend integration only computed weights in the legacy V0 trainer. The actual launcher uses `TaskRunnerV1`, whose batches live in TransferQueue as nested tensors. The backend patch now computes VA weights in the V1 trainer before sequence balancing: it retrieves the complete sibling group, converts it to a padded `DataProto`, calls the shared weighting implementation, and writes nested `va_opd_token_weights` / `va_opd_visual_advantage` fields back to TransferQueue. A 2-actor + 2-teacher preflight passed after this change.

The subsequent smoke reached the V1 weight hook but reported that the TransferQueue batch lacked `teacher_degraded_ids` and `teacher_degraded_logprobs`. `AgentLoopOutput.as_dict()` promoted the full-image teacher fields to top-level fields but left degraded fields inside `extra_fields`, so the V1 TransferQueue path dropped them. The backend now promotes degraded fields in `as_dict()` as well; preflight and backend SHA checks pass after the change.

After those fields reached the V1 hook, teacher ID identity initially failed because V1 may store teacher tensors as jagged/nested full-sequence tensors rather than padded prompt+response tensors. `_response_tokens` now handles both full-sequence and response-only transport, extracts the final response span, and retains the exact-ID gate.

The fixed-width final-span extraction was still wrong for padded V1 layouts because a short response can be followed by right padding. The alignment logic now searches each teacher ID row for the exact contiguous student response token sequence, extracts teacher values at those same positions, and only then enforces the exact-ID gate. This makes alignment independent of prompt-left-padding and response-right-padding assumptions.

**Gate C update, 2026-09-14:** VA-OPD 4-step smoke **passed** on the 4-GPU
2+2 layout. See `docs/va_opd_v1_smoke_incident_20260914.md` for the incident
chain and evidence. The next gate is the paired 50-step OPD and VA-OPD pilot.

## Required gates

Run these in order on a GPU node after the blockers above are resolved:

1. CPU/full preflight (**already passed on 2026-09-13**):

   ```bash
   bash scripts/hpc/run_va_opd_native.sh \
     --objective va_opd --profile paper --preflight-only --audit-all-images \
     --visible-gpus 0,1,2,3,4,5 --actor-gpus 4 --teacher-gpus 2 --teacher-tp 2 \
     --name paper_preflight
   ```

2. OPD 4-step smoke, using the same 2B/8B model pair, data, sampling, and response length:

   ```bash
   bash scripts/hpc/run_va_opd_native.sh \
     --objective opd --profile smoke --steps 4 \
     --visible-gpus 0,1,2,3,4,5 --actor-gpus 4 --teacher-gpus 2 --teacher-tp 2 \
     --student-model "$DTOPD_ROOT/models/Qwen3-VL-2B-Instruct" \
     --teacher-model "$DTOPD_ROOT/models/Qwen3-VL-8B-Instruct" \
     --train-data "$DTOPD_ROOT/fc-opd-storage/outputs/fc_opd/geometry3k_va_paper/train.parquet" \
     --val-data "$DTOPD_ROOT/fc-opd-storage/outputs/fc_opd/geometry3k_va_paper/val_monitor.parquet" \
     --name paper_opd_smoke
   ```

3. VA-OPD 4-step smoke, requiring `va_opd/mean` not identically zero, a non-extreme positive ratio, finite loss/grad/entropy, exact full/degraded/student response-ID equality, and every prompt group having exactly four siblings:

   ```bash
   bash scripts/hpc/run_va_opd_native.sh \
     --objective va_opd --profile smoke --steps 4 \
     --visible-gpus 0,1,2,3,4,5 --actor-gpus 4 --teacher-gpus 2 --teacher-tp 2 \
     --student-model "$DTOPD_ROOT/models/Qwen3-VL-2B-Instruct" \
     --teacher-model "$DTOPD_ROOT/models/Qwen3-VL-8B-Instruct" \
     --train-data "$DTOPD_ROOT/fc-opd-storage/outputs/fc_opd/geometry3k_va_paper/train.parquet" \
     --val-data "$DTOPD_ROOT/fc-opd-storage/outputs/fc_opd/geometry3k_va_paper/val_monitor.parquet" \
     --name paper_va_smoke
   ```

4. Run paired 50-step OPD and VA-OPD pilots, then compare best validation-monitor score, entropy, response clip ratio, VA mean/positive ratio, loss, gradient, and collapse behavior.

5. Only after both pilots are stable, launch the 5-epoch paper runs:

   ```bash
   bash scripts/hpc/run_va_opd_native.sh \
     --objective opd --profile paper \
     --visible-gpus 0,1,2,3,4,5 --actor-gpus 4 --teacher-gpus 2 --teacher-tp 2 \
     --name full5e_paper_opd
   ```

   ```bash
   bash scripts/hpc/run_va_opd_native.sh \
     --objective va_opd --profile paper \
     --visible-gpus 0,1,2,3,4,5 --actor-gpus 4 --teacher-gpus 2 --teacher-tp 2 \
     --name full5e_paper_va
   ```

The paper's headline comparison is OPD versus VA-OPD. The 4B or 32B extension settings should not be used to claim the primary result.

## Evaluation remains a separate gate

The paper reports avg@8 at temperature 1.0 on WeMath, MathVista, MathVerse, HallusionBench, AI2D, MMMU, MMStar, and OCRBench, using each benchmark's official protocol and GPT-4o judging where applicable. This repository does not yet have a single frozen paper-primary eight-benchmark eval pack tied to the new 2B checkpoints. Build and validate that pack before comparing numbers to the paper; the overlapping 200-row Geometry3K monitor is not a substitute.
