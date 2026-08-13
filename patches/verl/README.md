# STP-OPD verl backend wiring (2026-08-13)

## Wiring scheme (mirrors the validated FC-OPD thin patches)

Two thin patches against verl 0.7.1 (`third_party/verl` @ `bec9ef74`), both
config-gated so verl behaves identically without the STP-OPD section:

1. **`0001-stp-opd-ray-trainer-hook.patch`** — adds an optional
   `algorithm.stp_opd.post_rollout_hook` call in `ray_trainer.py` next to the
   existing FC-OPD hook.  The hook
   (`verl_stp_opd_integration.stp_opd_post_rollout_hook`) attaches:
   `stp_prefix_mask`, `stp_suffix_mask`, `stp_sampled_ids`,
   `stp_teacher_k1_log_probs`, `stp_valid_mask`, `stp_prefix_lengths` to the
   rollout batch.  The teacher service scores the full hybrid trajectory
   (fixed answer-free prefix + student suffix); the masks select the suffix
   region for RKL-K1 and GRPO.
2. **`0002-stp-opd-actor-loss.patch`** — in `dp_actor.py`, when
   `has_stp_opd_tensors(micro_batch)` is true, computes the regional loss via
   `compute_stp_opd_actor_loss` (delegates to
   `support_transition_train.train_step_loss` with arm/lambdas from
   `algorithm.stp_opd`) and adds `loss_coef * stp_opd_loss` to `policy_loss`
   with a global-normalizer denominator, exactly like the FC-OPD auxiliary
   loss path.

## Key design points

- RKL-K1 uses the **sampled-token K1 estimator**:
  `-mean log P_T(y_t)` over the suffix, where `log P_T(y_t)` is the teacher's
  log-prob of the student-sampled token on the same hybrid state
  (`stp_teacher_k1_log_probs`), matching the plan §4.4 "sampled-token K1
  reverse-KL estimator" and the FC-OPD teacher-sampled-log-prob channel.
- Scaffold branch is data-side: a scaffolded sample's response prefix region
  holds the teacher prefix tokens (`stp_prefix_mask` covers them); an
  unscaffolded sample has an empty prefix mask.  The hook derives masks from
  `stp_prefix_lengths`; the pilot fixes one verified answer-free horizon per
  prompt (`stp_prefix_manifest` JSON: prompt_id -> token_ids).
- `advantages` for the GRPO term come from verl and must be attached to the
  batch as `stp_advantages` by the trainer (not in the patch; trainer-side).

## Validation gate (handoff §6 P0)

The wiring is code-complete but **not yet validated end-to-end**.  Before any
arm launch in the pinned verl env:

- exact-token identity (prefix/suffix token hashes, tokenizer mapping) passes;
- online student scorer is exercised in the main A0/A3 paths;
- gradient isolation tests run against the verl micro-batch shapes;
- `stp_advantages` channel and `loss_coef` scaling verified;
- resume/save/load preserves global step, optimizer, schedule, arm identity;
- 4-step real-image canary passes independently for A0/A1/A2/A3.
