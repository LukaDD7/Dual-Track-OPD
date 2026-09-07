# STP-OPD verl backend wiring (2026-08-13)

## Wiring scheme (mirrors the validated FC-OPD thin patches)

Two thin patches against verl 0.7.1 (`third_party/verl` @ `bec9ef74`), both
config-gated so verl behaves identically without the STP-OPD section:

`stp_opd_combined.patch` — adds both sides (generated from the real verl diff,
verified with `git apply --check`):

1. **ray_trainer.py hook** — adds an optional
   `algorithm.stp_opd.post_rollout_hook` call in `ray_trainer.py` next to the
   existing FC-OPD hook.  The hook
   (`verl_stp_opd_integration.stp_opd_post_rollout_hook`) attaches:
   `stp_prefix_mask`, `stp_suffix_mask`, `stp_sampled_ids`,
   `stp_teacher_k1_log_probs`, `stp_valid_mask`, `stp_prefix_lengths` to the
   rollout batch.  The teacher service scores the full hybrid trajectory
   (fixed answer-free prefix + student suffix); the masks select the suffix
   region for RKL-K1 and GRPO.
2. **dp_actor.py actor loss** — in `dp_actor.py`, when
   `has_stp_opd_tensors(micro_batch)` is true, computes the regional loss via
   `compute_stp_opd_actor_loss` (delegates to
   `support_transition_train.train_step_loss` with arm/lambdas from
   `algorithm.stp_opd`) and adds `loss_coef * stp_opd_loss` to `policy_loss`
   with a global-normalizer denominator, exactly like the FC-OPD auxiliary
   loss path.

Apply with:

```bash
cd /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/Dual-Track-OPD/third_party/verl && \
git apply ../../patches/verl/fc_opd_ray_trainer_post_rollout_hook.patch && \
git apply ../../patches/verl/fc_opd_fsdp_actor_aux_kd.patch && \
git apply ../../patches/verl/stp_opd_combined.patch && \
git apply ../../patches/verl/0003-stp-opd-agent-scaffold.patch
```

Order matters: `stp_opd_combined.patch` was generated on top of the applied
FC-OPD patches (they share the response_mask / dp_actor regions), so FC-OPD
must be applied first.  The full chain (pristine verl 0.7.1 -> FC-OPD x2 ->
STP) was verified with `git apply --check` and `py_compile`.
`0003-stp-opd-agent-scaffold.patch` touches only
`single_turn_agent_loop.py` (independent file) and can be applied at any
point.

### Scaffold render (0003) and advantages

- `0003` renders the fixed answer-free teacher prefix as an assistant message
  in `SingleTurnAgentLoop.run` when the dataset row carries
  `stp_scaffolded` + `stp_prefix_text` (top-level row fields, set by
  `STPTransitionDataset`).
- GRPO advantages are reused directly from verl's standard
  `batch["advantages"]` (no separate `stp_advantages` attach needed);
  `compute_stp_opd_actor_loss` reads `stp_advantages` as an optional override.

## Environment validation (2026-08-13)

On the GPU instance (`fc-opd-verl071-cu128`, 2 GPUs):

- FC-OPD x2 + STP patches applied cleanly to verl 0.7.1;
- `run_verl_fc_opd_smoke.sh --gpus 2 --steps 1` completed with **exit code 0**
  (2/2 training steps; `actor/fc_opd_loss` ~0.156, `teacher_valid_ratio` 1.0);
- the trailing `DataLoader worker killed by signal` message is vLLM shutdown
  noise after training finished and does not affect the result.

Also fixed along the way: `teacher_transformers.get_rope_index` now filters
kwargs by the callable signature (Qwen3-VL + transformers version drift), and
the STP actor loss passes `stp_opd_loss_sum`/`stp_opd_active_weight_sum` via
`_forward_micro_batch` outputs like the FC-OPD path.

Environment is ready for STP-OPD canary wiring (scaffold render point,
`stp_advantages` attach, STP dataset).

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

## Data-side scaffold (dataloader)

verl 0.7.1 has no fixed-response-prefix mechanism, so the scaffold branch is
implemented with **path A** (minimal, reuses the FC-OPD actor aux-loss pattern):

- a scaffolded sample renders the verified answer-free teacher prefix as the
  assistant message content (`support_transition_dataset.build_samples`); the
  model continues generating the suffix after it;
- the rollout carries `stp_prefix_token_ids` + `stp_scaffolded` in
  `extra_info` (dataset side, mirroring `FCOPDDataset`);
- the FKL prefix region is scored by the actor on the prefix tokens: the actor
  forward covers prompt + prefix + suffix, the prefix positions take the
  teacher-forced CE (`prefix_fkl_ce`), suffix positions take RKL-K1/GRPO;
- the masks in `stp_opd_combined.patch` (dp_actor side) are derived from
  `stp_prefix_lengths` (prefix token count) instead of a rollout-side
  pre-filled response prefix.

Validation points that must be exercised in the pinned verl env: prefix token
ids survive the verl pipeline unchanged (`extra_info`), the actor forward
includes the prefix region with correct masks, and the scaffolded/unscaffolded
paired comparison uses identical prompts.

### Open item: scaffold render point in verl rollout

Where the fixed prefix becomes an assistant message in the verl pipeline
depends on the exact verl 0.7.1 data flow (dataset `raw_prompt` -> rollout
request `messages`).  That injection point must be confirmed **in the pinned
verl environment** (the version-specific call site is not derivable from this
repo alone).  The intended contract is:

- `STPTransitionDataset` (verl_stp_dataset.py) puts `stp_prefix_token_ids`,
  `stp_prefix_text`, and per-row scaffold flags into `extra_info`;
- at the rollout request construction, if `stp_scaffolded` + `stp_prefix_text`
  are present, append an assistant message with the prefix text so the model
  continues generating the suffix;
- an earlier draft patch (0003) was removed because it referenced a
  non-existent verl helper; the real patch must target the verified call site.

`stp_advantages` (GRPO advantages attached to the stp_* batch) is also
trainer-side and must be wired against the actual advantage computation.

## Validation gate (handoff §6 P0)

The wiring is code-complete but **not yet validated end-to-end**.  Before any
arm launch in the pinned verl env:

- exact-token identity (prefix/suffix token hashes, tokenizer mapping) passes;
- online student scorer is exercised in the main A0/A3 paths;
- gradient isolation tests run against the verl micro-batch shapes;
- `stp_advantages` channel and `loss_coef` scaling verified;
- resume/save/load preserves global step, optimizer, schedule, arm identity;
- 4-step real-image canary passes independently for A0/A1/A2/A3.

## TailSFT online filtering patch (2026-09-04, cu132 backend; refreshed 2026-09-05 for the 2x-loss aggregation + nested-metrics crashes)

`tailsft_online_filtering.patch` — additive TailSFT (arXiv:2608.25756)
sequence-level filtering for the **verl 0.9.0 cu132 SFT backend**
(`fc-opd-storage/backends/verl-qwen35-v090-cu132`, NOT the third_party/verl
0.7.1 tree above). Generated from the real backend `git diff` over exactly
three files:

1. `verl/trainer/config/sft_trainer_engine.yaml` — new `data.tailsft`
   section (`enabled: False` default, `filter_fraction`, `schedule`,
   `ramp_steps`). Stock behavior is byte-identical when disabled.
2. `verl/trainer/sft_trainer.py` — when `data.tailsft.enabled`, sets the
   actor's loss fn to `tailsft_loss` and injects the γ_t schedule value
   into each step's `meta_info` (computed by `tailsft_filter_fraction`).
   Also reduces `tailsft/*` metrics to scalars before `tracking.log`
   (LocalLogger's `concat_dict_to_str` only prints `numbers.Number`, so the
   per-rank per-micro-batch LIST values were silently invisible on the
   console — fixed 2026-09-05; counts sum, margin_mean averages). The
   first reduction attempt assumed FLAT lists and crashed at the first
   training step with `TypeError: unsupported operand type(s) for +:
   'int' and 'list'`: `allgather_dict_into_dict` APPENDS each dp rank's
   already-flat per-micro-batch list, so the collected value is NESTED
   (`[[f, ...], [f, ...]]`) and `sum()` starts at int 0. The 2026-09-05
   (v2) reduction flattens one level before sum/average and handles the
   flat, nested, and empty shapes.
3. `verl/workers/utils/losses.py` — `tailsft_loss` + helpers, purely
   additive next to the untouched stock `sft_loss`:
   - `tailsft_keep_mask` — rank-based (stable argsort) drop of exactly
     `k = floor(n * γ)` smallest-margin sequences; ties broken
     deterministically by position (threshold-based dropping kept ties and
     under-dropped — found by unit test, fixed 2026-09-04);
   - `_tailsft_per_example_ce` — length-normalized mean CE per sequence
     from the jagged (no_padding) flatten, mask already rolled;
   - `tailsft_loss` — falls back to stock `sft_loss` when `init_ce` is
     missing/partial-None, γ ≤ 0, pad_mode ≠ no_padding, or everything is
     dropped (safety net); otherwise gathers margins across the dp group
     (all_gather_object), builds the global keep-mask, zeroes dropped
     sequences' rolled mask, and divides by the STEP-LEVEL constant
     `batch_num_tokens * (1 - γ)` (fix 2026-09-05): the engine calls the
     loss once per micro-batch and SUMS the losses, which is only correct
     when every micro-batch divides by the same step-global constant
     (exactly how stock `sft_loss` works via its all-reduced
     `batch_num_tokens`). The original version all-reduced each
     micro-batch's own retained-token count, making each micro-batch loss
     a complete mean → M× too-large loss AND gradient at dynamic-bsz M=2
     steps (~4.5% of steps; observed as exact-2x `train/loss` + grad_norm
     spikes in `qwen3vl_sft_tailsft_mmf122k_1ep`).

Row-side plumbing lives in the repo (not the backend):
`scripts/sft_rl/tailsft_dataset.py` (TailsFTDataset reads the `init_ce`
parquet column AFTER super()._read_files_and_process so it stays aligned
with the selected dataframe) and
`scripts/sft_rl/annotate_tailsft_init_ce.py` (offline ℓ0 precompute under
the base model; `init_ce_from_logits` mirrors the loss-side roll).

Validated by `tests/sft_rl/test_tailsft_filtering.py` (23 tests, green in
env `vaopd-gkd-qwen35`: keep-mask floor/tie/degenerate semantics, γ
schedule, hand-computed per-example CE, ℓ0-vs-ℓt roll alignment, all
fallback paths, filtering math + gradient isolation + dp scaling, γ/init_ce
threading through index-select micro-batch slicing, and a
multi-micro-batch aggregation regression test that pins the
step-level-denominator invariant (sum of per-micro-batch losses == global
retained-token mean; the old per-micro-batch denominator fails it at
exactly 2x), and two metric-reduction regression tests pinning the
nested-list flattening (the flat-list assumption raised TypeError at the
first restart step).

## PTD-PO patch (2026-08-21, cu132 backend; refreshed 2026-09-04 for the crash fix)

`ptd_opd_20260821.patch` — Partial Trajectory Distillation + PPO
(arXiv:2606.07000) for the **verl 0.9.0 cu132 backend**
(`fc-opd-storage/backends/verl-qwen35-v090-cu132`, NOT the third_party/verl
0.7.1 tree above). Generated from the real backend `git diff` over exactly
13 files (excludes the three TailSFT files covered by
`tailsft_online_filtering.patch`):

`examples/on_policy_distillation_trainer/run_qwen3_5_4b_fsdp.sh`,
`verl/experimental/agent_loop/agent_loop.py`,
`verl/experimental/agent_loop/single_turn_agent_loop.py`,
`verl/trainer/config/algorithm.py`, `verl/trainer/config/ppo_trainer.yaml`,
`verl/trainer/distillation/fsdp/losses.py`,
`verl/trainer/distillation/losses.py`, `verl/trainer/ppo/ray_trainer.py`,
`verl/workers/config/__init__.py`, `verl/workers/config/distillation.py`,
`verl/workers/config/ptd.py`, `verl/workers/engine_workers.py`,
`verl/workers/rollout/vllm_rollout/vllm_async_server.py`.

2026-09-04 refresh captures the over-width-hint crash fix (in
`single_turn_agent_loop.py` + `agent_loop.py`): the hint is tokenized by the
HF processor with a large image patch grid, so image-heavy rows can produce a
hint wider than `rollout.prompt_length` even though the vLLM-tokenized student
prompt is hard-capped at 2048; concatenating such a hint with the uniform
2048-wide batch crashed `torch.cat`. Both sides now drop over-length hints
(no truncation — that would corrupt image-token ↔ pixel_values alignment) so
every surviving hint is exactly `prompt_length` wide; dropped hints are
excluded from the distillation loss by the existing `has_hint` gate.
