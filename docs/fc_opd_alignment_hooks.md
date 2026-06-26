# FC-OPD Alignment Hooks

Alignment hooks are present but default to no-op. They do not change existing
2C or 4C loss behavior.

Implemented schema fields:

- `outcome_metadata`
- `rollout_group_uid`
- `sibling_rollout_ids`
- `success_group_stats`
- `alignment_targets`

Implemented helper module:

```text
src/dual_track_opd/fc_opd/alignment.py
```

APIs:

- `compute_alignment_weights(record, strategy="none")`
- `apply_alignment_weights(loss_per_token, weights)`
- `compute_rollout_group_stats(records)`

Supported strategies:

- `none`
- `placeholder_noop`
- `visual_task_signal_only`
- `correctness_contrastive` schema placeholder
- `success_failure_rollout_contrastive` schema placeholder

Ground truth may be used only post-hoc for evaluation/alignment metadata. It is
never inserted into rollout prompts, evidence generation prompts, or teacher
condition inputs.
