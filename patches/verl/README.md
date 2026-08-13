# verl backend status for STP-OPD (2026-08-13)

## Decision: no source patch yet

verl 0.7.1 (`third_party/verl` @ `bec9ef74`) exposes two official extension
points that cover the STP-OPD pilot without modifying verl source:

1. **`register_policy_loss`** (`verl/trainer/ppo/core_algos.py`) — verl already
   resolves the policy loss by name from `actor_rollout_ref.actor.policy_loss`.
   `src/dual_track_opd/support_aware/verl_stp_opd_integration.py` registers an
   `stp_opd` entry (lazy import; no-op without verl installed).
2. **post-rollout hook** — the existing FC-OPD hook pattern
   (`fc_opd_post_rollout_hook`) attaches teacher-scored tensors to the rollout
   batch; STP-OPD attaches teacher logits, prefix/suffix masks, and sampled ids
   the same way.

## When a real patch becomes necessary

Only if the rollout-batch custom-field channel cannot carry the teacher RKL
inputs (teacher logits are needed at loss time).  In that case the smallest
patch would add a no-op-by-default hook call in
`verl/trainer/ppo/ray_trainer.py` next to the existing policy-loss call, gated
by a config flag, and this file would then document the exact diff.

## Validation gate

The end-to-end wiring must be validated in the pinned HPC verl environment
(`va-opd-native-e003-cu128-r595-v1` or the train-verified env) before any arm
launch: exact-token identity, online student scorer exercised in A0/A3,
gradient isolation, and resume/manifest checks (handoff §6 P0 items).
