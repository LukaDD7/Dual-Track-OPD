# Research Brief

The current research direction is **support-aware SFT-RL-OPD multi-stage
post-training for reasoning**.

The project studies how SFT/pre-RL initialization, reinforcement learning, and
on-policy distillation should divide roles and interact:

- SFT/pre-RL initialization establishes reusable capabilities and candidate
  reasoning modes.
- RL explores, selects, and recombines reasoning modes under verifiable reward.
- OPD uses teacher signals to guide or consolidate useful modes according to
  their effective policy support.

The primary question is not how to design another visual-token weighting rule.
It is whether OPD can improve the RL learning curve and compositional
generalization by intervening at the right support state, while avoiding
wrong-mode amplification and diversity collapse.

The first operational test uses Geometry3K for frozen-policy diagnostics and
micro-training, with DynaMath reserved for held-out OOD evaluation. The full
implementation contract is:

`docs/support_aware_sft_rl_opd_experiment_runbook.md`

Earlier token/chunk-level VA-OPD and FC-OPD work remains useful as mechanism and
infrastructure, especially for online rollout transport, teacher forced
scoring, reverse-KL, verifier gating, and training stability. It is no longer
the top-level research framing.
