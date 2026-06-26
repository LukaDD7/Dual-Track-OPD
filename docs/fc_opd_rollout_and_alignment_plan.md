# FC-OPD Rollout And Alignment Plan

The current single-response setup is a smoke/debug setup. It validates token
alignment, offline teacher scoring, FC-OPD loss plumbing, backward, and real
optimizer updates, but it is not the formal rollout recipe.

## Rollout Plan

- `K=1` is only for smoke and debugging.
- Formal OPD should use `K=4` by default.
- `K=8` should be treated as an ablation.

For each prompt `x`:

1. Sample `K` student responses `y_1 ... y_K`.
2. Freeze the exact `response_token_ids` for each sampled response.
3. Teacher-force score each `y_k` under all configured conditions.
4. Train the student against the stored offline scores.

Audit levels:

- `fixed_audit_response`: fixed non-gold response for protocol/path checks only.
- `dataset_target`: diagnostic target scoring only, not OPD.
- `student_rollout`: OPD-compatible signal audit; use this before training.

If rollouts are generated once and reused for multiple epochs, label the method
as offline or semi-offline distillation. If rollouts are periodically refreshed
from the current student policy, label it semi-on-policy or on-policy depending
on refresh frequency.

## Alignment Status

Do not add alignment-aware training loss yet.

Current implementation:

- condition-decomposed OPD;
- offline teacher condition scores;
- chunk/router-weighted FC-OPD distillation loss.

Current implementation is not success-conditioned alignment-aware OPD.

Explicit alignment would require:

```text
Align_c(u) = cos(g_c^KD(u), g_u^ideal(u))
```

`g_u^ideal` requires success, reward, or correctness estimates from
multi-rollout outcomes or from an external verifier. Those signals are not in
the current verified training path.

The dataset signal audit may report pairwise condition KD-gradient cosines:

- `cos(g_full, g_blur)`
- `cos(g_full, g_free)`
- `cos(g_full, g_task)`
- `cos(g_blur, g_task)`

This is a condition redundancy diagnostic. It can reveal collapse or redundant
condition prompts, but it is not ideal-gradient alignment and must not be
reported as success-conditioned alignment.
