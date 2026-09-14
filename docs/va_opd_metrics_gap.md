# VA-OPD Metrics Pipeline Gap

> 2026-07-05 Codex note: this report correctly identifies that actor-side VA
> diagnostics were not surfaced, but it was written against the older dp_actor
> patch shape. In faithful VA-OPD, rollout weights sum to 1 per prompt sibling
> group, so `rollout_weight_sum` should be approximately the number of prompts
> in the actor mini-batch, not the rollout batch size.

## Problem

`compute_va_opd_loss()` in `va_opd_loss.py` computes a rich `metrics` dict, but it is
**discarded** during the `VerlSparseKDOutput` conversion in `dp_actor.py`. The only
VA-OPD metric visible in training logs is `actor/fc_opd_loss` (the scalar loss).

## Lost metrics (all computed, none logged)

| Key | Type | Description |
|---|---|---|
| `va_opd/loss` | `torch.Tensor` (scalar) | Same as `actor/fc_opd_loss`, redundant |
| `va_opd/token_mean_loss` | `torch.Tensor` (scalar) | Per-token reverse-KL averaged over all valid tokens |
| `va_opd/rollout_weight_sum` | `float` (Python, `.item()` called) | Sum of rollout weights over batch |
| `va/mean` | `torch.Tensor` (scalar) | Mean rectified visual advantage over all valid tokens |
| `va/sparsity` | `torch.Tensor` (scalar) | Fraction of tokens where VA ≤ 0 (no visual dependence) |

## Root cause chain

1. `va_opd_loss.py:226-233` — metrics dict computed inside `VAOPDLossResult.metrics`
2. `dp_actor.py:967-969` — only `result.loss` is used; `result.metrics` ignored:
   ```python
   return VerlSparseKDOutput(
       per_token_loss=result.loss.expand(batch_size),
       active_weight=torch.ones(batch_size, device=student_logits.device),
   )
   ```
3. `VerlSparseKDOutput` (`verl_sparse_kd.py:16-19`) has no `metrics` field:
   ```python
   @dataclass(frozen=True)
   class VerlSparseKDOutput:
       per_token_loss: torch.Tensor
       active_weight: torch.Tensor
   ```
4. `dp_actor.py:424-426` — outputs dict only carries `per_token_loss` + `active_weight`,
   no metrics field.

## Where to fix (3 files, all on NFS → codex can access)

### File 1: `src/dual_track_opd/fc_opd/verl_sparse_kd.py` (line 16)

Add optional `metrics` field to `VerlSparseKDOutput`:

```python
@dataclass(frozen=True)
class VerlSparseKDOutput:
    per_token_loss: torch.Tensor
    active_weight: torch.Tensor
    metrics: dict[str, Any] | None = None  # ADD THIS
```

### File 2: `third_party/verl/verl/workers/actor/dp_actor.py` (line 967)

Pass `result.metrics` into the return value:

```python
return VerlSparseKDOutput(
    per_token_loss=result.loss.expand(batch_size),
    active_weight=torch.ones(batch_size, device=student_logits.device),
    metrics=result.metrics,  # ADD THIS
)
```

### File 3: `third_party/verl/verl/workers/actor/dp_actor.py` (lines 424-426 & 286-288)

Carry metrics through the outputs dict. Two locations (full batch + micro-batch paths):

```python
outputs = {"log_probs": log_probs}
if has_fc_opd:
    outputs["fc_opd_loss_per_token"] = fc_opd_loss_per_token
    outputs["fc_opd_active_weight"] = fc_opd_active_weight
    if fc_opd_output.metrics is not None:           # ADD
        outputs["fc_opd_metrics"] = fc_opd_output.metrics  # ADD
```

### File 4: `third_party/verl/verl/workers/actor/dp_actor.py` (lines 735-756)

Merge into `micro_batch_metrics` where the loss is consumed:

```python
fc_opd_loss_per_token = outputs.get("fc_opd_loss_per_token")
if fc_opd_loss_per_token is not None:
    fc_opd_active_weight = outputs["fc_opd_active_weight"]
    fc_opd_loss = fc_opd_loss_per_token.sum() / fc_opd_active_weight.sum().clamp_min(1.0)
    # ... existing loss logic ...
    micro_batch_metrics["actor/fc_opd_active_weight_mean"] = (
        fc_opd_active_weight.detach().float().mean().item()
    )
    # ADD: merge VA-OPD diagnostics
    fc_opd_metrics = outputs.get("fc_opd_metrics")
    if fc_opd_metrics is not None:
        for k, v in fc_opd_metrics.items():
            micro_batch_metrics[f"actor/fc_opd_{k}"] = (
                float(v) if isinstance(v, (int, float)) else float(v.detach().cpu().item())
            )
```

Note: `va_opd/rollout_weight_sum` is already a Python `float` (`.item()` called in
`va_opd_loss.py:229`), the others are `torch.Tensor` scalars. The merge loop above
handles both.

## Verification

After fix, training logs should show these additional metrics each step:

```
actor/fc_opd_va_opd/loss: 11.56
actor/fc_opd_va_opd/token_mean_loss: non-negative scalar KL diagnostic
actor/fc_opd_va_opd/rollout_weight_sum: ≈ number of prompts in the mini-batch
actor/fc_opd_va/mean: ~0.01 to 0.5
actor/fc_opd_va/sparsity: ~0.3 to 0.9
```
