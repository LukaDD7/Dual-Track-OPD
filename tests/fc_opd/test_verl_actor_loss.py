from __future__ import annotations

import numpy as np
import torch

from dual_track_opd.fc_opd.verl_actor_loss import (
    compute_verl_fc_opd_actor_loss,
    fc_opd_batch_denominator,
    has_fc_opd_tensors,
)


def test_va_opd_actor_loss_uses_rollout_weight_denominator() -> None:
    student_logits = torch.randn(2, 3, 7)
    response_mask = torch.ones(2, 3, dtype=torch.bool)
    responses = torch.tensor([[1, 2, 3], [1, 2, 4]])
    topk_ids = torch.tensor(
        [
            [[[1, 5], [2, 5], [3, 5]], [[1, 5], [2, 5], [3, 5]]],
            [[[1, 5], [2, 5], [4, 5]], [[1, 5], [2, 5], [4, 5]]],
        ],
        dtype=torch.long,
    )
    log_probs = torch.log_softmax(torch.zeros(2, 2, 3, 2), dim=-1)
    batch = {
        "responses": responses,
        "fc_teacher_topk_indices": topk_ids,
        "fc_teacher_topk_log_probs": log_probs,
        "fc_condition_weights": torch.ones(2, 2, 3),
        "fc_teacher_sampled_log_probs": torch.tensor(
            [
                [[-0.1, -0.2, -0.3], [-1.1, -0.2, -0.3]],
                [[-0.3, -0.2, -0.1], [-0.3, -1.2, -0.1]],
            ],
            dtype=torch.float32,
        ),
        "fc_rollout_weights": torch.tensor([0.25, 0.75]),
        "fc_condition_ids": np.array([[0, 1], [0, 1]]),
        "fc_opd_loss_mode": np.array(["va_opd", "va_opd"], dtype=object),
    }

    assert has_fc_opd_tensors(batch)
    assert torch.isclose(fc_opd_batch_denominator(batch, response_mask), torch.tensor(1.0))

    output = compute_verl_fc_opd_actor_loss(
        student_logits=student_logits,
        batch=batch,
        response_mask=response_mask,
        config={"fc_opd": {"loss_mode": "va_opd"}},
    )

    assert output.per_token_loss.shape == response_mask.shape
    assert output.active_weight.shape == response_mask.shape
    assert torch.isfinite(output.per_token_loss).all()
    assert torch.isclose(output.active_weight.sum(), torch.tensor(1.0))
    assert output.metrics is not None
    assert torch.isclose(
        torch.as_tensor(output.metrics["va_opd/rollout_weight_sum"]),
        torch.tensor(1.0),
    )
