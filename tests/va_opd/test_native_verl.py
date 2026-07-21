from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from dual_track_opd.va_opd.native_verl import (
    build_degraded_multi_modal_data,
    prepare_native_verl_batch,
)


def test_build_degraded_multimodal_uses_prepared_same_size_image(tmp_path):
    degraded_path = tmp_path / "diagram.lowres.png"
    Image.new("RGB", (20, 10), "gray").save(degraded_path)
    full = Image.new("RGB", (20, 10), "white")

    output = build_degraded_multi_modal_data(
        {"images": [full], "videos": None},
        {
            "extra_info": {
                "condition_inputs": {"degraded_image": {"path": str(degraded_path)}}
            }
        },
    )

    assert output["images"][0].size == (20, 10)
    assert output["images"][0].getpixel((0, 0)) == (128, 128, 128)
    assert output["videos"] is None


def test_build_degraded_multimodal_rejects_dimension_drift(tmp_path):
    degraded_path = tmp_path / "wrong.png"
    Image.new("RGB", (10, 10), "gray").save(degraded_path)
    with pytest.raises(ValueError, match="dimensions"):
        build_degraded_multi_modal_data(
            {"images": [Image.new("RGB", (20, 10), "white")]},
            {"condition_inputs": {"degraded_image": {"path": str(degraded_path)}}},
        )


def test_prepare_native_verl_batch_validates_ids_and_drops_degraded_transport():
    batch_size, prompt_len, response_len = 4, 2, 3
    responses = torch.tensor([[1, 2, 3]] * batch_size)
    full_ids = torch.cat((torch.tensor([[8, 9]] * batch_size), responses), dim=1).unsqueeze(-1)
    degraded_ids = full_ids.clone()
    full_lp = torch.full((batch_size, prompt_len + response_len, 1), -0.2)
    degraded_lp = full_lp.clone()
    degraded_lp[:, -response_len:] -= torch.tensor([0.4, 0.3, 0.2, 0.1]).view(-1, 1, 1)
    batch = SimpleNamespace(
        batch={
            "teacher_logprobs": full_lp,
            "teacher_ids": full_ids,
            "teacher_degraded_logprobs": degraded_lp,
            "teacher_degraded_ids": degraded_ids,
            "responses": responses,
            "response_mask": torch.ones_like(responses, dtype=torch.bool),
        },
        non_tensor_batch={"uid": np.array(["same"] * batch_size, dtype=object)},
    )
    config = {
        "va_opd": {"enabled": True},
        "actor_rollout_ref": {"rollout": {"n": 4}},
    }

    metrics = prepare_native_verl_batch(batch, config)

    assert batch.batch["va_opd_token_weights"].shape == responses.shape
    assert "teacher_degraded_logprobs" not in batch.batch
    assert "teacher_degraded_ids" not in batch.batch
    assert metrics["va_opd/group_weight_sum_max_error"] == pytest.approx(0.0, abs=1e-6)


def test_prepare_native_verl_batch_fails_on_response_id_mismatch():
    responses = torch.tensor([[1, 2]] * 4)
    ids = responses.unsqueeze(-1).clone()
    bad_ids = ids.clone()
    bad_ids[0, 1, 0] = 7
    batch = SimpleNamespace(
        batch={
            "teacher_logprobs": torch.full((4, 2, 1), -0.2),
            "teacher_ids": ids,
            "teacher_degraded_logprobs": torch.full((4, 2, 1), -0.3),
            "teacher_degraded_ids": bad_ids,
            "responses": responses,
            "response_mask": torch.ones_like(responses, dtype=torch.bool),
        },
        non_tensor_batch={"uid": np.array(["same"] * 4, dtype=object)},
    )
    config = {
        "va_opd": {"enabled": True},
        "actor_rollout_ref": {"rollout": {"n": 4}},
    }

    with pytest.raises(ValueError, match="degraded-image teacher IDs"):
        prepare_native_verl_batch(batch, config)
