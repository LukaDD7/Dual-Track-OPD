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
    full_path = tmp_path / "full.png"
    degraded_path = tmp_path / "diagram.lowres.png"
    Image.new("RGB", (20, 10), "white").save(full_path)
    Image.new("RGB", (20, 10), "gray").save(degraded_path)
    full = Image.new("RGB", (20, 10), "white")

    output = build_degraded_multi_modal_data(
        {"images": [full], "videos": None},
            {
                "extra_info": {
                    "condition_inputs": {
                        "full_image": {"path": str(full_path)},
                        "degraded_image": {"path": str(degraded_path)},
                    }
                }
            },
    )

    assert output["images"][0].size == (20, 10)
    assert output["images"][0].getpixel((0, 0)) == (128, 128, 128)
    assert output["videos"] is None


def test_build_degraded_multimodal_replaces_every_image_in_order(tmp_path):
    full_one = tmp_path / "full-one.png"
    full_two = tmp_path / "full-two.png"
    degraded_one = tmp_path / "degraded-one.png"
    degraded_two = tmp_path / "degraded-two.png"
    Image.new("RGB", (20, 10), "red").save(full_one)
    Image.new("RGB", (15, 15), "blue").save(full_two)
    Image.new("RGB", (20, 10), "black").save(degraded_one)
    Image.new("RGB", (15, 15), "white").save(degraded_two)

    output = build_degraded_multi_modal_data(
        {"images": [Image.open(full_one), Image.open(full_two)]},
        {
            "condition_inputs": {
                "full_images": [{"path": str(full_one)}, {"path": str(full_two)}],
                "degraded_images": [
                    {"path": str(degraded_one)},
                    {"path": str(degraded_two)},
                ],
            }
        },
    )

    assert [image.size for image in output["images"]] == [(20, 10), (15, 15)]
    assert output["images"][0].getpixel((0, 0)) == (0, 0, 0)
    assert output["images"][1].getpixel((0, 0)) == (255, 255, 255)


def test_build_degraded_multimodal_rejects_pair_count_mismatch(tmp_path):
    full = tmp_path / "full.png"
    degraded = tmp_path / "degraded.png"
    Image.new("RGB", (20, 10), "red").save(full)
    Image.new("RGB", (20, 10), "black").save(degraded)

    with pytest.raises(ValueError, match="image count must match"):
        build_degraded_multi_modal_data(
            {"images": [Image.open(full), Image.open(full)]},
            {
                "condition_inputs": {
                    "full_images": [{"path": str(full)}, {"path": str(full)}],
                    "degraded_images": [{"path": str(degraded)}],
                }
            },
        )


def test_build_degraded_multimodal_rejects_dimension_drift(tmp_path):
    full_path = tmp_path / "full.png"
    degraded_path = tmp_path / "wrong.png"
    Image.new("RGB", (20, 10), "white").save(full_path)
    Image.new("RGB", (10, 10), "gray").save(degraded_path)
    with pytest.raises(ValueError, match="dimensions"):
        build_degraded_multi_modal_data(
            {"images": [Image.new("RGB", (20, 10), "white")]},
            {
                "condition_inputs": {
                    "full_image": {"path": str(full_path)},
                    "degraded_image": {"path": str(degraded_path)},
                }
            },
        )


def test_build_degraded_multimodal_follows_runtime_resize(tmp_path):
    full_path = tmp_path / "full.png"
    degraded_path = tmp_path / "degraded.png"
    Image.new("RGB", (20, 10), "white").save(full_path)
    Image.new("RGB", (20, 10), "gray").save(degraded_path)

    output = build_degraded_multi_modal_data(
        {"images": [Image.new("RGB", (40, 20), "white")]},
        {
            "extra_info": {
                "condition_inputs": {
                    "full_image": {"path": str(full_path)},
                    "degraded_image": {"path": str(degraded_path)},
                }
            }
        },
    )

    assert output["images"][0].size == (40, 20)


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
