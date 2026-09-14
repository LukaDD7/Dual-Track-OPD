"""Thin integration helpers for verl's native on-policy-distillation trainer.

All VA research logic stays in this package.  The backend patch only calls the
entry points in this module and transports two extra teacher tensors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .objective import build_va_token_weights


def build_degraded_multi_modal_data(
    multi_modal_data: Mapping[str, Any] | None,
    sample_fields: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Replace the single full-resolution image with its prepared 10% image.

    The prepared dataset is the source of truth.  Runtime degradation is not a
    fallback because it can change image dimensions, interpolation, or visual
    tokenization without being captured by the experiment manifest.
    """

    source = dict(multi_modal_data or {})
    images = source.get("images")
    image_items = _as_items(images)
    if len(image_items) != 1:
        raise ValueError(f"VA-OPD currently requires exactly one image per sample, got {len(image_items)}")

    extra_info = _as_mapping((sample_fields or {}).get("extra_info"))
    condition_inputs = _as_mapping(
        (sample_fields or {}).get("condition_inputs")
        or extra_info.get("condition_inputs")
        or extra_info.get("fc_opd_condition_inputs")
    )
    full = _as_mapping(condition_inputs.get("full_image"))
    full_path = Path(str(full.get("path", ""))).expanduser()
    degraded = _as_mapping(condition_inputs.get("degraded_image"))
    degraded_path = Path(str(degraded.get("path", ""))).expanduser()
    if not full_path.is_file():
        raise FileNotFoundError(f"prepared VA-OPD full image is missing: {full_path}")
    if not degraded_path.is_file():
        raise FileNotFoundError(f"prepared VA-OPD degraded image is missing: {degraded_path}")

    from PIL import Image

    with Image.open(full_path) as handle:
        stored_full_size = handle.size
    with Image.open(degraded_path) as handle:
        degraded_image = handle.convert("RGB").copy()
    if degraded_image.size != stored_full_size:
        raise ValueError(
            "prepared degraded image dimensions differ from the prepared full image: "
            f"full={stored_full_size}, degraded={degraded_image.size}, path={degraded_path}"
        )

    original_size = _image_size(image_items[0])
    if original_size is not None and degraded_image.size != original_size:
        # vLLM may resize the full image before returning it to the agent loop.
        # The persisted image pair remains the provenance source, but the teacher
        # must consume the same runtime dimensions for both conditions so the
        # visual token count and sequence alignment stay unchanged.
        degraded_image = degraded_image.resize(original_size, Image.Resampling.NEAREST)

    source["images"] = _replace_single(images, degraded_image)
    return source


def prepare_native_verl_batch(batch: Any, config: Any) -> dict[str, float]:
    """Attach paper-faithful token weights before verl balances/microbatches data."""

    va_config = _config_get(_config_get(config, "va_opd", {}), "enabled", False)
    if not bool(va_config):
        return {}

    tensors = batch.batch
    required = {
        "teacher_logprobs",
        "teacher_ids",
        "teacher_degraded_logprobs",
        "teacher_degraded_ids",
        "responses",
        "response_mask",
    }
    missing = sorted(required - set(tensors.keys()))
    if missing:
        raise KeyError(f"native VA-OPD transport is missing tensors: {missing}")

    response_mask = tensors["response_mask"].bool()
    response_length = int(response_mask.shape[1])
    # Teacher tensors are full prompt+response sequences, but the V1 TransferQueue
    # path may already have converted them to response-length jagged tensors.
    # Preserve both layouts and never trust padding-only positions for identity.
    responses = tensors["responses"].long()
    valid = response_mask
    full_ids, full_log_probs = _aligned_teacher_pair(
        tensors["teacher_ids"],
        tensors["teacher_logprobs"],
        responses,
        response_mask,
    )
    degraded_ids, degraded_log_probs = _aligned_teacher_pair(
        tensors["teacher_degraded_ids"],
        tensors["teacher_degraded_logprobs"],
        responses,
        response_mask,
    )
    if not torch.equal(full_ids[valid], responses[valid]):
        mismatch = torch.nonzero(full_ids[valid] != responses[valid], as_tuple=False).flatten()
        raise ValueError(
            "full-image teacher IDs are not aligned to the exact student response IDs "
            f"(mismatch_count={mismatch.numel()})"
        )
    if not torch.equal(degraded_ids[valid], responses[valid]):
        mismatch = torch.nonzero(degraded_ids[valid] != responses[valid], as_tuple=False).flatten()
        raise ValueError(
            "degraded-image teacher IDs are not aligned to the exact student response IDs "
            f"(mismatch_count={mismatch.numel()})"
        )

    prompt_ids = list(batch.non_tensor_batch.get("uid", ()))
    if len(prompt_ids) != response_mask.shape[0]:
        raise ValueError("verl uid transport must contain one stable prompt-group id per rollout")
    rollout_n = int(_config_get(_config_get(_config_get(config, "actor_rollout_ref", {}), "rollout", {}), "n", 0))
    va_settings = _config_get(config, "va_opd", {})
    result = build_va_token_weights(
        full_log_probs,
        degraded_log_probs,
        response_mask=response_mask,
        prompt_ids=prompt_ids,
        expected_rollouts=rollout_n,
        top_fraction=float(_config_get(va_settings, "top_fraction", 0.20)),
        lambda_high=float(_config_get(va_settings, "lambda_high", 0.50)),
        tau=float(_config_get(va_settings, "tau", 1.0)),
    )
    tensors["va_opd_token_weights"] = result.token_weights.to(response_mask.device)
    tensors["va_opd_visual_advantage"] = result.visual_advantage.to(response_mask.device)

    # The degraded pass is only a fixed routing signal.  Drop it before actor
    # transfer/unpadding; the full-image teacher remains the KL target.
    tensors.pop("teacher_degraded_logprobs")
    tensors.pop("teacher_degraded_ids")

    group_sums = []
    for prompt_id in dict.fromkeys(prompt_ids):
        rows = torch.as_tensor(
            [i for i, value in enumerate(prompt_ids) if value == prompt_id],
            device=result.rollout_weights.device,
        )
        group_sums.append(result.rollout_weights[rows].sum())
    group_sums_tensor = torch.stack(group_sums)
    valid_va = result.visual_advantage[valid]
    return {
        "va_opd/mean": float(valid_va.mean().item()),
        "va_opd/positive_ratio": float((valid_va > 0).float().mean().item()),
        "va_opd/high_token_ratio": float(result.high_mask[valid].float().mean().item()),
        "va_opd/rollout_weight_min": float(result.rollout_weights.min().item()),
        "va_opd/rollout_weight_max": float(result.rollout_weights.max().item()),
        "va_opd/group_weight_sum_max_error": float((group_sums_tensor - 1.0).abs().max().item()),
        "va_opd/prompt_groups": float(result.prompt_group_count),
        "va_opd/degenerate_sequences": float(result.degenerate_sequence_count),
    }


def register_native_verl_loss(
    register_loss: Callable[..., Any],
    settings_type: type,
) -> None:
    """Register ``va_opd_k1`` after verl's registry has been constructed."""

    @register_loss(settings_type(names=["va_opd_k1"], use_estimator=True))
    def compute_va_opd_k1(config, distillation_config, model_output, data):
        from verl.trainer.ppo.core_algos import kl_penalty
        from verl.utils.metric import AggregationType, Metric
        from verl.workers.utils.padding import no_padding_2_padding

        student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
        teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
        response_mask = data["response_mask"]
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        weights = data["va_opd_token_weights"]
        if weights.is_nested:
            weights = weights.to_padded_tensor(0.0)
        if not (student_log_probs.shape == teacher_log_probs.shape == weights.shape == response_mask.shape):
            raise ValueError(
                "VA-OPD loss tensors are misaligned: "
                f"student={student_log_probs.shape}, teacher={teacher_log_probs.shape}, "
                f"weights={weights.shape}, mask={response_mask.shape}"
            )
        k1 = kl_penalty(logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty="k1")
        weighted = k1 * weights.to(device=k1.device, dtype=k1.dtype)
        valid = response_mask.bool()
        metrics = {
            "distillation/abs_loss": Metric(AggregationType.MEAN, k1[valid].abs().mean()),
            "va_opd/token_weight_mean": Metric(AggregationType.MEAN, weights[valid].float().mean()),
            "va_opd/token_weight_max": Metric(AggregationType.MAX, weights[valid].float().max()),
        }
        return weighted, metrics


def _response_scalar(tensor: torch.Tensor, response_length: int) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.ndim != 2 or tensor.shape[1] < response_length:
        raise ValueError(f"expected [B, prompt+response, 1] teacher tensor, got {tuple(tensor.shape)}")
    return tensor[:, -response_length:]


def _aligned_teacher_pair(
    teacher_ids: torch.Tensor,
    teacher_values: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Locate each student response inside teacher IDs and extract paired values.

    V1 stores teacher tensors either as full prompt+response tensors or as
    response-only tensors, with layout-dependent left/right padding.  Exact
    subsequence matching makes the alignment contract explicit instead of
    guessing a fixed slice offset.
    """
    if teacher_ids.is_nested:
        ids = teacher_ids.to_padded_tensor(0).long()
    else:
        ids = _squeeze_last_one(teacher_ids).long()
    if teacher_values.is_nested:
        values = teacher_values.to_padded_tensor(0.0)
    else:
        values = _squeeze_last_one(teacher_values)

    if ids.shape[0] != values.shape[0] or ids.shape[0] != responses.shape[0]:
        raise ValueError(
            "teacher/student batch sizes differ: "
            f"ids={ids.shape[0]}, values={values.shape[0]}, responses={responses.shape[0]}"
        )

    aligned_ids = torch.zeros_like(responses, dtype=torch.long)
    aligned_values = torch.zeros_like(responses, dtype=teacher_values.dtype)
    for row in range(responses.shape[0]):
        mask = response_mask[row].bool()
        response_tokens = responses[row, mask]
        count = int(mask.sum().item())
        if count == 0:
            continue
        if ids.shape[1] == responses.shape[1]:
            aligned_ids[row, mask] = ids[row, mask]
            aligned_values[row, mask] = values[row, mask]
            continue
        else:
            positions = _find_subsequence(ids[row], response_tokens)
        if positions is None or positions.numel() != count:
            raise ValueError(
                f"teacher response subsequence not found for row {row}: "
                f"teacher_width={ids.shape[1]}, response_tokens={count}"
            )
        aligned_ids[row, mask] = ids[row, positions]
        aligned_values[row, mask] = values[row, positions]
    return aligned_ids, aligned_values


def _find_subsequence(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    """Return exact contiguous source indices for ``target``."""
    source = source.flatten()
    target = target.flatten()
    if target.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=source.device)
    if source.numel() < target.numel():
        return None
    matches = (source == target[0]).nonzero(as_tuple=False).flatten()
    for start in matches.tolist():
        candidate = source[start : start + target.numel()]
        if candidate.numel() == target.numel() and torch.equal(candidate, target):
            return torch.arange(start, start + target.numel(), device=source.device)
    return None


def _squeeze_last_one(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    if tensor.ndim != 2:
        raise ValueError(f"expected a 2-D teacher tensor, got {tuple(tensor.shape)}")
    return tensor


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _as_mapping(value.item())
    if isinstance(value, np.ndarray):
        return _as_mapping(value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            return dict(value)
        except (TypeError, ValueError):
            return {}
    return {}


def _as_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _replace_single(original: Any, replacement: Any) -> Any:
    if isinstance(original, tuple):
        return (replacement,)
    if isinstance(original, list):
        return [replacement]
    return replacement


def _image_size(image: Any) -> tuple[int, int] | None:
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) == 2:
        return int(size[0]), int(size[1])
    if isinstance(image, (str, Path)) and Path(image).is_file():
        from PIL import Image

        with Image.open(image) as handle:
            return handle.size
    return None


def _config_get(config: Any, key: str, default: Any) -> Any:
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)
