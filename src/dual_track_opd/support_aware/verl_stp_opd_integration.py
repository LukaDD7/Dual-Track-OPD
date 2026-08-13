"""verl 0.7.1 integration for the STP-OPD pilot (thin, opt-in).

Follows the validated FC-OPD verl pattern (patches/verl/fc_opd_*.patch):

1. ``stp_opd_post_rollout_hook`` attaches teacher-scored tensors to the verl
   rollout batch (``stp_*`` keys).  The teacher service scores the full hybrid
   (fixed answer-free teacher prefix + student-sampled suffix) trajectory; the
   suffix masks select the suffix region for the RKL-K1 and GRPO terms.
2. ``has_stp_opd_tensors`` / ``compute_stp_opd_actor_loss`` are consumed by the
   thin verl actor patch (patches/verl/0002-stp-opd-actor-loss.patch).

Scaffold contract: the data pipeline must already encode the branch — a
scaffolded sample has the teacher prefix tokens in the response prefix region
(covered by ``stp_prefix_mask``) and the student suffix after it; an
unscaffolded sample has no prefix (empty prefix mask).  The hook derives the
masks from ``stp_prefix_lengths`` (per-sample prefix token count).

This module imports verl-related classes lazily and is import-safe without a
verl install.  The end-to-end wiring must be validated in the pinned HPC verl
environment (handoff §6 P0: exact-token identity, online scorer in A0/A3,
gradient isolation, resume/manifest).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .support_transition_train import SupportTransitionTrainConfig, train_step_loss


STP_TENSOR_KEYS = (
    "stp_prefix_mask",
    "stp_suffix_mask",
    "stp_sampled_ids",
    "stp_teacher_k1_log_probs",
    "stp_valid_mask",
    "stp_prefix_lengths",
)


def load_prefixes(path: str | Path) -> dict[str, tuple[int, ...]]:
    """Load the fixed verified answer-free teacher prefixes.

    JSON mapping ``prompt_id -> [token_ids]``; the pilot fixes one horizon per
    prompt (no horizon tuning inside the pilot, handoff §5).
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        str(prompt_id): tuple(int(value) for value in token_ids)
        for prompt_id, token_ids in raw.items()
    }


def _stp_config(config: Any) -> Mapping[str, Any]:
    algorithm = getattr(config, "algorithm", None) or (config.get("algorithm") if isinstance(config, Mapping) else {})
    return (algorithm or {}).get("stp_opd") or {}


def stp_opd_post_rollout_hook(
    *,
    batch: Any,
    tokenizer: Any,
    processor: Any | None,
    config: Any,
    global_steps: int,
    teacher_client: Any,
    prefixes: Mapping[str, tuple[int, ...]],
) -> tuple[Any, dict[str, float]]:
    """Attach STP-OPD tensors to a verl rollout batch (thin hook body).

    Call this from a registered verl hook entry (see patches/verl/0001-stp-opd-
    ray-trainer-hook.patch) or directly in a custom trainer.  The teacher client
    scores the hybrid trajectory; ``teacher_sampled_log_probs`` provides the
    K1 log-prob per token.
    """

    del processor, global_steps
    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"].bool()
    B, T = int(responses.shape[0]), int(responses.shape[1])
    device = responses.device
    prompt_ids = batch.non_tensor_batch.get("stp_prompt_ids")
    if prompt_ids is None:
        raise ValueError("stp_prompt_ids missing from non-tensor batch")

    prefix_lengths = torch.zeros(B, dtype=torch.long)
    sampled_ids = responses.clone()
    positions = torch.arange(T)
    prefix_mask = torch.zeros(B, T, dtype=torch.bool)
    for index in range(B):
        prompt_id = str(prompt_ids[index])
        prefix_ids = prefixes.get(prompt_id, ())
        prefix_length = min(len(prefix_ids), T)
        prefix_lengths[index] = prefix_length
        prefix_mask[index, :prefix_length] = True
        # sampled suffix ids are the student response tokens in the suffix region
        sampled_ids[index, :prefix_length] = -1
    suffix_mask = (~prefix_mask) & response_mask

    # Teacher K1 log-probs over the hybrid trajectory: delegate to the teacher
    # client exactly as FC-OPD does (score full hybrid response token ids).
    teacher_log_probs = torch.full((B, T), float("-inf"), dtype=torch.float32, device=device)
    for index in range(B):
        prompt_id = str(prompt_ids[index])
        response_ids = [
            int(value) for value in responses[index, response_mask[index]].tolist()
        ]
        scored = teacher_client.score_hybrid_k1(
            prompt_id=prompt_id,
            response_token_ids=response_ids,
        )
        valid = response_mask[index]
        teacher_log_probs[index, valid] = torch.tensor(
            scored, dtype=torch.float32, device=device
        )
    valid_mask = torch.isfinite(teacher_log_probs) & response_mask

    batch.batch["stp_prefix_mask"] = prefix_mask.to(device)
    batch.batch["stp_suffix_mask"] = suffix_mask.to(device)
    batch.batch["stp_sampled_ids"] = sampled_ids.to(device)
    batch.batch["stp_teacher_k1_log_probs"] = teacher_log_probs.to(device)
    batch.batch["stp_valid_mask"] = valid_mask.to(device)
    batch.batch["stp_prefix_lengths"] = prefix_lengths.to(device)
    metrics = {
        "actor/stp_prefix_tokens": int(prefix_mask.sum()),
        "actor/stp_suffix_tokens": int(suffix_mask.sum()),
    }
    return batch, metrics


def has_stp_opd_tensors(batch: Mapping[str, Any]) -> bool:
    """True when a verl micro-batch carries the STP-OPD tensor set."""

    present = [key in batch for key in STP_TENSOR_KEYS]
    if any(present) and not all(present):
        missing = [key for key, is_present in zip(STP_TENSOR_KEYS, present, strict=True) if not is_present]
        raise RuntimeError(f"incomplete STP-OPD tensor set; missing: {missing}")
    return all(present)


def compute_stp_opd_actor_loss(
    student_logits: torch.Tensor,
    batch: Mapping[str, Any],
    response_mask: torch.Tensor,
    config: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Assemble the STP-OPD regional loss for one actor mini-batch.

    ``student_logits`` has shape [B, T, V]; the batch carries the hook-attached
    ``stp_*`` tensors: prefix/suffix masks, sampled ids, teacher K1 log-probs,
    valid mask, and the verl-computed advantages (``batch["advantages"]``,
    the standard verl key; ``stp_advantages`` is accepted as an override).
    Delegates to
    ``train_step_loss`` with the arm from the config (A0..A3).
    """

    if not has_stp_opd_tensors(batch):
        raise RuntimeError("STP-OPD loss requested without stp_* tensors")
    stp = _stp_config(config)
    arm = str(stp.get("arm") or "A3")
    train_config = SupportTransitionTrainConfig(
        arm=arm,
        lambda_prefix=float(stp.get("lambda_prefix", 1.0)),
        lambda_distill=float(stp.get("lambda_distill", 1.0)),
        lambda_task=float(stp.get("lambda_task", 1.0)),
    )
    device = student_logits.device
    advantages = batch.get("stp_advantages") or batch.get("advantages")
    if advantages is None:
        raise RuntimeError(
            "advantages missing; verl must provide batch['advantages'] "
            "(or an stp_advantages override)"
        )
    total, terms = train_step_loss(
        student_logits,
        student_logits,
        prefix_ids=batch["stp_sampled_ids"].to(device),
        sampled_ids=batch["stp_sampled_ids"].to(device),
        advantages=advantages.to(device),
        arm=arm,
        response_length=int(student_logits.shape[1]),
        prefix_length=0,
        scaffolded=False,
        prefix_mask=batch["stp_prefix_mask"].to(device),
        suffix_mask=batch["stp_suffix_mask"].to(device),
        teacher_sampled_k1_log_probs=batch["stp_teacher_k1_log_probs"].to(device),
        valid_mask=batch["stp_valid_mask"].to(device),
        config=train_config,
    )
    metrics = {
        "actor/stp_loss": float(total.detach()),
        "actor/stp_prefix_fkl": float(terms["prefix_fkl"].detach()),
        "actor/stp_suffix_rkl": float(terms["suffix_rkl"].detach()),
        "actor/stp_suffix_pg": float(terms["suffix_pg"].detach()),
    }
    per_token = torch.zeros(
        batch["stp_suffix_mask"].shape, dtype=torch.float32, device=device
    )
    return per_token, metrics
