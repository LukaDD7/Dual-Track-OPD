"""GPU student forced-scorer callable via FQN from verl config.

This module provides a :class:`StudentScorer` that implements the
:class:`~.online_batch.StudentForcedScorer` protocol.  It loads a
HuggingFace VLM (Qwen3-VL or similar) on GPU, renders condition-specific
prompts, and runs forced-scoring forward passes over the current student
rollout token IDs.

Usage from a verl Hydra config::

    algorithm:
      fc_opd:
        student_scorer_fqn: dual_track_opd.fc_opd.student_scorer.StudentScorer
        student_scorer_kwargs:
          model_path: Qwen/Qwen3-VL-4B-Instruct
          device: cuda
          dtype: bfloat16
          top_k: 32

The scorer loads a **separate** copy of the student model.  For
memory-constrained setups the user may instead write a thin FQN wrapper
that re-uses the actor worker's live model; this module provides the
standalone baseline.
"""

from __future__ import annotations

import os
from typing import Sequence

import torch

from .conditions import Condition, ConditionInputs
from .online_batch import OnlineFCOPDSample, OnlineStudentScores
from .teacher_prompts import render_teacher_prompt


class StudentScorer:
    """Standalone GPU student forced-scorer.

    Each call runs ``len(conditions)`` forward passes through the VLM —
    one per condition prompt.  The ``FULL`` condition forward is kept with
    grad for loss back-propagation; all other conditions run under
    ``torch.no_grad()``.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        top_k: int = 32,
    ):
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._image_cls = Image
        resolved = os.path.expandvars(model_path.removeprefix("hf:"))
        self._processor = AutoProcessor.from_pretrained(resolved)
        torch_dtype = getattr(torch, dtype) if dtype != "float32" else torch.float32
        self._model = AutoModelForImageTextToText.from_pretrained(
            resolved,
            torch_dtype=torch_dtype,
            device_map=device if device == "cuda" else None,
        )
        if device != "cuda":
            self._model.to(device)
        self._model.eval()
        self._tokenizer = self._processor.tokenizer
        self._top_k = int(top_k)
        self._device = device

    # -- StudentForcedScorer protocol ----------------------------------------

    def __call__(
        self,
        sample: OnlineFCOPDSample,
        conditions: Sequence[Condition],
    ) -> OnlineStudentScores:
        """Score *sample.rollout_token_ids* under every requested condition."""
        if not conditions:
            raise ValueError("at least one condition is required")
        loss_logits = self._condition_logits(
            sample, Condition.FULL, grad=True
        )
        condition_log_probs: dict[Condition, torch.Tensor] = {}
        response_ids = torch.tensor(
            sample.rollout_token_ids, dtype=torch.long, device=loss_logits.device
        ).reshape(1, -1)
        with torch.no_grad():
            for condition in conditions:
                cond = Condition(condition)
                if cond is Condition.FULL:
                    # Re-use the grad-enabled logits (saves one forward pass).
                    log_probs = torch.log_softmax(loss_logits.float(), dim=-1)
                else:
                    logits = self._condition_logits(sample, cond, grad=False)
                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                condition_log_probs[cond] = (
                    log_probs.gather(-1, response_ids.unsqueeze(-1))
                    .squeeze(-1)
                    .cpu()
                )
        return OnlineStudentScores(
            loss_logits=loss_logits, condition_log_probs=condition_log_probs
        )

    # -- helpers -------------------------------------------------------------

    def _condition_logits(
        self,
        sample: OnlineFCOPDSample,
        condition: Condition,
        *,
        grad: bool,
    ) -> torch.Tensor:
        """Return logits ``[1, T, vocab]`` for the rollout positions."""
        rendered = render_teacher_prompt(
            condition, sample.question, sample.condition_inputs
        )
        images = None
        if rendered.image_paths:
            images = [
                self._image_cls.open(rendered.image_paths[0]).convert("RGB")
            ]
        prompt = self._processor.apply_chat_template(
            list(rendered.messages), tokenize=False, add_generation_prompt=True
        )
        proc = self._processor(
            text=[prompt], images=images, return_tensors="pt"
        )
        prompt_ids = proc["input_ids"]
        prompt_len = int(prompt_ids.shape[1])
        response_tensor = torch.tensor(
            [sample.rollout_token_ids], dtype=prompt_ids.dtype
        )
        full_ids = torch.cat([prompt_ids, response_tensor], dim=1).to(
            self._model.device
        )
        model_inputs: dict[str, torch.Tensor] = {}
        for key, value in proc.items():
            if key in {"input_ids", "attention_mask"}:
                continue
            model_inputs[key] = value.to(self._model.device)
        model_inputs["input_ids"] = full_ids
        model_inputs["attention_mask"] = torch.ones_like(full_ids)
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            return self._model(**model_inputs).logits[
                :,
                prompt_len - 1 : prompt_len - 1 + len(sample.rollout_token_ids),
                :,
            ]
