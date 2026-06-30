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
        sample: OnlineFCOPDSample | Sequence[OnlineFCOPDSample],
        conditions: Sequence[Condition],
    ) -> OnlineStudentScores | list[OnlineStudentScores]:
        """Score one or multiple samples under every requested condition."""
        if isinstance(sample, Sequence) and not isinstance(sample, (str, bytes)):
            samples_list = list(sample)
            if len(samples_list) == 0:
                raise ValueError("at least one sample is required")
            if len(samples_list) == 1:
                return self._score_one(samples_list[0], conditions)
            return self._score_batched(samples_list, conditions)
        return self._score_one(sample, conditions)  # type: ignore[arg-type]

    def _score_one(
        self,
        sample: OnlineFCOPDSample,
        conditions: Sequence[Condition],
    ) -> OnlineStudentScores:
        if not conditions:
            raise ValueError("at least one condition is required")
        loss_logits = self._condition_logits(sample, Condition.FULL, grad=True)
        condition_log_probs: dict[Condition, torch.Tensor] = {}
        response_ids = torch.tensor(
            sample.rollout_token_ids, dtype=torch.long, device=loss_logits.device
        ).reshape(1, -1)
        with torch.no_grad():
            for condition in conditions:
                cond = Condition(condition)
                if cond is Condition.FULL:
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

    _SCORE_MAX_SUB_BATCH = 4  # small batches to avoid OOM from large logits tensors [B, T, V]

    def _score_batched(
        self,
        samples: list[OnlineFCOPDSample],
        conditions: Sequence[Condition],
    ) -> list[OnlineStudentScores]:
        """Batch-score: one condition at a time, sub-batched to avoid OOM."""
        import sys as _sys, time as _time
        _t0 = _time.time()
        B = len(samples)
        norm_conditions = [Condition(c) for c in conditions]
        # Ensure FULL is first so we get loss_logits right away
        ordered = [c for c in norm_conditions if c is Condition.FULL]
        ordered += [c for c in norm_conditions if c is not Condition.FULL]

        # Per-sample loss_logits and condition_log_probs
        loss_logits_list: list[torch.Tensor] = [None] * B  # type: ignore[assignment]
        log_prob_maps: list[dict[Condition, torch.Tensor]] = [{} for _ in range(B)]

        for cond in ordered:
            for _start in range(0, B, self._SCORE_MAX_SUB_BATCH):
                _end = min(_start + self._SCORE_MAX_SUB_BATCH, B)
                chunk = samples[_start:_end]
                chunk_B = _end - _start

                # Build chunk_B copies of the condition prompt
                first = chunk[0]
                rendered = render_teacher_prompt(cond, first.question, first.condition_inputs)
                prompt_text = self._processor.apply_chat_template(
                    list(rendered.messages), tokenize=False, add_generation_prompt=True,
                )
                images = None
                if rendered.image_paths:
                    images = [self._image_cls.open(rendered.image_paths[0]).convert("RGB")]
                try:
                    encoded = self._processor(
                        text=[prompt_text] * chunk_B,
                        images=[images] * chunk_B if images else None,
                        padding=True, return_tensors="pt",
                    )
                finally:
                    for img in (images or []):
                        img.close()
                encoded = {k: v.to(self._model.device) for k, v in encoded.items()}
                prompt_ids = encoded["input_ids"]  # [chunk_B, Pp]

                # Pad responses to max length within this sub-batch
                max_R = max(len(s.rollout_token_ids) for s in chunk)
                resp_lens = [len(s.rollout_token_ids) for s in chunk]
                padded = torch.zeros(chunk_B, max_R, dtype=torch.long, device=self._model.device)
                for i, s in enumerate(chunk):
                    rlen = resp_lens[i]
                    padded[i, :rlen] = torch.tensor(s.rollout_token_ids, dtype=torch.long, device=self._model.device)

                input_ids = torch.cat([prompt_ids, padded], dim=1)
                resp_mask = (torch.arange(max_R, device=self._model.device).unsqueeze(0)
                             < torch.tensor(resp_lens, device=self._model.device).unsqueeze(1))
                am = torch.cat([encoded["attention_mask"], resp_mask.to(encoded["attention_mask"].dtype)], dim=1)

                model_inputs: dict[str, torch.Tensor] = {}
                for k, v in encoded.items():
                    if k in ("input_ids", "attention_mask"):
                        continue
                    model_inputs[k] = v
                model_inputs["input_ids"] = input_ids
                model_inputs["attention_mask"] = am
                for key in ("token_type_ids", "mm_token_type_ids"):
                    if key in model_inputs:
                        ext = torch.zeros(chunk_B, max_R, dtype=model_inputs[key].dtype, device=self._model.device)
                        model_inputs[key] = torch.cat([model_inputs[key], ext], dim=-1)

                # No grad needed — loss is computed in dp_actor, not in the scorer.
                with torch.no_grad():
                    outs = self._model(**model_inputs)

                for i, s in enumerate(chunk):
                    rlen = resp_lens[i]
                    sample_logits = outs.logits[i, -rlen:, :]
                    log_probs = torch.log_softmax(sample_logits.float(), dim=-1)
                    resp_ids = torch.tensor(s.rollout_token_ids, dtype=torch.long,
                                            device=sample_logits.device).unsqueeze(0)
                    gathered = log_probs[torch.arange(rlen, device=log_probs.device), resp_ids.squeeze(0)].cpu()
                    log_prob_maps[_start + i][cond] = gathered
                    if cond is Condition.FULL:
                        loss_logits_list[_start + i] = sample_logits.detach()

                del outs, model_inputs, encoded

        _dt = _time.time() - _t0
        print(f"[student] batched B={B} conditions={len(ordered)} sub_batch={self._SCORE_MAX_SUB_BATCH}: {_dt:.2f}s", file=_sys.stderr, flush=True)
        return [
            OnlineStudentScores(loss_logits=loss_logits_list[i], condition_log_probs=log_prob_maps[i])
            for i in range(B)
        ]

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
