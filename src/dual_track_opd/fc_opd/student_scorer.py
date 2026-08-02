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

import json
import os
from typing import Sequence

import torch

from .conditions import Condition
from .online_batch import OnlineFCOPDSample, OnlineStudentScores
from .teacher_prompts import RenderedTeacherPrompt, render_teacher_prompt


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
        """Batch-score exact rollout IDs without crossing prompt contexts.

        A verl rollout batch can contain several questions and images.  We
        therefore group by the fully rendered condition prompt and only batch
        responses that share the exact same context.  Within a group, response
        IDs are right-padded, so logits must be sliced from the prompt boundary
        rather than from the padded tensor tail.
        """
        import sys as _sys, time as _time
        _t0 = _time.time()
        B = len(samples)
        norm_conditions = list(dict.fromkeys(Condition(c) for c in conditions))
        if not norm_conditions:
            raise ValueError("at least one condition is required")
        # FULL logits are always required by OnlineStudentScores even when the
        # caller only asks for auxiliary routing conditions.
        ordered = [Condition.FULL]
        ordered += [c for c in norm_conditions if c is not Condition.FULL]

        # Per-sample loss_logits and condition_log_probs
        loss_logits_list: list[torch.Tensor] = [None] * B  # type: ignore[assignment]
        log_prob_maps: list[dict[Condition, torch.Tensor]] = [{} for _ in range(B)]
        total_prompt_groups = 0

        for cond in ordered:
            # Use the canonical rendered payload itself as the grouping key.
            # A digest alone would introduce an unnecessary collision risk.
            grouped: dict[
                str,
                list[tuple[int, OnlineFCOPDSample, RenderedTeacherPrompt]],
            ] = {}
            for original_index, sample in enumerate(samples):
                rendered = render_teacher_prompt(cond, sample.question, sample.condition_inputs)
                group_key = json.dumps(
                    {
                        "messages": rendered.messages,
                        "image_paths": rendered.image_paths,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                grouped.setdefault(group_key, []).append((original_index, sample, rendered))
            total_prompt_groups += len(grouped)

            for group in grouped.values():
                for _start in range(0, len(group), self._SCORE_MAX_SUB_BATCH):
                    indexed_chunk = group[_start:_start + self._SCORE_MAX_SUB_BATCH]
                    self._score_same_prompt_chunk(
                        indexed_chunk,
                        condition=cond,
                        include_condition_log_probs=cond in norm_conditions,
                        loss_logits_list=loss_logits_list,
                        log_prob_maps=log_prob_maps,
                    )

        if any(logits is None for logits in loss_logits_list):
            raise RuntimeError("FULL-condition student logits were not produced for every sample")
        _dt = _time.time() - _t0
        print(
            f"[student] batched B={B} prompt_groups={total_prompt_groups} "
            f"conditions={len(ordered)} sub_batch={self._SCORE_MAX_SUB_BATCH}: {_dt:.2f}s",
            file=_sys.stderr,
            flush=True,
        )
        return [
            OnlineStudentScores(loss_logits=loss_logits_list[i], condition_log_probs=log_prob_maps[i])
            for i in range(B)
        ]

    def _score_same_prompt_chunk(
        self,
        indexed_chunk: list[tuple[int, OnlineFCOPDSample, RenderedTeacherPrompt]],
        *,
        condition: Condition,
        include_condition_log_probs: bool,
        loss_logits_list: list[torch.Tensor],
        log_prob_maps: list[dict[Condition, torch.Tensor]],
    ) -> None:
        """Score one sub-batch whose rendered prompt/image is identical."""

        chunk_B = len(indexed_chunk)
        rendered = indexed_chunk[0][2]
        prompt_text = self._processor.apply_chat_template(
            list(rendered.messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        image = None
        if rendered.image_paths:
            image = self._image_cls.open(rendered.image_paths[0]).convert("RGB")
        try:
            encoded = self._processor(
                text=[prompt_text] * chunk_B,
                images=[image] * chunk_B if image is not None else None,
                padding=True,
                return_tensors="pt",
            )
        finally:
            if image is not None:
                image.close()
        encoded = {key: value.to(self._model.device) for key, value in encoded.items()}
        prompt_ids = encoded["input_ids"]
        prompt_width = int(prompt_ids.shape[1])

        response_lens = [len(sample.rollout_token_ids) for _, sample, _ in indexed_chunk]
        if any(length <= 0 for length in response_lens):
            raise ValueError("rollout_token_ids must be non-empty")
        max_response_len = max(response_lens)
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        padded_responses = torch.full(
            (chunk_B, max_response_len),
            int(pad_id),
            dtype=prompt_ids.dtype,
            device=self._model.device,
        )
        for row, (_, sample, _) in enumerate(indexed_chunk):
            padded_responses[row, : response_lens[row]] = torch.tensor(
                sample.rollout_token_ids,
                dtype=prompt_ids.dtype,
                device=self._model.device,
            )

        response_mask = (
            torch.arange(max_response_len, device=self._model.device).unsqueeze(0)
            < torch.tensor(response_lens, device=self._model.device).unsqueeze(1)
        )
        model_inputs: dict[str, torch.Tensor] = {
            key: value
            for key, value in encoded.items()
            if key not in {"input_ids", "attention_mask", "position_ids"}
        }
        model_inputs["input_ids"] = torch.cat([prompt_ids, padded_responses], dim=1)
        model_inputs["attention_mask"] = torch.cat(
            [
                encoded["attention_mask"],
                response_mask.to(encoded["attention_mask"].dtype),
            ],
            dim=1,
        )
        for key in ("token_type_ids", "mm_token_type_ids"):
            if key in model_inputs:
                extension = torch.zeros(
                    chunk_B,
                    max_response_len,
                    dtype=model_inputs[key].dtype,
                    device=self._model.device,
                )
                model_inputs[key] = torch.cat([model_inputs[key], extension], dim=-1)

        # The actor loss is computed elsewhere; this scorer only produces
        # current-policy signals and detached logits for the hook.
        with torch.no_grad():
            outputs = self._model(**model_inputs)

        for row, (original_index, sample, _) in enumerate(indexed_chunk):
            response_len = response_lens[row]
            response_logits = outputs.logits[
                row,
                prompt_width - 1 : prompt_width - 1 + response_len,
                :,
            ]
            if int(response_logits.shape[0]) != response_len:
                raise RuntimeError("student response logits do not align with rollout IDs")
            response_ids = torch.tensor(
                sample.rollout_token_ids,
                dtype=torch.long,
                device=response_logits.device,
            )
            log_probs = torch.log_softmax(response_logits.float(), dim=-1)
            gathered = log_probs[
                torch.arange(response_len, device=log_probs.device),
                response_ids,
            ].cpu()
            if include_condition_log_probs:
                log_prob_maps[original_index][condition] = gathered
            if condition is Condition.FULL:
                loss_logits_list[original_index] = response_logits.detach()

        del outputs, model_inputs, encoded

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
            if key in {"input_ids", "attention_mask", "position_ids"}:
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
