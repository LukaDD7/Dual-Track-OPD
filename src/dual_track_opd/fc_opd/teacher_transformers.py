"""Optional Transformers backend for exact-token Qwen3-VL teacher scoring."""

from __future__ import annotations

from typing import Sequence

import torch

from .teacher_protocol import (
    TeacherMetadata,
    TeacherScoreRequest,
    TeacherScoreResponse,
    ensure_exact_token_alignment,
    tokenizer_fingerprint,
)
from .teacher_prompts import render_teacher_prompt
from .teacher_scorer import TeacherScorer


class TransformersTeacherScorer(TeacherScorer):
    """Forced-score exact student response IDs with a Qwen3-VL teacher.

    Heavy dependencies and model weights are loaded only when this class is
    instantiated. The first real run must pass the GPU Smoke 2 probe before the
    scorer is used for optimization.
    """

    def __init__(
        self,
        *,
        model_id: str,
        top_k: int = 32,
        dtype: str = "bfloat16",
        device: str = "cuda",
        revision: str = "main",
    ):
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("install the GPU teacher environment before using this backend") from exc

        if not hasattr(torch, dtype):
            raise ValueError(f"unsupported torch dtype: {dtype}")
        torch_dtype = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.top_k = top_k
        vocab_size = int(
            getattr(
                getattr(self.model.config, "text_config", self.model.config),
                "vocab_size",
            )
        )
        if not 1 <= top_k < vocab_size:
            raise ValueError("top_k must be positive and smaller than the model vocabulary")
        resolved_revision = str(getattr(self.model.config, "_commit_hash", None) or revision)
        self._metadata = TeacherMetadata(
            model_id=model_id,
            tokenizer_hash=tokenizer_fingerprint(self.tokenizer),
            vocab_size=vocab_size,
            top_k=top_k,
            dtype=dtype,
            git_revision=resolved_revision,
        )

    @property
    def metadata(self) -> TeacherMetadata:
        return self._metadata

    def _check_response_text(self, request: TeacherScoreRequest) -> None:
        if request.response_text is None:
            return
        encoded = self.tokenizer.encode(request.response_text, add_special_tokens=False)
        ensure_exact_token_alignment(
            request.response_token_ids,
            encoded,
            context="teacher-side response retokenization",
        )

    def _prepare_prompt(self, request: TeacherScoreRequest) -> dict[str, torch.Tensor]:
        rendered = render_teacher_prompt(
            request.condition,
            request.question,
            request.condition_inputs,
        )
        prompt_text = self.processor.apply_chat_template(
            list(rendered.messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        images = None
        if rendered.image_paths:
            from PIL import Image

            images = [Image.open(path).convert("RGB") for path in rendered.image_paths]
        try:
            encoded = self.processor(
                text=[prompt_text],
                images=images,
                padding=True,
                return_tensors="pt",
            )
        finally:
            for image in images or []:
                image.close()
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _position_ids(self, model_inputs: dict[str, torch.Tensor]) -> torch.Tensor | None:
        kwargs = {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "image_grid_thw": model_inputs.get("image_grid_thw"),
            "video_grid_thw": model_inputs.get("video_grid_thw"),
        }
        candidates = (
            getattr(self.processor, "get_rope_index", None),
            getattr(self.model, "get_rope_index", None),
            getattr(getattr(self.model, "model", None), "get_rope_index", None),
        )
        for candidate in candidates:
            if candidate is None:
                continue
            result = candidate(**kwargs)
            return result[0] if isinstance(result, tuple) else result
        if kwargs["image_grid_thw"] is not None or kwargs["video_grid_thw"] is not None:
            raise RuntimeError("Qwen3-VL backend could not construct multimodal position IDs")
        return None

    @torch.inference_mode()
    def _score_one(self, request: TeacherScoreRequest) -> TeacherScoreResponse:
        if request.tokenizer_hash != self.metadata.tokenizer_hash:
            raise ValueError("request tokenizer hash does not match teacher metadata")
        if not request.response_token_ids:
            raise ValueError("response_token_ids must be non-empty")
        self._check_response_text(request)
        prompt_inputs = self._prepare_prompt(request)

        prompt_ids = prompt_inputs["input_ids"]
        prompt_length = prompt_ids.shape[1]
        response_ids = torch.tensor(
            [request.response_token_ids],
            dtype=prompt_ids.dtype,
            device=self.device,
        )
        model_inputs = dict(prompt_inputs)
        model_inputs["input_ids"] = torch.cat((prompt_ids, response_ids), dim=1)
        model_inputs["attention_mask"] = torch.cat(
            (
                prompt_inputs["attention_mask"],
                torch.ones_like(response_ids),
            ),
            dim=1,
        )
        for key in ("token_type_ids", "mm_token_type_ids"):
            if key in model_inputs:
                token_types = model_inputs[key]
                if token_types.shape[-1] != prompt_length:
                    raise RuntimeError(f"{key} does not align with prompt input IDs")
                extension = torch.zeros(
                    (*token_types.shape[:-1], response_ids.shape[1]),
                    dtype=token_types.dtype,
                    device=token_types.device,
                )
                model_inputs[key] = torch.cat((token_types, extension), dim=-1)
        model_inputs.pop("position_ids", None)
        position_ids = self._position_ids(model_inputs)
        if position_ids is not None:
            model_inputs["position_ids"] = position_ids

        outputs = self.model(**model_inputs, use_cache=False)
        response_logits = outputs.logits[:, prompt_length - 1 : prompt_length - 1 + response_ids.shape[1]]
        log_probs = torch.log_softmax(response_logits.float(), dim=-1)
        values, indices = torch.topk(log_probs, k=self.top_k, dim=-1)
        topk_mass = values.exp().sum(dim=-1)
        tail_log_prob = (1.0 - topk_mass).clamp_min(torch.finfo(torch.float32).tiny).log()
        probabilities = log_probs.exp()
        entropy = -(probabilities * log_probs).sum(dim=-1)

        response = TeacherScoreResponse(
            request_id=request.request_id,
            condition=request.condition,
            token_ids=request.response_token_ids,
            topk_token_ids=tuple(tuple(int(item) for item in row) for row in indices[0].cpu().tolist()),
            topk_log_probs=tuple(
                tuple(float(item) for item in row) for row in values[0].cpu().tolist()
            ),
            tail_log_prob=tuple(float(item) for item in tail_log_prob[0].cpu().tolist()),
            teacher_entropy=tuple(float(item) for item in entropy[0].cpu().tolist()),
        )
        ensure_exact_token_alignment(
            request.response_token_ids,
            response.token_ids,
            context="transformers teacher scoring",
        )
        return response

    def score_batch(
        self,
        requests: Sequence[TeacherScoreRequest],
    ) -> list[TeacherScoreResponse]:
        # Correctness-first implementation. Length/condition batching is added
        # after the single-request GPU probe establishes exact alignment.
        return [self._score_one(request) for request in requests]
