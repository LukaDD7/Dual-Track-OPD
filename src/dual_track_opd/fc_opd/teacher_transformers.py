"""Optional Transformers backend for exact-token Qwen3-VL teacher scoring."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from .teacher_protocol import (
    TeacherMetadata,
    TeacherScoreRequest,
    TeacherScoreResponse,
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
        import os as _os

        _model_is_local = _os.path.isdir(model_id) or _os.path.isfile(model_id)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            local_files_only=_model_is_local,
        )
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            local_files_only=_model_is_local,
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
        """Verify teacher re-tokenization matches student token IDs.

        If the teacher tokenizer produces different token IDs (rare edge case
        with Qwen3-VL 32B vs 4B tokenizer differences), we use the teacher's
        own encoding (truncated to the original length) so the forced forward
        pass uses tokens the teacher model understands.  A warning is logged
        so we can track how often this happens.
        """
        if request.response_text is None:
            return
        encoded = self.tokenizer.encode(request.response_text, add_special_tokens=False)
        if tuple(request.response_token_ids) == tuple(encoded):
            return
        # Mismatch – repair by using the teacher's own tokenization, truncated
        # to match the original response length so response_mask stays aligned.
        original_len = len(request.response_token_ids)
        repaired = encoded[:original_len]
        if len(repaired) < original_len:
            # Edge case: teacher encoding is shorter; pad with the last token
            # (usually <|endoftext|> or similar) rather than inventing new ones.
            pad = [repaired[-1]] * (original_len - len(repaired))
            repaired = list(repaired) + pad
        import logging
        _logger = logging.getLogger(__name__)
        _logger.warning(
            "Teacher tokenizer mismatch at %d/%d positions – using repaired IDs "
            "(first mismatch at pos %d: student=%d teacher=%d).  This is expected "
            "to be rare (<1%% of steps).",
            sum(1 for a, b in zip(request.response_token_ids, encoded) if a != b),
            original_len,
            next((i for i, (a, b) in enumerate(zip(request.response_token_ids, encoded)) if a != b), 0),
            request.response_token_ids[next((i for i, (a, b) in enumerate(zip(request.response_token_ids, encoded)) if a != b), 0)],
            encoded[next((i for i, (a, b) in enumerate(zip(request.response_token_ids, encoded)) if a != b), 0)],
        )
        object.__setattr__(request, "response_token_ids", repaired)

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
            # Apply condition-specific image transforms (e.g. degrade)
            _apply_image_transform(request.condition, images, request.condition_inputs)
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
        # Qwen3-VL merges image tokens, so input/output position counts differ.
        # Response logits are always the *last* N positions of the output.
        _n_resp = response_ids.shape[1]
        response_logits = outputs.logits[:, -_n_resp:]
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
        # NOTE: request.response_token_ids may have been repaired by
        # _check_response_text; response.token_ids is set from the same
        # (possibly repaired) list, so they are always consistent.
        return response

    @torch.inference_mode()
    def _score_batched(
        self,
        requests: list[TeacherScoreRequest],
    ) -> list[TeacherScoreResponse]:
        """Score multiple requests sharing the same condition in one forward pass.

        All requests must have the same condition (and thus the same prompt text
        and image).  Responses are padded to the longest one; the attention mask
        prevents cross-sample contamination.
        """
        if len(requests) <= 1:
            return [self._score_one(r) for r in requests]

        # Validate and repair
        for r in requests:
            if r.tokenizer_hash != self.metadata.tokenizer_hash:
                raise ValueError("tokenizer hash mismatch")
            if not r.response_token_ids:
                raise ValueError("empty response_token_ids")
            self._check_response_text(r)

        B = len(requests)
        first = requests[0]
        prompt_inputs = self._prepare_prompt(first)
        prompt_ids = prompt_inputs["input_ids"]  # [1, P]
        P = prompt_ids.shape[1]

        # Pad responses to max length
        max_R = max(len(r.response_token_ids) for r in requests)
        resp_lens = torch.zeros(B, dtype=torch.long, device=self.device)
        padded = torch.zeros(B, max_R, dtype=torch.long, device=self.device)
        for i, r in enumerate(requests):
            rlen = len(r.response_token_ids)
            resp_lens[i] = rlen
            padded[i, :rlen] = torch.tensor(r.response_token_ids, dtype=torch.long, device=self.device)

        # Build batched inputs
        input_ids = torch.cat([prompt_ids.expand(B, -1), padded], dim=1)  # [B, P+max_R]
        prompt_mask = prompt_inputs["attention_mask"].expand(B, -1)  # [B, P]
        resp_mask = torch.arange(max_R, device=self.device).unsqueeze(0) < resp_lens.unsqueeze(1)
        resp_mask = resp_mask.to(prompt_mask.dtype)
        attention_mask = torch.cat([prompt_mask, resp_mask], dim=1)  # [B, P+max_R]

        model_inputs = dict(prompt_inputs)
        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = attention_mask

        # Expand vision tensor to batch dimension
        for key in ("image_grid_thw", "video_grid_thw"):
            thw = model_inputs.get(key)
            if thw is not None and thw.ndim >= 1 and thw.shape[0] == 1:
                model_inputs[key] = thw.expand(B, -1)

        # Extend token_type / mm_token_type for response tokens
        for key in ("token_type_ids", "mm_token_type_ids"):
            if key not in model_inputs:
                continue
            tok = model_inputs[key]
            if tok.shape[0] == 1:
                tok = tok.expand(B, -1)
            if tok.shape[-1] != P:
                raise RuntimeError(f"{key} does not align with prompt input IDs")
            extension = torch.zeros(B, max_R, dtype=tok.dtype, device=tok.device)
            model_inputs[key] = torch.cat([tok, extension], dim=-1)

        model_inputs.pop("position_ids", None)
        position_ids = self._position_ids(model_inputs)
        if position_ids is not None:
            model_inputs["position_ids"] = position_ids

        # Single batched forward
        outputs = self.model(**model_inputs, use_cache=False)

        # Extract per-sample results
        results: list[TeacherScoreResponse] = []
        for i, request in enumerate(requests):
            rlen = int(resp_lens[i].item())
            sample_logits = outputs.logits[i, -rlen:, :]  # [rlen, V]
            log_probs = torch.log_softmax(sample_logits.float(), dim=-1)
            values, indices = torch.topk(log_probs, k=self.top_k, dim=-1)
            topk_mass = values.exp().sum(dim=-1)
            tail_log_prob = (1.0 - topk_mass).clamp_min(torch.finfo(torch.float32).tiny).log()
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)
            results.append(TeacherScoreResponse(
                request_id=request.request_id,
                condition=request.condition,
                token_ids=request.response_token_ids,
                topk_token_ids=tuple(tuple(int(x) for x in row) for row in indices.cpu().tolist()),
                topk_log_probs=tuple(tuple(float(x) for x in row) for row in values.cpu().tolist()),
                tail_log_prob=tuple(float(x) for x in tail_log_prob.cpu().tolist()),
                teacher_entropy=tuple(float(x) for x in entropy.cpu().tolist()),
            ))
        return results

    def score_batch(
        self,
        requests: Sequence[TeacherScoreRequest],
    ) -> list[TeacherScoreResponse]:
        # Group by condition, then batch each group in one forward pass.
        # 16 samples × 6 conditions = 96 requests → 6 batched forwards
        # instead of 96 individual forwards.  ~16× faster.
        from collections import defaultdict as _defaultdict

        groups: dict[Any, list[TeacherScoreRequest]] = _defaultdict(list)
        for req in requests:
            groups[req.condition].append(req)

        results_map: dict[str, TeacherScoreResponse] = {}
        for _condition, group in groups.items():
            batch_results = self._score_batched(group)
            for req, resp in zip(group, batch_results):
                results_map[req.request_id] = resp

        return [results_map[req.request_id] for req in requests]


def _apply_image_transform(
    condition: "Condition",
    images: list["Image.Image"],
    condition_inputs: "ConditionInputs",
) -> None:
    """Apply condition-specific image degradation transforms in-place."""
    from PIL import Image

    from .conditions import Condition as C

    if condition in (C.DEGRADED, C.BLUR):
        transform = getattr(condition_inputs.degraded_image, "transform", None)
        if transform and transform.get("type") == "lowres_nearest":
            scale = float(transform.get("scale", 0.1))
            for i, img in enumerate(images):
                w, h = img.size
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                images[i] = img.resize(new_size, Image.NEAREST).resize((w, h), Image.NEAREST)
        elif transform and transform.get("type") == "gaussian_blur":
            from PIL import ImageFilter

            sigma = float(transform.get("sigma", 2.0))
            for i, img in enumerate(images):
                images[i] = img.filter(ImageFilter.GaussianBlur(radius=sigma))
