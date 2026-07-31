"""Optional Transformers backend for exact-token Qwen3-VL teacher scoring."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Sequence

import torch

from .conditions import Condition
from .teacher_protocol import (
    TeacherMetadata,
    TeacherScoreRequest,
    TeacherScoreResponse,
    tokenizer_fingerprint,
)
from .teacher_prompts import render_teacher_prompt
from .teacher_scorer import TeacherScorer


def response_prediction_logits(full_logits: torch.Tensor, num_response_tokens: int) -> torch.Tensor:
    """Return causal-LM logits that predict each appended response token.

    The final ``T`` input positions contain the response tokens themselves.
    Logits at those positions predict the *following* tokens, so scoring the
    response requires the preceding ``T`` positions: ``[-T-1:-1]``.
    """

    if full_logits.ndim != 3:
        raise ValueError("full_logits must have shape [batch, sequence, vocab]")
    if num_response_tokens < 1:
        raise ValueError("num_response_tokens must be positive")
    if full_logits.shape[1] <= num_response_tokens:
        raise ValueError("full_logits must include at least one prompt position before the response")
    return full_logits[:, -num_response_tokens - 1 : -1, :]


def inject_image_placeholders(
    messages: Sequence[dict[str, Any]],
    image_paths: Sequence[str],
) -> list[dict[str, Any]]:
    """Mirror verl RLHFDataset's ``<image>`` → structured-content conversion."""

    output: list[dict[str, Any]] = []
    image_offset = 0
    for original in messages:
        message = dict(original)
        content = message.get("content")
        if isinstance(content, str):
            content_list: list[dict[str, Any]] = []
            for segment in filter(None, re.split("(<image>)", content)):
                if segment == "<image>":
                    if image_offset >= len(image_paths):
                        raise ValueError("prompt contains more <image> placeholders than supplied images")
                    content_list.append({"type": "image", "image": str(image_paths[image_offset])})
                    image_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list
        elif isinstance(content, list):
            normalized_content = []
            for item in content:
                normalized = dict(item)
                if normalized.get("type") == "image":
                    if image_offset >= len(image_paths):
                        raise ValueError("structured prompt contains more images than supplied inputs")
                    normalized["image"] = str(image_paths[image_offset])
                    normalized.pop("bytes", None)
                    image_offset += 1
                normalized_content.append(normalized)
            message["content"] = normalized_content
        output.append(message)
    if image_offset != len(image_paths):
        raise ValueError(
            f"prompt consumed {image_offset} image placeholders but {len(image_paths)} images were supplied"
        )
    return output


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
        """Warn on response text round-trip drift without changing token IDs."""
        if request.response_text is None:
            return
        encoded = self.tokenizer.encode(request.response_text, add_special_tokens=False)
        if tuple(request.response_token_ids) == tuple(encoded):
            return
        import logging
        _logger = logging.getLogger(__name__)
        compared_len = min(len(request.response_token_ids), len(encoded))
        first_mismatch = next(
            (i for i, (a, b) in enumerate(zip(request.response_token_ids, encoded)) if a != b),
            compared_len,
        )
        student_token = (
            request.response_token_ids[first_mismatch]
            if first_mismatch < len(request.response_token_ids)
            else -1
        )
        teacher_token = encoded[first_mismatch] if first_mismatch < len(encoded) else -1
        _logger.warning(
            "Teacher tokenizer round-trip mismatch at %d/%d compared positions "
            "(student_len=%d teacher_len=%d, first mismatch at pos %d: student=%d teacher=%d). "
            "Keeping the original student response_token_ids for forced scoring.",
            sum(1 for a, b in zip(request.response_token_ids, encoded) if a != b),
            compared_len,
            len(request.response_token_ids),
            len(encoded),
            first_mismatch,
            student_token,
            teacher_token,
        )

    def _prepare_prompt(self, request: TeacherScoreRequest) -> dict[str, torch.Tensor]:
        rendered = render_teacher_prompt(
            request.condition,
            request.question,
            request.condition_inputs,
        )
        # For image-preserving GKD, use the exact rollout messages.  Rebuilding
        # them from ``question`` used to add a "Question:\n" prefix that the
        # student never saw, so teacher and student conditioned on different
        # histories.  Other FC-OPD text conditions still use their deliberate
        # condition-specific rendering.
        messages = (
            inject_image_placeholders(list(request.prompt), rendered.image_paths)
            if request.prompt is not None and request.condition in {Condition.FULL, Condition.DEGRADED}
            else list(rendered.messages)
        )
        template_kwargs = dict(request.chat_template_kwargs or {})
        add_generation_prompt = bool(template_kwargs.pop("add_generation_prompt", True))
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
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
        # Qwen3-VL with transformers ≥ 5.x requires mm_token_type_ids for
        # get_rope_index().  The processor populates it; forward it if present.
        for extra in ("mm_token_type_ids", "pixel_values", "pixel_values_videos"):
            if extra in model_inputs:
                kwargs[extra] = model_inputs[extra]
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
        _n_resp = response_ids.shape[1]
        response_logits = response_prediction_logits(outputs.logits, _n_resp)
        log_probs = torch.log_softmax(response_logits.float(), dim=-1)
        values, indices = torch.topk(log_probs, k=self.top_k, dim=-1)
        topk_mass = values.exp().sum(dim=-1)
        tail_log_prob = (1.0 - topk_mass).clamp_min(torch.finfo(torch.float32).tiny).log()
        probabilities = log_probs.exp()
        entropy = -(probabilities * log_probs).sum(dim=-1)
        # ── Exact log P_T(y_t | condition) via direct indexing ──
        _t_idx = torch.arange(_n_resp, device=response_logits.device)
        _r_ids = torch.tensor(request.response_token_ids, dtype=torch.long, device=response_logits.device)
        if log_probs.ndim == 3:
            sampled_lp = log_probs[0, _t_idx, _r_ids]  # [T]
        else:
            sampled_lp = log_probs[_t_idx, _r_ids]      # [T]
        sampled_lp = sampled_lp.unsqueeze(0)              # [1, T]

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
            sampled_token_log_probs=tuple(float(item) for item in sampled_lp[0].cpu().tolist()),
        )
        return response

    @torch.inference_mode()
    def diagnose_generation_alignment(
        self,
        request: TeacherScoreRequest,
        *,
        max_new_tokens: int = 4,
    ) -> dict[str, Any]:
        """Compare native generation scores with one-pass forced scoring."""

        prompt_inputs = self._prepare_prompt(request)
        generated = self.model.generate(
            **prompt_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        generated_ids = tuple(int(item) for item in generated.sequences[0, prompt_length:].tolist())
        if not generated_ids or not generated.scores:
            raise RuntimeError("teacher native generate returned no diagnostic tokens/scores")

        raw_logits = getattr(generated, "logits", None)
        if not raw_logits:
            raise RuntimeError("teacher generate did not return raw logits; output_logits is required")
        native_log_probs = torch.stack(
            [torch.log_softmax(logits[0].float(), dim=-1) for logits in raw_logits[: len(generated_ids)]],
            dim=0,
        )
        processed_log_probs = torch.stack(
            [torch.log_softmax(score[0].float(), dim=-1) for score in generated.scores[: len(generated_ids)]],
            dim=0,
        )
        k = min(self.top_k, int(native_log_probs.shape[-1]))
        native_values, native_indices = torch.topk(native_log_probs, k=k, dim=-1)
        forced_request = replace(
            request,
            response_token_ids=generated_ids,
            response_text=self.tokenizer.decode(list(generated_ids), skip_special_tokens=False),
        )
        forced = self._score_one(forced_request)
        forced_indices = torch.tensor(forced.topk_token_ids, device=native_indices.device)
        forced_values = torch.tensor(forced.topk_log_probs, device=native_values.device)

        # Compare via per-position token overlap rather than strict equality.
        # KV-cache generation vs full-forward scoring routinely swaps a few
        # tail tokens (≪1 %) because of floating-point noise; requiring 256/256
        # exact match is too strict for a diagnostic whose real goal is to
        # catch protocol-level mismatches (wrong position IDs, image encoding,
        # or tokenizer fingerprint).
        _overlap = 0
        _max_diff = 0.0
        _n_tokens, _k = native_indices.shape
        for _t in range(_n_tokens):
            _n_lookup = {
                int(_tid): float(_lp)
                for _tid, _lp in zip(native_indices[_t].tolist(), native_values[_t].tolist())
            }
            for _i in range(_k):
                _tid = int(forced_indices[_t, _i].item())
                if _tid in _n_lookup:
                    _overlap += 1
                    _diff = abs(float(forced_values[_t, _i].item()) - _n_lookup[_tid])
                    if _diff > _max_diff:
                        _max_diff = _diff
        _overlap_ratio = _overlap / (_n_tokens * _k)
        ids_match = _overlap_ratio >= 0.95
        max_logprob_diff = _max_diff if _overlap > 0 else None
        head_k = min(10, _k)
        head_overlap = 0
        for _t in range(_n_tokens):
            head_overlap += len(set(native_indices[_t, :head_k].tolist()) & set(forced_indices[_t, :head_k].tolist()))
        head_overlap_ratio = head_overlap / (_n_tokens * head_k)
        top1_match = bool(torch.equal(native_indices[:, 0], forced_indices[:, 0]))
        top1_logprob_diff = float((native_values[:, 0] - forced_values[:, 0]).abs().max().cpu().item())

        eos_raw = getattr(self.tokenizer, "eos_token_id", None)
        eos_ids = [] if eos_raw is None else ([int(eos_raw)] if isinstance(eos_raw, int) else [int(x) for x in eos_raw])
        first_eos_prob = float(native_log_probs[0, eos_ids].exp().sum().cpu().item()) if eos_ids else 0.0
        processed_first_eos_prob = (
            float(processed_log_probs[0, eos_ids].exp().sum().cpu().item()) if eos_ids else 0.0
        )
        return {
            "request_id": request.request_id,
            "condition": request.condition.value,
            "chat_template_kwargs": request.chat_template_kwargs,
            "prompt_length": prompt_length,
            "prompt_input_ids_tail": prompt_inputs["input_ids"][0, -32:].detach().cpu().tolist(),
            "image_grid_thw": (
                None
                if prompt_inputs.get("image_grid_thw") is None
                else prompt_inputs["image_grid_thw"].detach().cpu().tolist()
            ),
            "generated_token_ids": list(generated_ids),
            "generated_text": self.tokenizer.decode(list(generated_ids), skip_special_tokens=False),
            "eos_token_ids": eos_ids,
            "native_first_eos_probability": first_eos_prob,
            "processed_first_eos_probability": processed_first_eos_prob,
            "native_topk_token_ids": native_indices.detach().cpu().tolist(),
            "native_topk_log_probs": native_values.detach().cpu().tolist(),
            "forced_topk_token_ids": [list(row) for row in forced.topk_token_ids],
            "forced_topk_log_probs": [list(row) for row in forced.topk_log_probs],
            "native_forced_topk_ids_match": ids_match,
            "native_forced_max_logprob_diff": max_logprob_diff,
            "native_forced_top10_overlap_ratio": head_overlap_ratio,
            "native_forced_top1_match": top1_match,
            "native_forced_top1_logprob_diff": top1_logprob_diff,
        }
    @torch.inference_mode()
    def _score_batched(
        self,
        requests: list[TeacherScoreRequest],
    ) -> list[TeacherScoreResponse]:
        """Score independently until a verified multimodal batch path exists.

        The previous processor-native batch path replicated the first request's
        prompt/image across the whole condition group and sliced variable-length
        responses from padded tails. Both errors corrupt token-level VA, so the
        faithful path favors correctness over throughput.
        """
        return [self._score_one(request) for request in requests]

    def score_batch(
        self,
        requests: Sequence[TeacherScoreRequest],
    ) -> list[TeacherScoreResponse]:
        # Group by condition → 6 batched forwards instead of 96.
        from collections import defaultdict as _dd
        import sys as _sys

        groups: dict[Any, list[TeacherScoreRequest]] = _dd(list)
        for req in requests:
            groups[req.condition].append(req)

        print(f"[teacher] score_batch: {len(requests)} requests → {len(groups)} groups "
              f"{ {str(c): len(g) for c, g in groups.items()} }", file=_sys.stderr, flush=True)

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
        if transform and transform.get("type") == "precomputed_degraded":
            return
        if transform and transform.get("type") in {"lowres_nearest", "lowres_bilinear_nearest"}:
            scale = float(transform.get("scale", 0.1))
            downsample = Image.BILINEAR if transform.get("type") == "lowres_bilinear_nearest" else Image.NEAREST
            for i, img in enumerate(images):
                w, h = img.size
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                images[i] = img.resize(new_size, downsample).resize((w, h), Image.NEAREST)
        elif transform and transform.get("type") == "gaussian_blur":
            from PIL import ImageFilter

            sigma = float(transform.get("sigma", 2.0))
            for i, img in enumerate(images):
                images[i] = img.filter(ImageFilter.GaussianBlur(radius=sigma))
