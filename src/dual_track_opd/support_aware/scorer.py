"""Teacher and student forced-scoring for frozen-policy diagnostics.

TeacherScorer
    Thin wrapper around the battle-tested ``TeacherClient`` from FC-OPD.
    Sends response token IDs and receives ``sampled_token_log_probs`` —
    the exact log-probability the teacher assigns to each response token.

StudentScorer
    Loads Qwen3-VL-4B-Instruct locally and runs a forced forward pass
    to compute per-token log-probabilities for a given response.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from PIL import Image

from dual_track_opd.fc_opd.dataset_signal_audit import hash_token_ids


# ---------------------------------------------------------------------------
# Teacher scorer — wraps the verified TeacherClient
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherScorerConfig:
    base_url: str = "http://127.0.0.1:18080"
    timeout_seconds: float = 120.0
    expected_tokenizer_hash: str | None = None


class TeacherScorer:
    """Teacher forced-scorer backed by the verified FC-OPD TeacherClient."""

    def __init__(self, config: TeacherScorerConfig | None = None):
        from dual_track_opd.fc_opd.teacher_client import TeacherClient

        self._cfg = config or TeacherScorerConfig()
        self._client = TeacherClient(
            self._cfg.base_url,
            expected_tokenizer_hash=self._cfg.expected_tokenizer_hash,
            timeout_seconds=self._cfg.timeout_seconds,
        )

    @property
    def tokenizer_hash(self) -> str:
        return self._client.metadata.tokenizer_hash

    @property
    def model_id(self) -> str:
        return self._client.metadata.model_id

    @property
    def metadata(self):
        return self._client.metadata

    def health(self) -> bool:
        return self._client.health()

    # -- scoring ---------------------------------------------------------------

    @dataclass(frozen=True)
    class ScoreResult:
        request_id: str
        sampled_token_log_probs: tuple[float, ...]
        mean_logp: float
        scored_token_ids: tuple[int, ...] = ()
        scored_token_hash: str = ""
        response_mask: tuple[bool, ...] = ()
        error: str | None = None

    def score(
        self,
        *,
        request_id: str,
        question: str,
        image_path: str,
        prompt_text: str,
        response_token_ids: Sequence[int],
        response_text: str = "",
    ) -> "TeacherScorer.ScoreResult":
        """Teacher-force score one response under condition=FULL."""
        from dual_track_opd.fc_opd.conditions import Condition, ConditionInputs, ImageInput
        from dual_track_opd.fc_opd.teacher_client import score_teacher_conditions

        condition_inputs = ConditionInputs(
            full_image=ImageInput(path=image_path),
            degraded_image=ImageInput(
                path=image_path,
                transform={"type": "gaussian_blur", "sigma": 2.0},
            ),
            free_caption="placeholder",
            task_evidence="placeholder",
        )
        prompt = (
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text},
                ],
            },
        )

        try:
            scores = score_teacher_conditions(
                response_token_ids=response_token_ids,
                question=question,
                condition_inputs=condition_inputs,
                conditions=[Condition.FULL],
                teacher_client=self._client,
                response_text=response_text,
                prompt=prompt,
                request_prefix=request_id,
            )
        except Exception as exc:
            return TeacherScorer.ScoreResult(
                request_id=request_id,
                sampled_token_log_probs=(),
                mean_logp=float("nan"),
                error=str(exc),
            )

        topk = scores.get(Condition.FULL)
        if topk is None or topk.sampled_log_probs is None:
            return TeacherScorer.ScoreResult(
                request_id=request_id,
                sampled_token_log_probs=(),
                mean_logp=float("nan"),
                error="teacher returned no sampled_log_probs",
            )

        sampled = tuple(float(v) for v in topk.sampled_log_probs.flatten().tolist())
        mean_logp = float(sum(sampled) / max(len(sampled), 1))
        scored_token_ids = tuple(int(token_id) for token_id in response_token_ids)

        return TeacherScorer.ScoreResult(
            request_id=request_id,
            sampled_token_log_probs=sampled,
            mean_logp=mean_logp,
            scored_token_ids=scored_token_ids,
            scored_token_hash=hash_token_ids(scored_token_ids),
            response_mask=(True,) * len(scored_token_ids),
        )

    def score_batch(
        self,
        *,
        request_ids: Sequence[str],
        questions: Sequence[str],
        image_paths: Sequence[str],
        prompt_texts: Sequence[str],
        response_token_ids_list: Sequence[Sequence[int]],
        response_texts: Sequence[str] | None = None,
    ) -> list["TeacherScorer.ScoreResult"]:
        """Teacher-force score multiple responses in a single HTTP batch call.

        All responses must share the same question/image (same prompt, different
        rollouts).  Uses ``score_teacher_conditions_multi_sample`` under the hood.
        """
        from dual_track_opd.fc_opd.conditions import Condition, ConditionInputs, ImageInput
        from dual_track_opd.fc_opd.teacher_client import score_teacher_conditions_multi_sample

        n = len(request_ids)
        if response_texts is None:
            response_texts = [""] * n
        lengths = {
            len(request_ids),
            len(questions),
            len(image_paths),
            len(prompt_texts),
            len(response_token_ids_list),
            len(response_texts),
        }
        if n == 0 or lengths != {n}:
            raise ValueError("teacher batch scorer inputs must have one non-empty shared length")

        # Build samples — all share the same condition_inputs (same prompt).
        # Pass the exact student prompt (image + text) so the teacher conditions
        # on the same history the student saw.  Omitting it makes the backend
        # fall back to render_teacher_prompt(), which prepends a "Question:\n"
        # prefix the student never saw and drops the step-by-step instruction —
        # different conditioning, different forced log-probs.
        samples: list[
            tuple[Sequence[int], str, ConditionInputs, Sequence[Mapping[str, Any]]]
        ] = []
        for i in range(n):
            condition_inputs = ConditionInputs(
                full_image=ImageInput(path=image_paths[i]),
                degraded_image=ImageInput(
                    path=image_paths[i],
                    transform={"type": "gaussian_blur", "sigma": 2.0},
                ),
                free_caption="placeholder",
                task_evidence="placeholder",
            )
            prompt = (
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_paths[i]},
                        {"type": "text", "text": prompt_texts[i]},
                    ],
                },
            )
            samples.append((
                tuple(int(t) for t in response_token_ids_list[i]),
                questions[i],
                condition_inputs,
                prompt,
            ))

        try:
            batch_results = score_teacher_conditions_multi_sample(
                samples=samples,
                conditions=[Condition.FULL],
                teacher_client=self._client,
                response_texts=list(response_texts),
                request_prefix=request_ids[0].rsplit(":", 1)[0] if request_ids else "batch",
            )
        except Exception as exc:
            return [
                TeacherScorer.ScoreResult(
                    request_id=rid,
                    sampled_token_log_probs=(),
                    mean_logp=float("nan"),
                    error=str(exc),
                )
                for rid in request_ids
            ]

        results: list[TeacherScorer.ScoreResult] = []
        if len(batch_results) != n:
            return [
                TeacherScorer.ScoreResult(
                    request_id=rid,
                    sampled_token_log_probs=(),
                    mean_logp=float("nan"),
                    error="teacher batch result count does not match request count",
                )
                for rid in request_ids
            ]
        for i, cond_map in enumerate(batch_results):
            topk = cond_map.get(Condition.FULL)
            if topk is None or topk.sampled_log_probs is None:
                results.append(TeacherScorer.ScoreResult(
                    request_id=request_ids[i],
                    sampled_token_log_probs=(),
                    mean_logp=float("nan"),
                    error="teacher returned no sampled_log_probs",
                ))
            else:
                sampled = tuple(float(v) for v in topk.sampled_log_probs.flatten().tolist())
                mean_logp = float(sum(sampled) / max(len(sampled), 1))
                scored_token_ids = tuple(
                    int(token_id) for token_id in response_token_ids_list[i]
                )
                results.append(TeacherScorer.ScoreResult(
                    request_id=request_ids[i],
                    sampled_token_log_probs=sampled,
                    mean_logp=mean_logp,
                    scored_token_ids=scored_token_ids,
                    scored_token_hash=hash_token_ids(scored_token_ids),
                    response_mask=(True,) * len(scored_token_ids),
                ))
        return results


# ---------------------------------------------------------------------------
# Student scorer (local model)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudentScorerConfig:
    model_path: str = ""
    device: str = "cuda:0"
    dtype: str = "bfloat16"


class StudentScorer:
    """Local Qwen3-VL-4B student forced-scorer.

    Loads the model once, then runs forced forward passes for each
    (prompt, response) pair.

    Usage::

        scorer = StudentScorer(StudentScorerConfig(model_path="..."))
        result = scorer.score(
            question="Find x.",
            image=<PIL.Image>,
            prompt_text="Find x.\\n\\nThink step by step...",
            response_token_ids=[...],
        )
        # result.mean_logp, result.sampled_token_log_probs
    """

    def __init__(self, config: StudentScorerConfig, *, model=None, processor=None):
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be provided together")
        if model is None:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            resolved = os.path.expandvars(config.model_path.removeprefix("hf:"))
            processor = AutoProcessor.from_pretrained(resolved)
            torch_dtype = (
                getattr(torch, config.dtype) if config.dtype != "float32" else torch.float32
            )
            model = AutoModelForImageTextToText.from_pretrained(
                resolved, torch_dtype=torch_dtype, device_map="auto"
            )
        self._processor = processor
        self._model = model
        self._model.eval()
        self._tokenizer = self._processor.tokenizer
        self._device_str = config.device  # stored for reference
        self._image_cls = Image

    @property
    def _model_device(self) -> torch.device:
        """The device where the model actually resides."""
        return next(self._model.parameters()).device

    @property
    def tokenizer(self):
        return self._tokenizer

    @dataclass(frozen=True)
    class ScoreResult:
        sampled_token_log_probs: tuple[float, ...]
        mean_logp: float
        token_count: int
        scored_token_ids: tuple[int, ...] = ()
        scored_token_hash: str = ""
        response_mask: tuple[bool, ...] = ()
        error: str | None = None

    def build_chat_prompt(
        self,
        question: str,
        image: Image.Image,
        prompt_text: str,
    ) -> str:
        """Build the chat-formatted prompt string (without generation prompt)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        return self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def score(
        self,
        *,
        question: str,
        image: Image.Image,
        prompt_text: str,
        response_text: str,
        response_token_ids: Sequence[int],
    ) -> "StudentScorer.ScoreResult":
        """Force-score the exact generated token IDs for one response.

        ``response_text`` is display/verifier metadata only.  It is
        intentionally never re-tokenized because decoding is not invertible.
        """

        return self.score_batch(
            questions=[question],
            images=[image],
            prompt_texts=[prompt_text],
            response_texts=[response_text],
            response_token_ids_list=[response_token_ids],
        )[0]

    def score_batch(
        self,
        *,
        questions: Sequence[str],
        images: Sequence[Image.Image],
        prompt_texts: Sequence[str],
        response_texts: Sequence[str],
        response_token_ids_list: Sequence[Sequence[int]],
    ) -> list["StudentScorer.ScoreResult"]:
        """Force-score exact IDs for responses sharing one prompt/image.

        All items must reference the same prompt (same question + image) so the
        chat template prefix is identical.  Items differ only in the response suffix.
        """
        import torch.nn.functional as F

        n = len(questions)
        if n == 0:
            return []
        lengths = {
            len(questions),
            len(images),
            len(prompt_texts),
            len(response_texts),
            len(response_token_ids_list),
        }
        if lengths != {n}:
            raise ValueError("student batch scorer inputs must have the same length")
        if any(question != questions[0] for question in questions):
            raise ValueError("student batch scorer requires one shared question")
        if any(prompt_text != prompt_texts[0] for prompt_text in prompt_texts):
            raise ValueError("student batch scorer requires one shared prompt")

        exact_ids = [tuple(int(token_id) for token_id in ids) for ids in response_token_ids_list]
        valid_indices = [index for index, ids in enumerate(exact_ids) if ids]
        results: list[StudentScorer.ScoreResult] = [
            StudentScorer.ScoreResult(
                sampled_token_log_probs=(),
                mean_logp=float("nan"),
                token_count=0,
                error="empty response",
            )
            for _ in range(n)
        ]
        if not valid_indices:
            return results

        # Build chat prompt for the shared prefix (use first item's inputs)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[0]},
                    {"type": "text", "text": prompt_texts[0]},
                ],
            }
        ]
        chat_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process the prompt/image only.  The response suffix is appended from
        # the exact IDs returned by generation; display text is never an input.
        batch_size = len(valid_indices)
        prompt_enc = self._processor(
            text=[chat_text] * batch_size,
            images=[images[index] for index in valid_indices],
            return_tensors="pt",
            padding=True,
        )
        device = self._model_device
        prompt_enc = {key: value.to(device) for key, value in prompt_enc.items()}
        prompt_ids = prompt_enc["input_ids"]
        prompt_width = int(prompt_ids.shape[1])

        valid_ids = [exact_ids[index] for index in valid_indices]
        response_lens = [len(ids) for ids in valid_ids]
        max_response_len = max(response_lens)
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        padded_responses = torch.full(
            (batch_size, max_response_len),
            int(pad_id),
            dtype=prompt_ids.dtype,
            device=device,
        )
        for row, ids in enumerate(valid_ids):
            padded_responses[row, : len(ids)] = torch.tensor(
                ids,
                dtype=prompt_ids.dtype,
                device=device,
            )
        response_mask = (
            torch.arange(max_response_len, device=device).unsqueeze(0)
            < torch.tensor(response_lens, device=device).unsqueeze(1)
        )

        model_inputs: dict[str, torch.Tensor] = {
            key: value
            for key, value in prompt_enc.items()
            if key not in {"input_ids", "attention_mask", "position_ids"}
        }
        model_inputs["input_ids"] = torch.cat([prompt_ids, padded_responses], dim=1)
        model_inputs["attention_mask"] = torch.cat(
            [
                prompt_enc["attention_mask"],
                response_mask.to(prompt_enc["attention_mask"].dtype),
            ],
            dim=1,
        )
        for key in ("token_type_ids", "mm_token_type_ids"):
            if key in model_inputs:
                extension = torch.zeros(
                    batch_size,
                    max_response_len,
                    dtype=model_inputs[key].dtype,
                    device=device,
                )
                model_inputs[key] = torch.cat([model_inputs[key], extension], dim=-1)

        with torch.no_grad():
            logits = self._model(**model_inputs).logits

        for row, original_index in enumerate(valid_indices):
            response_ids = valid_ids[row]
            response_len = len(response_ids)
            response_logits = logits[
                row,
                prompt_width - 1 : prompt_width - 1 + response_len,
                :,
            ]
            if int(response_logits.shape[0]) != response_len:
                raise RuntimeError("student response logits do not align with exact response IDs")
            response_ids_tensor = torch.tensor(
                response_ids,
                dtype=torch.long,
                device=device,
            )

            log_probs = F.log_softmax(response_logits.float(), dim=-1)
            sampled = log_probs[
                torch.arange(response_len, device=device),
                response_ids_tensor,
            ]

            sampled_tuple = tuple(float(v.item()) for v in sampled)
            mean_logp = float(sum(sampled_tuple) / len(sampled_tuple))

            results[original_index] = StudentScorer.ScoreResult(
                sampled_token_log_probs=sampled_tuple,
                mean_logp=mean_logp,
                token_count=len(sampled_tuple),
                scored_token_ids=response_ids,
                scored_token_hash=hash_token_ids(response_ids),
                response_mask=(True,) * response_len,
            )

        return results

    def encode_response(self, response_text: str) -> tuple[int, ...]:
        """Tokenize a response string into token IDs."""
        return tuple(int(t) for t in self._tokenizer.encode(response_text))

    def tokenizer_hash(self) -> str:
        """Compute a deterministic hash of the tokenizer (matches teacher fingerprint)."""
        from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint

        return tokenizer_fingerprint(self._tokenizer)
