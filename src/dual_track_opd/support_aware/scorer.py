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

import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Teacher scorer — wraps the verified TeacherClient
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherScorerConfig:
    base_url: str = "http://127.0.0.1:18080"
    timeout_seconds: float = 120.0


class TeacherScorer:
    """Teacher forced-scorer backed by the verified FC-OPD TeacherClient."""

    def __init__(self, config: TeacherScorerConfig | None = None):
        from dual_track_opd.fc_opd.teacher_client import TeacherClient

        self._cfg = config or TeacherScorerConfig()
        self._client = TeacherClient(
            self._cfg.base_url,
            timeout_seconds=self._cfg.timeout_seconds,
        )

    @property
    def tokenizer_hash(self) -> str:
        return self._client.metadata.tokenizer_hash

    @property
    def model_id(self) -> str:
        return self._client.metadata.model_id

    def health(self) -> bool:
        return self._client.health()

    # -- scoring ---------------------------------------------------------------

    @dataclass(frozen=True)
    class ScoreResult:
        request_id: str
        sampled_token_log_probs: tuple[float, ...]
        mean_logp: float
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

        return TeacherScorer.ScoreResult(
            request_id=request_id,
            sampled_token_log_probs=sampled,
            mean_logp=mean_logp,
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

        # Build samples — all share the same condition_inputs (same prompt)
        samples: list[tuple[Sequence[int], str, ConditionInputs]] = []
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
            samples.append((
                tuple(int(t) for t in response_token_ids_list[i]),
                questions[i],
                condition_inputs,
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
                results.append(TeacherScorer.ScoreResult(
                    request_id=request_ids[i],
                    sampled_token_log_probs=sampled,
                    mean_logp=mean_logp,
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

    def __init__(self, config: StudentScorerConfig):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        resolved = os.path.expandvars(config.model_path.removeprefix("hf:"))
        self._processor = AutoProcessor.from_pretrained(resolved)
        torch_dtype = (
            getattr(torch, config.dtype) if config.dtype != "float32" else torch.float32
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            resolved, torch_dtype=torch_dtype, device_map="auto"
        )
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
        """Student-force score one response.

        Concatenates prompt + response, runs one forward pass, and extracts
        the log-probability of each response token.

        Args:
            question: The geometry problem text (used in chat template).
            image: The diagram as a PIL Image.
            prompt_text: The full prompt text shown to the student.
            response_text: The decoded response (used for tokenization).
            response_token_ids: Pre-computed token IDs of the response.
        """
        import torch.nn.functional as F

        response_ids = tuple(int(t) for t in response_token_ids)
        if not response_ids:
            return StudentScorer.ScoreResult(
                sampled_token_log_probs=(),
                mean_logp=float("nan"),
                token_count=0,
                error="empty response",
            )

        # Build the chat prompt (without generation prompt so we can append response)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        chat_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize prompt + response together for exact alignment
        # We tokenize the full text, then identify where the response starts
        full_text = chat_text + response_text
        full_enc = self._processor(text=[full_text], images=[image], return_tensors="pt")
        full_ids = full_enc["input_ids"]

        # Tokenize just the prompt (with generation prompt) to find boundary
        prompt_enc = self._processor(text=[chat_text], images=[image], return_tensors="pt")
        prompt_len = prompt_enc["input_ids"].shape[1]

        # Move to device
        model_inputs: dict[str, torch.Tensor] = {}
        for key, value in full_enc.items():
            if key in {"input_ids", "attention_mask"}:
                continue
            model_inputs[key] = value.to(self._model_device)
        model_inputs["input_ids"] = full_ids.to(self._model_device)
        model_inputs["attention_mask"] = torch.ones_like(full_ids, device=self._model_device)

        with torch.no_grad():
            logits = self._model(**model_inputs).logits

        # Extract log-probs at response positions
        # logits[t] predicts token t+1, so for response positions [prompt_len, prompt_len+T-1],
        # we use logits[prompt_len-1 : prompt_len+T-1]
        T = full_ids.shape[1] - prompt_len
        if T <= 0:
            return StudentScorer.ScoreResult(
                sampled_token_log_probs=(),
                mean_logp=float("nan"),
                token_count=0,
                error="response tokenization misalignment",
            )

        response_logits = logits[0, prompt_len - 1 : prompt_len + T - 1, :]
        response_ids_tensor = full_ids[0, prompt_len:].to(self._model_device)

        log_probs = F.log_softmax(response_logits.float(), dim=-1)
        sampled = log_probs[range(len(response_ids_tensor)), response_ids_tensor]

        sampled_tuple = tuple(float(v.item()) for v in sampled)
        mean_logp = float(sum(sampled_tuple) / len(sampled_tuple))

        return StudentScorer.ScoreResult(
            sampled_token_log_probs=sampled_tuple,
            mean_logp=mean_logp,
            token_count=len(sampled_tuple),
        )

    def score_batch(
        self,
        *,
        questions: Sequence[str],
        images: Sequence[Image.Image],
        prompt_texts: Sequence[str],
        response_texts: Sequence[str],
        response_token_ids_list: Sequence[Sequence[int]],
    ) -> list["StudentScorer.ScoreResult"]:
        """Student-force score multiple responses in a single batched forward pass.

        All items must reference the same prompt (same question + image) so the
        chat template prefix is identical.  Items differ only in the response suffix.
        """
        import torch.nn.functional as F

        n = len(questions)
        if n == 0:
            return []

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

        # Tokenize each (prompt + response) pair independently, then pad
        prompt_enc = self._processor(text=[chat_text], images=[images[0]], return_tensors="pt")
        prompt_len = prompt_enc["input_ids"].shape[1]

        # Build padded batch
        all_input_ids: list[torch.Tensor] = []
        all_response_lens: list[int] = []
        for i in range(n):
            resp_ids = tuple(int(t) for t in response_token_ids_list[i])
            if not resp_ids:
                all_input_ids.append(prompt_enc["input_ids"][0])
                all_response_lens.append(0)
                continue
            full_text = chat_text + response_texts[i]
            full_enc = self._processor(
                text=[full_text], images=[images[0]], return_tensors="pt"
            )
            all_input_ids.append(full_enc["input_ids"][0])
            all_response_lens.append(full_enc["input_ids"].shape[1] - prompt_len)

        # Pad to max length
        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_token_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        padded_ids = torch.full((n, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((n, max_len), dtype=torch.long)
        for i, ids in enumerate(all_input_ids):
            L = ids.shape[0]
            padded_ids[i, :L] = ids
            attention_mask[i, :L] = 1

        # Move to device
        device = self._model_device
        model_inputs: dict[str, torch.Tensor] = {
            "input_ids": padded_ids.to(device),
            "attention_mask": attention_mask.to(device),
        }
        # Use prompt_enc for image-related tensors (shared across all items)
        for key, value in prompt_enc.items():
            if key in {"input_ids", "attention_mask"}:
                continue
            if isinstance(value, torch.Tensor):
                model_inputs[key] = value.to(device)

        with torch.no_grad():
            logits = self._model(**model_inputs).logits  # (n, max_len, vocab)

        # Extract per-item log-probs
        results: list[StudentScorer.ScoreResult] = []
        for i in range(n):
            T = all_response_lens[i]
            if T <= 0:
                results.append(StudentScorer.ScoreResult(
                    sampled_token_log_probs=(),
                    mean_logp=float("nan"),
                    token_count=0,
                    error="empty response" if T == 0 else "response tokenization misalignment",
                ))
                continue

            # Response tokens are at positions [prompt_len, prompt_len+T)
            # logits[t] predicts token t+1, so we use logits[i, prompt_len-1 : prompt_len+T-1]
            response_logits = logits[i, prompt_len - 1 : prompt_len + T - 1, :]
            response_ids = padded_ids[i, prompt_len : prompt_len + T]

            log_probs = F.log_softmax(response_logits.float(), dim=-1)
            sampled = log_probs[range(len(response_ids)), response_ids]

            sampled_tuple = tuple(float(v.item()) for v in sampled)
            mean_logp = float(sum(sampled_tuple) / len(sampled_tuple))

            results.append(StudentScorer.ScoreResult(
                sampled_token_log_probs=sampled_tuple,
                mean_logp=mean_logp,
                token_count=len(sampled_tuple),
            ))

        return results

    def encode_response(self, response_text: str) -> tuple[int, ...]:
        """Tokenize a response string into token IDs."""
        return tuple(int(t) for t in self._tokenizer.encode(response_text))

    def tokenizer_hash(self) -> str:
        """Compute a deterministic hash of the tokenizer (matches teacher fingerprint)."""
        from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint

        return tokenizer_fingerprint(self._tokenizer)
