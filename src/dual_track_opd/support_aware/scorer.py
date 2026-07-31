"""Teacher and student forced-scoring for frozen-policy diagnostics.

TeacherScorer
    Thin HTTP client for the standalone Qwen3-VL-32B teacher service.
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
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Teacher scorer (HTTP)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeacherScorerConfig:
    base_url: str = "http://127.0.0.1:18080"
    timeout_seconds: float = 120.0


class TeacherServiceError(RuntimeError):
    pass


class TeacherScorer:
    """HTTP client for teacher forced-scoring."""

    def __init__(self, config: TeacherScorerConfig | None = None):
        self._cfg = config or TeacherScorerConfig()
        self._metadata: dict[str, Any] | None = None

    # -- connection ------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            self._metadata = self._get_json("/metadata")
        return self._metadata

    @property
    def tokenizer_hash(self) -> str:
        return str(self.metadata["tokenizer_hash"])

    @property
    def model_id(self) -> str:
        return str(self.metadata["model_id"])

    def health(self) -> bool:
        try:
            resp = self._get_json("/health")
            return isinstance(resp, dict) and resp.get("status") == "ok"
        except TeacherServiceError:
            return False

    def _get_json(self, path: str) -> Any:
        return self._request("GET", path)

    def _request(
        self,
        method: str,
        path: str,
        payload: object | None = None,
    ) -> Any:
        data: bytes | None = None
        headers: dict[str, str] = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        url = f"{self._cfg.base_url.rstrip('/')}{path}"
        req = Request(url, method=method, data=data, headers=headers)
        try:
            with urlopen(req, timeout=self._cfg.timeout_seconds) as resp:
                return json.loads(resp.read())
        except HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise TeacherServiceError(
                f"teacher HTTP {exc.code}: {body[:500]}"
            ) from exc
        except URLError as exc:
            raise TeacherServiceError(
                f"teacher unreachable at {self._cfg.base_url}: {exc.reason}"
            ) from exc

    # -- scoring ---------------------------------------------------------------

    @dataclass(frozen=True)
    class ScoreResult:
        """Result of a single teacher forced-scoring request."""

        request_id: str
        sampled_token_log_probs: tuple[float, ...]
        mean_logp: float
        teacher_entropy: tuple[float, ...] | None = None
        error: str | None = None

    def score(
        self,
        *,
        request_id: str,
        question: str,
        image_path: str,
        response_token_ids: Sequence[int],
        response_text: str = "",
    ) -> "TeacherScorer.ScoreResult":
        """Teacher-force score one response.

        Args:
            request_id: Unique identifier for this scoring request.
            question: The geometry problem text.
            image_path: Path to the diagram image.
            response_token_ids: The exact token IDs of the student response.
            response_text: The decoded response text (for logging only).

        Returns:
            ScoreResult with per-token log-probs and mean.
        """
        meta = self.metadata
        payload = {
            "requests": [
                {
                    "request_id": request_id,
                    "condition": "full",
                    "question": question,
                    "condition_inputs": {
                        "full_image": {"path": image_path},
                        "degraded_image": {"path": image_path, "transform": {"type": "gaussian_blur", "sigma": 2.0}},
                        "free_caption": "placeholder",
                        "task_evidence": "placeholder",
                        "task_visible_evidence": None,
                        "task_infer_evidence": None,
                        "task_solve_evidence": None,
                        "verified_facts": None,
                        "verified_facts_source": None,
                    },
                    "response_token_ids": list(response_token_ids),
                    "tokenizer_hash": str(meta["tokenizer_hash"]),
                    "response_text": response_text,
                }
            ]
        }

        raw = self._request("POST", "/score", payload)

        if not isinstance(raw, dict) or not isinstance(raw.get("responses"), list):
            raise TeacherServiceError("invalid score response")
        responses = raw["responses"]
        if len(responses) != 1:
            raise TeacherServiceError(
                f"expected 1 response, got {len(responses)}"
            )

        resp = responses[0]
        if resp.get("request_id") != request_id:
            raise TeacherServiceError("response request_id mismatch")

        error = resp.get("error")
        if error:
            return TeacherScorer.ScoreResult(
                request_id=request_id,
                sampled_token_log_probs=(),
                mean_logp=float("nan"),
                error=str(error),
            )

        sampled = tuple(float(v) for v in resp["sampled_token_log_probs"])
        mean_logp = float(sum(sampled) / max(len(sampled), 1))
        entropy = (
            tuple(float(v) for v in resp["teacher_entropy"])
            if resp.get("teacher_entropy")
            else None
        )

        return TeacherScorer.ScoreResult(
            request_id=request_id,
            sampled_token_log_probs=sampled,
            mean_logp=mean_logp,
            teacher_entropy=entropy,
        )


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

    def encode_response(self, response_text: str) -> tuple[int, ...]:
        """Tokenize a response string into token IDs."""
        return tuple(int(t) for t in self._tokenizer.encode(response_text))

    def tokenizer_hash(self) -> str:
        """Compute a deterministic hash of the tokenizer (matches teacher fingerprint)."""
        from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint

        return tokenizer_fingerprint(self._tokenizer)
