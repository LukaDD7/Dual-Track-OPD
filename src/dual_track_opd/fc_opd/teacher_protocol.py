"""JSON protocol shared by the standalone teacher service and client."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .conditions import Condition, ConditionInputs


PROTOCOL_VERSION = "fc-opd-teacher-v2-exact-prompt"

_BYTES_MARKER = "__fc_opd_bytes_b64__"


def _is_pil_image(obj: object) -> bool:
    try:
        from PIL.Image import Image

        return isinstance(obj, Image)
    except ImportError:
        return False


def _serialize_prompt_value(obj: object) -> object:
    """Recursively convert bytes/PIL.Image → {__fc_opd_bytes_b64__: <base64>} for JSON safety."""
    if isinstance(obj, bytes):
        return {_BYTES_MARKER: base64.b64encode(obj).decode("ascii")}
    if _is_pil_image(obj):
        from io import BytesIO

        buf = BytesIO()
        obj.save(buf, format="PNG")
        return {_BYTES_MARKER: base64.b64encode(buf.getvalue()).decode("ascii")}
    if isinstance(obj, dict):
        return {str(k): _serialize_prompt_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_prompt_value(v) for v in obj]
    return obj


def _deserialize_prompt_value(obj: object) -> object:
    """Recursively restore {__fc_opd_bytes_b64__: <base64>} → bytes."""
    if isinstance(obj, dict):
        if _BYTES_MARKER in obj and len(obj) == 1:
            return base64.b64decode(obj[_BYTES_MARKER])
        return {str(k): _deserialize_prompt_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deserialize_prompt_value(v) for v in obj]
    return obj


class FingerprintTokenizer(Protocol):
    def get_vocab(self) -> Mapping[str, int]: ...


@dataclass(frozen=True)
class TeacherMetadata:
    model_id: str
    tokenizer_hash: str
    vocab_size: int
    top_k: int
    dtype: str
    git_revision: str
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if not self.model_id or not self.tokenizer_hash:
            raise ValueError("teacher model ID and tokenizer hash must be non-empty")
        if self.vocab_size <= 1 or not 1 <= self.top_k < self.vocab_size:
            raise ValueError("teacher top_k must be positive and smaller than vocab_size")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "tokenizer_hash": self.tokenizer_hash,
            "vocab_size": self.vocab_size,
            "top_k": self.top_k,
            "dtype": self.dtype,
            "git_revision": self.git_revision,
            "protocol_version": self.protocol_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TeacherMetadata":
        return cls(
            model_id=str(value["model_id"]),
            tokenizer_hash=str(value["tokenizer_hash"]),
            vocab_size=int(value["vocab_size"]),
            top_k=int(value["top_k"]),
            dtype=str(value["dtype"]),
            git_revision=str(value["git_revision"]),
            protocol_version=str(value["protocol_version"]),
        )


@dataclass(frozen=True)
class TeacherScoreRequest:
    request_id: str
    condition: Condition
    question: str
    condition_inputs: ConditionInputs
    response_token_ids: tuple[int, ...]
    tokenizer_hash: str
    response_text: str | None = None
    # Exact chat messages used by the rollout.  FULL/DEGRADED GKD must reuse
    # these messages instead of reconstructing a semantically similar prompt.
    prompt: tuple[dict[str, Any], ...] | None = None

    def __post_init__(self) -> None:
        if not self.request_id or not self.question.strip() or not self.tokenizer_hash:
            raise ValueError("request ID, question, and tokenizer hash must be non-empty")
        if not self.response_token_ids or any(token_id < 0 for token_id in self.response_token_ids):
            raise ValueError("response_token_ids must contain non-negative token IDs")
        self.condition_inputs.validate()
        if self.prompt is not None and (
            not self.prompt or any(not isinstance(message, Mapping) for message in self.prompt)
        ):
            raise ValueError("prompt must be a non-empty sequence of message mappings")

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "condition": self.condition.value,
            "question": self.question,
            "condition_inputs": self.condition_inputs.to_dict(),
            "response_token_ids": list(self.response_token_ids),
            "tokenizer_hash": self.tokenizer_hash,
            "response_text": self.response_text,
            "prompt": None if self.prompt is None else _serialize_prompt_value(list(self.prompt)),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TeacherScoreRequest":
        raw_inputs = value["condition_inputs"]
        if not isinstance(raw_inputs, Mapping):
            raise ValueError("condition_inputs must be a mapping")
        full = raw_inputs["full_image"]
        degraded = raw_inputs["degraded_image"]
        if not isinstance(full, Mapping) or not isinstance(degraded, Mapping):
            raise ValueError("image condition inputs must be mappings")
        from .conditions import ImageInput

        inputs = ConditionInputs(
            full_image=ImageInput(path=str(full["path"])),
            degraded_image=ImageInput(
                path=str(degraded["path"]),
                transform=dict(degraded.get("transform", {})),
            ),
            free_caption=str(raw_inputs["free_caption"]),
            task_evidence=str(raw_inputs["task_evidence"]),
            task_visible_evidence=(
                None if raw_inputs.get("task_visible_evidence") is None else str(raw_inputs["task_visible_evidence"])
            ),
            task_infer_evidence=(
                None if raw_inputs.get("task_infer_evidence") is None else str(raw_inputs["task_infer_evidence"])
            ),
            task_solve_evidence=(
                None if raw_inputs.get("task_solve_evidence") is None else str(raw_inputs["task_solve_evidence"])
            ),
            verified_facts=(
                None if raw_inputs.get("verified_facts") is None else str(raw_inputs["verified_facts"])
            ),
            verified_facts_source=(
                None
                if raw_inputs.get("verified_facts_source") is None
                else str(raw_inputs["verified_facts_source"])
            ),
        )
        inputs.validate()
        return cls(
            request_id=str(value["request_id"]),
            condition=Condition(str(value["condition"])),
            question=str(value["question"]),
            condition_inputs=inputs,
            response_token_ids=tuple(int(item) for item in value["response_token_ids"]),
            tokenizer_hash=str(value["tokenizer_hash"]),
            response_text=None if value.get("response_text") is None else str(value["response_text"]),
            prompt=(
                None
                if value.get("prompt") is None
                else tuple(
                    dict(message) for message in _deserialize_prompt_value(value["prompt"])
                )
            ),
        )


@dataclass(frozen=True)
class TeacherScoreResponse:
    request_id: str
    condition: Condition
    token_ids: tuple[int, ...]
    topk_token_ids: tuple[tuple[int, ...], ...]
    topk_log_probs: tuple[tuple[float, ...], ...]
    tail_log_prob: tuple[float, ...] | None
    teacher_entropy: tuple[float, ...]
    sampled_token_log_probs: tuple[float, ...]  # exact log P_T(y_t | cond) per response token

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "condition": self.condition.value,
            "token_ids": list(self.token_ids),
            "topk_token_ids": [list(row) for row in self.topk_token_ids],
            "topk_log_probs": [list(row) for row in self.topk_log_probs],
            "tail_log_prob": None if self.tail_log_prob is None else list(self.tail_log_prob),
            "teacher_entropy": list(self.teacher_entropy),
            "sampled_token_log_probs": list(self.sampled_token_log_probs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TeacherScoreResponse":
        return cls(
            request_id=str(value["request_id"]),
            condition=Condition(str(value["condition"])),
            token_ids=tuple(int(item) for item in value["token_ids"]),
            topk_token_ids=tuple(
                tuple(int(item) for item in row) for row in value["topk_token_ids"]
            ),
            topk_log_probs=tuple(
                tuple(float(item) for item in row) for row in value["topk_log_probs"]
            ),
            tail_log_prob=(
                None
                if value.get("tail_log_prob") is None
                else tuple(float(item) for item in value["tail_log_prob"])
            ),
            teacher_entropy=tuple(float(item) for item in value["teacher_entropy"]),
            sampled_token_log_probs=tuple(float(item) for item in value["sampled_token_log_probs"]),
        )


def tokenizer_fingerprint(tokenizer: FingerprintTokenizer) -> str:
    """Hash tokenizer vocabulary and special-token metadata deterministically."""

    def json_safe(value: object) -> object:
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            return [json_safe(item) for item in value]
        return str(value)

    vocab = sorted((str(token), int(token_id)) for token, token_id in tokenizer.get_vocab().items())
    special_tokens = getattr(tokenizer, "special_tokens_map", {})
    added_vocab = getattr(tokenizer, "get_added_vocab", lambda: {})()
    payload = {
        "vocab": vocab,
        "special_tokens_map": json_safe(special_tokens),
        "added_vocab": sorted(
            (str(token), int(token_id)) for token, token_id in dict(added_vocab).items()
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_score_response(response: TeacherScoreResponse, metadata: TeacherMetadata) -> None:
    """Validate response shapes, ranges, and finite numeric values."""

    import math

    length = len(response.token_ids)
    if len(response.topk_token_ids) != length or len(response.topk_log_probs) != length:
        raise ValueError("teacher response positions do not align with token_ids")
    if len(response.teacher_entropy) != length:
        raise ValueError("teacher_entropy does not align with token_ids")
    if len(response.sampled_token_log_probs) != length:
        raise ValueError("sampled_token_log_probs does not align with token_ids")
    if response.tail_log_prob is not None and len(response.tail_log_prob) != length:
        raise ValueError("tail_log_prob does not align with token_ids")

    for position, (token_ids, log_probs) in enumerate(zip(
        response.topk_token_ids,
        response.topk_log_probs,
        strict=True,
    )):
        if len(token_ids) != metadata.top_k or len(log_probs) != metadata.top_k:
            raise ValueError("teacher top-k width does not match metadata")
        if any(token_id < 0 or token_id >= metadata.vocab_size for token_id in token_ids):
            raise ValueError("teacher returned a token outside the vocabulary")
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("teacher returned duplicate token IDs at one position")
        if not all(math.isfinite(value) and value <= 1e-6 for value in log_probs):
            raise ValueError("teacher top-k log probabilities must be finite and non-positive")
        topk_mass = sum(math.exp(value) for value in log_probs)
        tail_mass = (
            0.0
            if response.tail_log_prob is None
            else math.exp(response.tail_log_prob[position])
        )
        if topk_mass > 1.0 + 1e-4 or not math.isclose(topk_mass + tail_mass, 1.0, abs_tol=1e-4):
            raise ValueError("teacher top-k and tail probability mass must sum to one")
    if not all(math.isfinite(value) and value >= 0 for value in response.teacher_entropy):
        raise ValueError("teacher entropy must be finite and non-negative")
    if not all(math.isfinite(value) and value <= 1e-6 for value in response.sampled_token_log_probs):
        raise ValueError("sampled_token_log_probs must be finite and non-positive (log-probs)")
    if response.tail_log_prob is not None and not all(
        math.isfinite(value) and value <= 1e-6 for value in response.tail_log_prob
    ):
        raise ValueError("teacher tail log probabilities must be finite and non-positive")


def ensure_exact_token_alignment(
    expected: Sequence[int],
    actual: Sequence[int],
    *,
    context: str,
) -> None:
    if tuple(expected) != tuple(actual):
        raise ValueError(f"response token IDs differ during {context}")
