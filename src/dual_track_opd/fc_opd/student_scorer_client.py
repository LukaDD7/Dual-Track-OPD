"""HTTP client for a dedicated FC-OPD student scorer service."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch

from .conditions import Condition, build_condition_inputs
from .online_batch import OnlineFCOPDSample, OnlineStudentScores


class StudentScorerServiceError(RuntimeError):
    pass


class StudentScorerClient:
    """Synchronous scorer proxy backed by a dedicated GPU HTTP service."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

    def health(self) -> bool:
        response = self._request_json("GET", "/health")
        return isinstance(response, Mapping) and response.get("status") == "ok"

    def __call__(
        self,
        sample: OnlineFCOPDSample | Sequence[OnlineFCOPDSample],
        conditions: Sequence[Condition | str],
    ) -> OnlineStudentScores | list[OnlineStudentScores]:
        samples = list(sample) if isinstance(sample, Sequence) and not isinstance(sample, (str, bytes)) else [sample]
        payload = {
            "samples": [_sample_to_dict(item) for item in samples],
            "conditions": [Condition(condition).value for condition in conditions],
        }
        raw = self._request_json("POST", "/score", payload)
        if not isinstance(raw, Mapping) or not isinstance(raw.get("responses"), list):
            raise StudentScorerServiceError("student scorer returned an invalid payload")
        responses = [_scores_from_dict(item) for item in raw["responses"]]
        if len(responses) != len(samples):
            raise StudentScorerServiceError("student scorer response count does not match request count")
        return responses if len(samples) != 1 or isinstance(sample, Sequence) else responses[0]

    def _request_json(self, method: str, path: str, payload: object | None = None) -> object:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        request = Request(f"{self.base_url}{path}", method=method, data=data, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise StudentScorerServiceError(f"student scorer returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise StudentScorerServiceError(f"student scorer is unreachable: {exc.reason}") from exc


def _sample_to_dict(sample: OnlineFCOPDSample) -> dict[str, object]:
    return {
        "sample_uid": sample.sample_uid,
        "question": sample.question,
        "condition_inputs": sample.condition_inputs.to_dict(),
        "rollout_token_ids": list(sample.rollout_token_ids),
        "rollout_text": sample.rollout_text,
        "prompt": _serialize_prompt(sample.prompt),
        "images": _serialize_images(sample.images),
        "choices": list(sample.choices),
        "answer_metadata": sample.answer_metadata,
        "metadata": dict(sample.metadata),
    }


def _sample_from_dict(payload: Mapping[str, object]) -> OnlineFCOPDSample:
    return OnlineFCOPDSample(
        sample_uid=str(payload["sample_uid"]),
        question=str(payload["question"]),
        condition_inputs=build_condition_inputs({"condition_inputs": payload["condition_inputs"]}),
        rollout_token_ids=tuple(int(item) for item in payload["rollout_token_ids"]),  # type: ignore[index]
        rollout_text=str(payload["rollout_text"]),
        prompt=_deserialize_prompt(payload.get("prompt")),
        images=_deserialize_images(payload.get("images")),
        choices=tuple(str(item) for item in payload.get("choices", ()) or ()),  # type: ignore[arg-type]
        answer_metadata=payload.get("answer_metadata"),
        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {},
    )


def _serialize_prompt(prompt: object) -> object:
    """Prompt is typically a list of chat-message dicts — already JSON-serializable."""
    return prompt


def _deserialize_prompt(raw: object) -> object:
    return raw


def _serialize_images(images: object) -> object:
    """Encode any ``bytes`` items inside the images field as base64 strings."""
    if images is None:
        return None
    if isinstance(images, (str, bytes)):
        return images if isinstance(images, str) else base64.b64encode(images).decode("ascii")
    if isinstance(images, dict):
        return {k: _serialize_images(v) for k, v in images.items()}
    if isinstance(images, (list, tuple)):
        return [_serialize_images(item) for item in images]
    return images


def _deserialize_images(raw: object) -> object:
    """Decode base64-encoded bytes back.  Strings that look like paths are left alone."""
    if raw is None:
        return None
    if isinstance(raw, str):
        # Heuristic: a short-ish base64 string that is all ASCII — try to decode.
        # Longer than 200 chars is almost certainly base64 image data.
        if len(raw) > 200:
            try:
                return base64.b64decode(raw)
            except Exception:
                return raw
        return raw
    if isinstance(raw, dict):
        return {k: _deserialize_images(v) for k, v in raw.items()}
    if isinstance(raw, list):
        return [_deserialize_images(item) for item in raw]
    return raw


def _scores_to_dict(scores: OnlineStudentScores) -> dict[str, object]:
    return {
        "condition_log_probs": {
            Condition(condition).value: _values_to_list(values)
            for condition, values in scores.condition_log_probs.items()
        }
    }


def _scores_from_dict(payload: Mapping[str, object]) -> OnlineStudentScores:
    raw = payload.get("condition_log_probs")
    if not isinstance(raw, Mapping):
        raise StudentScorerServiceError("student scorer response is missing condition_log_probs")
    return OnlineStudentScores(
        loss_logits=None,
        condition_log_probs={
            Condition(condition): torch.tensor(values, dtype=torch.float32)
            for condition, values in raw.items()
        },
    )


def _values_to_list(values: Sequence[float] | torch.Tensor) -> list[float]:
    tensor = values.detach().cpu().float().reshape(-1) if isinstance(values, torch.Tensor) else torch.tensor(list(values), dtype=torch.float32)
    return [float(value) for value in tensor.tolist()]
