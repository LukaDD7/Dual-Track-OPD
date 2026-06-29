"""Synchronous client for the standalone FC-OPD teacher service."""

from __future__ import annotations

import json
from typing import Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch

from .conditions import Condition, ConditionInputs
from .signal_decomposer import TeacherTopK
from .teacher_protocol import (
    PROTOCOL_VERSION,
    TeacherMetadata,
    TeacherScoreRequest,
    TeacherScoreResponse,
    validate_score_response,
)


class TeacherServiceError(RuntimeError):
    pass


class TeacherClient:
    def __init__(
        self,
        base_url: str,
        *,
        expected_tokenizer_hash: str | None = None,
        timeout_seconds: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.metadata = TeacherMetadata.from_dict(self._request_json("GET", "/metadata"))
        if self.metadata.protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"unsupported teacher protocol: {self.metadata.protocol_version}"
            )
        if expected_tokenizer_hash is not None and self.metadata.tokenizer_hash != expected_tokenizer_hash:
            raise ValueError("student and teacher tokenizer hashes do not match")

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
            raise TeacherServiceError(f"teacher service returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise TeacherServiceError(f"teacher service is unreachable: {exc.reason}") from exc

    def health(self) -> bool:
        response = self._request_json("GET", "/health")
        return isinstance(response, dict) and response.get("status") == "ok"

    def score(self, requests: Sequence[TeacherScoreRequest]) -> list[TeacherScoreResponse]:
        if not requests:
            raise ValueError("at least one teacher request is required")
        for request in requests:
            if request.tokenizer_hash != self.metadata.tokenizer_hash:
                raise ValueError("request tokenizer hash does not match teacher metadata")

        payload = {"requests": [request.to_dict() for request in requests]}
        raw = self._request_json("POST", "/score", payload)
        if not isinstance(raw, dict) or not isinstance(raw.get("responses"), list):
            raise TeacherServiceError("teacher service returned an invalid score payload")
        responses = [TeacherScoreResponse.from_dict(item) for item in raw["responses"]]
        if len(responses) != len(requests):
            raise TeacherServiceError("teacher service response count does not match request count")

        for request, response in zip(requests, responses, strict=True):
            if response.request_id != request.request_id or response.condition != request.condition:
                raise TeacherServiceError("teacher response identity does not match its request")
            # NOTE: token_ids may differ from the original request when the
            # teacher repaired them for tokenizer alignment (32B vs 4B).
            # We skip strict alignment here; shape validation happens later
            # in _validate_teacher_scores (online_batch.py).
            validate_score_response(response, self.metadata)
        return responses


def score_teacher_conditions(
    response_token_ids: Sequence[int],
    question: str,
    condition_inputs: ConditionInputs,
    conditions: Iterable[Condition | str],
    teacher_client: TeacherClient,
    *,
    response_text: str | None = None,
    request_prefix: str = "score",
) -> dict[Condition, TeacherTopK]:
    """Score one student response under multiple teacher conditions."""

    requests = [
        TeacherScoreRequest(
            request_id=f"{request_prefix}:{Condition(condition).value}",
            condition=Condition(condition),
            question=question,
            condition_inputs=condition_inputs,
            response_token_ids=tuple(int(item) for item in response_token_ids),
            tokenizer_hash=teacher_client.metadata.tokenizer_hash,
            response_text=response_text,
        )
        for condition in conditions
    ]
    responses = teacher_client.score(requests)
    output: dict[Condition, TeacherTopK] = {}
    for response in responses:
        output[response.condition] = _topk_from_response(response)
    return output


def score_teacher_conditions_multi_sample(
    samples: Sequence[tuple[Sequence[int], str, ConditionInputs]],
    conditions: Iterable[Condition | str],
    teacher_client: TeacherClient,
    *,
    response_texts: Sequence[str | None] | None = None,
    request_prefix: str = "score",
) -> list[dict[Condition, TeacherTopK]]:
    """Score multiple student responses under multiple teacher conditions.

    All B×C requests are sent in a single HTTP POST.  Returns one dict
    per input sample, each mapping condition → TeacherTopK.
    """
    conditions = [Condition(c) for c in conditions]
    if not conditions:
        raise ValueError("at least one condition is required")
    texts = response_texts or [None] * len(samples)
    if len(texts) != len(samples):
        raise ValueError("response_texts must match samples length")

    requests: list[TeacherScoreRequest] = []
    for idx, (token_ids, question, condition_inputs) in enumerate(samples):
        for condition in conditions:
            requests.append(
                TeacherScoreRequest(
                    request_id=f"{request_prefix}:{idx}:{condition.value}",
                    condition=condition,
                    question=question,
                    condition_inputs=condition_inputs,
                    response_token_ids=tuple(int(t) for t in token_ids),
                    tokenizer_hash=teacher_client.metadata.tokenizer_hash,
                    response_text=texts[idx],
                )
            )

    responses = teacher_client.score(requests)
    # Group responses back by sample index and condition.
    by_sample: list[dict[str, TeacherTopK]] = [
        {} for _ in samples
    ]
    for response in responses:
        # request_id is "prefix:idx:condition_value"
        parts = response.request_id.rsplit(":", 1)
        sample_idx_str = parts[0].rsplit(":", 1)[-1]
        try:
            sample_idx = int(sample_idx_str)
        except ValueError:
            raise TeacherServiceError(
                f"cannot parse sample index from response request_id: {response.request_id}"
            )
        by_sample[sample_idx][response.condition.value] = _topk_from_response(response)

    return [
        {Condition(k): v for k, v in sample_dict.items()}
        for sample_dict in by_sample
    ]


def _topk_from_response(response: TeacherScoreResponse) -> TeacherTopK:
    return TeacherTopK(
        token_ids=torch.tensor([response.topk_token_ids], dtype=torch.int64),
        log_probs=torch.tensor([response.topk_log_probs], dtype=torch.float32),
        tail_log_prob=(
            None
            if response.tail_log_prob is None
            else torch.tensor([response.tail_log_prob], dtype=torch.float32)
        ),
        entropy=torch.tensor([response.teacher_entropy], dtype=torch.float32),
    )
