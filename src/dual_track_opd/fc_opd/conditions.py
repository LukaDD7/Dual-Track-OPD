"""Condition schema and deterministic condition-input construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class Condition(str, Enum):
    """Teacher input conditions used by failure-calibrated OPD."""

    FULL = "full"
    BLUR = "blur"
    DEGRADED = "degraded"
    FREE = "free"
    TASK = "task"
    TASK_VISIBLE = "task_visible"
    TASK_INFER = "task_infer"
    TASK_SOLVE = "task_solve"
    FACT = "fact"


@dataclass(frozen=True)
class ImageInput:
    path: str
    transform: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ConditionInputs:
    """Validated condition inputs for one question."""

    full_image: ImageInput
    degraded_image: ImageInput
    free_caption: str
    task_evidence: str
    task_visible_evidence: str | None = None
    task_infer_evidence: str | None = None
    task_solve_evidence: str | None = None
    verified_facts: str | None = None
    verified_facts_source: str | None = None

    def validate(self, *, require_paths: bool = False) -> None:
        if not self.full_image.path.strip() or not self.degraded_image.path.strip():
            raise ValueError("condition image paths must be non-empty")
        if not self.free_caption.strip() or not self.task_evidence.strip():
            raise ValueError("caption and task evidence must be non-empty")
        for field, value in (
            ("task_visible_evidence", self.task_visible_evidence),
            ("task_infer_evidence", self.task_infer_evidence),
            ("task_solve_evidence", self.task_solve_evidence),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"{field} must be non-empty when provided")
        transform = self.degraded_image.transform
        if not isinstance(transform, Mapping):
            raise ValueError("degraded image must record a transform")
        transform_type = transform.get("type")
        if transform_type == "gaussian_blur":
            sigma = transform.get("sigma")
            if not isinstance(sigma, int | float) or sigma <= 0:
                raise ValueError("gaussian_blur sigma must be positive")
        elif transform_type == "lowres_nearest":
            scale = transform.get("scale")
            if not isinstance(scale, int | float) or not 0 < float(scale) < 1:
                raise ValueError("lowres_nearest scale must be in (0, 1)")
        else:
            raise ValueError("degraded image transform must be gaussian_blur or lowres_nearest")
        _validate_verified_facts(self.verified_facts, self.verified_facts_source)
        if require_paths:
            for path in (self.full_image.path, self.degraded_image.path):
                if not Path(path).is_file():
                    raise FileNotFoundError(f"condition image path does not exist: {path}")

    def available_conditions(self) -> tuple[Condition, ...]:
        conditions = [Condition.FULL, Condition.BLUR, Condition.FREE, Condition.TASK]
        if self.task_visible_evidence is not None:
            conditions.append(Condition.TASK_VISIBLE)
        if self.task_infer_evidence is not None:
            conditions.append(Condition.TASK_INFER)
        if self.task_solve_evidence is not None:
            conditions.append(Condition.TASK_SOLVE)
        if self.verified_facts is not None:
            conditions.append(Condition.FACT)
        return tuple(conditions)

    def to_dict(self) -> dict[str, object]:
        return {
            "full_image": {
                "path": self.full_image.path,
            },
            "degraded_image": {
                "path": self.degraded_image.path,
                "transform": dict(self.degraded_image.transform or {}),
            },
            "free_caption": self.free_caption,
            "task_evidence": self.task_evidence,
            "task_visible_evidence": self.task_visible_text,
            "task_infer_evidence": self.task_infer_evidence,
            "task_solve_evidence": self.task_solve_evidence,
            "verified_facts": self.verified_facts,
            "verified_facts_source": self.verified_facts_source,
        }

    @property
    def task_visible_text(self) -> str:
        return self.task_visible_evidence or self.task_evidence


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, field)


def _resolve_path(path: str, root: str | Path | None) -> str:
    candidate = Path(path).expanduser()
    if root is not None and not candidate.is_absolute():
        candidate = Path(root).expanduser() / candidate
    return str(candidate.resolve(strict=False))


def _validate_verified_facts(facts: str | None, source: str | None) -> None:
    if (facts is None) != (source is None):
        raise ValueError("verified_facts and verified_facts_source must be provided together")


def build_condition_inputs(
    record: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> ConditionInputs:
    """Build validated condition inputs without performing image transformations.

    The degraded image must already exist as a deterministic prepared artifact.
    This function records and validates its transform metadata; it deliberately
    does not mutate source images.
    """

    config = config or {}
    root = config.get("data_root")
    require_paths = bool(config.get("require_paths", False))
    default_blur = config.get("blur", {"type": "gaussian_blur", "sigma": 2.0})

    raw = record.get("condition_inputs")
    if not isinstance(raw, Mapping):
        raise ValueError("record.condition_inputs must be a mapping")

    full_raw = raw.get("full_image")
    degraded_raw = raw.get("degraded_image")
    if not isinstance(full_raw, Mapping) or not isinstance(degraded_raw, Mapping):
        raise ValueError("full_image and degraded_image must be mappings")

    full_path = _resolve_path(_nonempty_string(full_raw.get("path"), "full_image.path"), root)
    degraded_path = _resolve_path(
        _nonempty_string(degraded_raw.get("path"), "degraded_image.path"),
        root,
    )

    transform = degraded_raw.get("transform", default_blur)
    if not isinstance(transform, Mapping) or not transform:
        raise ValueError("degraded_image.transform must be a non-empty mapping")
    if transform.get("type") == "gaussian_blur":
        sigma = transform.get("sigma")
        if not isinstance(sigma, int | float) or sigma <= 0:
            raise ValueError("gaussian_blur sigma must be positive")
    elif transform.get("type") == "lowres_nearest":
        scale = transform.get("scale")
        if not isinstance(scale, int | float) or not 0 < float(scale) < 1:
            raise ValueError("lowres_nearest scale must be in (0, 1)")
    else:
        raise ValueError("degraded image transform must be gaussian_blur or lowres_nearest")

    if require_paths:
        for field, path in (("full_image.path", full_path), ("degraded_image.path", degraded_path)):
            if not Path(path).is_file():
                raise FileNotFoundError(f"{field} does not resolve to a file: {path}")

    facts = _optional_string(raw.get("verified_facts"), "verified_facts")
    facts_source = _optional_string(raw.get("verified_facts_source"), "verified_facts_source")
    _validate_verified_facts(facts, facts_source)

    inputs = ConditionInputs(
        full_image=ImageInput(path=full_path),
        degraded_image=ImageInput(path=degraded_path, transform=dict(transform)),
        free_caption=_nonempty_string(raw.get("free_caption"), "free_caption"),
        task_evidence=_nonempty_string(raw.get("task_evidence"), "task_evidence"),
        task_visible_evidence=_optional_string(raw.get("task_visible_evidence"), "task_visible_evidence"),
        task_infer_evidence=_optional_string(raw.get("task_infer_evidence"), "task_infer_evidence"),
        task_solve_evidence=_optional_string(raw.get("task_solve_evidence"), "task_solve_evidence"),
        verified_facts=facts,
        verified_facts_source=facts_source,
    )
    inputs.validate(require_paths=require_paths)
    return inputs
