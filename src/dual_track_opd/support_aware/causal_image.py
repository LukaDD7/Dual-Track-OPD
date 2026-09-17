"""Deterministic in-memory image interventions for causal probes."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image, ImageFilter

from dual_track_opd.fc_opd.degradation import degraded_transform


@dataclass(frozen=True)
class ImageConditions:
    full: Image.Image
    degraded: Image.Image
    null: Image.Image
    metadata: Mapping[str, Any]


def image_sha256(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _apply_transform(image: Image.Image, mode: str) -> Image.Image:
    original = image.convert("RGB")
    transform = degraded_transform(mode)
    kind = transform["type"]
    width, height = original.size
    if kind == "lowres_bilinear_nearest":
        scale = float(transform["scale"])
        low = original.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.Resampling.BILINEAR,
        )
        return low.resize((width, height), Image.Resampling.NEAREST)
    if kind == "gaussian_blur":
        return original.filter(ImageFilter.GaussianBlur(radius=float(transform["sigma"])))
    if kind == "jpeg":
        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=int(transform["quality"]), optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as value:
            return value.convert("RGB")
    if kind == "blank_control":
        return Image.new("RGB", (width, height), "white")
    raise ValueError(f"unsupported causal image degradation: {mode}")


def build_image_conditions(
    image: Image.Image,
    *,
    degraded_mode: str = "blur_sigma_2",
    null_value: int = 255,
) -> ImageConditions:
    """Build same-shape full/degraded/null images with auditable metadata."""

    if not 0 <= null_value <= 255:
        raise ValueError("null image value must lie in [0, 255]")
    full = image.convert("RGB").copy()
    degraded = _apply_transform(full, degraded_mode)
    null = Image.new("RGB", full.size, (null_value, null_value, null_value))
    if degraded.size != full.size or null.size != full.size:
        raise RuntimeError("causal image interventions must preserve image size")
    metadata = {
        "deterministic": True,
        "seed": None,
        "degraded_mode": degraded_mode,
        "degraded_transform": degraded_transform(degraded_mode),
        "null_transform": {"type": "constant_rgb", "value": null_value},
        "size": list(full.size),
        "mode": "RGB",
        "sha256": {
            "full": image_sha256(full),
            "degraded": image_sha256(degraded),
            "null": image_sha256(null),
        },
    }
    return ImageConditions(full=full, degraded=degraded, null=null, metadata=metadata)
