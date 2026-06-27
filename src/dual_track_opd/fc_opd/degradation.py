"""Deterministic degraded-image materialization for FC-OPD conditions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .geometry3k_adapter import default_degraded_image_dir

DEGRADED_MODES: tuple[str, ...] = (
    "lowres_75_bilinear_nearest",
    "lowres_50_bilinear_nearest",
    "lowres_33_bilinear_nearest",
    "lowres_20_bilinear_nearest",
    "lowres_10_bilinear_nearest",
    "blur_sigma_1",
    "blur_sigma_2",
    "jpeg_q20",
    "blank_control",
    "lowres_10pct_nearest",
    "gaussian_blur_s2",
)


def degraded_output_root(degraded_dir: str | None = None) -> Path:
    if degraded_dir:
        return Path(degraded_dir).expanduser()
    return default_degraded_image_dir()


def degraded_transform(mode: str, *, blur_sigma: float = 2.0) -> dict[str, Any]:
    if mode.startswith("lowres_") and mode.endswith("_bilinear_nearest"):
        pct = int(mode.split("_")[1])
        return {
            "type": "lowres_bilinear_nearest",
            "scale": pct / 100.0,
            "downsample": "bilinear",
            "upsample": "nearest",
            "degraded_mode": mode,
        }
    if mode == "lowres_10pct_nearest":
        return {"type": "lowres_nearest", "scale": 0.1, "degraded_mode": mode}
    if mode.startswith("blur_sigma_"):
        sigma = float(mode.rsplit("_", 1)[1])
        return {"type": "gaussian_blur", "sigma": sigma, "degraded_mode": mode}
    if mode == "gaussian_blur_s2":
        return {"type": "gaussian_blur", "sigma": float(blur_sigma), "degraded_mode": mode}
    if mode == "jpeg_q20":
        return {"type": "jpeg", "quality": 20, "degraded_mode": mode}
    if mode == "blank_control":
        return {"type": "blank_control", "degraded_mode": mode}
    raise ValueError(f"degraded_mode must be one of {DEGRADED_MODES}")


def materialize_degraded_image(
    image_path: str,
    *,
    mode: str,
    degraded_dir: str | None = None,
    blur_sigma: float = 2.0,
) -> str:
    """Write degraded image under DTOPD_OUTPUT_ROOT/fc_opd/degraded_images.

    The source sample directory is never used as the output root unless the
    caller explicitly passes such a path through ``degraded_dir``.
    """

    source = Path(os.path.expandvars(image_path)).expanduser()
    suffix = source.suffix or ".png"
    output_root = degraded_output_root(degraded_dir)
    target_suffix = ".jpg" if mode == "jpeg_q20" else suffix
    target = output_root / _safe_relative_name(source, mode, target_suffix)
    if target.is_file():
        return str(target)

    from PIL import Image, ImageFilter

    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        original = image.convert("RGB")
        width, height = original.size
        if mode.startswith("lowres_") and mode.endswith("_bilinear_nearest"):
            scale = degraded_transform(mode)["scale"]
            low = original.resize(
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                Image.Resampling.BILINEAR,
            )
            result = low.resize((width, height), Image.Resampling.NEAREST)
            result.save(target)
        elif mode == "lowres_10pct_nearest":
            low = original.resize((max(1, width // 10), max(1, height // 10)), Image.Resampling.NEAREST)
            low.resize((width, height), Image.Resampling.NEAREST).save(target)
        elif mode.startswith("blur_sigma_") or mode == "gaussian_blur_s2":
            sigma = float(degraded_transform(mode, blur_sigma=blur_sigma)["sigma"])
            original.filter(ImageFilter.GaussianBlur(radius=sigma)).save(target)
        elif mode == "jpeg_q20":
            original.save(target, format="JPEG", quality=20, optimize=False)
        elif mode == "blank_control":
            Image.new("RGB", (width, height), "white").save(target)
        else:
            raise ValueError(f"degraded_mode must be one of {DEGRADED_MODES}")
    return str(target)


def _safe_relative_name(source: Path, mode: str, suffix: str) -> Path:
    parts = [part for part in source.parts if part not in {"/", ""}]
    if len(parts) >= 3:
        stem_dir = Path(*parts[-3:-1])
    else:
        stem_dir = Path("images")
    return stem_dir / f"{source.stem}.{mode}{suffix}"
