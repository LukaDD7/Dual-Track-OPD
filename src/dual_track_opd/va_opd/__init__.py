"""Visual-Advantage On-Policy Distillation research primitives."""

from .objective import (
    VA_LAMBDA,
    VA_TAU,
    VA_TOP_FRACTION,
    VAWeightingResult,
    build_va_token_weights,
    compute_rollout_weights,
    split_high_low_va,
)

__all__ = [
    "VA_LAMBDA",
    "VA_TAU",
    "VA_TOP_FRACTION",
    "VAWeightingResult",
    "build_va_token_weights",
    "compute_rollout_weights",
    "split_high_low_va",
]
