"""Compatibility helpers for the upstream SCOPED RL data pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_sample_object_array(value: Any) -> np.ndarray:
    """Return a one-dimensional object array whose entries are samples.

    NumPy infers an extra dimension when every sample is an equal-length list
    (for example, ``raw_prompt_ids`` in a batch of same-length prompts).  The
    upstream DataProto contract expects non-tensor fields to use dimension zero
    as the sample axis, so preserve each sample as one object instead.
    """
    items = list(value)
    result = np.empty(len(items), dtype=object)
    result[:] = [
        list(item) if isinstance(item, (np.ndarray, list, tuple)) else item
        for item in items
    ]
    return result


def normalize_non_tensor_batch(mapping: dict[str, Any]) -> dict[str, Any]:
    """Normalize object-valued DataProto fields to one sample axis.

    This is deliberately conservative: numeric arrays keep their shape, while
    object arrays are reduced to the DataProto sample axis.  Equal-length
    Python lists can otherwise be inferred as a rectangular 2-D object array
    by NumPy, which breaks concatenation against batches whose entries have
    different lengths.
    """
    normalized = {}
    for key, value in mapping.items():
        if isinstance(value, np.ndarray) and value.dtype == np.dtype(object):
            normalized[key] = as_sample_object_array(value)
        else:
            normalized[key] = value
    return normalized
