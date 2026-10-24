"""Apply local SCOPED runtime compatibility patches without editing upstream."""

from __future__ import annotations

try:
    import torch
    import verl.utils.dataset as upstream_dataset
except Exception:
    # Do not make unrelated Python entry points fail when the RL stack is absent.
    pass
else:
    from dual_track_opd.scoped_runtime.collate import as_sample_object_array

    _upstream_collate_fn = upstream_dataset.collate_fn

    def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
        result = _upstream_collate_fn(features)
        return {
            key: (
                value
                if torch.is_tensor(value)
                else as_sample_object_array(value)
            )
            for key, value in result.items()
        }

    upstream_dataset.collate_fn = collate_fn
