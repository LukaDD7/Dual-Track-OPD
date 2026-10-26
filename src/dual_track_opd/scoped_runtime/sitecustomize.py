"""Apply local SCOPED runtime compatibility patches without editing upstream."""

from __future__ import annotations

try:
    import numpy as np
    import torch
    import verl.protocol as upstream_protocol
    import verl.utils.dataset as upstream_dataset
except Exception:
    # Do not make unrelated Python entry points fail when the RL stack is absent.
    pass
else:
    from dual_track_opd.scoped_runtime.collate import (
        as_sample_object_array,
        normalize_non_tensor_batch,
    )

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

    _upstream_concat = upstream_protocol.DataProto.concat

    @staticmethod
    def concat(data: list["upstream_protocol.DataProto"]) -> "upstream_protocol.DataProto":
        prepared = []
        for proto in data:
            prepared.append(
                upstream_protocol.DataProto(
                    batch=proto.batch,
                    non_tensor_batch=normalize_non_tensor_batch(proto.non_tensor_batch),
                    meta_info=proto.meta_info,
                )
            )
        return _upstream_concat(prepared)

    upstream_protocol.DataProto.concat = concat
