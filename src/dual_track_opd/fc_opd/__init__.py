"""Framework-independent failure-calibrated OPD primitives."""

from .chunk_parser import ChunkMasks, parse_response_chunks
from .conditions import Condition, ConditionInputs, build_condition_inputs
from .loss import FCOPDLossConfig, compute_fc_opd_loss, sparse_forward_kl
from .router import RouterConfig, route_condition_weights
from .signal_decomposer import TeacherTopK, compute_condition_signals, jensen_shannon_topk

__all__ = [
    "ChunkMasks",
    "Condition",
    "ConditionInputs",
    "FCOPDLossConfig",
    "RouterConfig",
    "TeacherTopK",
    "build_condition_inputs",
    "compute_condition_signals",
    "compute_fc_opd_loss",
    "jensen_shannon_topk",
    "parse_response_chunks",
    "route_condition_weights",
    "sparse_forward_kl",
]
