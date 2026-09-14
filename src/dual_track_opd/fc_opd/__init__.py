"""Framework-independent failure-calibrated OPD primitives."""


def _missing_optional_class(name: str, requirement: str):
    class _MissingOptional:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise ModuleNotFoundError(f"{name} requires optional packages: {requirement}")

    _MissingOptional.__name__ = name
    return _MissingOptional


from .chunk_parser import ChunkMasks, parse_response_chunks
from .conditions import Condition, ConditionInputs, build_condition_inputs
from .loss import FCOPDLossConfig, compute_fc_opd_loss, sparse_forward_kl
from .online_batch import (
    OnlineFCOPDBatchOutput,
    OnlineFCOPDConfig,
    OnlineFCOPDSample,
    OnlineFCOPDSampleOutput,
    OnlineStudentScores,
    compute_online_fc_opd_batch,
)
from .offline_loss import (
    MinTrainReport,
    MinTrainStep,
    OfflineLossResult,
    OfflineLossSmokeReport,
    OfflineRecordTensors,
    load_offline_score_records,
    make_student_logits,
    offline_record_to_tensors,
    run_offline_loss_backward,
    run_offline_loss_smoke,
    run_offline_min_train_smoke,
)
from .real_student_smoke import (
    HFStudentProvider,
    RealMinTrainReport,
    RealMinTrainStep,
    RealStudentConfig,
    RealStudentResult,
    RealStudentSmokeReport,
    StudentForwardOutput,
    condition_inputs_from_record,
    detect_tied_parameter_groups,
    lm_head_embed_tied,
    response_logit_slice,
    run_real_student_min_train,
    run_real_student_record,
    run_real_student_smoke,
    teacher_scores_to_device,
)
from .offline_scoring import (
    ByteTokenizer,
    OfflineScoreRecord,
    OfflineScoringConfig,
    OfflineScoringResult,
    build_condition_inputs_for_record,
    default_output_dir,
    iter_offline_scores,
    load_student_responses,
    load_vision_opd_records,
    make_smoke_dataset,
    score_record,
    write_offline_scores,
)
from .router import RouterConfig, route_condition_weights
from .signal_decomposer import TeacherTopK, compute_condition_signals, jensen_shannon_topk
from .student_scorer import StudentScorer
from .ray_student_scorer import RayStudentScorerProxy, build_ray_student_scorer_proxy
from .teacher_client import TeacherClient, score_teacher_conditions, score_teacher_conditions_multi_sample
from .teacher_protocol import (
    TeacherMetadata,
    TeacherScoreRequest,
    TeacherScoreResponse,
    tokenizer_fingerprint,
)
from .teacher_prompts import render_teacher_prompt
try:
    from .verl_dataset import FCOPDDataset
except ModuleNotFoundError as exc:
    if exc.name not in {"transformers", "verl"}:
        raise
    FCOPDDataset = _missing_optional_class("FCOPDDataset", "transformers and verl")
from .verl_integration import (
    DEFAULT_VERL_CONDITION_ORDER,
    VERL_CONDITION_IDS,
    VerlFCOPDTensors,
    online_batch_output_to_verl_tensors,
    online_sample_outputs_to_verl_tensors,
)
from .verl_post_rollout_hook import fc_opd_post_rollout_hook
from .verl_sparse_kd import VerlSparseKDOutput, compute_verl_sparse_reverse_kl, compute_verl_sparse_topk_kd

__all__ = [
    "ByteTokenizer",
    "ChunkMasks",
    "Condition",
    "ConditionInputs",
    "FCOPDLossConfig",
    "DEFAULT_VERL_CONDITION_ORDER",
    "HFStudentProvider",
    "MinTrainReport",
    "RealMinTrainReport",
    "RealMinTrainStep",
    "MinTrainStep",
    "OfflineLossResult",
    "OfflineLossSmokeReport",
    "OfflineRecordTensors",
    "OfflineScoreRecord",
    "OfflineScoringConfig",
    "OfflineScoringResult",
    "OnlineFCOPDBatchOutput",
    "OnlineFCOPDConfig",
    "OnlineFCOPDSample",
    "OnlineFCOPDSampleOutput",
    "OnlineStudentScores",
    "RayStudentScorerProxy",
    "RealStudentConfig",
    "RealStudentResult",
    "RealStudentSmokeReport",
    "RouterConfig",
    "StudentScorer",
    "StudentForwardOutput",
    "TeacherTopK",
    "TeacherClient",
    "TeacherMetadata",
    "TeacherScoreRequest",
    "TeacherScoreResponse",
    "VERL_CONDITION_IDS",
    "VerlFCOPDTensors",
    "VerlSparseKDOutput",
    "build_condition_inputs",
    "build_condition_inputs_for_record",
    "build_ray_student_scorer_proxy",
    "compute_condition_signals",
    "compute_fc_opd_loss",
    "compute_online_fc_opd_batch",
    "compute_verl_sparse_reverse_kl",
    "compute_verl_sparse_topk_kd",
    "condition_inputs_from_record",
    "default_output_dir",
    "detect_tied_parameter_groups",
    "fc_opd_post_rollout_hook",
    "iter_offline_scores",
    "jensen_shannon_topk",
    "lm_head_embed_tied",
    "load_offline_score_records",
    "load_student_responses",
    "load_vision_opd_records",
    "make_smoke_dataset",
    "make_student_logits",
    "offline_record_to_tensors",
    "online_batch_output_to_verl_tensors",
    "online_sample_outputs_to_verl_tensors",
    "parse_response_chunks",
    "run_offline_loss_backward",
    "run_offline_loss_smoke",
    "run_offline_min_train_smoke",
    "render_teacher_prompt",
    "response_logit_slice",
    "route_condition_weights",
    "run_real_student_min_train",
    "run_real_student_record",
    "run_real_student_smoke",
    "score_record",
    "score_teacher_conditions",
    "score_teacher_conditions_multi_sample",
    "sparse_forward_kl",
    "teacher_scores_to_device",
    "tokenizer_fingerprint",
    "write_offline_scores",
]
