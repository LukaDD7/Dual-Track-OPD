"""Offline reachability-proxy exporter and evaluator (user brief 2026-08-14).

Implements the smallest inference/export-only measurement path for the
teacher-prefix reachability study:

    teacher-prefix transport gold  ->  position / cumulative student NLL /
                                       cumulative teacher-Top-100 FKL+tail

The student and teacher checkpoints stay frozen.  No loss, autograd,
optimizer, actor update, Ray training, or router change is introduced.

CLI phases:

    preflight   CPU-only input/hash/schema/join validation
    score       frozen student+teacher forced-forward scoring (one forward
                per model per verified teacher trace, with an explicit
                OOM-safe chunk fallback)
    analyze     CPU-only CSV construction and scalar proxy evaluation
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from dual_track_opd.fc_opd.dataset_signal_audit import hash_token_ids
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint

from .causal_dataset import read_jsonl, sha256_file
from .causal_runtime import RuntimeModels, _prompt_inputs, load_runtime_models, response_chunk_logits
from .diagnostic import build_prompt, extract_image


PRIMARY_COLUMNS = (
    "prompt_id",
    "teacher_trace_id",
    "horizon",
    "rescue_gold",
    "position",
    "cum_student_nll",
    "cum_top100_fkl_tail",
)

DEFAULT_HORIZONS = (64, 128, 256, 512)
DEFAULT_EPSILON = 1.0e-12
DEFAULT_MASS_TOLERANCE = 1.0e-3


@dataclass(frozen=True)
class ProxyConfig:
    output_dir: str
    proposal_dir: str
    k32_run_dir: str
    cohort_dir: str
    pool256_dir: str
    causal_dir: str
    rescue_dir: str
    student_model_path: str
    teacher_model_path: str
    cohort_parquet_path: str | None = None
    student_device: str = "cuda:0"
    teacher_device: str = "cuda:1"
    dtype: str = "bfloat16"
    response_format: str = "legacy_answer"
    top_k: int = 100
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    epsilon: float = DEFAULT_EPSILON
    mass_tolerance: float = DEFAULT_MASS_TOLERANCE
    chunk_size: int = 64
    max_prompts: int | None = None

    def validate(self) -> None:
        if not self.output_dir or not self.proposal_dir or not self.cohort_dir:
            raise ValueError("output/proposal/cohort paths are required")
        if not self.student_model_path or not self.teacher_model_path:
            raise ValueError("student and teacher model paths are required")
        if not 1 <= self.top_k:
            raise ValueError("top_k must be positive")
        if not self.horizons or any(value <= 0 for value in self.horizons):
            raise ValueError("horizons must be positive")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be unique and increasing")
        if not 0 < self.epsilon <= 1e-3:
            raise ValueError("epsilon must be a small positive float")
        if not 0 < self.mass_tolerance <= 1e-2:
            raise ValueError("mass_tolerance must be a small positive float")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.max_prompts is not None and self.max_prompts <= 0:
            raise ValueError("max_prompts must be positive")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _nested(raw: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = raw
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def load_config(path: str | Path, args: argparse.Namespace | None = None) -> ProxyConfig:
    import yaml

    raw = _expand(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def override(name: str, default: Any = None) -> Any:
        if args is not None and getattr(args, name, None) is not None:
            return getattr(args, name)
        return default

    horizons = tuple(
        int(value)
        for value in (override("horizons", None) or _nested(raw, "proxy", "horizons", default=DEFAULT_HORIZONS))
    )
    config = ProxyConfig(
        output_dir=str(override("output_dir", _nested(raw, "output", "dir", default=""))),
        proposal_dir=str(_nested(raw, "data", "proposal_dir", default="")),
        k32_run_dir=str(_nested(raw, "data", "k32_run_dir", default="")),
        cohort_dir=str(_nested(raw, "data", "cohort_dir", default="")),
        cohort_parquet_path=(
            None
            if _nested(raw, "data", "cohort_parquet_path") in (None, "")
            else str(_nested(raw, "data", "cohort_parquet_path"))
        ),
        pool256_dir=str(_nested(raw, "data", "pool256_dir", default="")),
        causal_dir=str(_nested(raw, "data", "causal_dir", default="")),
        rescue_dir=str(_nested(raw, "data", "rescue_dir", default="")),
        student_model_path=str(_nested(raw, "models", "student", default="")),
        teacher_model_path=str(_nested(raw, "models", "teacher", default="")),
        student_device=str(override("student_device", _nested(raw, "hardware", "student_device", default="cuda:0"))),
        teacher_device=str(override("teacher_device", _nested(raw, "hardware", "teacher_device", default="cuda:1"))),
        dtype=str(_nested(raw, "hardware", "dtype", default="bfloat16")),
        response_format=str(_nested(raw, "data", "response_format", default="legacy_answer")),
        top_k=int(_nested(raw, "proxy", "top_k", default=100)),
        horizons=horizons,
        epsilon=float(_nested(raw, "proxy", "epsilon", default=DEFAULT_EPSILON)),
        mass_tolerance=float(_nested(raw, "proxy", "mass_tolerance", default=DEFAULT_MASS_TOLERANCE)),
        chunk_size=int(_nested(raw, "proxy", "chunk_size", default=64)),
        max_prompts=override("max_prompts"),
    )
    config.validate()
    return config


def _git_state(repo: str | Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "", True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _read_jsonl_checked(path: str | Path) -> list[dict[str, Any]]:
    rows = read_jsonl(Path(path))
    if not rows:
        raise ValueError(f"{path}: empty JSONL input")
    return rows


def _sha256(path: str | Path) -> str:
    return sha256_file(Path(path))


def _resolve_output(config: ProxyConfig) -> Path:
    output = Path(config.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    return output


def _canonical_inputs(config: ProxyConfig) -> dict[str, str]:
    root = Path(config.rescue_dir).expanduser()
    return {
        "pool256_summary": str(Path(config.pool256_dir) / "summary.json"),
        "pool256_rollouts": str(Path(config.pool256_dir) / "rollouts.jsonl"),
        "pool256_prompt_support": str(Path(config.pool256_dir) / "prompt_support_summary.jsonl"),
        "pool256_frontier": str(Path(config.pool256_dir) / "frontier_analysis" / "frontier_prompts.jsonl"),
        "k32_validation": str(Path(config.k32_run_dir) / "k32_validation.json"),
        "k32_rollouts": str(Path(config.k32_run_dir) / "rollouts.jsonl"),
        "causal_summary": str(Path(config.causal_dir) / "summary.json"),
        "causal_records": str(Path(config.causal_dir) / "causal_state_records.jsonl"),
        "causal_windows": str(Path(config.causal_dir) / "candidate_windows.csv"),
        "retained_proposals": str(Path(config.proposal_dir) / "retained_proposals.jsonl"),
        "rescue_comparisons": str(root / "rescue_comparisons.jsonl"),
        "minimal_rescue": str(root / "minimal_rescue_prefixes.jsonl"),
        "intervention_units": str(root / "intervention_units.jsonl"),
        "cohort": str(Path(config.cohort_dir) / "cohort.parquet"),
    }


def _selected_teacher_trace(retained_rows: Sequence[Mapping[str, Any]], sample_uid: str) -> dict[str, Any]:
    """Experiment-B rule: lowest numeric reachability_rank, correct + retained_for_fkl."""

    candidates = [
        row
        for row in retained_rows
        if str(row.get("sample_uid")) == sample_uid
        and row.get("correct") is True
        and row.get("retained_for_fkl") is True
    ]
    if not candidates:
        raise ValueError(f"{sample_uid}: no correct retained_for_fkl teacher proposal")
    candidates.sort(
        key=lambda row: (
            int(row.get("reachability_rank") or 10**9),
            int(row.get("proposal_id") or 0),
        )
    )
    return candidates[0]


def teacher_trace_id(trace: Mapping[str, Any]) -> str:
    uid = str(trace.get("sample_uid") or "")
    proposal_id = str(trace.get("proposal_id") or "")
    token_hash = str(trace.get("response_token_hash") or "")
    return f"{uid}:proposal-{proposal_id}:{token_hash[:16]}"


def run_preflight(config: ProxyConfig) -> dict[str, Any]:
    """CPU-only input/hash/schema/join validation.  Never loads models."""

    output = _resolve_output(config)
    inputs = _canonical_inputs(config)
    report: dict[str, Any] = {
        "schema_version": "reachability-proxy-preflight-v1",
        "config": asdict(config),
    }

    files: dict[str, Any] = {}
    for name, path in inputs.items():
        p = Path(path)
        if not p.is_file() or not os.access(p, os.R_OK):
            files[name] = {"path": path, "readable": False}
            raise FileNotFoundError(f"{name}: missing or unreadable: {path}")
        info: dict[str, Any] = {"path": str(p), "readable": True, "bytes": p.stat().st_size}
        if p.suffix in (".jsonl", ".json"):
            info["sha256"] = _sha256(p)
            info["lines"] = sum(1 for _ in p.open(encoding="utf-8"))
        elif p.suffix == ".parquet":
            info["sha256"] = _sha256(p)
        files[name] = info
    report["files"] = files

    coverage: dict[str, Any] = {}
    pool_rollouts = _read_jsonl_checked(inputs["pool256_rollouts"])
    coverage["pool256_rollout_rows"] = len(pool_rollouts)
    coverage["pool256_unique_prompts"] = len({str(r["sample_uid"]) for r in pool_rollouts})
    support = _read_jsonl_checked(inputs["pool256_prompt_support"])
    coverage["pool256_prompt_support_rows"] = len(support)
    frontier = _read_jsonl_checked(inputs["pool256_frontier"])
    coverage["pool256_frontier_rows"] = len(frontier)
    coverage["pool256_frontier_strata"] = dict(
        Counter(str(r.get("observed_support_stratum")) for r in frontier)
    )

    k32 = json.loads(Path(inputs["k32_validation"]).read_text(encoding="utf-8"))
    coverage["k32_valid"] = k32.get("valid")
    coverage["k32_num_prompts"] = k32.get("num_prompts")
    coverage["k32_rollout_rows"] = k32.get("rollout_rows")
    k32_rollouts = _read_jsonl_checked(inputs["k32_rollouts"])
    coverage["k32_actual_rollout_rows"] = len(k32_rollouts)
    coverage["k32_actual_unique_prompts"] = len({str(r["sample_uid"]) for r in k32_rollouts})

    causal_summary = json.loads(Path(inputs["causal_summary"]).read_text(encoding="utf-8"))
    causal_records = _read_jsonl_checked(inputs["causal_records"])
    coverage["causal_prompt_count"] = causal_summary.get("prompt_count")
    coverage["causal_record_count"] = causal_summary.get("record_count")
    coverage["causal_records_prompt_ids"] = len({str(r["prompt_id"]) for r in causal_records})
    trajectory_ids = Counter(str(r.get("trajectory_id") or "") for r in causal_records)
    coverage["causal_trajectory_ids"] = len(trajectory_ids)
    coverage["causal_duplicate_trajectory_ids"] = [k for k, v in trajectory_ids.items() if v > 1]
    coverage["causal_trajectory_outcomes"] = dict(
        Counter(bool(r.get("trajectory_correct")) for r in causal_records)
    )

    retained = _read_jsonl_checked(inputs["retained_proposals"])
    coverage["retained_rows"] = len(retained)
    coverage["retained_correct"] = sum(1 for r in retained if r.get("correct") is True)
    coverage["retained_retained_for_fkl"] = sum(1 for r in retained if r.get("retained_for_fkl") is True)
    coverage["retained_missing_token_ids_or_hash"] = sum(
        1 for r in retained if not r.get("response_token_ids") or not r.get("response_token_hash")
    )
    coverage["retained_duplicate_identity"] = [
        key
        for key, count in Counter(
            (str(r.get("sample_uid")), str(r.get("proposal_id")), str(r.get("response_token_hash")))
            for r in retained
        ).items()
        if count > 1
    ]

    rescue = _read_jsonl_checked(inputs["rescue_comparisons"])
    coverage["rescue_rows"] = len(rescue)
    coverage["rescue_prompts"] = len({str(r["sample_uid"]) for r in rescue})
    coverage["rescue_rows_meeting_rule"] = sum(
        1 for r in rescue if r.get("meets_preregistered_rescue_rule") is True
    )
    coverage["rescue_duplicate_prompt_horizon"] = [
        key
        for key, count in Counter((str(r["sample_uid"]), int(r["horizon"])) for r in rescue).items()
        if count > 1
    ]
    minimal = _read_jsonl_checked(inputs["minimal_rescue"])
    coverage["minimal_rescue_rows"] = len(minimal)
    coverage["minimal_prompts"] = sorted(str(r["sample_uid"]) for r in minimal)
    units = _read_jsonl_checked(inputs["intervention_units"])
    coverage["intervention_units_rows"] = len(units)
    coverage["intervention_units_prompts"] = len({str(r["sample_uid"]) for r in units})

    if coverage["pool256_unique_prompts"] != 256:
        raise ValueError("256-pool rollout coverage is not 256 prompts")
    if coverage["pool256_rollout_rows"] != 2304:
        raise ValueError("256-pool rollout rows != 2304")
    if coverage["k32_valid"] is not True or coverage["k32_num_prompts"] != 64:
        raise ValueError("K=32 validation is not valid/64 prompts")
    if coverage["causal_prompt_count"] != 64 or coverage["causal_record_count"] != 101:
        raise ValueError("causal probe coverage is not 64 prompts / 101 records")
    if coverage["causal_duplicate_trajectory_ids"]:
        raise ValueError("causal records contain duplicate trajectory IDs")
    if coverage["rescue_prompts"] != 12:
        raise ValueError("rescue comparisons do not cover exactly 12 prompts")
    if coverage["minimal_rescue_rows"] != 7:
        raise ValueError("minimal rescue prefixes do not reproduce 7 positives")
    if coverage["rescue_duplicate_prompt_horizon"]:
        raise ValueError("rescue comparisons contain duplicate (prompt, horizon) keys")
    if coverage["retained_duplicate_identity"]:
        raise ValueError("retained proposals contain duplicate trace identity")
    report["coverage"] = coverage

    join: dict[str, Any] = {}
    for uid in sorted({str(r["sample_uid"]) for r in rescue}):
        trace = _selected_teacher_trace(retained, uid)
        token_hash = str(trace.get("response_token_hash") or "")
        unit_hashes = {
            str(u.get("source_response_token_hash")) for u in units if str(u.get("sample_uid")) == uid
        }
        join[uid] = {
            "teacher_trace_id": teacher_trace_id(trace),
            "selected_proposal_id": trace.get("proposal_id"),
            "reachability_rank": trace.get("reachability_rank"),
            "trace_token_count": len(trace.get("response_token_ids") or ()),
            "response_token_hash_prefix": token_hash[:16],
            "response_token_hash_present_in_intervention": token_hash in unit_hashes,
            "rescue_horizons": sorted({int(r["horizon"]) for r in rescue if str(r["sample_uid"]) == uid}),
        }
        if token_hash not in unit_hashes:
            raise ValueError(f"{uid}: selected trace hash absent from intervention records")
    report["join_selected_trace"] = join

    k32_tokenizer_hashes = {
        str(r.get("student_tokenizer_hash") or r.get("tokenizer_hash") or "")
        for r in k32_rollouts
    } - {""}
    report["k32_tokenizer_hash"] = (
        next(iter(k32_tokenizer_hashes)) if len(k32_tokenizer_hashes) == 1 else None
    )
    report["models"] = {
        "student": _model_identity(config.student_model_path),
        "teacher": _model_identity(config.teacher_model_path),
    }

    commit, dirty = _git_state(_repo_root())
    report["repo"] = {"commit": commit, "dirty": dirty}
    report_path = output / "preflight_report.json"
    _write_json_atomic(report_path, report)
    return report


def _model_identity(model_path: str) -> dict[str, Any]:
    path = Path(os.path.expandvars(model_path)).expanduser()
    identity: dict[str, Any] = {"path": str(path), "is_local": path.exists()}
    if path.is_dir():
        for name in ("config.json", "generation_config.json", "tokenizer_config.json"):
            candidate = path / name
            if candidate.is_file():
                identity[f"{name}_sha256"] = _sha256(candidate)
    return identity


# ---------------------------------------------------------------------------
# Pure scalar math (unit-tested without models)
# ---------------------------------------------------------------------------


def coarse_topk_tail_fkl(
    *,
    q_logp_topk: Sequence[float],
    q_tail_logp: float,
    p_logp_topk: Sequence[float],
    p_tail_prob: float,
    epsilon: float,
) -> float:
    """Coarse Top-100-plus-one-tail-bucket forward KL (never exact vocab KL).

    ``q`` is the teacher's Top-100 distribution plus its tail mass; ``p`` is
    the student's probability mass on the same teacher Top-100 IDs plus the
    student tail ``1 - sum(p_i)``.  The student tail probability is clamped at
    ``epsilon`` only for the log; neither tail is renormalized away.
    """

    q_logp = [float(value) for value in q_logp_topk]
    p_logp = [float(value) for value in p_logp_topk]
    if len(q_logp) != len(p_logp):
        raise ValueError("teacher and student Top-K log-prob lists must have equal length")
    if not q_logp:
        raise ValueError("Top-K log-prob lists must be non-empty")
    q_tail = math.exp(float(q_tail_logp))
    p_tail = max(float(p_tail_prob), epsilon)
    total = 0.0
    for q_lp, p_lp in zip(q_logp, p_logp):
        q_i = math.exp(q_lp)
        total += q_i * (q_lp - p_lp)
    total += q_tail * (float(q_tail_logp) - math.log(p_tail))
    return total


def horizon_aggregates(
    *,
    token_rows: Sequence[Mapping[str, Any]],
    trace_length: int,
    horizons: Sequence[int],
) -> list[dict[str, Any]]:
    """Aggregate per-token scalars into cumulative horizon rows."""

    if trace_length <= 0:
        raise ValueError("trace_length must be positive")
    if any(int(row["position_index"]) != index for index, row in enumerate(token_rows)):
        raise ValueError("token rows must be contiguous and start at position 0")
    if len(token_rows) != trace_length:
        raise ValueError("token rows do not cover the full teacher trace")
    nll_cum = 0.0
    fkl_cum = 0.0
    rows: list[dict[str, Any]] = []
    horizon_set = sorted(int(value) for value in set(horizons))
    by_position = {int(row["position_index"]): row for row in token_rows}
    for t in range(trace_length):
        row = by_position[t]
        nll_cum += float(row["student_nll"])
        fkl_cum += float(row["d_coarse_fkl"])
        if (t + 1) in horizon_set:
            h = t + 1
            rows.append(
                {
                    "horizon": h,
                    "position": h / trace_length,
                    "cum_student_nll": nll_cum,
                    "cum_top100_fkl_tail": fkl_cum,
                    "token_count": h,
                }
            )
    return rows


def build_proxy_study_rows(
    *,
    token_rows: Sequence[Mapping[str, Any]],
    rescue_rows: Sequence[Mapping[str, Any]],
    selected_traces: Mapping[str, Mapping[str, Any]],
    horizons: Sequence[int],
) -> list[dict[str, Any]]:
    """Strict one-to-one join of scored horizon rows with rescue gold."""

    horizon_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for trace_uid, rows in _group_by_prompt(token_rows).items():
        if trace_uid not in selected_traces:
            # Pre-scored extras without rescue gold are not part of this study.
            continue
        trace = selected_traces[trace_uid]
        trace_length = len(trace.get("response_token_ids") or ())
        for aggregate in horizon_aggregates(
            token_rows=rows, trace_length=trace_length, horizons=horizons
        ):
            horizon_rows[(trace_uid, int(aggregate["horizon"]))] = aggregate

    study: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for rescue in rescue_rows:
        uid = str(rescue["sample_uid"])
        horizon = int(rescue["horizon"])
        key = (uid, horizon)
        if key in seen:
            raise ValueError(f"{uid}: duplicate rescue (prompt, horizon) key at {horizon}")
        seen.add(key)
        if key not in horizon_rows:
            raise ValueError(f"{uid}: scored horizon {horizon} missing for rescue row")
        aggregate = horizon_rows[key]
        trace = selected_traces[uid]
        row = {
            "prompt_id": uid,
            "teacher_trace_id": teacher_trace_id(trace),
            "horizon": horizon,
            "rescue_gold": bool(rescue.get("meets_preregistered_rescue_rule")),
            "position": float(aggregate["position"]),
            "cum_student_nll": float(aggregate["cum_student_nll"]),
            "cum_top100_fkl_tail": float(aggregate["cum_top100_fkl_tail"]),
            "teacher_trace_length": int(len(trace.get("response_token_ids") or ())),
            "teacher_trace_hash": str(trace.get("response_token_hash") or ""),
            "token_count": int(aggregate["token_count"]),
            "rescue_row_source": "rescue_comparisons.jsonl",
        }
        study.append(row)
    study.sort(key=lambda row: (row["prompt_id"], row["horizon"]))
    return study


def _group_by_prompt(token_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in token_rows:
        grouped[str(row["prompt_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["position_index"]))
    return grouped


def write_proxy_study_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise ValueError("proxy study rows are empty")
    columns = list(PRIMARY_COLUMNS) + [
        column for column in rows[0] if column not in PRIMARY_COLUMNS
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return _sha256(path)


def _auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    positives = sum(1 for _, label in pairs if label)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum = 0.0
    for index, (_, label) in enumerate(pairs, start=1):
        if label:
            rank_sum += index
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def tie_correct_auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Tie-correct (Mann-Whitney) AUROC; ties share the average rank."""

    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    positives = sum(1 for _, label in pairs if label)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(1 for _, label in pairs[index:end] if label)
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def fit_l2_logistic(
    features: Sequence[Sequence[float]],
    labels: Sequence[bool],
    *,
    l2: float = 1.0,
    iterations: int = 50,
) -> tuple[list[float], float]:
    """L2-regularized logistic regression via iteratively reweighted least squares."""

    import numpy as np

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("features and labels must align")
    n, d = X.shape
    if n < 2 or d < 1:
        raise ValueError("need at least two rows and one feature")
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Z = (X - mean) / std
    weights = np.ones(n)
    beta = np.zeros(d + 1)
    for _ in range(iterations):
        design = np.concatenate([np.ones((n, 1)), Z], axis=1)
        eta = design @ beta
        eta = np.clip(eta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-9, None)
        W = np.diag(w)
        reg = np.eye(d + 1)
        reg[0, 0] = 0.0
        H = design.T @ W @ design + l2 * reg
        grad = design.T @ (y - p)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta += step
    scale = list((beta[1:] / std).tolist())
    intercept = float(beta[0] - np.sum(beta[1:] * mean / std))
    return scale, intercept


def logistic_predict_proba(
    features: Sequence[Sequence[float]],
    weights: Sequence[float],
    intercept: float,
) -> list[float]:
    """Sigmoid predictions from L2-logistic weights (standardized at fit time)."""

    import numpy as np

    X = np.asarray(features, dtype=np.float64)
    eta = X @ np.asarray(weights) + intercept
    return (1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))).tolist()


def cluster_bootstrap_metrics(
    prompt_ids: Sequence[str],
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    seed: int,
    resamples: int = 2000,
) -> dict[str, list[float | None]]:
    """Prompt-cluster bootstrap: resample prompts, carrying all their rows."""

    import random

    rng = random.Random(seed)
    grouped: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for uid, score, label in zip(prompt_ids, scores, labels):
        grouped[str(uid)].append((float(score), bool(label)))
    keys = sorted(grouped)
    auroc_values: list[float] = []
    auprc_values: list[float] = []
    for _ in range(resamples):
        sampled_scores: list[float] = []
        sampled_labels: list[bool] = []
        for _ in range(len(keys)):
            for score, label in grouped[rng.choice(keys)]:
                sampled_scores.append(score)
                sampled_labels.append(label)
        if not any(sampled_labels) or all(sampled_labels):
            continue
        auroc_values.append(tie_correct_auroc(sampled_scores, sampled_labels))
        auprc_values.append(_auprc(sampled_scores, sampled_labels))
    return {
        "auroc_ci95": _percentile_ci(auroc_values),
        "auprc_ci95": _percentile_ci(auprc_values),
    }


def grouped_cv_auroc(
    prompt_ids: Sequence[str],
    features: Sequence[Sequence[float]],
    labels: Sequence[bool],
    *,
    n_folds: int = 5,
    repeats: int = 5,
    seed: int = 20260815,
    fit_model: bool = False,
) -> dict[str, Any]:
    """Grouped stratified repeated CV; prompts never split across folds."""

    import random

    rng = random.Random(seed)
    by_prompt: dict[str, list[tuple[list[float], bool]]] = defaultdict(list)
    for uid, feature_row, label in zip(prompt_ids, features, labels):
        by_prompt[str(uid)].append((list(feature_row), bool(label)))
    keys = sorted(by_prompt)
    fold_aurocs: list[float] = []
    fold_auprcs: list[float] = []
    for repeat in range(repeats):
        shuffled = list(keys)
        rng.shuffle(shuffled)
        folds: list[list[str]] = [[] for _ in range(n_folds)]
        for index, uid in enumerate(shuffled):
            folds[index % n_folds].append(uid)
        for fold_index in range(n_folds):
            test_uids = set(folds[fold_index])
            train_uids = [uid for uid in keys if uid not in test_uids]
            train_features = [
                row for uid in train_uids for row, _ in by_prompt[uid]
            ]
            train_labels = [label for uid in train_uids for _, label in by_prompt[uid]]
            test_features = [row for uid in sorted(test_uids) for row, _ in by_prompt[uid]]
            test_labels = [label for uid in sorted(test_uids) for _, label in by_prompt[uid]]
            if fit_model:
                if not any(train_labels) or all(train_labels) or not test_labels:
                    continue
                weights, intercept = fit_l2_logistic(train_features, train_labels)
                test_scores = logistic_predict_proba(test_features, weights, intercept)
            else:
                test_scores = [row[0] for row in test_features]
            fold_aurocs.append(tie_correct_auroc(test_scores, test_labels))
            fold_auprcs.append(_auprc(test_scores, test_labels))
    return {
        "n_folds": n_folds,
        "repeats": repeats,
        "mean_heldout_auroc": float(sum(fold_aurocs) / len(fold_aurocs)) if fold_aurocs else None,
        "mean_heldout_auprc": float(sum(fold_auprcs) / len(fold_auprcs)) if fold_auprcs else None,
        "fold_count": len(fold_aurocs),
    }


def _auprc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: -item[0])
    if not any(labels):
        return float("nan")
    tp = 0
    precision_sum = 0.0
    for index, (_, label) in enumerate(pairs, start=1):
        if label:
            tp += 1
            precision_sum += tp / index
    return precision_sum / sum(labels)


def _prompt_level_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[list[float], list[bool]]]:
    """One fixed-orientation score per prompt; higher = more scaffoldable."""

    by_prompt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row["prompt_id"])].append(row)
    result: dict[str, tuple[list[float], list[bool]]] = {
        "position": ([], []),
        "cum_student_nll": ([], []),
        "cum_top100_fkl_tail": ([], []),
    }
    for prompt_rows in by_prompt.values():
        scaffoldable = any(bool(row["rescue_gold"]) for row in prompt_rows)
        if scaffoldable:
            minimal = min(
                (row for row in prompt_rows if row["rescue_gold"]),
                key=lambda row: row["horizon"],
            )
            chosen = minimal
        else:
            chosen = max(prompt_rows, key=lambda row: row["horizon"])
        for name in result:
            value = float(chosen[name])
            result[name][0].append(-value)
            result[name][1].append(scaffoldable)
    return result


def _within_prompt_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    by_prompt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row["prompt_id"])].append(row)
    metrics: dict[str, dict[str, float]] = {
        "position": {"recall_at_1": 0.0, "plus_minus_one_bin": 0.0, "mean_abs_bin_distance": 0.0, "pairwise_agreement": 0.0, "n_prompts": 0.0},
        "cum_student_nll": {"recall_at_1": 0.0, "plus_minus_one_bin": 0.0, "mean_abs_bin_distance": 0.0, "pairwise_agreement": 0.0, "n_prompts": 0.0},
        "cum_top100_fkl_tail": {"recall_at_1": 0.0, "plus_minus_one_bin": 0.0, "mean_abs_bin_distance": 0.0, "pairwise_agreement": 0.0, "n_prompts": 0.0},
    }
    for prompt_rows in by_prompt.values():
        scaffoldable = [row for row in prompt_rows if row["rescue_gold"]]
        if not scaffoldable:
            continue
        h_star = min(row["horizon"] for row in scaffoldable)
        ordered = sorted(prompt_rows, key=lambda row: row["horizon"])
        horizons = [row["horizon"] for row in ordered]
        star_index = horizons.index(h_star)
        for name in metrics:
            scalar_by_h = {row["horizon"]: float(row[name]) for row in ordered}
            ranked = sorted(scalar_by_h.items(), key=lambda item: item[1])
            rank_of_star = next(index for index, (h, _) in enumerate(ranked) if h == h_star)
            metrics[name]["recall_at_1"] += 1.0 if rank_of_star == 0 else 0.0
            metrics[name]["plus_minus_one_bin"] += 1.0 if rank_of_star <= 1 else 0.0
            metrics[name]["mean_abs_bin_distance"] += float(rank_of_star)
            agreements = 0.0
            for other_h in horizons:
                if other_h == h_star:
                    continue
                agreements += 1.0 if scalar_by_h[h_star] <= scalar_by_h[other_h] else 0.0
            metrics[name]["pairwise_agreement"] += agreements / max(1, len(horizons) - 1)
            metrics[name]["n_prompts"] += 1.0
    for name in metrics:
        count = metrics[name]["n_prompts"]
        if count > 0:
            for key in ("recall_at_1", "plus_minus_one_bin", "pairwise_agreement"):
                metrics[name][key] /= count
            metrics[name]["mean_abs_bin_distance"] /= count
    return metrics


def evaluate_proxy_study(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    """Prompt-level AUROC/AUPRC with prompt bootstrap plus within-prompt ranking."""

    prompt_scores = _prompt_level_scores(rows)
    within = _within_prompt_metrics(rows)
    rng = __import__("random").Random(seed)
    prompt_ids = sorted({str(row["prompt_id"]) for row in rows})
    evaluation: dict[str, Any] = {}
    for name, (scores, labels) in prompt_scores.items():
        point = {"auroc": _auroc(scores, labels), "auprc": _auprc(scores, labels)}
        bootstrap = {"auroc": [], "auprc": []}
        for _ in range(bootstrap_resamples):
            sampled = [rng.randrange(len(scores)) for _ in range(len(scores))]
            boot_scores = [scores[index] for index in sampled]
            boot_labels = [labels[index] for index in sampled]
            if not any(boot_labels) or all(boot_labels):
                continue
            bootstrap["auroc"].append(_auroc(boot_scores, boot_labels))
            bootstrap["auprc"].append(_auprc(boot_scores, boot_labels))
        evaluation[name] = {
            "point": point,
            "bootstrap_ci95": {
                "auroc": _percentile_ci(bootstrap["auroc"]),
                "auprc": _percentile_ci(bootstrap["auprc"]),
            },
            "n_prompts": len(prompt_ids),
        }
    evaluation["within_prompt"] = within
    evaluation["prompt_ids"] = prompt_ids
    evaluation["n_rows"] = len(rows)
    return evaluation


def _percentile_ci(values: Sequence[float]) -> list[float | None]:
    if not values:
        return [None, None]
    ordered = sorted(float(value) for value in values if math.isfinite(value))
    if not ordered:
        return [None, None]
    lower = ordered[int(0.025 * (len(ordered) - 1))]
    upper = ordered[int(0.975 * (len(ordered) - 1))]
    return [lower, upper]


def _horizon_bin(horizon: int, horizons: Sequence[int]) -> int:
    return list(horizons).index(horizon)


def _fit_horizon_tau(
    rows_by_prompt: Mapping[str, Sequence[Mapping[str, Any]]],
    scaffoldable_prompts: Sequence[str],
    scalar: str,
    horizons: Sequence[int],
) -> float:
    """Calibrate one scalar threshold on training prompts (min mean bin distance)."""

    values = sorted(
        {
            float(row[scalar])
            for uid in scaffoldable_prompts
            for row in rows_by_prompt[uid]
        }
    )
    if not values:
        raise ValueError("no training scalar values for threshold calibration")
    grid = values
    if len(grid) > 40:
        step = (len(grid) - 1) / 39
        grid = [grid[int(round(index * step))] for index in range(40)]

    def mean_bin_distance(tau: float) -> float:
        total = 0.0
        for uid in scaffoldable_prompts:
            rows = rows_by_prompt[uid]
            h_star = min(int(row["horizon"]) for row in rows if row["rescue_gold"])
            scalar_by_h = {int(row["horizon"]): float(row[scalar]) for row in rows}
            h_hat = min(scalar_by_h, key=lambda h: abs(scalar_by_h[h] - tau))
            total += abs(_horizon_bin(h_hat, horizons) - _horizon_bin(h_star, horizons))
        return total / max(1, len(scaffoldable_prompts))

    return min(grid, key=mean_bin_distance)


def evaluate_proxy_heldout(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    test_fraction: float = 0.4,
) -> dict[str, Any]:
    """Formal §9 evaluation: fit thresholds on train prompts, score test prompts."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie in (0, 1)")
    prompt_ids = sorted({str(row["prompt_id"]) for row in rows})
    rng = __import__("random").Random(seed)
    shuffled = list(prompt_ids)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    test_set = set(shuffled[:n_test])
    train_set = set(shuffled[n_test:])
    horizons = sorted({int(row["horizon"]) for row in rows})
    by_prompt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row["prompt_id"])].append(row)
    scaffoldable_train = [
        uid for uid in sorted(train_set) if any(row["rescue_gold"] for row in by_prompt[uid])
    ]
    result: dict[str, Any] = {
        "seed": seed,
        "test_fraction": test_fraction,
        "train_prompts": sorted(train_set),
        "test_prompts": sorted(test_set),
    }
    for name in ("position", "cum_student_nll", "cum_top100_fkl_tail"):
        tau = (
            _fit_horizon_tau(by_prompt, scaffoldable_train, name, horizons)
            if scaffoldable_train
            else None
        )
        scores: list[float] = []
        labels: list[bool] = []
        for uid in sorted(test_set):
            prompt_rows = by_prompt[uid]
            scaffoldable = any(row["rescue_gold"] for row in prompt_rows)
            chosen = (
                min((row for row in prompt_rows if row["rescue_gold"]), key=lambda row: row["horizon"])
                if scaffoldable
                else max(prompt_rows, key=lambda row: row["horizon"])
            )
            scores.append(-float(chosen[name]))
            labels.append(scaffoldable)
        recall_at_1 = plus_minus_one = bin_distance = 0.0
        n_within = 0
        if tau is not None:
            for uid in sorted(test_set):
                prompt_rows = by_prompt[uid]
                gold_rows = [row for row in prompt_rows if row["rescue_gold"]]
                if not gold_rows:
                    continue
                h_star = min(int(row["horizon"]) for row in gold_rows)
                scalar_by_h = {int(row["horizon"]): float(row[name]) for row in prompt_rows}
                h_hat = min(scalar_by_h, key=lambda h: abs(scalar_by_h[h] - tau))
                distance = abs(_horizon_bin(h_hat, horizons) - _horizon_bin(h_star, horizons))
                recall_at_1 += 1.0 if distance == 0 else 0.0
                plus_minus_one += 1.0 if distance <= 1 else 0.0
                bin_distance += distance
                n_within += 1
        result[name] = {
            "tau": tau,
            "test_auroc": _auroc(scores, labels),
            "test_auprc": _auprc(scores, labels),
            "within_prompt": {
                "n_prompts": n_within,
                "recall_at_1": recall_at_1 / n_within if n_within else None,
                "plus_minus_one_bin": plus_minus_one / n_within if n_within else None,
                "mean_abs_bin_distance": bin_distance / n_within if n_within else None,
            },
        }
    return result


# ---------------------------------------------------------------------------
# GPU scoring
# ---------------------------------------------------------------------------


def _cohort_frame(cohort_dir: str, cohort_parquet_path: str | None = None):
    import pandas as pd

    cohort_path = (
        Path(cohort_parquet_path).expanduser().resolve()
        if cohort_parquet_path
        else Path(cohort_dir) / "cohort.parquet"
    )
    frame = pd.read_parquet(cohort_path).copy()
    frame.index = frame["sample_uid"].astype(str)
    if frame.index.duplicated().any():
        raise ValueError("cohort contains duplicate sample_uid")
    return frame


def _prompt_inputs_checked(
    models: RuntimeModels,
    *,
    image,
    prompt_text: str,
    student_device: str,
    teacher_device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], str]:
    student_inputs = _prompt_inputs(models.student_processor, image, prompt_text)
    teacher_inputs = _prompt_inputs(models.teacher_processor, image, prompt_text)
    if not torch.equal(
        student_inputs["input_ids"].detach().cpu(),
        teacher_inputs["input_ids"].detach().cpu(),
    ):
        raise RuntimeError("teacher/student rendered prompt IDs differ")
    prompt_hash = hash_token_ids(
        tuple(int(value) for value in student_inputs["input_ids"][0].cpu().tolist())
    )
    return student_inputs, teacher_inputs, prompt_hash


def _full_logits(model: Any, inputs: Mapping[str, torch.Tensor], response_ids: Sequence[int], device: str, chunk_size: int):
    """One forward per model; explicit OOM-safe per-chunk fallback."""

    try:
        logits, used_keep = response_chunk_logits(
            model,
            inputs,
            response_ids,
            start=0,
            end=len(response_ids),
            device=device,
        )
        return logits, used_keep, "full"
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        chunks = []
        used_keep = True
        for start in range(0, len(response_ids), chunk_size):
            end = min(len(response_ids), start + chunk_size)
            chunk, keep = response_chunk_logits(
                model,
                inputs,
                response_ids,
                start=start,
                end=end,
                device=device,
            )
            chunks.append(chunk)
            used_keep = used_keep and keep
        return torch.cat(chunks, dim=0), used_keep, "chunked"


def _score_trace(
    models: RuntimeModels,
    *,
    image,
    prompt_text: str,
    trace: Mapping[str, Any],
    config: ProxyConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    response_ids = tuple(int(value) for value in trace["response_token_ids"])
    if not response_ids:
        raise ValueError("teacher trace has no response token IDs")
    expected_hash = str(trace.get("response_token_hash") or "")
    actual_hash = hash_token_ids(response_ids)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError("teacher trace response_token_hash does not match its token IDs")

    student_inputs, teacher_inputs, prompt_hash = _prompt_inputs_checked(
        models,
        image=image,
        prompt_text=prompt_text,
        student_device=config.student_device,
        teacher_device=config.teacher_device,
    )
    student_logits, student_keep, student_mode = _full_logits(
        models.student_model, student_inputs, response_ids, config.student_device, config.chunk_size
    )
    teacher_logits, teacher_keep, teacher_mode = _full_logits(
        models.teacher_model, teacher_inputs, response_ids, config.teacher_device, config.chunk_size
    )
    if int(student_logits.shape[0]) != len(response_ids) or int(teacher_logits.shape[0]) != len(response_ids):
        raise RuntimeError("forced-forward logits do not align with teacher trace length")

    trace_id = teacher_trace_id(trace)
    rows: list[dict[str, Any]] = []
    ids = torch.tensor(response_ids, dtype=torch.long, device=config.student_device)
    for start in range(0, len(response_ids), config.chunk_size):
        end = min(len(response_ids), start + config.chunk_size)
        student_chunk = student_logits[start:end]
        teacher_chunk = teacher_logits[start:end]
        student_logp = torch.log_softmax(student_chunk.float(), dim=-1)
        teacher_logp = torch.log_softmax(teacher_chunk.float(), dim=-1)

        stats = _per_token_symmetric_stats(
            teacher_logp=teacher_logp,
            student_logp=student_logp,
            response_token_ids=ids[start:end],
            top_k=config.top_k,
            epsilon=config.epsilon,
            mass_tolerance=config.mass_tolerance,
        )

        for offset in range(end - start):
            position = start + offset
            token_id = int(ids[start + offset].item())
            token_stats = stats[offset]
            rows.append(
                {
                    "prompt_id": str(trace["sample_uid"]),
                    "teacher_trace_id": trace_id,
                    "position_index": position,
                    "token_id": token_id,
                    "student_nll": float(-token_stats["student_sampled_logp"]),
                    "d_coarse_fkl": float(token_stats["fkl_teacher_support"]),
                    "teacher_q_top100_ids": tuple(int(value) for value in token_stats["teacher_top100_ids"]),
                    "teacher_q_logp_top100": tuple(float(value) for value in token_stats["teacher_top100_logp"]),
                    "teacher_q_tail_logp": float(token_stats["teacher_tail_logp"]),
                    "student_p_logp_top100_at_teacher_ids": tuple(
                        float(value) for value in token_stats["student_logp_at_teacher_ids"]
                    ),
                    "student_p_tail_prob": float(token_stats["student_tail_on_teacher_support"]),
                    "teacher_mass_residual": float(token_stats["teacher_mass_residual"]),
                    "student_mass_residual": float(token_stats["student_mass_residual"]),
                    "student_top100_ids": tuple(int(value) for value in token_stats["student_top100_ids"]),
                    "student_top100_logp": tuple(float(value) for value in token_stats["student_top100_logp"]),
                    "teacher_logp_at_student_top100_ids": tuple(
                        float(value) for value in token_stats["teacher_logp_at_student_ids"]
                    ),
                    "student_tail_on_student_support": float(
                        token_stats["student_tail_on_student_support"]
                    ),
                    "teacher_tail_on_student_support": float(
                        token_stats["teacher_tail_on_student_support"]
                    ),
                    "teacher_entropy": float(token_stats["teacher_entropy"]),
                    "student_entropy": float(token_stats["student_entropy"]),
                    "teacher_top1_top2_margin": float(token_stats["teacher_top1_top2_margin"]),
                    "student_top1_top2_margin": float(token_stats["student_top1_top2_margin"]),
                    "C_h_16": float(token_stats["teacher_mass_on_student_top16"]),
                    "O_h_16": float(token_stats["overlap_top16"]),
                }
            )
        del student_chunk, teacher_chunk, student_logp, teacher_logp
    del student_logits, teacher_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    audit = {
        "trace_id": trace_id,
        "sample_uid": str(trace["sample_uid"]),
        "proposal_id": trace.get("proposal_id"),
        "trace_length": len(response_ids),
        "prompt_token_hash": prompt_hash,
        "response_token_hash": actual_hash,
        "scoring_mode": f"student={student_mode};teacher={teacher_mode}",
        "logits_to_keep": bool(student_keep and teacher_keep),
    }
    return rows, audit


def _per_token_symmetric_stats(
    *,
    teacher_logp: torch.Tensor,
    student_logp: torch.Tensor,
    response_token_ids: torch.Tensor,
    top_k: int,
    epsilon: float,
    mass_tolerance: float,
) -> list[dict[str, Any]]:
    """Per-token symmetric Top-K stats (pure, torch; unit-testable).

    One teacher and one student log-probability matrix per token, aligned at
    the same response positions.  Returns every field needed by the symmetric
    cache schema plus mass-conservation checks.
    """

    if teacher_logp.shape != student_logp.shape:
        raise ValueError("teacher/student log-prob shapes must match")
    length = int(teacher_logp.shape[0])
    if int(response_token_ids.shape[0]) != length:
        raise ValueError("response token IDs must align with log-prob rows")

    teacher_values, teacher_indices = torch.topk(teacher_logp, k=top_k, dim=-1)
    student_values, student_indices = torch.topk(student_logp, k=top_k, dim=-1)
    teacher_topk_mass = teacher_values.exp().sum(dim=-1)
    teacher_tail = torch.clamp(1.0 - teacher_topk_mass, min=epsilon)
    teacher_tail_logp = teacher_tail.log()

    sampled = student_logp[
        torch.arange(length, device=student_logp.device), response_token_ids
    ]
    teacher_ids_on_student = teacher_indices.to(student_logp.device)
    student_at_teacher_ids = student_logp.gather(dim=-1, index=teacher_ids_on_student)
    student_mass_on_teacher = student_at_teacher_ids.exp().sum(dim=-1)
    student_tail_on_teacher = torch.clamp(
        1.0 - student_mass_on_teacher, min=epsilon
    )

    student_ids_on_teacher = student_indices.to(teacher_logp.device)
    teacher_at_student_ids = teacher_logp.gather(dim=-1, index=student_ids_on_teacher)
    teacher_mass_on_student = teacher_at_student_ids.exp().sum(dim=-1)
    teacher_tail_on_student = torch.clamp(
        1.0 - teacher_mass_on_student, min=epsilon
    )
    student_tail_on_student = torch.clamp(
        1.0 - student_values.exp().sum(dim=-1), min=epsilon
    )

    teacher_on_student = teacher_values.to(student_logp.device)
    teacher_tail_t = teacher_tail.to(student_logp.device)
    teacher_tail_logp_t = teacher_tail_logp.to(student_logp.device)
    q_i = teacher_on_student.exp()
    fkl_teacher_support = (q_i * (teacher_on_student - student_at_teacher_ids)).sum(
        dim=-1
    ) + (teacher_tail_t * (teacher_tail_logp_t - student_tail_on_teacher.log()))

    probs_t = teacher_logp.exp()
    probs_s = student_logp.exp()
    teacher_entropy = -(probs_t * teacher_logp).sum(dim=-1)
    student_entropy = -(probs_s * student_logp).sum(dim=-1)

    teacher_mass_residual = teacher_topk_mass + teacher_tail - 1.0
    student_mass_residual = student_mass_on_teacher + student_tail_on_teacher - 1.0
    if float(teacher_mass_residual.abs().max()) > mass_tolerance:
        raise RuntimeError("teacher Top-K+tail mass conservation failed")
    if float(student_mass_residual.abs().max()) > mass_tolerance:
        raise RuntimeError("student Top-K+tail mass conservation failed")

    teacher_mass_on_student_top16 = (
        teacher_at_student_ids[:, :16].exp().sum(dim=-1)
    )
    overlap_top16 = torch.tensor(
        [
            len(
                set(teacher_indices[t, :16].cpu().tolist())
                & set(student_indices[t, :16].cpu().tolist())
            )
            / 16
            for t in range(length)
        ],
        device=teacher_logp.device,
    )

    stats: list[dict[str, Any]] = []
    for t in range(length):
        stats.append(
            {
                "student_sampled_logp": float(sampled[t].cpu()),
                "fkl_teacher_support": float(fkl_teacher_support[t].cpu()),
                "teacher_top100_ids": teacher_indices[t].cpu().tolist(),
                "teacher_top100_logp": teacher_values[t].cpu().tolist(),
                "teacher_tail_logp": float(teacher_tail_logp[t].cpu()),
                "student_logp_at_teacher_ids": student_at_teacher_ids[t].cpu().tolist(),
                "student_tail_on_teacher_support": float(student_tail_on_teacher[t].cpu()),
                "teacher_mass_residual": float(teacher_mass_residual[t].cpu()),
                "student_mass_residual": float(student_mass_residual[t].cpu()),
                "student_top100_ids": student_indices[t].cpu().tolist(),
                "student_top100_logp": student_values[t].cpu().tolist(),
                "teacher_logp_at_student_ids": teacher_at_student_ids[t].cpu().tolist(),
                "student_tail_on_student_support": float(student_tail_on_student[t].cpu()),
                "teacher_tail_on_student_support": float(teacher_tail_on_student[t].cpu()),
                "teacher_entropy": float(teacher_entropy[t].cpu()),
                "student_entropy": float(student_entropy[t].cpu()),
                "teacher_top1_top2_margin": float(
                    (teacher_values[t, 0] - teacher_values[t, 1]).cpu()
                ),
                "student_top1_top2_margin": float(
                    (student_values[t, 0] - student_values[t, 1]).cpu()
                ),
                "teacher_mass_on_student_top16": float(
                    teacher_mass_on_student_top16[t].cpu()
                ),
                "overlap_top16": float(overlap_top16[t].cpu()),
            }
        )
    return stats


def run_score(
    config: ProxyConfig,
    *,
    prompt_uids: Sequence[str] | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    output = _resolve_output(config)
    lock_name = "score.lock" if num_shards <= 1 else f"score.s{shard_index}.lock"
    with _exclusive_lock(output / lock_name):
        return _run_score_impl(
            config,
            prompt_uids=prompt_uids,
            shard_index=shard_index,
            num_shards=num_shards,
        )


@contextlib.contextmanager
def _exclusive_lock(path: Path):
    """Best-effort exclusive lock so concurrent score runs cannot interleave."""

    handle = None
    try:
        import fcntl

        handle = path.open("w")
        fcntl.flock(handle, fcntl.LOCK_EX)
    except (ImportError, OSError):
        handle = None
    try:
        yield
    finally:
        if handle is not None:
            try:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_UN)
            except Exception:
                pass
            handle.close()


def _run_score_impl(
    config: ProxyConfig,
    *,
    prompt_uids: Sequence[str] | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    output = _resolve_output(config)
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard assignment")
    retained = _read_jsonl_checked(Path(config.proposal_dir) / "retained_proposals.jsonl")
    retained_uids = {str(row["sample_uid"]) for row in retained}
    if prompt_uids:
        requested = set(prompt_uids)
        unknown = requested - retained_uids
        if unknown:
            raise ValueError(f"prompts without retained teacher traces: {sorted(unknown)}")
        selected_uids = sorted(requested)
    else:
        rescue = _read_jsonl_checked(Path(config.rescue_dir) / "rescue_comparisons.jsonl")
        rescue_uids = {str(row["sample_uid"]) for row in rescue}
        selected_uids = sorted(rescue_uids)
        missing_traces = [uid for uid in selected_uids if uid not in retained_uids]
        if missing_traces:
            raise ValueError(f"rescue prompts without retained traces: {missing_traces[:10]}")
    selected_uids = [uid for uid in selected_uids if uid in retained_uids]
    if config.max_prompts is not None:
        selected_uids = selected_uids[: config.max_prompts]
    start = len(selected_uids) * shard_index // num_shards
    end = len(selected_uids) * (shard_index + 1) // num_shards
    selected_uids = selected_uids[start:end]

    frame = _cohort_frame(config.cohort_dir, config.cohort_parquet_path)
    models = load_runtime_models(
        student_model_path=config.student_model_path,
        student_device=config.student_device,
        dtype=config.dtype,
        teacher_model_path=config.teacher_model_path,
        teacher_device=config.teacher_device,
    )
    token_rows_path = output / (
        "proxy_token_rows.jsonl" if num_shards <= 1 else f"proxy_token_rows.s{shard_index}.jsonl"
    )
    expected_traces = {
        uid: _selected_teacher_trace(retained, uid) for uid in selected_uids
    }
    completed = _complete_scored_prompts(token_rows_path, expected_traces)
    pending = [uid for uid in selected_uids if uid not in completed]
    skipped = [uid for uid in selected_uids if uid in completed]
    if pending and token_rows_path.is_file():
        # Re-scoring must be idempotent: drop stale partial rows for prompts
        # that will be (re)scored so appended rows never duplicate positions.
        _purge_scored_rows(token_rows_path, set(pending))
    manifest: dict[str, Any] = {
        "schema_version": "reachability-proxy-score-v1",
        "config": asdict(config),
        "repo": {"commit": _git_state(_repo_root())[0], "dirty": _git_state(_repo_root())[1]},
        "tokenizer_hash": models.tokenizer_hash,
        "model_identity": {
            "student": _model_identity(config.student_model_path),
            "teacher": _model_identity(config.teacher_model_path),
        },
        "prompts": {},
    }
    with token_rows_path.open("w" if not token_rows_path.is_file() else "a", encoding="utf-8") as handle:
        for uid in pending:
            trace = expected_traces[uid]
            cohort_row = frame.loc[uid].to_dict()
            question = str(cohort_row.get("question") or "").strip()
            gold_answer = cohort_row.get("answer")
            prompt_text = build_prompt(question, config.response_format)
            image = extract_image(cohort_row).convert("RGB")
            rows, audit = _score_trace(
                models,
                image=image,
                prompt_text=prompt_text,
                trace=trace,
                config=config,
            )
            audit["gold_answer"] = str(gold_answer)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            manifest["prompts"][uid] = audit
    for uid in skipped:
        manifest["prompts"][uid] = {
            "sample_uid": uid,
            "skipped_existing": True,
            "trace_id": teacher_trace_id(expected_traces[uid]),
        }
    manifest["scored_prompts"] = len(pending)
    manifest["skipped_existing_prompts"] = skipped
    manifest_path = output / (
        "run_manifest.json" if num_shards <= 1 else f"run_manifest.s{shard_index}.json"
    )
    manifest["shard_index"] = shard_index
    manifest["num_shards"] = num_shards
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _purge_scored_rows(token_rows_path: Path, pending_uids: set[str]) -> None:
    kept: list[str] = []
    with token_rows_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if str(json.loads(line).get("prompt_id")) in pending_uids:
                continue
            kept.append(line)
    with token_rows_path.open("w", encoding="utf-8") as handle:
        handle.writelines(kept)


def _complete_scored_prompts(
    token_rows_path: Path,
    expected_traces: Mapping[str, Mapping[str, Any]],
    *,
    require_symmetric_cache: bool = True,
) -> set[str]:
    """Prompts whose token rows fully cover the trace and match its hash.

    With ``require_symmetric_cache`` (Phase-B default) a prompt is only
    complete when its rows carry the symmetric Top-100 fields, so stale v1
    caches are re-scored instead of being treated as done.
    """

    if not token_rows_path.is_file():
        return set()
    rows_by_prompt: dict[str, dict[int, str]] = defaultdict(dict)
    symmetric_ok: dict[str, bool] = defaultdict(bool)
    for line in token_rows_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        uid = str(row["prompt_id"])
        rows_by_prompt[uid][int(row["position_index"])] = str(row["teacher_trace_id"])
        if "student_top100_ids" in row:
            symmetric_ok[uid] = True
    complete: set[str] = set()
    for uid, trace in expected_traces.items():
        positions = rows_by_prompt.get(uid)
        if positions is None:
            continue
        if require_symmetric_cache and not symmetric_ok.get(uid):
            continue
        expected_length = len(trace.get("response_token_ids") or ())
        expected_hash_prefix = str(trace.get("response_token_hash") or "")[:16]
        if len(positions) != expected_length or set(positions) != set(range(expected_length)):
            continue
        trace_ids = {value for value in positions.values()}
        if len(trace_ids) != 1 or expected_hash_prefix not in next(iter(trace_ids)):
            continue
        complete.add(uid)
    return complete


def run_analyze(config: ProxyConfig, *, prompt_uids: Sequence[str] | None = None) -> dict[str, Any]:
    output = _resolve_output(config)
    token_rows_path = output / "proxy_token_rows.jsonl"
    if not token_rows_path.is_file():
        raise FileNotFoundError("run score before analyze")
    token_rows = _read_jsonl_checked(token_rows_path)
    rescue = _read_jsonl_checked(Path(config.rescue_dir) / "rescue_comparisons.jsonl")
    scored_uids = sorted({str(row["prompt_id"]) for row in token_rows})
    if prompt_uids:
        requested = set(prompt_uids)
        missing = requested - set(scored_uids)
        if missing:
            raise ValueError(f"analyze requested prompts with no scored token rows: {sorted(missing)}")
        rescue = [row for row in rescue if str(row["sample_uid"]) in requested]
    else:
        missing = {str(row["sample_uid"]) for row in rescue} - set(scored_uids)
        if missing:
            raise ValueError(
                f"rescue rows without scored tokens: {sorted(missing)}; "
                "score every Experiment-B prompt or pass --prompt-uids"
            )
    retained = _read_jsonl_checked(Path(config.proposal_dir) / "retained_proposals.jsonl")
    selected_traces = {
        uid: _selected_teacher_trace(retained, uid)
        for uid in sorted({str(row["sample_uid"]) for row in rescue})
    }
    study_rows = build_proxy_study_rows(
        token_rows=token_rows,
        rescue_rows=rescue,
        selected_traces=selected_traces,
        horizons=config.horizons,
    )
    csv_path = output / "proxy_study.csv"
    csv_hash = write_proxy_study_csv(csv_path, study_rows)

    minimal = _read_jsonl_checked(Path(config.rescue_dir) / "minimal_rescue_prefixes.jsonl")
    expected_minimal = {
        str(row["sample_uid"]): int(row["horizon"])
        for row in minimal
        if row.get("meets_preregistered_rescue_rule") is True
    }
    reproduced: dict[str, Any] = {}
    for uid, expected_h in expected_minimal.items():
        gold_rows = [
            row for row in study_rows if row["prompt_id"] == uid and row["rescue_gold"]
        ]
        if not gold_rows:
            reproduced[uid] = {
                "expected_minimal_horizon": expected_h,
                "reproduced_minimal_horizon": None,
                "missing_from_study": True,
            }
            continue
        reproduced[uid] = {
            "expected_minimal_horizon": expected_h,
            "reproduced_minimal_horizon": min(row["horizon"] for row in gold_rows),
        }
    present_uids = {str(row["prompt_id"]) for row in study_rows}
    coverage = {
        "prompts": len({row["prompt_id"] for row in study_rows}),
        "rows": len(study_rows),
        "rescue_positive_prompts_in_study": sum(
            1 for uid in expected_minimal if uid in present_uids
        ),
        "minimal_horizons_reproduced": all(
            value["expected_minimal_horizon"] == value["reproduced_minimal_horizon"]
            for value in reproduced.values()
            if not value.get("missing_from_study")
        ),
    }
    evaluation = evaluate_proxy_study(study_rows, seed=20260814)
    heldout = evaluate_proxy_heldout(study_rows, seed=20260815)
    analysis: dict[str, Any] = {
        "schema_version": "reachability-proxy-analyze-v1",
        "coverage": coverage,
        "minimal_horizon_reproduction": reproduced,
        "evaluation": evaluation,
        "heldout_evaluation": heldout,
        "disclaimer": "Expansion-stage evaluation: descriptive full-set metrics plus "
                      "held-out §9 metrics (thresholds fit on train prompts, evaluated "
                      "on disjoint test prompts). Selection is a GO/NO-GO decision for "
                      "the user based on held-out evidence.",
        "csv": {"path": str(csv_path), "sha256": csv_hash},
    }
    analysis_path = output / "proxy_analysis.json"
    _write_json_atomic(analysis_path, analysis)
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["analysis"] = {
            "proxy_study_csv_sha256": csv_hash,
            "proxy_analysis_sha256": _sha256(analysis_path),
            "coverage": coverage,
        }
        _write_json_atomic(manifest_path, manifest)
    resolved_config = output / "resolved_config.yaml"
    import yaml

    resolved_config.write_text(
        yaml.safe_dump(asdict(config), allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    return analysis


def _merge_sources(
    *,
    token_dirs: Sequence[str | Path],
    rescue_dirs: Sequence[str | Path],
    proposal_dirs: Sequence[str | Path],
    horizons: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    """Merge token/rescue/proposal sources into one strict combined study."""

    token_rows: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for source in token_dirs:
        rows = _read_jsonl_checked(Path(source) / "proxy_token_rows.jsonl")
        for row in rows:
            if str(row["prompt_id"]) in seen_prompts:
                continue
            token_rows.append(row)
        seen_prompts.update(str(row["prompt_id"]) for row in rows)

    rescue_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()
    for source in rescue_dirs:
        for row in _read_jsonl_checked(Path(source) / "rescue_comparisons.jsonl"):
            key = (str(row["sample_uid"]), int(row["horizon"]))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rescue_rows.append(row)

    retained_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_retained: set[tuple[str, str, str]] = set()
    for source in proposal_dirs:
        for row in _read_jsonl_checked(Path(source) / "retained_proposals.jsonl"):
            identity = (
                str(row.get("sample_uid")),
                str(row.get("proposal_id")),
                str(row.get("response_token_hash")),
            )
            if identity in seen_retained:
                continue
            seen_retained.add(identity)
            retained_by_uid[identity[0]].append(row)

    rescue_uids = sorted({str(row["sample_uid"]) for row in rescue_rows})
    selected_traces: dict[str, dict[str, Any]] = {}
    for uid in rescue_uids:
        if uid not in retained_by_uid:
            raise ValueError(f"{uid}: rescue gold without retained teacher trace")
        selected_traces[uid] = _selected_teacher_trace(retained_by_uid[uid], uid)

    study_rows = build_proxy_study_rows(
        token_rows=token_rows,
        rescue_rows=rescue_rows,
        selected_traces=selected_traces,
        horizons=horizons,
    )
    expected_minimal: dict[str, int] = {}
    for source in rescue_dirs:
        for row in read_jsonl(Path(source) / "minimal_rescue_prefixes.jsonl"):
            if row.get("meets_preregistered_rescue_rule") is True:
                expected_minimal.setdefault(str(row["sample_uid"]), int(row["horizon"]))
    return study_rows, expected_minimal, rescue_rows


def run_analyze_combined(
    *,
    token_dirs: Sequence[str | Path],
    rescue_dirs: Sequence[str | Path],
    proposal_dirs: Sequence[str | Path],
    output_dir: str | Path,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    heldout_seed: int = 20260815,
) -> dict[str, Any]:
    """CPU-only combined proxy study across multiple gold sources (e.g., 12+33+19)."""

    study_rows, expected_minimal, _ = _merge_sources(
        token_dirs=token_dirs,
        rescue_dirs=rescue_dirs,
        proposal_dirs=proposal_dirs,
        horizons=horizons,
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "proxy_study.csv"
    csv_hash = write_proxy_study_csv(csv_path, study_rows)

    reproduced = {
        uid: {
            "expected_minimal_horizon": horizon,
            "reproduced_minimal_horizon": min(
                row["horizon"] for row in study_rows if row["prompt_id"] == uid and row["rescue_gold"]
            ),
        }
        for uid, horizon in expected_minimal.items()
    }
    coverage = {
        "prompts": len({row["prompt_id"] for row in study_rows}),
        "rows": len(study_rows),
        "rescue_positive_prompts": sum(
            1 for uid in expected_minimal if any(
                row["prompt_id"] == uid and row["rescue_gold"] for row in study_rows
            )
        ),
        "minimal_horizons_reproduced": all(
            value["expected_minimal_horizon"] == value["reproduced_minimal_horizon"]
            for value in reproduced.values()
        ),
    }
    analysis: dict[str, Any] = {
        "schema_version": "reachability-proxy-combined-analyze-v1",
        "coverage": coverage,
        "minimal_horizon_reproduction": reproduced,
        "evaluation": evaluate_proxy_study(study_rows, seed=20260814),
        "heldout_evaluation": evaluate_proxy_heldout(study_rows, seed=heldout_seed),
        "sources": {
            "token_dirs": [str(Path(value)) for value in token_dirs],
            "rescue_dirs": [str(Path(value)) for value in rescue_dirs],
            "proposal_dirs": [str(Path(value)) for value in proposal_dirs],
        },
        "disclaimer": "Combined gold from Experiment-B + expansion cohorts; held-out "
                      "thresholds fit on train prompts and evaluated on disjoint prompts.",
    }
    analysis_path = output / "proxy_analysis.json"
    _write_json_atomic(analysis_path, analysis)
    (output / "resolved_config.yaml").write_text(
        __import__("yaml").safe_dump(
            {
                "horizons": list(horizons),
                "heldout_seed": heldout_seed,
                "token_dirs": [str(Path(value)) for value in token_dirs],
                "rescue_dirs": [str(Path(value)) for value in rescue_dirs],
                "proposal_dirs": [str(Path(value)) for value in proposal_dirs],
            },
            allow_unicode=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return analysis


def derive_salvaged_features(
    token_rows: Sequence[Mapping[str, Any]],
    study_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Enrich (prompt, horizon) rows with features from the existing cache.

    From the symmetric v2 cache we derive: position, both directional support
    masses M_h^K / C_h^K and overlap O_h^K for K in {4,16,64} (K slicing of the
    cached Top-100 arrays), endpoint/window NLL and FKL, handoff contrasts,
    and cumulative barriers.  Endpoint-alignment sensitivity variants
    (``*_next``) read the state at position h instead of h-1.
    """

    by_prompt = _group_by_prompt(token_rows)
    enriched: list[dict[str, Any]] = []
    for row in study_rows:
        uid = str(row["prompt_id"])
        horizon = int(row["horizon"])
        tokens = by_prompt.get(uid)
        if tokens is None:
            raise ValueError(f"{uid}: scored token rows missing for study row")
        trace_length = len(tokens)

        def window_mean(key: str, start: int, length: int) -> float | None:
            low = max(0, start)
            high = min(trace_length, start + length)
            if high <= low:
                return None
            values = [float(tokens[index][key]) for index in range(low, high)]
            return sum(values) / len(values)

        def support_compat(endpoint_index: int) -> tuple[list[float], list[int], list[int]]:
            """Teacher logps at student Top-100, student ids, teacher ids."""

            if endpoint_index < 0 or endpoint_index >= trace_length:
                return [], [], []
            token = tokens[endpoint_index]
            teacher_ids = token.get("teacher_q_top100_ids")
            if teacher_ids is None:
                teacher_ids = token.get("teacher_top100_ids", [])
            return (
                [float(value) for value in token.get("teacher_logp_at_student_top100_ids", [])],
                [int(value) for value in token.get("student_top100_ids", [])],
                [int(value) for value in teacher_ids],
            )

        student_at_teacher = (
            [
                float(value)
                for value in tokens[horizon - 1]["student_p_logp_top100_at_teacher_ids"]
            ]
            if horizon - 1 >= 0
            else []
        )
        feature_row = dict(row)
        feature_row["trace_length"] = trace_length
        for K in (4, 16, 64, 100):
            feature_row[f"M_h_{K}"] = sum(
                math.exp(value) for value in student_at_teacher[:K]
            )
        feature_row["D_endpoint_fkl"] = (
            float(tokens[horizon - 1]["d_coarse_fkl"]) if horizon - 1 >= 0 else None
        )
        endpoint_tokens = tokens[horizon - 1] if horizon - 1 >= 0 else {}
        feature_row["C_h_16"] = (
            float(endpoint_tokens["C_h_16"])
            if horizon - 1 >= 0 and "C_h_16" in endpoint_tokens
            else None
        )
        feature_row["O_h_16"] = (
            float(endpoint_tokens["O_h_16"])
            if horizon - 1 >= 0 and "O_h_16" in endpoint_tokens
            else None
        )
        teacher_at_student, student_ids, teacher_ids = support_compat(horizon - 1)
        # K=16 keeps the cached endpoint fields; K in {4,64} are derived K slices.
        for K in (4, 64):
            feature_row[f"C_h_{K}"] = (
                sum(math.exp(value) for value in teacher_at_student[:K])
                if teacher_at_student
                else None
            )
            feature_row[f"O_h_{K}"] = (
                len(set(student_ids[:K]) & set(teacher_ids[:K])) / K
                if student_ids and teacher_ids
                else None
            )
        feature_row["C_h_16_derived"] = feature_row["C_h_16"]
        feature_row["O_h_16_derived"] = feature_row["O_h_16"]
        if teacher_at_student:
            feature_row["C_h_16_derived"] = (
                sum(math.exp(value) for value in teacher_at_student[:16])
            )
        if student_ids and teacher_ids:
            feature_row["O_h_16_derived"] = (
                len(set(student_ids[:16]) & set(teacher_ids[:16])) / 16
            )
        next_teacher_at_student, next_student_ids, next_teacher_ids = support_compat(horizon)
        for K in (4, 16, 64):
            feature_row[f"C_h_{K}_next"] = (
                sum(math.exp(value) for value in next_teacher_at_student[:K])
                if next_teacher_at_student
                else None
            )
            feature_row[f"O_h_{K}_next"] = (
                len(set(next_student_ids[:K]) & set(next_teacher_ids[:K])) / K
                if next_student_ids and next_teacher_ids
                else None
            )
        for window in (32, 64):
            feature_row[f"fkl_takeoff_{window}"] = window_mean(
                "d_coarse_fkl", horizon, window
            )
            feature_row[f"nll_takeoff_{window}"] = window_mean(
                "student_nll", horizon, window
            )
            feature_row[f"fkl_past_{window}"] = window_mean(
                "d_coarse_fkl", horizon - window, window
            )
            feature_row[f"nll_past_{window}"] = window_mean(
                "student_nll", horizon - window, window
            )
        past_32 = feature_row["fkl_past_32"]
        takeoff_32 = feature_row["fkl_takeoff_32"]
        feature_row["delta_handoff_32"] = (
            past_32 - takeoff_32 if past_32 is not None and takeoff_32 is not None else None
        )
        past_64 = feature_row["fkl_past_64"]
        takeoff_64 = feature_row["fkl_takeoff_64"]
        feature_row["delta_handoff_64"] = (
            past_64 - takeoff_64 if past_64 is not None and takeoff_64 is not None else None
        )
        enriched.append(feature_row)
    return enriched


SALVAGE_FEATURES: dict[str, str] = {
    "position": "position",
    "C_h_16": "C_h_16",
    "C_h_4": "C_h_4",
    "C_h_64": "C_h_64",
    "C_h_4_next": "C_h_4_next",
    "C_h_16_next": "C_h_16_next",
    "C_h_64_next": "C_h_64_next",
    "M_h_4": "M_h_4",
    "M_h_16": "M_h_16",
    "M_h_64": "M_h_64",
    "M_h_100": "M_h_100",
    "O_h_16": "O_h_16",
    "O_h_4": "O_h_4",
    "O_h_64": "O_h_64",
    "O_h_4_next": "O_h_4_next",
    "O_h_16_next": "O_h_16_next",
    "O_h_64_next": "O_h_64_next",
    "D_endpoint_fkl": "D_endpoint_fkl",
    "fkl_takeoff_32": "fkl_takeoff_32",
    "fkl_takeoff_64": "fkl_takeoff_64",
    "nll_takeoff_32": "nll_takeoff_32",
    "nll_takeoff_64": "nll_takeoff_64",
    "delta_handoff_32": "delta_handoff_32",
    "delta_handoff_64": "delta_handoff_64",
    "cum_student_nll": "cum_student_nll",
    "cum_top100_fkl_tail": "cum_top100_fkl_tail",
}

MODEL_SETS: dict[str, tuple[str, ...]] = {
    "M0_position": ("position",),
    "M1_compat_partial": ("M_h_16",),
    "M1_compat": ("C_h_16", "M_h_16", "O_h_16"),
    "M2_takeoff": ("fkl_takeoff_32", "fkl_takeoff_64"),
    "M3_mechanistic_lite": ("position", "M_h_16", "fkl_takeoff_64", "delta_handoff_64"),
}


def evaluate_prefix_study(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 20260815,
    n_folds: int = 5,
    repeats: int = 5,
    strata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Prefix-level evaluation: every (prompt, horizon) row scored before labels."""

    complete = [row for row in rows if row.get("rescue_gold") is not None]
    prompt_ids = [str(row["prompt_id"]) for row in complete]
    labels = [bool(row["rescue_gold"]) for row in complete]
    evaluation: dict[str, Any] = {
        "unit": "(prompt_id, horizon) rows",
        "n_rows": len(complete),
        "n_prompts": len({str(row["prompt_id"]) for row in complete}),
        "n_positive_rows": sum(1 for value in labels if value),
        "single_features": {},
        "models": {},
    }
    # Registered single-feature direction: higher oriented score = more viable.
    direction: dict[str, int] = {
        "position": 1,
        "C_h_16": 1,
        "C_h_4": 1,
        "C_h_64": 1,
        "C_h_4_next": 1,
        "C_h_16_next": 1,
        "C_h_64_next": 1,
        "M_h_4": 1,
        "M_h_16": 1,
        "M_h_64": 1,
        "M_h_100": 1,
        "O_h_16": 1,
        "O_h_4": 1,
        "O_h_64": 1,
        "O_h_4_next": 1,
        "O_h_16_next": 1,
        "O_h_64_next": 1,
        "D_endpoint_fkl": -1,
        "fkl_takeoff_32": -1,
        "fkl_takeoff_64": -1,
        "nll_takeoff_32": -1,
        "nll_takeoff_64": -1,
        "delta_handoff_32": 1,
        "delta_handoff_64": 1,
        "cum_student_nll": -1,
        "cum_top100_fkl_tail": -1,
    }
    for name, feature_key in SALVAGE_FEATURES.items():
        valid = [
            (uid, score, label)
            for uid, row, label in zip(prompt_ids, complete, labels)
            if (score := row.get(feature_key)) is not None
        ]
        if len(valid) < 4 or not any(label for _, _, label in valid) or all(
            label for _, _, label in valid
        ):
            continue
        scores = [float(score) for _, score, _ in valid]
        row_labels = [bool(label) for _, _, label in valid]
        row_uids = [str(uid) for uid, _, _ in valid]
        evaluation["single_features"][name] = {
            "auroc": tie_correct_auroc(scores, row_labels),
            "auprc": _auprc(scores, row_labels),
            "direction": direction.get(name, 1),
            "cluster_bootstrap": cluster_bootstrap_metrics(
                row_uids, scores, row_labels, seed=seed
            ),
            "grouped_cv": grouped_cv_auroc(
                row_uids, [[score] for score in scores], row_labels,
                n_folds=n_folds, repeats=repeats, seed=seed, fit_model=False,
            ),
            "n_rows": len(valid),
            "within_prompt_hstar": _within_prompt_from_horizon_map(
                _feature_horizon_rows(complete, feature_key, direction.get(name, 1)),
                sorted({int(row["horizon"]) for row in complete}),
            ),
        }
    for name, feature_keys in MODEL_SETS.items():
        available = [key for key in feature_keys if key in SALVAGE_FEATURES]
        if not available:
            continue
        feature_rows: list[list[float]] = []
        valid_uids: list[str] = []
        valid_labels: list[bool] = []
        for row in complete:
            values = [row.get(SALVAGE_FEATURES[key]) for key in available]
            if any(value is None for value in values):
                continue
            feature_rows.append([float(value) for value in values])
            valid_uids.append(str(row["prompt_id"]))
            valid_labels.append(bool(row["rescue_gold"]))
        if len(feature_rows) < 8 or not any(valid_labels) or all(valid_labels):
            continue
        evaluation["models"][name] = {
            "features": available,
            "grouped_cv": grouped_cv_auroc(
                valid_uids, feature_rows, valid_labels,
                n_folds=n_folds, repeats=repeats, seed=seed, fit_model=True,
            ),
            "n_rows": len(feature_rows),
        }
    evaluation["models"].update(
        registered_model_evaluation(
            complete,
            MODEL_SETS,
            n_folds=n_folds,
            repeats=repeats,
            seed=seed,
        )
    )

    if strata:
        strata_metrics: dict[str, dict[str, dict[str, Any]]] = defaultdict(
            lambda: defaultdict(lambda: {"n_rows": 0, "n_positive": 0, "scores": [], "labels": []})
        )
        for name, feature_key in SALVAGE_FEATURES.items():
            for uid, row, label in zip(prompt_ids, complete, labels):
                stratum = strata.get(str(uid))
                if not stratum:
                    continue
                score = row.get(feature_key)
                if score is None:
                    continue
                oriented = direction.get(name, 1) * float(score)
                bucket = strata_metrics[stratum][name]
                bucket["n_rows"] += 1
                bucket["n_positive"] += 1 if label else 0
                bucket["scores"].append(oriented)
                bucket["labels"].append(bool(label))
        for stratum, features in strata_metrics.items():
            for feature_name, bucket in features.items():
                if bucket["n_rows"] >= 4 and 0 < bucket["n_positive"] < bucket["n_rows"]:
                    bucket["auroc"] = tie_correct_auroc(bucket["scores"], bucket["labels"])
                bucket.pop("scores", None)
                bucket.pop("labels", None)
        evaluation["strata"] = {
            stratum: dict(features) for stratum, features in strata_metrics.items()
        }

    # Null-feature audit: a random feature must stay at chance.
    import random

    rng = random.Random(seed)
    null_scores = [rng.random() for _ in complete]
    evaluation["null_feature_audit"] = {
        "auroc": tie_correct_auroc(null_scores, labels),
        "cluster_bootstrap": cluster_bootstrap_metrics(
            prompt_ids, null_scores, labels, seed=seed, resamples=500
        ),
    }
    return evaluation


def _feature_horizon_rows(
    rows: Sequence[Mapping[str, Any]],
    feature_key: str,
    direction: int,
) -> dict[str, dict[int, tuple[float, bool]]]:
    """Per-(prompt, horizon) oriented feature scores with labels."""

    by_prompt: dict[str, dict[int, tuple[float, bool]]] = defaultdict(dict)
    for row in rows:
        score = row.get(feature_key)
        if score is None:
            continue
        by_prompt[str(row["prompt_id"])][int(row["horizon"])] = (
            direction * float(score),
            bool(row["rescue_gold"]),
        )
    return by_prompt


def _within_prompt_from_horizon_map(
    by_prompt_horizon: Mapping[str, Mapping[int, tuple[float, bool]]],
    horizons: Sequence[int],
) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {
        "n_prompts": 0,
        "recall_at_1": 0.0,
        "plus_minus_one_bin": 0.0,
        "mean_abs_bin_distance": 0.0,
        "pairwise_agreement": 0.0,
    }
    for uid, horizon_rows in by_prompt_horizon.items():
        gold_horizons = [
            horizon for horizon, (_, label) in horizon_rows.items() if label
        ]
        if not gold_horizons:
            continue
        h_star = min(gold_horizons)
        h_hat = max(horizon_rows, key=lambda horizon: horizon_rows[horizon][0])
        bin_star = list(horizons).index(h_star)
        bin_hat = list(horizons).index(h_hat)
        distance = abs(bin_hat - bin_star)
        metrics["recall_at_1"] = float(metrics["recall_at_1"]) + (1.0 if distance == 0 else 0.0)
        metrics["plus_minus_one_bin"] = float(metrics["plus_minus_one_bin"]) + (
            1.0 if distance <= 1 else 0.0
        )
        metrics["mean_abs_bin_distance"] = float(metrics["mean_abs_bin_distance"]) + distance
        agreements = sum(
            1.0
            for horizon in horizon_rows
            if horizon != h_star and horizon_rows[horizon][0] <= horizon_rows[h_star][0]
        )
        metrics["pairwise_agreement"] = float(metrics["pairwise_agreement"]) + agreements / max(
            1, len(horizon_rows) - 1
        )
        metrics["n_prompts"] = int(metrics["n_prompts"]) + 1
    count = int(metrics["n_prompts"])
    if count:
        for key in ("recall_at_1", "plus_minus_one_bin", "mean_abs_bin_distance", "pairwise_agreement"):
            metrics[key] = float(metrics[key]) / count
    return metrics


def registered_model_evaluation(
    rows: Sequence[Mapping[str, Any]],
    model_sets: Mapping[str, Sequence[str]],
    *,
    n_folds: int = 5,
    repeats: int = 5,
    seed: int = 20260815,
) -> dict[str, Any]:
    """OOF grouped-CV model evaluation with paired deltas and h* prediction."""

    import random

    rng = random.Random(seed)
    by_prompt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row["prompt_id"])].append(row)
    keys = sorted(by_prompt)
    horizons = sorted({int(row["horizon"]) for row in rows})

    def feature_rows(prompt_rows: Sequence[Mapping[str, Any]], feature_keys: Sequence[str]):
        result: list[list[float]] = []
        for row in prompt_rows:
            values = [row.get(SALVAGE_FEATURES[key]) for key in feature_keys]
            if any(value is None for value in values):
                return None
            result.append([float(value) for value in values])
        return result

    folds_by_repeat: list[list[set[str]]] = []
    for _ in range(repeats):
        shuffled = list(keys)
        rng.shuffle(shuffled)
        folds = [set() for _ in range(n_folds)]
        for index, uid in enumerate(shuffled):
            folds[index % n_folds].add(uid)
        folds_by_repeat.append(folds)

    result: dict[str, Any] = {}
    label_by_key = {
        (str(row["prompt_id"]), int(row["horizon"])): bool(row["rescue_gold"])
        for row in rows
    }
    model_feature_keys = {
        name: [key for key in feature_keys if key in SALVAGE_FEATURES]
        for name, feature_keys in model_sets.items()
    }
    for name, feature_keys in model_feature_keys.items():
        if not feature_keys:
            continue
        flat_rows = [row for uid in keys for row in by_prompt[uid]]
        if feature_rows(flat_rows, feature_keys) is None:
            # No row exposes every required feature (e.g., v1 cache lacking C_h).
            continue
        oof: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        fold_aurocs: list[float] = []
        fold_auprcs: list[float] = []
        for folds in folds_by_repeat:
            for fold_index, test_uids in enumerate(folds):
                train_uids = [uid for uid in keys if uid not in test_uids]
                train_rows = [row for uid in train_uids for row in by_prompt[uid]]
                train_features = feature_rows(train_rows, feature_keys)
                if train_features is None or not any(
                    bool(row["rescue_gold"]) for row in train_rows
                ) or all(bool(row["rescue_gold"]) for row in train_rows):
                    continue
                weights, intercept = fit_l2_logistic(
                    train_features, [bool(row["rescue_gold"]) for row in train_rows]
                )
                test_rows = [row for uid in sorted(test_uids) for row in by_prompt[uid]]
                test_features = feature_rows(test_rows, feature_keys)
                if test_features is None or not test_rows:
                    continue
                probabilities = logistic_predict_proba(
                    test_features, weights, intercept
                )
                test_labels = [bool(row["rescue_gold"]) for row in test_rows]
                fold_aurocs.append(tie_correct_auroc(probabilities, test_labels))
                fold_auprcs.append(_auprc(probabilities, test_labels))
                for row, probability in zip(test_rows, probabilities):
                    oof[str(row["prompt_id"])][int(row["horizon"])].append(probability)
        mean_probabilities: dict[str, dict[int, float]] = {}
        for uid, horizon_probs in oof.items():
            mean_probabilities[uid] = {
                horizon: sum(values) / len(values)
                for horizon, values in horizon_probs.items()
            }
        within = _within_prompt_from_horizon_map(
            {
                uid: {
                    horizon: (prob, label_by_key[(uid, horizon)])
                    for horizon, prob in horizons_map.items()
                }
                for uid, horizons_map in mean_probabilities.items()
            },
            horizons,
        )
        result[name] = {
            "features": feature_keys,
            "mean_heldout_auroc": (
                float(sum(fold_aurocs) / len(fold_aurocs)) if fold_aurocs else None
            ),
            "mean_heldout_auprc": (
                float(sum(fold_auprcs) / len(fold_auprcs)) if fold_auprcs else None
            ),
            "fold_aurocs": fold_aurocs,
            "within_prompt_hstar": within,
        }

    baseline_folds = result.get("M0_position", {}).get("fold_aurocs")
    if baseline_folds:
        for name, model in result.items():
            if name == "M0_position" or not model.get("fold_aurocs"):
                continue
            deltas = [
                model_fold - baseline_fold
                for model_fold, baseline_fold in zip(
                    model["fold_aurocs"], baseline_folds
                )
            ]
            model["paired_delta_vs_M0"] = {
                "mean_delta_auroc": float(sum(deltas) / len(deltas)) if deltas else None,
                "pct_folds_ge_M0": (
                    float(sum(1 for value in deltas if value >= 0)) / len(deltas)
                    if deltas
                    else None
                ),
            }
    return result


def run_salvage_analysis(
    *,
    token_dirs: Sequence[str | Path],
    rescue_dirs: Sequence[str | Path],
    proposal_dirs: Sequence[str | Path],
    output_dir: str | Path,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    seed: int = 20260815,
) -> dict[str, Any]:
    """Phase-A CPU-only prefix-level analysis on the existing asymmetric cache."""

    study_rows, expected_minimal, _ = _merge_sources(
        token_dirs=token_dirs,
        rescue_dirs=rescue_dirs,
        proposal_dirs=proposal_dirs,
        horizons=horizons,
    )
    token_rows: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for source in token_dirs:
        for row in _read_jsonl_checked(Path(source) / "proxy_token_rows.jsonl"):
            if str(row["prompt_id"]) in seen_prompts:
                continue
            token_rows.append(row)
        seen_prompts.update(str(row["prompt_id"]) for row in token_rows)
    enriched = derive_salvaged_features(token_rows, study_rows)
    strata: dict[str, str] = {}
    for proposal_dir in proposal_dirs:
        for row in read_jsonl(Path(proposal_dir) / "retained_proposals.jsonl"):
            stratum = str(row.get("observed_stratum") or row.get("support_state") or "")
            if stratum:
                strata.setdefault(str(row["sample_uid"]), stratum)
    for row in enriched:
        row["stratum"] = strata.get(str(row["prompt_id"]), "unknown")
    evaluation = evaluate_prefix_study(enriched, seed=seed, strata=strata)
    analysis: dict[str, Any] = {
        "schema_version": "reachability-prefix-salvage-v1",
        "coverage": {
            "prompts": len({row["prompt_id"] for row in study_rows}),
            "rows": len(study_rows),
            "rescue_positive_prompts": sum(
                1
                for uid in expected_minimal
                if any(row["prompt_id"] == uid and row["rescue_gold"] for row in study_rows)
            ),
        },
        "evaluation": evaluation,
        "cache_limitations": [
            "visual JS features not yet included (M4 deferred)",
        ],
        "label": "full registered features on symmetric v2 cache (Phase C)",
    }
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_table = output / "proxy_feature_table.jsonl"
    with feature_table.open("w", encoding="utf-8") as handle:
        for row in enriched:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    analysis["feature_table"] = {
        "path": str(feature_table),
        "rows": len(enriched),
        "sha256": _sha256(feature_table),
    }
    analysis_path = output / "proxy_prefix_salvage_analysis.json"
    _write_json_atomic(analysis_path, analysis)
    return analysis


def merge_token_shards(
    token_files: Sequence[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Merge sharded symmetric-cache token files into one canonical JSONL."""

    if not token_files:
        raise ValueError("at least one token file is required")
    rows_by_prompt: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    prompt_order: list[str] = []
    input_hashes: dict[str, str] = {}
    for source in token_files:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        input_hashes[str(path)] = _sha256(path)
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            uid = str(row["prompt_id"])
            if uid not in rows_by_prompt:
                prompt_order.append(uid)
            rows_by_prompt[uid][int(row["position_index"])] = row
    for uid in prompt_order:
        positions = rows_by_prompt[uid]
        if not positions:
            raise ValueError(f"{uid}: empty token rows")
        trace_length = max(positions) + 1
        if set(positions) != set(range(trace_length)):
            raise ValueError(f"{uid}: non-contiguous token rows")
        trace_ids = {str(positions[index].get("teacher_trace_id")) for index in range(trace_length)}
        if len(trace_ids) != 1:
            raise ValueError(f"{uid}: token rows disagree on teacher trace identity")

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    token_path = output / "proxy_token_rows.jsonl"
    row_count = 0
    with token_path.open("w", encoding="utf-8") as handle:
        for uid in sorted(prompt_order):
            positions = rows_by_prompt[uid]
            for index in sorted(positions):
                handle.write(json.dumps(positions[index], ensure_ascii=False, sort_keys=True) + "\n")
                row_count += 1
    result = {
        "schema_version": "reachability-token-merge-v1",
        "token_files": [str(Path(value).expanduser().resolve()) for value in token_files],
        "input_sha256": input_hashes,
        "prompts": len(prompt_order),
        "rows": row_count,
        "output": str(token_path),
        "output_sha256": _sha256(token_path),
    }
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["token_merge"] = result
        _write_json_atomic(manifest_path, manifest)
    (output / "token_merge_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--output-dir")
    score = sub.add_parser("score")
    score.add_argument("--config", required=True)
    score.add_argument("--output-dir")
    score.add_argument("--student-device")
    score.add_argument("--teacher-device")
    score.add_argument("--max-prompts", type=int)
    score.add_argument("--prompt-uids", nargs="+")
    score.add_argument("--shard-index", type=int, default=0)
    score.add_argument("--num-shards", type=int, default=1)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--output-dir")
    analyze.add_argument("--prompt-uids", nargs="+")
    combined = sub.add_parser("analyze-combined")
    combined.add_argument("--token-dirs", nargs="+", required=True)
    combined.add_argument("--rescue-dirs", nargs="+", required=True)
    combined.add_argument("--proposal-dirs", nargs="+", required=True)
    combined.add_argument("--output-dir", required=True)
    combined.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    combined.add_argument("--heldout-seed", type=int, default=20260815)
    salvage = sub.add_parser("salvage")
    salvage.add_argument("--token-dirs", nargs="+", required=True)
    salvage.add_argument("--rescue-dirs", nargs="+", required=True)
    salvage.add_argument("--proposal-dirs", nargs="+", required=True)
    salvage.add_argument("--output-dir", required=True)
    salvage.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    salvage.add_argument("--seed", type=int, default=20260815)
    merge_tokens = sub.add_parser("merge-tokens")
    merge_tokens.add_argument("--token-files", nargs="+", required=True)
    merge_tokens.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.phase == "analyze-combined":
        run_analyze_combined(
            token_dirs=args.token_dirs,
            rescue_dirs=args.rescue_dirs,
            proposal_dirs=args.proposal_dirs,
            output_dir=args.output_dir,
            horizons=tuple(args.horizons),
            heldout_seed=args.heldout_seed,
        )
        return 0
    if args.phase == "salvage":
        run_salvage_analysis(
            token_dirs=args.token_dirs,
            rescue_dirs=args.rescue_dirs,
            proposal_dirs=args.proposal_dirs,
            output_dir=args.output_dir,
            horizons=tuple(args.horizons),
            seed=args.seed,
        )
        return 0
    if args.phase == "merge-tokens":
        merge_token_shards(args.token_files, args.output_dir)
        return 0
    config = load_config(args.config, args)
    if args.phase == "preflight":
        run_preflight(config)
    elif args.phase == "score":
        run_score(
            config,
            prompt_uids=args.prompt_uids,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    elif args.phase == "analyze":
        run_analyze(config, prompt_uids=args.prompt_uids)
    else:
        parser.error(f"unknown phase: {args.phase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
