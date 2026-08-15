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
    cohort_parquet_path: str | None = None
    pool256_dir: str
    causal_dir: str
    rescue_dir: str
    student_model_path: str
    teacher_model_path: str
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

        teacher_values, teacher_indices = torch.topk(teacher_logp, k=config.top_k, dim=-1)
        teacher_topk_mass = teacher_values.exp().sum(dim=-1)
        teacher_tail = torch.clamp(1.0 - teacher_topk_mass, min=config.epsilon)
        teacher_tail_logp = teacher_tail.log()

        student_sampled = student_logp[
            torch.arange(end - start, device=student_logp.device), ids[start:end]
        ]
        teacher_ids_on_student = teacher_indices.to(student_logp.device)
        student_at_teacher_ids = student_logp.gather(
            dim=-1, index=teacher_ids_on_student
        )
        student_mass = student_at_teacher_ids.exp().sum(dim=-1)
        student_tail = torch.clamp(1.0 - student_mass, min=config.epsilon)
        student_tail_logp = student_tail.log()

        # Cross-model KL: teacher Top-100 values and tail live on the teacher
        # device; move them to the student device for the per-token math.
        teacher_values_on_student = teacher_values.to(student_logp.device)
        teacher_tail_on_student = teacher_tail.to(student_logp.device)
        teacher_tail_logp_on_student = teacher_tail_logp.to(student_logp.device)
        q_i = teacher_values_on_student.exp()
        d_t = (q_i * (teacher_values_on_student - student_at_teacher_ids)).sum(dim=-1) + (
            teacher_tail_on_student
            * (teacher_tail_logp_on_student - student_tail_logp)
        )
        teacher_mass_residual = teacher_topk_mass + teacher_tail - 1.0
        student_mass_residual = student_mass + student_tail - 1.0
        if float(teacher_mass_residual.abs().max()) > config.mass_tolerance:
            raise RuntimeError("teacher Top-K+tail mass conservation failed")
        if float(student_mass_residual.abs().max()) > config.mass_tolerance:
            raise RuntimeError("student Top-K+tail mass conservation failed")

        for offset in range(end - start):
            position = start + offset
            token_id = int(ids[start + offset].item())
            rows.append(
                {
                    "prompt_id": str(trace["sample_uid"]),
                    "teacher_trace_id": trace_id,
                    "position_index": position,
                    "token_id": token_id,
                    "student_nll": float(-student_sampled[offset].cpu()),
                    "d_coarse_fkl": float(d_t[offset].cpu()),
                    "teacher_q_top100_ids": tuple(int(value) for value in teacher_indices[offset].cpu().tolist()),
                    "teacher_q_logp_top100": tuple(float(value) for value in teacher_values[offset].cpu().tolist()),
                    "teacher_q_tail_logp": float(teacher_tail_logp[offset].cpu()),
                    "student_p_logp_top100_at_teacher_ids": tuple(
                        float(value) for value in student_at_teacher_ids[offset].cpu().tolist()
                    ),
                    "student_p_tail_prob": float(student_tail[offset].cpu()),
                    "teacher_mass_residual": float(teacher_mass_residual[offset].cpu()),
                    "student_mass_residual": float(student_mass_residual[offset].cpu()),
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


def run_score(config: ProxyConfig, *, prompt_uids: Sequence[str] | None = None) -> dict[str, Any]:
    output = _resolve_output(config)
    rescue = _read_jsonl_checked(Path(config.rescue_dir) / "rescue_comparisons.jsonl")
    retained = _read_jsonl_checked(Path(config.proposal_dir) / "retained_proposals.jsonl")
    selected_uids = sorted({str(row["sample_uid"]) for row in rescue})
    if prompt_uids:
        requested = set(prompt_uids)
        unknown = requested - set(selected_uids)
        if unknown:
            raise ValueError(f"unknown rescue prompts: {sorted(unknown)}")
        selected_uids = [uid for uid in selected_uids if uid in requested]
    if config.max_prompts is not None:
        selected_uids = selected_uids[: config.max_prompts]

    frame = _cohort_frame(config.cohort_dir, config.cohort_parquet_path)
    models = load_runtime_models(
        student_model_path=config.student_model_path,
        student_device=config.student_device,
        dtype=config.dtype,
        teacher_model_path=config.teacher_model_path,
        teacher_device=config.teacher_device,
    )
    token_rows_path = output / "proxy_token_rows.jsonl"
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
    with token_rows_path.open("w", encoding="utf-8") as handle:
        for uid in selected_uids:
            trace = _selected_teacher_trace(retained, uid)
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
    manifest_path = output / "run_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return manifest


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
    analysis: dict[str, Any] = {
        "schema_version": "reachability-proxy-analyze-v1",
        "coverage": coverage,
        "minimal_horizon_reproduction": reproduced,
        "evaluation": evaluation,
        "disclaimer": "12-prompt plumbing study: seven positives cannot select a proxy; "
                     "metrics are plumbing evidence only.",
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
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--output-dir")
    analyze.add_argument("--prompt-uids", nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config, args)
    if args.phase == "preflight":
        run_preflight(config)
    elif args.phase == "score":
        run_score(config, prompt_uids=args.prompt_uids)
    elif args.phase == "analyze":
        run_analyze(config, prompt_uids=args.prompt_uids)
    else:
        parser.error(f"unknown phase: {args.phase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
