"""Operator-Need Diagnostic (frozen plan Question B, CPU-only).

Derives per-token local support quantities from the symmetric Top-100 cache
(exact cross-gather, never intersection-only):

    C_t = sum_{v in S_t^S(K)} p_T(v)                      (compatibility)
    D_t = KL( pbar_T^U || pbar_S^U )                       (union disagreement)
    Q_t = 1 - H(p_T^off) / log|A_t^off|                   (off-support clarity)
    R_t = Dtilde_t * Ctilde_t                              (RKL need, TA-OPD)
    F_t = Dtilde_t * (1 - Ctilde_t) * Q_t                  (FKL need hypothesis)

``Ctilde`` and ``Dtilde`` use TA-OPD's bounded 5th/95th-percentile
normalization over the diagnostic token bank.  They are never unbounded
z-scores, so ``1 - Ctilde`` remains a valid incompatibility factor.

Then aggregates tokens to reasoning blocks (via ``reasoning_blocks``) and
writes the Operator-Need Report plus a block feature table.  No model forward,
no training.  K=16 primary; K=8/32/64 sensitivity derived from the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .reachability_proxy import teacher_trace_id
from .reasoning_blocks import blocks_with_token_spans


SCHEMA_VERSION = "support-aware-operator-need-v2"
DEFAULT_KS = (16,)
SENSITIVITY_KS = (8, 32, 64)
EPS = 1e-9


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _selected_teacher_trace(
    retained_rows: Sequence[Mapping[str, Any]], sample_uid: str
) -> dict[str, Any]:
    """Experiment-B rule: lowest numeric reachability_rank among retained."""

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


def _load_sources(
    *,
    token_dirs: Sequence[str],
    rescue_dirs: Sequence[str],
    proposal_dirs: Sequence[str],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, int],
    dict[str, str],
]:
    """Load token rows by trace, retained proposals, rescue gold, strata."""

    token_rows_by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in token_dirs:
        path = Path(source) / "proxy_token_rows.jsonl"
        for row in _iter_jsonl(path):
            token_rows_by_trace[str(row["teacher_trace_id"])].append(row)

    retained_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in proposal_dirs:
        path = Path(source) / "retained_proposals.jsonl"
        for row in _iter_jsonl(path):
            retained_by_uid[str(row.get("sample_uid"))].append(row)

    minimal_horizon: dict[str, int] = {}
    strata: dict[str, str] = {}
    for source in rescue_dirs:
        for row in _iter_jsonl(Path(source) / "minimal_rescue_prefixes.jsonl"):
            uid = str(row["sample_uid"])
            if row.get("meets_preregistered_rescue_rule") is True:
                minimal_horizon.setdefault(uid, int(row["horizon"]))
            if row.get("observed_stratum"):
                strata.setdefault(uid, str(row["observed_stratum"]))
        for row in _iter_jsonl(Path(source) / "rescue_comparisons.jsonl"):
            uid = str(row["sample_uid"])
            if row.get("observed_stratum"):
                strata.setdefault(uid, str(row["observed_stratum"]))

    traces: dict[str, dict[str, Any]] = {}
    for uid, rows in retained_by_uid.items():
        try:
            traces[uid] = _selected_teacher_trace(rows, uid)
        except ValueError:
            continue
    return token_rows_by_trace, traces, minimal_horizon, strata


def _token_scalars(
    row: Mapping[str, Any], k: int
) -> tuple[float, float, float, float, list[int], np.ndarray]:
    """C_t (exact), D_t (union KL), Q_t, off-support count for one row, K."""

    student_ids = row["student_top100_ids"][:k]
    teacher_ids = row["teacher_q_top100_ids"][:k]
    student_set = set(student_ids)
    teacher_set = set(teacher_ids)

    teacher_logp_at_student = np.asarray(
        row["teacher_logp_at_student_top100_ids"][:k], dtype=np.float64
    )
    teacher_logp_at_teacher = np.asarray(row["teacher_q_logp_top100"][:k], dtype=np.float64)
    student_logp_at_student = np.asarray(row["student_top100_logp"][:k], dtype=np.float64)
    student_logp_at_teacher = np.asarray(
        row["student_p_logp_top100_at_teacher_ids"][:k], dtype=np.float64
    )

    teacher_prob_at_student = np.exp(np.clip(teacher_logp_at_student, -80.0, 0.0))
    c_t = float(teacher_prob_at_student.sum())

    # Union distribution, exact via cross-gather.
    union = sorted(set(student_ids) | teacher_set)
    p_t: dict[int, float] = {}
    p_s: dict[int, float] = {}
    student_index = {token: index for index, token in enumerate(student_ids)}
    teacher_index = {token: index for index, token in enumerate(teacher_ids)}
    for token in union:
        if token in student_index:
            p_t[token] = float(np.exp(np.clip(teacher_logp_at_student[student_index[token]], -80.0, 0.0)))
            p_s[token] = float(np.exp(np.clip(student_logp_at_student[student_index[token]], -80.0, 0.0)))
        else:
            p_t[token] = float(np.exp(np.clip(teacher_logp_at_teacher[teacher_index[token]], -80.0, 0.0)))
            p_s[token] = float(np.exp(np.clip(student_logp_at_teacher[teacher_index[token]], -80.0, 0.0)))
    sum_t = sum(p_t.values())
    sum_s = sum(p_s.values())
    d_t = 0.0
    for token in union:
        pt = p_t[token] / max(sum_t, EPS)
        ps = p_s[token] / max(sum_s, EPS)
        if pt > EPS:
            d_t += pt * math.log(max(pt, EPS) / max(ps, EPS))

    # Off-support clarity over teacher top-K minus student top-K.
    off_ids = [token for token in teacher_ids if token not in student_set]
    off_logp = np.asarray(
        [float(np.exp(np.clip(teacher_logp_at_teacher[teacher_index[token]], -80.0, 0.0))) for token in off_ids]
    )
    if len(off_ids) == 0:
        q_t = 0.0
    elif len(off_ids) == 1:
        q_t = 1.0
    else:
        off_p = off_logp / max(off_logp.sum(), EPS)
        entropy = float(-np.sum(off_p * np.log(np.clip(off_p, EPS, 1.0))))
        q_t = 1.0 - entropy / math.log(len(off_ids))
    return c_t, d_t, q_t, len(off_ids), off_ids, off_logp


def _robust_unit_normalize(
    values: Sequence[float],
    *,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> list[float]:
    """TA-OPD-style quantile normalization clipped to ``[0, 1]``."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return []
    if not 0.0 <= q_low < q_high <= 1.0:
        raise ValueError("normalization quantiles must satisfy 0 <= q_low < q_high <= 1")
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return [0.0] * len(values)
    lower, upper = np.quantile(finite, [q_low, q_high])
    denominator = float(upper - lower)
    if abs(denominator) <= EPS:
        return [0.0] * len(values)
    normalized = np.clip((array - lower) / denominator, 0.0, 1.0)
    normalized[~np.isfinite(array)] = 0.0
    return [float(value) for value in normalized]


def _load_tokenizer(model_path: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(os.path.expandvars(model_path)).tokenizer


def run_operator_need(
    *,
    token_dirs: Sequence[str],
    rescue_dirs: Sequence[str],
    proposal_dirs: Sequence[str],
    teacher_model: str,
    output_dir: str,
    primary_k: int = 16,
    seed: int = 20260815,
) -> dict[str, Any]:
    """Derive token/block operator-need features and write the report."""

    token_rows_by_trace, traces, minimal_horizon, strata = _load_sources(
        token_dirs=token_dirs,
        rescue_dirs=rescue_dirs,
        proposal_dirs=proposal_dirs,
    )
    tokenizer = _load_tokenizer(teacher_model)

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    per_token: list[dict[str, Any]] = []
    per_trace: dict[str, list[float]] = defaultdict(list)
    sensitivity: dict[int, list[float]] = {k: [] for k in SENSITIVITY_KS}
    primary_c: list[float] = []
    covered_tokens = 0
    missing_traces: list[str] = []
    span_mismatch: list[str] = []

    for uid, trace in sorted(traces.items()):
        trace_id = teacher_trace_id(trace)
        token_ids = list(trace.get("response_token_ids") or [])
        rows = token_rows_by_trace.get(trace_id)
        if not rows:
            missing_traces.append(uid)
            continue
        rows.sort(key=lambda row: int(row["position_index"]))
        cache_positions = {int(row["position_index"]) for row in rows}
        expected = set(range(len(token_ids)))
        if cache_positions != expected:
            span_mismatch.append(uid)
            continue
        text, blocks = blocks_with_token_spans(tokenizer, token_ids)
        block_by_token: list[int | None] = [None] * len(token_ids)
        for index, block in enumerate(blocks):
            for token_index in range(block.start_token, block.end_token):
                if 0 <= token_index < len(block_by_token):
                    block_by_token[token_index] = index

        for row in rows:
            c_t, d_t, q_t, _, _, _ = _token_scalars(row, primary_k)
            primary_c.append(c_t)
            for k in SENSITIVITY_KS:
                c_k, _, _, _, _, _ = _token_scalars(row, k)
                sensitivity[k].append(c_k)
            per_token.append(
                {
                    "prompt_id": uid,
                    "trace_id": trace_id,
                    "position_index": int(row["position_index"]),
                    "token_id": int(row["token_id"]),
                    "block_index": block_by_token[int(row["position_index"])],
                    "C_t": c_t,
                    "D_t": d_t,
                    "Q_t": q_t,
                }
            )
        covered_tokens += len(rows)

    # TA-OPD normalizes a token batch with robust quantiles before composing
    # learnable/incompatible disagreement.  The offline token bank is the
    # corresponding comparison batch for this diagnostic.
    c_norm = _robust_unit_normalize([record["C_t"] for record in per_token])
    d_norm = _robust_unit_normalize([record["D_t"] for record in per_token])
    for record, c_tilde, d_tilde in zip(per_token, c_norm, d_norm, strict=True):
        record["C_tilde"] = c_tilde
        record["D_tilde"] = d_tilde
        record["R_t"] = d_tilde * c_tilde
        record["F_t"] = d_tilde * (1.0 - c_tilde) * float(record["Q_t"])
        per_trace[str(record["prompt_id"])].append(float(record["F_t"]))

    # Block aggregation.
    block_rows: list[dict[str, Any]] = []
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in per_token:
        by_uid[record["prompt_id"]].append(record)
    for uid, records in sorted(by_uid.items()):
        trace = traces[uid]
        token_ids = list(trace.get("response_token_ids") or [])
        _, blocks = blocks_with_token_spans(tokenizer, token_ids)
        h_star = minimal_horizon.get(uid)
        token_by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record["block_index"] is None:
                continue
            token_by_block[int(record["block_index"])].append(record)
        for index, block in enumerate(blocks):
            token_records = token_by_block.get(index, [])
            if not token_records:
                continue
            c_mean = float(np.mean([r["C_t"] for r in token_records]))
            d_mean = float(np.mean([r["D_t"] for r in token_records]))
            q_mean = float(np.mean([r["Q_t"] for r in token_records]))
            r_mean = float(np.mean([r["R_t"] for r in token_records]))
            f_mean = float(np.mean([r["F_t"] for r in token_records]))
            block_rows.append(
                {
                    "prompt_id": uid,
                    "stratum": strata.get(uid),
                    "h_star": h_star,
                    "block_index": index,
                    "block_type": block.block_type,
                    "start_token": block.start_token,
                    "end_token": block.end_token,
                    "start_char": block.start_char,
                    "end_char": block.end_char,
                    "text_sha256": hashlib.sha256(block.text.encode("utf-8")).hexdigest(),
                    "text_excerpt": block.text[:200],
                    "token_count": len(token_records),
                    "C_i": c_mean,
                    "D_i": d_mean,
                    "Q_i": q_mean,
                    "R_i": r_mean,
                    "F_i": f_mean,
                    "pre_hstar": h_star is None or block.end_token <= h_star,
                }
            )

    feature_table_path = output / "operator_need_blocks.jsonl"
    with feature_table_path.open("w", encoding="utf-8") as handle:
        for row in block_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    f_values = np.asarray([row["F_i"] for row in block_rows], dtype=np.float64)
    r_values = np.asarray([row["R_i"] for row in block_rows], dtype=np.float64)
    c_values = np.asarray([row["C_i"] for row in block_rows], dtype=np.float64)
    sensitivity_corr = {
        str(k): float(np.corrcoef(
            primary_c[: len(sensitivity[k])],
            sensitivity[k],
        )[0, 1])
        for k in SENSITIVITY_KS
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "coverage": {
            "prompts": len(by_uid),
            "rescue_positive_prompts": sum(1 for uid in by_uid if uid in minimal_horizon),
            "traces": len(per_trace),
            "tokens": covered_tokens,
            "blocks": len(block_rows),
            "missing_traces": missing_traces,
            "span_mismatch": span_mismatch,
        },
        "settings": {
            "primary_k": primary_k,
            "sensitivity_ks": list(SENSITIVITY_KS),
            "seed": seed,
            "normalization": "token_bank_quantile_05_95_clipped_0_1",
        },
        "block_stats": {
            "F_i": {
                "mean": float(f_values.mean()) if len(f_values) else None,
                "median": float(np.median(f_values)) if len(f_values) else None,
                "p10": float(np.percentile(f_values, 10)) if len(f_values) else None,
                "p90": float(np.percentile(f_values, 90)) if len(f_values) else None,
            },
            "R_i": {
                "mean": float(r_values.mean()) if len(r_values) else None,
                "median": float(np.median(r_values)) if len(r_values) else None,
            },
            "C_i": {
                "mean": float(c_values.mean()) if len(c_values) else None,
                "median": float(np.median(c_values)) if len(c_values) else None,
            },
        },
        "sensitivity_C_k_corr_vs_K16": sensitivity_corr,
        "feature_table": {
            "path": str(feature_table_path),
            "sha256": hashlib.sha256(feature_table_path.read_bytes()).hexdigest(),
        },
    }
    report_path = output / "operator_need_report.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-dirs", nargs="+", required=True)
    parser.add_argument("--rescue-dirs", nargs="+", required=True)
    parser.add_argument("--proposal-dirs", nargs="+", required=True)
    parser.add_argument("--teacher-model", default="${DTOPD_MODEL_ROOT}/Qwen3-VL-32B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--primary-k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_operator_need(
        token_dirs=args.token_dirs,
        rescue_dirs=args.rescue_dirs,
        proposal_dirs=args.proposal_dirs,
        teacher_model=args.teacher_model,
        output_dir=args.output_dir,
        primary_k=args.primary_k,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
