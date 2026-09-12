"""Offline robustness analysis for exact-token teacher gaps.

The K=32 diagnostic already stores aligned teacher/student log-probabilities
for every response token.  This module reuses those immutable arrays to test
whether the observed length failure is caused by token outliers, position
drift, or truncation.  It never regenerates a response and never trains.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "support-aware-gap-robustness-v1"
DEFAULT_PREFIX_HORIZONS = (128, 256, 512)
DEFAULT_WINDOWS = ((0, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048))


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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def exact_content_gaps(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned token gaps and student NLLs under the content mask."""

    teacher = tuple(float(value) for value in row.get("teacher_sampled_token_log_probs") or ())
    student = tuple(float(value) for value in row.get("student_sampled_token_log_probs") or ())
    token_ids = tuple(int(value) for value in row.get("response_token_ids") or ())
    mask = tuple(bool(value) for value in row.get("content_mask") or ())
    lengths = {len(teacher), len(student), len(token_ids), len(mask)}
    if len(lengths) != 1 or not teacher:
        raise ValueError(f"{row.get('sample_uid')}: token arrays are empty or misaligned")
    if row.get("exact_token_alignment") is not True:
        raise ValueError(f"{row.get('sample_uid')}: exact-token alignment is not true")
    indices = [index for index, keep in enumerate(mask) if keep]
    if not indices:
        raise ValueError(f"{row.get('sample_uid')}: no content tokens")
    gaps = np.asarray([teacher[index] - student[index] for index in indices], dtype=np.float64)
    student_nll = np.asarray([-student[index] for index in indices], dtype=np.float64)
    if not np.isfinite(gaps).all() or not np.isfinite(student_nll).all():
        raise ValueError(f"{row.get('sample_uid')}: non-finite per-token score")
    return gaps, student_nll


def trek_trim_indices(
    student_nll: Sequence[float],
    *,
    low_fraction: float = 0.10,
    high_fraction: float = 0.02,
) -> np.ndarray:
    """Indices retained by TREK's two-sided student-NLL quantile trim."""

    values = np.asarray(student_nll, dtype=np.float64)
    if values.ndim != 1 or not values.size:
        raise ValueError("student_nll must be a non-empty vector")
    if not 0 <= low_fraction < 1 or not 0 <= high_fraction < 1:
        raise ValueError("trim fractions must lie in [0, 1)")
    if low_fraction + high_fraction >= 1:
        raise ValueError("trim fractions must sum to less than one")
    order = np.argsort(values, kind="stable")
    low_count = math.floor(values.size * low_fraction)
    high_count = math.floor(values.size * high_fraction)
    stop = values.size - high_count if high_count else values.size
    retained = order[low_count:stop]
    if not retained.size:
        raise ValueError("trim removes every token")
    return np.sort(retained)


def score_variants(
    row: Mapping[str, Any],
    *,
    prefix_horizons: Sequence[int] = DEFAULT_PREFIX_HORIZONS,
    windows: Sequence[tuple[int, int]] = DEFAULT_WINDOWS,
    suffix_tokens: int = 64,
) -> dict[str, float | None]:
    gaps, student_nll = exact_content_gaps(row)
    retained = trek_trim_indices(student_nll)
    scores: dict[str, float | None] = {
        "mean_gap": float(gaps.mean()),
        "trek_trimmed_gap": float(gaps[retained].mean()),
        "suffix_64_gap": float(gaps[-min(suffix_tokens, gaps.size):].mean()),
    }
    for horizon in prefix_horizons:
        scores[f"prefix_{horizon}_gap"] = (
            float(gaps[:horizon].mean()) if gaps.size >= horizon else None
        )
    for start, end in windows:
        scores[f"window_{start}_{end}_gap"] = (
            float(gaps[start:end].mean()) if gaps.size >= end else None
        )
    scores["window_2048_end_gap"] = float(gaps[2048:].mean()) if gaps.size > 2048 else None
    return scores


def cross_fitted_length_residuals(rows: Sequence[Mapping[str, Any]], folds: int = 5) -> dict[str, float]:
    """Remove unlabeled log-length/finish drift without using correctness."""

    if folds < 2:
        raise ValueError("folds must be at least two")
    usable = [row for row in rows if _finite(row.get("mean_gap"))]
    result: dict[str, float] = {}
    for fold in range(folds):
        test = [row for row in usable if int(row["fold"]) == fold]
        train = [row for row in usable if int(row["fold"]) != fold]
        if not test or len(train) < 3:
            continue

        def features(row: Mapping[str, Any]) -> list[float]:
            length = max(int(row.get("content_token_count") or 0), 1)
            return [1.0, math.log1p(length), float(row.get("finish_reason") == "length")]

        matrix = np.asarray([features(row) for row in train], dtype=np.float64)
        targets = np.asarray([float(row["mean_gap"]) for row in train], dtype=np.float64)
        coefficients, *_ = np.linalg.lstsq(matrix, targets, rcond=None)
        for row in test:
            prediction = float(np.asarray(features(row)) @ coefficients)
            result[str(row["rollout_key"])] = float(row["mean_gap"]) - prediction
    return result


def _pairwise_auc(correct: Sequence[float], wrong: Sequence[float]) -> float:
    credits = [
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in correct
        for negative in wrong
    ]
    return float(np.mean(credits))


def paired_prompt_auc(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    baseline_key: str = "mean_gap",
    seed: int = 42,
    resamples: int = 10_000,
) -> dict[str, Any]:
    """Compare one score with baseline on exactly the same rollout/prompt set."""

    by_prompt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("is_greedy") or row.get("correct") not in {True, False}:
            continue
        if not _finite(row.get(score_key)) or not _finite(row.get(baseline_key)):
            continue
        by_prompt[str(row["sample_uid"])].append(row)
    prompt_rows: list[dict[str, Any]] = []
    for uid, prompt_rows_raw in sorted(by_prompt.items()):
        correct = [row for row in prompt_rows_raw if row.get("correct") is True]
        wrong = [row for row in prompt_rows_raw if row.get("correct") is False]
        if not correct or not wrong:
            continue
        metric_auc = _pairwise_auc(
            [float(row[score_key]) for row in correct],
            [float(row[score_key]) for row in wrong],
        )
        baseline_auc = _pairwise_auc(
            [float(row[baseline_key]) for row in correct],
            [float(row[baseline_key]) for row in wrong],
        )
        prompt_rows.append({
            "sample_uid": uid,
            "rollout_count": len(prompt_rows_raw),
            "correct_count": len(correct),
            "metric_auc": metric_auc,
            "baseline_auc": baseline_auc,
            "delta_auc": metric_auc - baseline_auc,
        })
    if not prompt_rows:
        return {
            "eligible_prompt_count": 0,
            "metric_auc": None,
            "baseline_auc_on_same_rows": None,
            "paired_delta_auc": None,
            "paired_delta_ci95": [None, None],
            "per_prompt": [],
        }
    metric_values = np.asarray([row["metric_auc"] for row in prompt_rows])
    baseline_values = np.asarray([row["baseline_auc"] for row in prompt_rows])
    delta_values = metric_values - baseline_values
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(prompt_rows), size=(resamples, len(prompt_rows)))
    boot = delta_values[samples].mean(axis=1)
    return {
        "eligible_prompt_count": len(prompt_rows),
        "metric_auc": float(metric_values.mean()),
        "baseline_auc_on_same_rows": float(baseline_values.mean()),
        "paired_delta_auc": float(delta_values.mean()),
        "paired_delta_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "per_prompt": prompt_rows,
    }


def run_analysis(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    prefix_horizons: Sequence[int] = DEFAULT_PREFIX_HORIZONS,
    bootstrap_seed: int = 42,
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    source = Path(run_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    required = (source / "rollouts.jsonl", source / "summary.json", source / "k32_validation.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing K32 inputs: {missing}")
    validation = json.loads((source / "k32_validation.json").read_text(encoding="utf-8"))
    if validation.get("valid") is not True or int(validation.get("K") or 0) != 32:
        raise ValueError("gap robustness requires a protocol-valid K=32 run")
    raw_rows = _read_jsonl(source / "rollouts.jsonl")
    scored_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if row.get("is_greedy"):
            continue
        variants = score_variants(row, prefix_horizons=prefix_horizons)
        existing_gap = row.get("teacher_gap")
        if not _finite(existing_gap) or not math.isclose(
            float(existing_gap),
            float(variants["mean_gap"]),
            rel_tol=1e-7,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"{row.get('sample_uid')}: recomputed exact-token mean gap "
                "does not match stored teacher_gap"
            )
        rollout_key = f"{row['sample_uid']}:{int(row.get('rollout_id') or 0)}"
        fold = int(hashlib.sha256(str(row["sample_uid"]).encode()).hexdigest()[:8], 16) % 5
        scored_rows.append({
            "schema_version": SCHEMA_VERSION,
            "rollout_key": rollout_key,
            "sample_uid": str(row["sample_uid"]),
            "rollout_id": int(row.get("rollout_id") or 0),
            "is_greedy": False,
            "correct": row.get("correct"),
            "malformed": bool(row.get("malformed")),
            "finish_reason": row.get("finish_reason"),
            "response_token_count": int(row.get("response_token_count") or 0),
            "content_token_count": int(row.get("content_token_count") or 0),
            "fold": fold,
            **variants,
        })
    residuals = cross_fitted_length_residuals(scored_rows)
    for row in scored_rows:
        row["length_residual_gap"] = residuals.get(str(row["rollout_key"]))
    score_keys = [
        key for key in scored_rows[0]
        if key.endswith("_gap") and key not in {"mean_gap"}
    ] if scored_rows else []
    analyses: dict[str, dict[str, Any]] = {}
    for index, key in enumerate(score_keys):
        metric_seed = bootstrap_seed + index * 1001
        result = paired_prompt_auc(
            scored_rows,
            score_key=key,
            seed=metric_seed,
            resamples=bootstrap_resamples,
        )
        result["by_finish_reason"] = {
            finish_reason: paired_prompt_auc(
                [row for row in scored_rows if row.get("finish_reason") == finish_reason],
                score_key=key,
                seed=metric_seed + 100 + reason_index,
                resamples=bootstrap_resamples,
            )
            for reason_index, finish_reason in enumerate(("stop", "length"))
        }
        length_bins = {
            "le_512": lambda length: length <= 512,
            "513_2048": lambda length: 512 < length <= 2048,
            "gt_2048": lambda length: length > 2048,
        }
        result["by_response_length"] = {
            label: paired_prompt_auc(
                [
                    row for row in scored_rows
                    if predicate(int(row.get("response_token_count") or 0))
                ],
                score_key=key,
                seed=metric_seed + 200 + bin_index,
                resamples=bootstrap_resamples,
            )
            for bin_index, (label, predicate) in enumerate(length_bins.items())
        }
        analyses[key] = result
    git_commit, git_dirty = _git_state()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_run_dir": str(source),
        "source_rollouts_sha256": _sha256_file(source / "rollouts.jsonl"),
        "source_summary_sha256": _sha256_file(source / "summary.json"),
        "source_validation_sha256": _sha256_file(source / "k32_validation.json"),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "stochastic_rollout_count": len(scored_rows),
        "prefix_horizons": list(prefix_horizons),
        "primary_interpretation": (
            "paired_delta compares each alternative against mean_gap on exactly "
            "the same rollouts and eligible prompts; this is diagnostic, not a router gate"
        ),
        "score_analyses": analyses,
    }
    output.mkdir(parents=True)
    _write_jsonl(output / "gap_scores.jsonl", scored_rows)
    _write_json(output / "gap_analysis.json", summary)
    with (output / "gap_metric_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "score_key", "eligible_prompt_count", "metric_auc",
            "baseline_auc_on_same_rows", "paired_delta_auc", "ci95_low", "ci95_high",
        ))
        writer.writeheader()
        for key, result in analyses.items():
            low, high = result["paired_delta_ci95"]
            writer.writerow({
                "score_key": key,
                "eligible_prompt_count": result["eligible_prompt_count"],
                "metric_auc": result["metric_auc"],
                "baseline_auc_on_same_rows": result["baseline_auc_on_same_rows"],
                "paired_delta_auc": result["paired_delta_auc"],
                "ci95_low": low,
                "ci95_high": high,
            })
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix-horizons", nargs="+", type=int, default=list(DEFAULT_PREFIX_HORIZONS))
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_analysis(
            args.run_dir,
            args.output_dir,
            prefix_horizons=tuple(args.prefix_horizons),
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_resamples=args.bootstrap_resamples,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
