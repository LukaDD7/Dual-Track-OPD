"""Build and validate the fixed K=32 support-confirmation cohort.

The K=8 diagnostic is a screening measurement.  This module freezes a
deterministic, length-balanced cohort across observed stochastic support strata
and materializes a small parquet that the existing exact-token diagnostic can
consume without adding a second dataset path to the training code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from dual_track_opd.fc_opd.dataset_signal_audit import hash_token_ids


SCHEMA_VERSION = "support-aware-k32-cohort-v1"
STRATUM_ORDER = (
    "no_correct_observed",
    "rare_success",
    "mixed_support",
    "all_correct_observed",
)


def observed_stratum(correct_count: int, K: int) -> str:
    """Classify stochastic support without using the greedy verdict."""

    if K <= 0:
        raise ValueError("K must be positive")
    if not 0 <= correct_count <= K:
        raise ValueError(f"correct_count must be in [0, K], got {correct_count}/{K}")
    if correct_count == 0:
        return "no_correct_observed"
    if correct_count == K:
        return "all_correct_observed"
    if correct_count / K <= 0.25:
        return "rare_success"
    return "mixed_support"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_key(seed: int, sample_uid: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_uid}".encode()).hexdigest()


def _diagnostic_uid_order(sample_uids: Sequence[str]) -> list[str]:
    return sorted(sample_uids, key=lambda uid: hashlib.sha256(uid.encode()).hexdigest())


def build_candidates(
    prompt_summaries: Sequence[Mapping[str, Any]],
    rollouts: Sequence[Mapping[str, Any]],
    *,
    length_cutoff: int = 2048,
    seed: int = 20260804,
) -> list[dict[str, Any]]:
    """Join K=8 summaries to rollout quality statistics."""

    stochastic_by_uid: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for rollout in rollouts:
        if rollout.get("is_greedy"):
            continue
        uid = str(rollout.get("sample_uid") or "")
        if uid:
            stochastic_by_uid[uid].append(rollout)

    candidates: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    for summary in prompt_summaries:
        uid = str(summary.get("sample_uid") or "")
        if not uid or uid in seen_uids:
            raise ValueError(f"empty or duplicate sample_uid in prompt summaries: {uid!r}")
        seen_uids.add(uid)
        K = int(summary.get("K") or 0)
        correct_count = int(summary.get("correct_count") or 0)
        prompt_rollouts = stochastic_by_uid.get(uid, [])
        if len(prompt_rollouts) != K:
            raise ValueError(
                f"{uid}: expected {K} stochastic rollouts, found {len(prompt_rollouts)}"
            )
        lengths = [int(row.get("response_token_count") or 0) for row in prompt_rollouts]
        if any(length <= 0 for length in lengths):
            raise ValueError(f"{uid}: missing response_token_count")
        median_tokens = float(statistics.median(lengths))
        finish_length_count = sum(
            1 for row in prompt_rollouts if row.get("finish_reason") == "length"
        )
        malformed_count = sum(1 for row in prompt_rollouts if row.get("malformed"))
        length_bucket = "le_cutoff" if median_tokens <= length_cutoff else "gt_cutoff"
        candidates.append({
            "sample_uid": uid,
            "screening_K": K,
            "screening_correct_count": correct_count,
            "screening_pass_rate": correct_count / K,
            "observed_stratum": observed_stratum(correct_count, K),
            "legacy_support_state": summary.get("support_state"),
            "greedy_correct": summary.get("greedy_correct"),
            "median_stochastic_response_tokens": median_tokens,
            "stochastic_truncation_rate": finish_length_count / K,
            "stochastic_malformed_rate": malformed_count / K,
            "length_bucket": length_bucket,
            "selection_key": _selection_key(seed, uid),
        })
    return candidates


def select_balanced_cohort(
    candidates: Sequence[Mapping[str, Any]],
    *,
    per_stratum: int = 16,
) -> list[dict[str, Any]]:
    """Select equal strata and approximately equal short/long prompts."""

    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive")
    target_short = per_stratum // 2
    target_long = per_stratum - target_short
    selected: list[dict[str, Any]] = []

    for stratum in STRATUM_ORDER:
        pool = [dict(row) for row in candidates if row["observed_stratum"] == stratum]
        if len(pool) < per_stratum:
            raise ValueError(
                f"stratum {stratum!r} has {len(pool)} candidates; need {per_stratum}"
            )
        short = sorted(
            (row for row in pool if row["length_bucket"] == "le_cutoff"),
            key=lambda row: row["selection_key"],
        )
        long = sorted(
            (row for row in pool if row["length_bucket"] == "gt_cutoff"),
            key=lambda row: row["selection_key"],
        )
        chosen = short[:target_short] + long[:target_long]
        chosen_uids = {row["sample_uid"] for row in chosen}
        if len(chosen) < per_stratum:
            backfill = sorted(
                (row for row in pool if row["sample_uid"] not in chosen_uids),
                key=lambda row: row["selection_key"],
            )
            chosen.extend(backfill[: per_stratum - len(chosen)])
        if len(chosen) != per_stratum:
            raise RuntimeError(f"failed to fill stratum {stratum!r}")
        for rank, row in enumerate(sorted(chosen, key=lambda x: x["selection_key"]), start=1):
            row["selection_rank_within_stratum"] = rank
            selected.append(row)

    return selected


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


def prepare_cohort(
    diagnostic_run: str | Path,
    source_dataset: str | Path,
    output_dir: str | Path,
    *,
    per_stratum: int = 16,
    length_cutoff: int = 2048,
    seed: int = 20260804,
    num_shards: int = 4,
) -> dict[str, Any]:
    """Materialize a deterministic cohort parquet and provenance manifest."""

    run = Path(diagnostic_run).expanduser().resolve()
    dataset = Path(source_dataset).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    required = (
        run / "prompt_support_summary.jsonl",
        run / "rollouts.jsonl",
        run / "summary.json",
        run / "run_manifest.json",
        dataset,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required inputs: {missing}")

    prompt_summaries = _read_jsonl(run / "prompt_support_summary.jsonl")
    rollouts = _read_jsonl(run / "rollouts.jsonl")
    source_summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    source_run_manifest = json.loads(
        (run / "run_manifest.json").read_text(encoding="utf-8")
    )
    source_protocol_checks = {
        "complete_exit_status": source_run_manifest.get("exit_status") in {"PASS", "GATE_FAIL"},
        "exact_token_alignment": float(
            source_summary.get("exact_token_alignment_rate") or 0.0
        ) == 1.0,
        "prompt_token_hash_available": float(
            source_summary.get("prompt_token_hash_rate") or 0.0
        ) == 1.0,
        "complete_rollout_coverage": float(
            source_summary.get("shard_completeness_rate") or 0.0
        ) == 1.0,
        "zero_missing_images": int(source_summary.get("missing_image_count") or 0) == 0,
        "finite_scores": int(source_summary.get("non_finite_score_count") or 0) == 0,
    }
    failed_source_checks = [
        name for name, passed in source_protocol_checks.items() if not passed
    ]
    if failed_source_checks:
        raise ValueError(
            "source diagnostic violates required protocol checks: "
            f"{failed_source_checks}"
        )
    screening_k_values = {int(row.get("K") or 0) for row in prompt_summaries}
    if screening_k_values != {8}:
        raise ValueError(f"source diagnostic must be K=8, got {screening_k_values}")
    candidates = build_candidates(
        prompt_summaries,
        rollouts,
        length_cutoff=length_cutoff,
        seed=seed,
    )
    selected = select_balanced_cohort(candidates, per_stratum=per_stratum)
    selected_uids = [str(row["sample_uid"]) for row in selected]
    diagnostic_order = _diagnostic_uid_order(selected_uids)

    import pandas as pd

    frame = pd.read_parquet(dataset)
    if "sample_uid" not in frame.columns:
        raise ValueError("source dataset has no sample_uid column")
    uid_series = frame["sample_uid"].astype(str)
    if uid_series.duplicated().any():
        raise ValueError("source dataset contains duplicate sample_uid values")
    indexed = frame.copy()
    indexed.index = uid_series
    missing_uids = sorted(set(selected_uids) - set(indexed.index))
    if missing_uids:
        raise ValueError(f"selected UIDs missing from source dataset: {missing_uids[:10]}")
    cohort_frame = indexed.loc[diagnostic_order].reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        cohort_path = temp_dir / "cohort.parquet"
        cohort_frame.to_parquet(cohort_path, index=False)
        commit, dirty = _git_state()
        interval_boundaries = [
            {
                "shard_index": shard_index,
                "prompt_start": len(diagnostic_order) * shard_index // num_shards,
                "prompt_end": len(diagnostic_order) * (shard_index + 1) // num_shards,
            }
            for shard_index in range(num_shards)
        ]
        selected_by_uid = {str(row["sample_uid"]): dict(row) for row in selected}
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "git_dirty": dirty,
            "source_diagnostic_run": str(run),
            "source_dataset": str(dataset),
            "source_prompt_summary_sha256": _sha256_file(run / "prompt_support_summary.jsonl"),
            "source_rollouts_sha256": _sha256_file(run / "rollouts.jsonl"),
            "source_summary_sha256": _sha256_file(run / "summary.json"),
            "source_dataset_sha256": _sha256_file(dataset),
            "source_run_git_commit": source_run_manifest.get("git_commit"),
            "source_run_git_dirty": source_run_manifest.get("git_dirty"),
            "source_protocol_checks": source_protocol_checks,
            "cohort_dataset_sha256": _sha256_file(cohort_path),
            "selection_seed": seed,
            "length_cutoff": length_cutoff,
            "per_stratum": per_stratum,
            "num_prompts": len(diagnostic_order),
            "num_shards": num_shards,
            "diagnostic_uid_order": diagnostic_order,
            "diagnostic_selection_sha256": hashlib.sha256(
                json.dumps(diagnostic_order, sort_keys=True).encode()
            ).hexdigest(),
            "shard_plan": interval_boundaries,
            "selected_prompts": [selected_by_uid[uid] for uid in diagnostic_order],
        }
        (temp_dir / "cohort_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        fieldnames = list(manifest["selected_prompts"][0])
        with (temp_dir / "cohort_prompts.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest["selected_prompts"])
        temp_dir.rename(output)
    except Exception:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return manifest


def validate_merged_run(
    run_dir: str | Path,
    cohort_dir: str | Path,
    *,
    expected_K: int = 32,
) -> dict[str, Any]:
    """Fail closed on incomplete or token-misaligned merged K=32 output."""

    run = Path(run_dir).expanduser().resolve()
    cohort = Path(cohort_dir).expanduser().resolve()
    required = (
        run / "summary.json",
        run / "run_manifest.json",
        run / "rollouts.jsonl",
        run / "prompt_support_summary.jsonl",
        run / "selected_prompts.jsonl",
        cohort / "cohort_manifest.json",
        cohort / "cohort.parquet",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing validation inputs: {missing}")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    cohort_manifest = json.loads(
        (cohort / "cohort_manifest.json").read_text(encoding="utf-8")
    )
    rollouts = _read_jsonl(run / "rollouts.jsonl")
    prompt_summaries = _read_jsonl(run / "prompt_support_summary.jsonl")
    selected_prompts = _read_jsonl(run / "selected_prompts.jsonl")

    expected_uids = list(cohort_manifest["diagnostic_uid_order"])
    actual_uids = [str(row.get("sample_uid") or "") for row in selected_prompts]
    errors: list[str] = []
    if actual_uids != expected_uids:
        errors.append("selected prompt order/content differs from cohort manifest")
    if len(prompt_summaries) != len(expected_uids):
        errors.append("prompt summary count differs from cohort manifest")
    if int(summary.get("rollouts_per_prompt") or 0) != expected_K:
        errors.append("summary K differs from expected K")
    if int(run_manifest.get("rollouts_per_prompt") or 0) != expected_K:
        errors.append("run manifest K differs from expected K")
    if run_manifest.get("exit_status") not in {"PASS", "GATE_FAIL"}:
        errors.append(f"run is not complete: {run_manifest.get('exit_status')}")
    if _sha256_file(cohort / "cohort.parquet") != cohort_manifest["cohort_dataset_sha256"]:
        errors.append("cohort parquet hash differs from cohort manifest")

    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollouts:
        uid = str(row.get("sample_uid") or "")
        by_uid[uid].append(row)
        response_ids = tuple(int(token_id) for token_id in row.get("response_token_ids") or ())
        response_hash = hash_token_ids(response_ids) if response_ids else ""
        if (
            not response_ids
            or row.get("exact_token_alignment") is not True
            or not row.get("prompt_token_hash")
            or row.get("response_token_hash") != response_hash
            or row.get("teacher_scored_token_hash") != response_hash
            or row.get("student_scored_token_hash") != response_hash
        ):
            errors.append(f"{uid}:rollout-{row.get('rollout_id')}: token contract failure")
            if len(errors) >= 20:
                break
    expected_keys = {(True, 0)} | {(False, index) for index in range(1, expected_K + 1)}
    for uid in expected_uids:
        keys = {
            (bool(row.get("is_greedy")), int(row.get("rollout_id", -1)))
            for row in by_uid.get(uid, [])
        }
        if keys != expected_keys:
            errors.append(f"{uid}: incomplete greedy+K rollout key set")

    gate_failures = [
        {"name": name, "detail": detail}
        for name, passed, detail in summary.get("acceptance_gates", [])
        if not passed
    ]
    result = {
        "schema_version": "support-aware-k32-validation-v1",
        "valid": not errors,
        "run_dir": str(run),
        "cohort_dir": str(cohort),
        "num_prompts": len(expected_uids),
        "K": expected_K,
        "rollout_rows": len(rollouts),
        "gate_failures_are_reported_not_suppressed": gate_failures,
        "protocol_errors": errors,
    }
    (run / "k32_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if errors:
        raise ValueError("; ".join(errors[:10]))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build the immutable K=32 cohort")
    build.add_argument("--diagnostic-run", required=True)
    build.add_argument("--source-dataset", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--per-stratum", type=int, default=16)
    build.add_argument("--length-cutoff", type=int, default=2048)
    build.add_argument("--seed", type=int, default=20260804)
    build.add_argument("--num-shards", type=int, default=4)

    validate = subparsers.add_parser("validate", help="validate a merged K=32 run")
    validate.add_argument("--run-dir", required=True)
    validate.add_argument("--cohort-dir", required=True)
    validate.add_argument("--expected-k", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            result = prepare_cohort(
                args.diagnostic_run,
                args.source_dataset,
                args.output_dir,
                per_stratum=args.per_stratum,
                length_cutoff=args.length_cutoff,
                seed=args.seed,
                num_shards=args.num_shards,
            )
        else:
            result = validate_merged_run(
                args.run_dir,
                args.cohort_dir,
                expected_K=args.expected_k,
            )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
