"""Join immutable K=32 trajectories, cohort rows, and teacher proposals."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ProbeInput:
    sample_uid: str
    cohort: Mapping[str, Any]
    support_summary: Mapping[str, Any]
    student_rollout: Mapping[str, Any]
    teacher_proposals: tuple[Mapping[str, Any], ...]
    wrong_control_rollout: Mapping[str, Any] | None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _rollout_rank(row: Mapping[str, Any]) -> tuple[float, int, str]:
    logp = row.get("student_mean_logp")
    numeric = float(logp) if logp is not None else float("nan")
    # More likely rollouts sort first; missing/non-finite values sort last.
    first = -numeric if math.isfinite(numeric) else math.inf
    return first, int(row.get("rollout_id") or 0), str(row.get("response_token_hash") or "")


def select_probe_inputs(
    *,
    k32_run_dir: str | Path,
    cohort_dir: str | Path,
    proposal_dir: str | Path | None = None,
    trajectory_outcomes: Sequence[str] = ("correct", "wrong"),
    max_trajectories_per_outcome: int = 1,
    prompt_uids: Sequence[str] | None = None,
    max_prompts: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> tuple[list[ProbeInput], dict[str, Any]]:
    """Build deterministic, outcome-stratified Phase-I work units."""

    allowed = {"correct", "wrong"}
    outcomes = tuple(str(value) for value in trajectory_outcomes)
    if not outcomes or any(value not in allowed for value in outcomes):
        raise ValueError(f"trajectory_outcomes must be a subset of {sorted(allowed)}")
    if max_trajectories_per_outcome <= 0:
        raise ValueError("max_trajectories_per_outcome must be positive")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid shard assignment")
    if max_prompts is not None and max_prompts <= 0:
        raise ValueError("max_prompts must be positive")

    k32 = Path(k32_run_dir).expanduser().resolve()
    cohort = Path(cohort_dir).expanduser().resolve()
    required = (
        k32 / "k32_validation.json",
        k32 / "rollouts.jsonl",
        k32 / "prompt_support_summary.jsonl",
        cohort / "cohort.parquet",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing causal-probe inputs: {missing}")
    validation = json.loads((k32 / "k32_validation.json").read_text(encoding="utf-8"))
    if validation.get("valid") is not True or int(validation.get("K") or 0) != 32:
        raise ValueError("causal-state probe requires a protocol-valid K=32 run")

    summaries = read_jsonl(k32 / "prompt_support_summary.jsonl")
    summary_by_uid: dict[str, dict[str, Any]] = {}
    for row in summaries:
        uid = str(row.get("sample_uid") or "")
        if not uid or uid in summary_by_uid or int(row.get("K") or 0) != 32:
            raise ValueError(f"invalid K=32 support summary for {uid!r}")
        summary_by_uid[uid] = row

    rollout_groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    rollout_rows = read_jsonl(k32 / "rollouts.jsonl")
    student_tokenizer_hashes = {
        str(row.get("student_tokenizer_hash") or row.get("tokenizer_hash") or "")
        for row in rollout_rows
    } - {""}
    teacher_tokenizer_hashes = {
        str(row.get("teacher_tokenizer_hash") or "")
        for row in rollout_rows
    } - {""}
    if len(student_tokenizer_hashes) != 1 or len(teacher_tokenizer_hashes) != 1:
        raise ValueError("K=32 rollouts must expose one student and one teacher tokenizer hash")
    if student_tokenizer_hashes != teacher_tokenizer_hashes:
        raise ValueError("K=32 student/teacher tokenizers were not aligned")
    k32_tokenizer_hash = next(iter(student_tokenizer_hashes))
    for row in rollout_rows:
        uid = str(row.get("sample_uid") or "")
        if uid not in summary_by_uid or row.get("is_greedy") is True:
            continue
        if row.get("exact_token_alignment") is not True or not row.get("response_token_ids"):
            continue
        outcome = "correct" if row.get("correct") is True else "wrong" if row.get("correct") is False else None
        if outcome is None:
            continue
        rollout_groups.setdefault(uid, {"correct": [], "wrong": []})[outcome].append(row)
    for group in rollout_groups.values():
        group["correct"].sort(key=_rollout_rank)
        group["wrong"].sort(key=_rollout_rank)

    proposals_by_uid: dict[str, list[dict[str, Any]]] = {}
    proposal_hash = None
    proposal_tokenizer_hash = None
    if proposal_dir is not None:
        proposal_root = Path(proposal_dir).expanduser().resolve()
        proposal_path = proposal_root / "retained_proposals.jsonl"
        if not proposal_path.is_file():
            raise FileNotFoundError(f"missing retained teacher proposals: {proposal_path}")
        proposal_hash = sha256_file(proposal_path)
        result_paths = sorted((proposal_root / "prompt_results").glob("*.json"))
        if not result_paths:
            raise FileNotFoundError("proposal cache has no prompt_results tokenizer evidence")
        proposal_hashes = {
            str(json.loads(path.read_text(encoding="utf-8")).get("tokenizer_hash") or "")
            for path in result_paths
        } - {""}
        if len(proposal_hashes) != 1:
            raise ValueError("proposal prompt results disagree on tokenizer hash")
        proposal_tokenizer_hash = next(iter(proposal_hashes))
        if proposal_tokenizer_hash != k32_tokenizer_hash:
            raise ValueError("proposal and immutable K=32 tokenizers differ")
        for row in read_jsonl(proposal_path):
            if row.get("correct") is True and row.get("retained_for_fkl"):
                proposals_by_uid.setdefault(str(row.get("sample_uid") or ""), []).append(row)
        for rows in proposals_by_uid.values():
            rows.sort(key=lambda row: (
                int(row.get("reachability_rank") or 10**9),
                int(row.get("proposal_id") or 0),
            ))

    import pandas as pd

    frame = pd.read_parquet(cohort / "cohort.parquet").copy()
    if "sample_uid" not in frame.columns:
        raise ValueError("cohort parquet has no sample_uid column")
    frame.index = frame["sample_uid"].astype(str)
    if frame.index.duplicated().any():
        raise ValueError("cohort parquet contains duplicate sample_uid values")

    requested = None if prompt_uids is None else {str(value) for value in prompt_uids}
    eligible_uids = [
        uid
        for uid, grouped in rollout_groups.items()
        if uid in frame.index
        and (requested is None or uid in requested)
        and any(grouped[outcome] for outcome in outcomes)
    ]
    eligible_uids.sort(key=_selection_key)
    start = len(eligible_uids) * shard_index // num_shards
    end = len(eligible_uids) * (shard_index + 1) // num_shards
    selected_uids = eligible_uids[start:end]
    if max_prompts is not None:
        selected_uids = selected_uids[:max_prompts]

    inputs: list[ProbeInput] = []
    for uid in selected_uids:
        cohort_row = frame.loc[uid].to_dict()
        cohort_row["sample_uid"] = uid
        for outcome in outcomes:
            for rollout in rollout_groups[uid][outcome][:max_trajectories_per_outcome]:
                inputs.append(ProbeInput(
                    sample_uid=uid,
                    cohort=cohort_row,
                    support_summary=summary_by_uid[uid],
                    student_rollout=rollout,
                    teacher_proposals=tuple(proposals_by_uid.get(uid, ())),
                    wrong_control_rollout=(
                        rollout_groups[uid]["wrong"][0]
                        if rollout_groups[uid]["wrong"]
                        else None
                    ),
                ))
    provenance = {
        "k32_validation_sha256": sha256_file(k32 / "k32_validation.json"),
        "k32_rollouts_sha256": sha256_file(k32 / "rollouts.jsonl"),
        "k32_support_summary_sha256": sha256_file(k32 / "prompt_support_summary.jsonl"),
        "cohort_sha256": sha256_file(cohort / "cohort.parquet"),
        "retained_proposals_sha256": proposal_hash,
        "k32_tokenizer_hash": k32_tokenizer_hash,
        "proposal_tokenizer_hash": proposal_tokenizer_hash,
        "trajectory_outcomes": list(outcomes),
        "all_eligible_uids": eligible_uids,
        "selected_uids": selected_uids,
        "work_unit_count": len(inputs),
        "shard_index": shard_index,
        "num_shards": num_shards,
    }
    return inputs, provenance
