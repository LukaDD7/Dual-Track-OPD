"""Offline TOPD-style short-window OT probing on cached trajectories (CPU).

Replicates the *detection* half of TOPD (Jiang & Ferraro, arXiv 2606.00305)
without any regeneration: the student trajectory comes from cached K=32 wrong
rollouts (per-token teacher/student log-probs already aligned), the teacher
trajectory comes from the cached verified teacher proposals, and short-window
OT distances are computed in the student's text-embedding space.  The probe
then tests whether OT-based locators predict Experiment B rescue lifts
(Gate C'), i.e. whether the OT locator can replace the failed gap locator.

This is diagnostic only: it never trains, never regenerates, and never uses
the verifier answer during embedding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .prefix_intervention import prefix_leakage_reason


SCHEMA_VERSION = "support-aware-offline-topd-probe-v1"
DEFAULT_HORIZONS = (64, 128, 256, 512)
DEFAULT_WINDOWS = ((0, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048))
WINDOW_END = 2048


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


def _selection_key(uid: str) -> str:
    return hashlib.sha256(uid.encode()).hexdigest()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class ProbeConfig:
    proposal_dir: str
    k32_run_dir: str
    intervention_dir: str
    student_model_path: str
    output_dir: str
    window_size: int = 50
    stride: int = 1
    high_loss_quantile: float = 0.80
    ot_quantile: float = 0.80
    alignment: str = "warped"
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    bootstrap_seed: int = 42
    bootstrap_resamples: int = 10_000
    max_prompts: int | None = None
    num_shards: int = 1
    shard_index: int = 0
    dtype: str = "float32"
    device: str = "cpu"

    def validate(self) -> None:
        if self.window_size <= 0 or self.stride <= 0:
            raise ValueError("window_size and stride must be positive")
        if not 0 < self.high_loss_quantile < 1 or not 0 < self.ot_quantile < 1:
            raise ValueError("quantiles must lie strictly in (0, 1)")
        if self.alignment not in {"warped", "absolute"}:
            raise ValueError("alignment must be 'warped' or 'absolute'")
        if not self.horizons or any(value <= 0 for value in self.horizons):
            raise ValueError("horizons must be positive")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be unique and increasing")
        if not 0 <= self.shard_index < self.num_shards:
            raise ValueError("invalid shard assignment")
        if self.max_prompts is not None and self.max_prompts <= 0:
            raise ValueError("max_prompts must be positive")


def load_config(path: str | Path, args: argparse.Namespace) -> ProbeConfig:
    raw: dict[str, Any] = {}
    if path:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    probe = raw.get("probe", {})
    data = raw.get("data", {})
    config = ProbeConfig(
        proposal_dir=str(args.proposal_dir or data.get("proposal_dir") or ""),
        k32_run_dir=str(args.k32_run_dir or data.get("k32_run_dir") or ""),
        intervention_dir=str(args.intervention_dir or data.get("intervention_dir") or ""),
        student_model_path=str(args.student_model_path or raw.get("model", {}).get("student") or ""),
        output_dir=str(args.output_dir or raw.get("output", {}).get("dir") or ""),
        window_size=int(args.window_size or probe.get("window_size", 50)),
        stride=int(args.stride or probe.get("stride", 1)),
        high_loss_quantile=float(
            args.high_loss_quantile or probe.get("high_loss_quantile", 0.80)
        ),
        ot_quantile=float(args.ot_quantile or probe.get("ot_quantile", 0.80)),
        alignment=str(args.alignment or probe.get("alignment", "warped")),
        horizons=tuple(
            int(value)
            for value in (args.horizons or probe.get("horizons") or DEFAULT_HORIZONS)
        ),
        bootstrap_seed=int(args.bootstrap_seed or probe.get("bootstrap_seed", 42)),
        bootstrap_resamples=int(
            args.bootstrap_resamples or probe.get("bootstrap_resamples", 10_000)
        ),
        max_prompts=args.max_prompts,
        num_shards=int(args.num_shards),
        shard_index=int(args.shard_index),
        dtype=str(probe.get("dtype", "float32")),
        device=str(probe.get("device", "cpu")),
    )
    config = ProbeConfig(**{
        key: os.path.expandvars(value) if isinstance(value, str) else value
        for key, value in asdict(config).items()
    })
    config.validate()
    return config


def select_probe_records(
    config: ProbeConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    proposal_dir = Path(config.proposal_dir).expanduser().resolve()
    k32_dir = Path(config.k32_run_dir).expanduser().resolve()
    intervention_dir = Path(config.intervention_dir).expanduser().resolve()
    required = (
        proposal_dir / "retained_proposals.jsonl",
        k32_dir / "rollouts.jsonl",
        k32_dir / "k32_validation.json",
        intervention_dir / "rescue_comparisons.jsonl",
        intervention_dir / "minimal_rescue_prefixes.jsonl",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing offline probe inputs: {missing}")
    validation = json.loads((k32_dir / "k32_validation.json").read_text(encoding="utf-8"))
    if validation.get("valid") is not True or int(validation.get("K") or 0) != 32:
        raise ValueError("offline probe requires a protocol-valid K=32 run")

    # Top-1 retained teacher proposal per uid (same selection as Experiment B).
    proposal_by_uid: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(proposal_dir / "retained_proposals.jsonl"):
        uid = str(row.get("sample_uid") or "")
        if row.get("correct") is not True or not row.get("retained_for_fkl"):
            continue
        current = proposal_by_uid.get(uid)
        if current is None or int(row.get("reachability_rank") or 10**9) < int(
            current.get("reachability_rank") or 10**9
        ):
            proposal_by_uid[uid] = row

    # Wrong student rollouts: same filter and ordering as Experiment B
    # (most student-probable wrong path is the primary trajectory).
    wrong_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(k32_dir / "rollouts.jsonl"):
        if (
            not row.get("is_greedy")
            and row.get("correct") is False
            and row.get("exact_token_alignment") is True
            and row.get("response_token_ids")
        ):
            wrong_by_uid[str(row["sample_uid"])].append(row)
    for rows in wrong_by_uid.values():
        rows.sort(key=lambda row: (
            -float(row["student_mean_logp"])
            if row.get("student_mean_logp") is not None
            and math.isfinite(float(row["student_mean_logp"]))
            else math.inf,
            int(row.get("rollout_id") or 0),
            row.get("response_token_hash", ""),
        ))

    # Rescue outcomes from Experiment B.
    rescue_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(intervention_dir / "rescue_comparisons.jsonl"):
        rescue_by_uid[str(row.get("sample_uid") or "")].append(row)
    minimal_by_uid: dict[str, int] = {}
    for row in _read_jsonl(intervention_dir / "minimal_rescue_prefixes.jsonl"):
        if row.get("meets_preregistered_rescue_rule") is True:
            minimal_by_uid[str(row.get("sample_uid") or "")] = int(row["horizon"])

    available_uids = sorted(
        set(proposal_by_uid).intersection(wrong_by_uid).intersection(rescue_by_uid),
        key=_selection_key,
    )
    start = len(available_uids) * config.shard_index // config.num_shards
    end = len(available_uids) * (config.shard_index + 1) // config.num_shards
    shard_uids = available_uids[start:end]
    if config.max_prompts is not None:
        shard_uids = shard_uids[: config.max_prompts]

    records: list[dict[str, Any]] = []
    for uid in shard_uids:
        proposal = proposal_by_uid[uid]
        rescue_rows = rescue_by_uid[uid]
        best_lift = max(
            (float(row["teacher_minus_wrong_posterior_mean"])
             for row in rescue_rows if _finite(row.get("teacher_minus_wrong_posterior_mean"))),
            default=None,
        )
        best_gt_wrong = max(
            (float(row["teacher_gt_wrong_probability"])
             for row in rescue_rows if _finite(row.get("teacher_gt_wrong_probability"))),
            default=None,
        )
        records.append({
            "sample_uid": uid,
            "teacher_proposal": proposal,
            "wrong_rollouts": wrong_by_uid[uid],
            "rescue_rows": rescue_rows,
            "observed_minimal_horizon": minimal_by_uid.get(uid),
            "best_lift": best_lift,
            "best_gt_wrong_probability": best_gt_wrong,
        })
    provenance = {
        "proposal_retained_sha256": _sha256_file(proposal_dir / "retained_proposals.jsonl"),
        "k32_rollouts_sha256": _sha256_file(k32_dir / "rollouts.jsonl"),
        "k32_validation_sha256": _sha256_file(k32_dir / "k32_validation.json"),
        "rescue_comparisons_sha256": _sha256_file(intervention_dir / "rescue_comparisons.jsonl"),
        "minimal_rescue_sha256": _sha256_file(
            intervention_dir / "minimal_rescue_prefixes.jsonl"
        ),
        "all_selected_uids": available_uids,
        "expected_uids": shard_uids,
        "shard_start": start,
        "shard_end": end,
    }
    return records, provenance


def _load_embedder(model_path: str, dtype: str, device: str):
    """Return embed_tokens of the student text tower (CPU, no generation)."""

    import torch
    from transformers import AutoModelForImageTextToText

    torch_dtype = getattr(torch, dtype) if dtype != "float32" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=Path(model_path).exists(),
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    text_model = getattr(model.model, "language_model", None)
    if text_model is not None and hasattr(text_model, "embed_tokens"):
        embed_tokens = text_model.embed_tokens
    elif hasattr(model.model, "embed_tokens"):
        embed_tokens = model.model.embed_tokens
    else:
        raise AttributeError("could not locate text embed_tokens in student model")

    def embed_ids(token_ids: Sequence[int]) -> tuple[np.ndarray, dict[int, int]]:
        unique = sorted({int(value) for value in token_ids})
        index = {token: position for position, token in enumerate(unique)}
        with torch.no_grad():
            vectors = (
                embed_tokens(torch.tensor(unique, dtype=torch.long))
                .float()
                .numpy()
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)
        return vectors, index

    return embed_ids


def _content_arrays(row: Mapping[str, Any]) -> tuple[list[int], np.ndarray]:
    teacher = tuple(float(value) for value in row.get("teacher_sampled_token_log_probs") or ())
    student = tuple(float(value) for value in row.get("student_sampled_token_log_probs") or ())
    token_ids = tuple(int(value) for value in row.get("response_token_ids") or ())
    mask = tuple(bool(value) for value in row.get("content_mask") or ())
    lengths = {len(teacher), len(student), len(token_ids), len(mask)}
    if len(lengths) != 1 or not teacher:
        raise ValueError(f"{row.get('sample_uid')}: token arrays empty or misaligned")
    if row.get("exact_token_alignment") is not True:
        raise ValueError(f"{row.get('sample_uid')}: exact-token alignment is not true")
    indices = [index for index, keep in enumerate(mask) if keep]
    if not indices:
        raise ValueError(f"{row.get('sample_uid')}: no content tokens")
    gaps = np.asarray(
        [teacher[index] - student[index] for index in indices], dtype=np.float64
    )
    if not np.isfinite(gaps).all():
        raise ValueError(f"{row.get('sample_uid')}: non-finite per-token gap")
    return [token_ids[index] for index in indices], gaps


def _content_ids(row: Mapping[str, Any]) -> list[int]:
    """Token IDs under the content mask for a trajectory without aligned scores."""

    token_ids = tuple(int(value) for value in row.get("response_token_ids") or ())
    mask = tuple(bool(value) for value in row.get("content_mask") or ())
    if not token_ids or len(token_ids) != len(mask):
        raise ValueError(f"{row.get('sample_uid')}: token ids/mask empty or misaligned")
    indices = [index for index, keep in enumerate(mask) if keep]
    if not indices:
        raise ValueError(f"{row.get('sample_uid')}: no content tokens")
    return [token_ids[index] for index in indices]


def window_ot_distance(
    teacher_vectors: np.ndarray,
    student_vectors: np.ndarray,
) -> tuple[float, float]:
    """Exact min-cost matching with cosine ground cost (K==K, uniform marginals)."""

    if teacher_vectors.shape != student_vectors.shape:
        raise ValueError("windows must have equal length and embedding width")
    cosine = teacher_vectors @ student_vectors.T
    cosine = np.clip(cosine, -1.0, 1.0)
    cost = 1.0 - cosine
    rows, cols = linear_sum_assignment(cost)
    matched = float(cost[rows, cols].mean())
    return matched, float(1.0 - matched)


def _teacher_start(
    student_start: int,
    teacher_length: int,
    student_length: int,
    window_size: int,
    alignment: str,
) -> int:
    if alignment == "warped":
        start = (student_start * teacher_length) // student_length
    else:
        start = student_start
    return min(max(start, 0), max(teacher_length - window_size, 0))


def compute_prompt_profile(
    record: Mapping[str, Any],
    embed_ids,
    *,
    window_size: int,
    stride: int,
    high_loss_quantile: float,
    ot_quantile: float,
    alignment: str,
    position_pairs: list[tuple[str, float, float]] | None = None,
) -> dict[str, Any]:
    uid = str(record["sample_uid"])
    proposal = record["teacher_proposal"]
    student_row = record["wrong_rollouts"][0]  # most student-probable wrong path
    teacher_ids = _content_ids(proposal)
    student_ids, gaps = _content_arrays(student_row)
    teacher_vectors, teacher_index = embed_ids(teacher_ids)
    student_vectors, student_index = embed_ids(student_ids)

    teacher_length = len(teacher_ids)
    student_length = len(student_ids)
    last_start = max(student_length - window_size, 0)
    starts = list(range(0, last_start + 1, stride))
    ot_values: dict[int, float] = {}
    for start in starts:
        teacher_start = _teacher_start(
            start, teacher_length, student_length, window_size, alignment
        )
        teacher_window = teacher_vectors[
            [teacher_index[teacher_ids[teacher_start + offset]]
             for offset in range(window_size)]
        ]
        student_window = student_vectors[
            [student_index[student_ids[start + offset]]
             for offset in range(window_size)]
        ]
        ot_values[start] = window_ot_distance(teacher_window, student_window)[0]

    valid_starts = np.asarray(sorted(ot_values), dtype=np.int64)
    ot_array = np.asarray([ot_values[int(position)] for position in valid_starts])
    gap_at_valid = gaps[valid_starts]
    loss_threshold = float(np.quantile(gaps, high_loss_quantile))
    ot_threshold = float(np.quantile(ot_array, ot_quantile))
    high_loss = gap_at_valid >= loss_threshold
    high_ot = ot_array >= ot_threshold
    divergent = high_loss & high_ot
    if position_pairs is not None:
        position_pairs.extend(
            (uid, float(left), float(right))
            for left, right in zip(gap_at_valid.tolist(), ot_array.tolist())
        )

    def window_mean(position_mask: np.ndarray, values: np.ndarray) -> float | None:
        selected = values[position_mask]
        return float(selected.mean()) if selected.size else None

    stats: dict[str, Any] = {
        "sample_uid": uid,
        "observed_stratum": proposal.get("observed_stratum"),
        "teacher_content_token_count": teacher_length,
        "student_content_token_count": student_length,
        "wrong_rollout_count": len(record["wrong_rollouts"]),
        "window_count": int(valid_starts.size),
        "loss_threshold": loss_threshold,
        "ot_threshold": ot_threshold,
        "mean_gap": float(gaps.mean()),
        "spearman_gap_ot": float(_spearman(gap_at_valid, ot_array)),
        "pearson_gap_ot": float(np.corrcoef(gap_at_valid, ot_array)[0, 1]),
        "predicted_minimal_horizon": None,
        "observed_minimal_horizon": record.get("observed_minimal_horizon"),
        "best_lift": record.get("best_lift"),
        "best_gt_wrong_probability": record.get("best_gt_wrong_probability"),
    }
    total_divergent = int(divergent.sum())
    stats["divergent_token_count"] = total_divergent
    for start, end in DEFAULT_WINDOWS + ((WINDOW_END, 10**9),):
        label = f"window_{start}_{end}_gap" if end != 10**9 else "window_2048_end_gap"
        mask = (valid_starts >= start) & (valid_starts < end)
        stats[f"ot_{label}"] = window_mean(mask, ot_array)
        stats[f"div_{label}"] = (
            float(divergent[mask].mean()) if mask.any() else None
        )
    for horizon in DEFAULT_HORIZONS:
        prefix_mask = valid_starts < horizon
        stats[f"ot_prefix_{horizon}"] = window_mean(prefix_mask, ot_array)
        prefix_divergent = int(divergent[prefix_mask].sum())
        stats[f"div_prefix_{horizon}_mass"] = (
            float(prefix_divergent / total_divergent) if total_divergent else None
        )
    for horizon in DEFAULT_HORIZONS:
        mass = stats.get(f"div_prefix_{horizon}_mass")
        if mass is not None and mass >= 0.5:
            stats["predicted_minimal_horizon"] = horizon
            break
    if stats["predicted_minimal_horizon"] is None:
        stats["predicted_minimal_horizon"] = int(DEFAULT_HORIZONS[-1])
    return stats


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    if (
        len(left) < 2
        or len(set(float(value) for value in left)) < 2
        or len(set(float(value) for value in right)) < 2
    ):
        return math.nan
    correlation, _ = spearmanr(left, right)
    return float(correlation)


def _spearman_ci(
    pairs: Sequence[tuple[float, float]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    values = [(float(x), float(y)) for x, y in pairs if _finite(x) and _finite(y)]
    observed = _spearman([x for x, _ in values], [y for _, y in values])
    if len(values) < 2:
        return {
            "n": len(values),
            "spearman": None,
            "ci95": [None, None],
            "finite_bootstrap_samples": 0,
        }
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    samples = []
    for _ in range(resamples):
        draw = array[rng.integers(0, len(array), size=len(array))]
        sample = _spearman(draw[:, 0].tolist(), draw[:, 1].tolist())
        if math.isfinite(sample):
            samples.append(sample)
    if not samples:
        return {
            "n": len(values),
            "spearman": observed,
            "ci95": [None, None],
            "finite_bootstrap_samples": 0,
        }
    samples = np.asarray(samples)
    return {
        "n": len(values),
        "spearman": observed,
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "finite_bootstrap_samples": int(samples.size),
    }


def pooled_spearman_ci(
    per_uid_pairs: Mapping[str, Sequence[tuple[float, float]]],
    *,
    seed: int,
    resamples: int = 1000,
    cap_per_prompt: int = 300,
) -> dict[str, Any]:
    """Pooled gap/OT correlation with prompt-level bootstrap (subsampled pairs)."""

    uids = sorted(per_uid_pairs)
    by_uid: dict[str, np.ndarray] = {}
    for uid in uids:
        pairs = [(float(x), float(y)) for x, y in per_uid_pairs[uid]]
        if not pairs:
            continue
        if len(pairs) > cap_per_prompt:
            stride = math.ceil(len(pairs) / cap_per_prompt)
            pairs = pairs[::stride][:cap_per_prompt]
        by_uid[uid] = np.asarray(pairs, dtype=np.float64)
    if len(by_uid) < 2:
        return {"n": len(by_uid), "spearman": None, "ci95": [None, None]}
    uid_list = sorted(by_uid)
    observed = _spearman(
        np.concatenate([by_uid[uid][:, 0] for uid in uid_list]).tolist(),
        np.concatenate([by_uid[uid][:, 1] for uid in uid_list]).tolist(),
    )
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(resamples):
        draw = [
            by_uid[uid_list[index]]
            for index in rng.integers(0, len(uid_list), size=len(uid_list))
        ]
        sample = _spearman(
            np.concatenate([block[:, 0] for block in draw]).tolist(),
            np.concatenate([block[:, 1] for block in draw]).tolist(),
        )
        if math.isfinite(sample):
            samples.append(sample)
    if not samples:
        return {
            "n": len(uid_list),
            "spearman": observed,
            "ci95": [None, None],
            "finite_bootstrap_samples": 0,
        }
    samples = np.asarray(samples)
    return {
        "n": len(uid_list),
        "spearman": observed,
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "finite_bootstrap_samples": int(samples.size),
        "cap_per_prompt": cap_per_prompt,
    }


def analyze_locators(
    profiles: Sequence[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    locators = [
        "div_window_512_1024_gap",
        "ot_window_512_1024_gap",
        "ot_prefix_256",
        "divergent_token_count",
    ]
    outcome_keys = ["best_lift", "best_gt_wrong_probability"]
    results: dict[str, Any] = {"locators": {}, "horizon_match": {}}
    for locator in locators:
        results["locators"][locator] = {}
        for outcome in outcome_keys:
            pairs = [
                (float(profile[locator]), float(profile[outcome]))
                for profile in profiles
                if profile.get(locator) is not None and _finite(profile.get(outcome))
            ]
            results["locators"][locator][outcome] = _spearman_ci(
                pairs, seed=seed, resamples=resamples
            )
    rescued = [
        profile
        for profile in profiles
        if profile.get("observed_minimal_horizon") is not None
    ]
    exact = 0
    within_one = 0
    for profile in rescued:
        predicted = int(profile["predicted_minimal_horizon"])
        observed = int(profile["observed_minimal_horizon"])
        if predicted == observed:
            exact += 1
        steps = DEFAULT_HORIZONS
        if abs(steps.index(predicted) - steps.index(observed)) <= 1:
            within_one += 1
    results["horizon_match"] = {
        "rescued_prompt_count": len(rescued),
        "exact_match_count": exact,
        "within_one_step_count": within_one,
        "per_prompt": [
            {
                "sample_uid": str(profile["sample_uid"]),
                "observed_minimal_horizon": profile.get("observed_minimal_horizon"),
                "predicted_minimal_horizon": profile.get("predicted_minimal_horizon"),
            }
            for profile in rescued
        ],
    }
    return results


def _answer_free_check(
    record: Mapping[str, Any],
    tokenizer,
    *,
    horizons: Sequence[int],
) -> tuple[bool, list[str]]:
    problems: list[str] = []
    proposal = record["teacher_proposal"]
    token_ids = [int(value) for value in proposal.get("response_token_ids") or ()]
    checked = [
        int(record.get("observed_minimal_horizon") or 0)
        if record.get("observed_minimal_horizon") is not None
        else int(horizons[0])
    ]
    for horizon in checked:
        if horizon <= 0 or horizon > len(token_ids):
            problems.append(f"{record['sample_uid']}: horizon {horizon} exceeds source length")
            continue
        prefix = token_ids[:horizon]
        text = tokenizer.decode(
            prefix, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        reason = prefix_leakage_reason(text, proposal.get("gold_answer"))
        if reason is not None:
            problems.append(f"{record['sample_uid']} h={horizon}: {reason}")
    return not problems, problems


def run_analysis(config: ProbeConfig) -> dict[str, Any]:
    records, provenance = select_probe_records(config)
    if not records:
        raise ValueError("no probe records selected")
    embed_ids = _load_embedder(config.student_model_path, config.dtype, config.device)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        config.student_model_path, local_files_only=True
    )
    tokenizer = processor.tokenizer

    profiles: list[dict[str, Any]] = []
    position_pairs: list[tuple[str, float, float]] = []
    for record in records:
        profile = compute_prompt_profile(
            record,
            embed_ids,
            window_size=config.window_size,
            stride=config.stride,
            high_loss_quantile=config.high_loss_quantile,
            ot_quantile=config.ot_quantile,
            alignment=config.alignment,
            position_pairs=position_pairs,
        )
        ok, problems = _answer_free_check(record, tokenizer, horizons=config.horizons)
        profile["answer_free_check_ok"] = ok
        profile["answer_free_check_problems"] = problems
        profiles.append(profile)

    feasibility = {
        "expected_prompt_count": len(records),
        "probed_prompt_count": len(profiles),
        "missing_uids": [
            str(record["sample_uid"])
            for record in records
            if not any(
                profile["sample_uid"] == record["sample_uid"] for profile in profiles
            )
        ],
        "answer_free_failures": [
            profile["sample_uid"]
            for profile in profiles
            if profile.get("answer_free_check_ok") is not True
        ],
    }
    feasibility["complete"] = (
        feasibility["probed_prompt_count"] == feasibility["expected_prompt_count"]
        and not feasibility["missing_uids"]
        and not feasibility["answer_free_failures"]
    )

    locators = analyze_locators(
        profiles, seed=config.bootstrap_seed, resamples=config.bootstrap_resamples
    )
    per_uid_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for uid, gap, ot_value in position_pairs:
        per_uid_pairs[uid].append((gap, ot_value))
    pooled = {
        "per_prompt": [
            {
                "sample_uid": str(profile["sample_uid"]),
                "spearman_gap_ot": profile.get("spearman_gap_ot"),
            }
            for profile in profiles
        ],
        "pooled": pooled_spearman_ci(
            per_uid_pairs,
            seed=config.bootstrap_seed,
            resamples=min(config.bootstrap_resamples, 1000),
        ),
    }

    primary = locators["locators"]["div_window_512_1024_gap"]["best_lift"]
    primary_pass = bool(
        primary.get("spearman") is not None
        and primary["spearman"] > 0
        and primary["ci95"][0] is not None
        and primary["ci95"][0] > 0
    )
    horizon = locators["horizon_match"]
    horizon_pass = bool(
        horizon["exact_match_count"] >= 3 and horizon["within_one_step_count"] >= 5
    )
    g2 = pooled.get("pooled") or {}
    g2_pass = bool(
        g2.get("spearman") is not None
        and g2["spearman"] > 0
        and g2["ci95"][0] is not None
        and g2["ci95"][0] > 0
    )
    g3_pass = primary_pass or horizon_pass
    gates = {
        "G1_data_feasibility": {
            "pass": feasibility["complete"],
            "detail": feasibility,
        },
        "G2_pooled_loss_ot_sanity": {
            "pass": g2_pass,
            "detail": g2,
        },
        "G3_ot_locator_predicts_rescue": {
            "pass": g3_pass,
            "primary_locator_pass": primary_pass,
            "horizon_match_pass": horizon_pass,
            "detail": locators,
        },
    }
    decision = (
        "GO: design minimal-bridge arm with per-prompt horizon from OT profile"
        if g3_pass
        else "NO-GO: offline OT locator does not recover Gate C; keep rare_success verified-FKL"
    )

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "probe_rows.jsonl", profiles)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": _git_state()[0],
        "git_dirty": _git_state()[1],
        "complete": feasibility["complete"],
        "expected_prompt_count": len(records),
        "completed_prompt_count": len(profiles),
        "completed_uids": [str(record["sample_uid"]) for record in records],
        "gates": gates,
        "decision": decision,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "locator_analysis.json", {
        "gates": gates,
        "decision": decision,
        "provenance": provenance,
    })
    _write_json(output_dir / "run_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "provenance": provenance,
        "git_commit": _git_state()[0],
        "git_dirty": _git_state()[1],
    })
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run the offline TOPD probe")
    run.add_argument("--config", default=None)
    run.add_argument("--proposal-dir", default=None)
    run.add_argument("--k32-run-dir", default=None)
    run.add_argument("--intervention-dir", default=None)
    run.add_argument("--student-model-path", default=None)
    run.add_argument("--output-dir", default=None)
    run.add_argument("--window-size", type=int, default=None)
    run.add_argument("--stride", type=int, default=None)
    run.add_argument("--high-loss-quantile", type=float, default=None)
    run.add_argument("--ot-quantile", type=float, default=None)
    run.add_argument("--alignment", default=None)
    run.add_argument("--horizons", type=int, nargs="+", default=None)
    run.add_argument("--bootstrap-seed", type=int, default=None)
    run.add_argument("--bootstrap-resamples", type=int, default=None)
    run.add_argument("--max-prompts", type=int, default=None)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        config = load_config(args.config, args)
        summary = run_analysis(config)
        print(json.dumps({k: summary.get(k) for k in (
            "complete", "expected_prompt_count", "completed_prompt_count", "decision"
        )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
