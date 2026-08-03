"""Post-hoc support/frontier analysis for frozen-policy diagnostic runs.

The diagnostic deliberately keeps the greedy rollout separate from the
stochastic support estimate.  This module consumes only the ``K`` stochastic
rollouts recorded in ``prompt_support_summary.jsonl`` and estimates the chance
that a fresh RL group of size ``G`` contains both a success and a failure:

    U_G(p) = 1 - p**G - (1 - p)**G.

``U_G`` is an operational prediction of a gradient-bearing binary-reward
group, not a claim that every mixed group produces a useful optimization
step.  A Jeffreys Beta posterior avoids treating ``0/K`` and ``K/K`` as known
probabilities of exactly zero and one.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_PRIOR_ALPHA = 0.5
DEFAULT_PRIOR_BETA = 0.5
DEFAULT_GROUP_SIZE = 8
DEFAULT_MC_SAMPLES = 20_000
DEFAULT_SEED = 42
DEFAULT_USEFUL_GROUP_THRESHOLD = 0.5
DEFAULT_READY_POSTERIOR_PROBABILITY = 0.8


@dataclass(frozen=True)
class FrontierConfig:
    """Analysis settings recorded with every output artifact."""

    group_size: int = DEFAULT_GROUP_SIZE
    prior_alpha: float = DEFAULT_PRIOR_ALPHA
    prior_beta: float = DEFAULT_PRIOR_BETA
    mc_samples: int = DEFAULT_MC_SAMPLES
    seed: int = DEFAULT_SEED
    useful_group_threshold: float = DEFAULT_USEFUL_GROUP_THRESHOLD
    ready_posterior_probability: float = DEFAULT_READY_POSTERIOR_PROBABILITY

    def validate(self) -> None:
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2")
        if self.prior_alpha <= 0 or self.prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive")
        if self.mc_samples < 100:
            raise ValueError("mc_samples must be at least 100")
        if not 0 <= self.useful_group_threshold <= 1:
            raise ValueError("useful_group_threshold must be in [0, 1]")
        if not 0 <= self.ready_posterior_probability <= 1:
            raise ValueError("ready_posterior_probability must be in [0, 1]")


def useful_group_probability(pass_rate: float, group_size: int) -> float:
    """Return the probability that a binary-reward group is mixed."""

    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    if not 0 <= pass_rate <= 1:
        raise ValueError("pass_rate must be in [0, 1]")
    return 1.0 - pass_rate**group_size - (1.0 - pass_rate) ** group_size


def _beta_moment(alpha: float, beta: float, power: int) -> float:
    """Compute E[p**power] for p ~ Beta(alpha, beta)."""

    return math.exp(
        math.lgamma(alpha + power)
        + math.lgamma(alpha + beta)
        - math.lgamma(alpha)
        - math.lgamma(alpha + beta + power)
    )


def posterior_expected_useful_group_probability(
    alpha: float,
    beta: float,
    group_size: int,
) -> float:
    """Compute the exact posterior expectation of ``U_G(p)``."""

    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta posterior parameters must be positive")
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    return 1.0 - _beta_moment(alpha, beta, group_size) - _beta_moment(
        beta, alpha, group_size
    )


def observed_support_stratum(correct_count: int, K: int) -> str:
    """Assign a transparent observed-count stratum independent of greedy."""

    if K <= 0:
        raise ValueError("K must be positive")
    if not 0 <= correct_count <= K:
        raise ValueError("correct_count must be in [0, K]")
    if correct_count == 0:
        return "no_correct_observed"
    if correct_count == K:
        return "all_correct_observed"
    if correct_count / K <= 0.25:
        return "rare_success"
    return "mixed_support"


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sequence")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def analyze_prompt(
    row: Mapping[str, Any],
    config: FrontierConfig,
    *,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Add posterior support and predicted RL-group fields to one prompt."""

    config.validate()
    uid = str(row.get("sample_uid") or "")
    if not uid:
        raise ValueError("prompt row is missing sample_uid")
    K = int(row.get("K") or 0)
    correct_count = int(row.get("correct_count") or 0)
    if K <= 0 or not 0 <= correct_count <= K:
        raise ValueError(
            f"{uid}: invalid stochastic counts correct_count={correct_count}, K={K}"
        )

    posterior_alpha = correct_count + config.prior_alpha
    posterior_beta = K - correct_count + config.prior_beta
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    observed_rate = correct_count / K
    expected_useful = posterior_expected_useful_group_probability(
        posterior_alpha,
        posterior_beta,
        config.group_size,
    )
    plugin_useful = useful_group_probability(observed_rate, config.group_size)

    rng = random.Random(config.seed + seed_offset)
    pass_rate_samples = sorted(
        rng.betavariate(posterior_alpha, posterior_beta)
        for _ in range(config.mc_samples)
    )
    useful_samples = sorted(
        useful_group_probability(value, config.group_size)
        for value in pass_rate_samples
    )
    ready_probability = sum(
        value >= config.useful_group_threshold for value in useful_samples
    ) / config.mc_samples

    return {
        "sample_uid": uid,
        "K": K,
        "correct_count": correct_count,
        "greedy_correct": row.get("greedy_correct"),
        "legacy_support_state": row.get("support_state"),
        "observed_support_stratum": observed_support_stratum(correct_count, K),
        "observed_pass_rate": observed_rate,
        "posterior_alpha": posterior_alpha,
        "posterior_beta": posterior_beta,
        "posterior_mean_pass_rate": posterior_mean,
        "posterior_pass_rate_ci95": [
            _quantile(pass_rate_samples, 0.025),
            _quantile(pass_rate_samples, 0.975),
        ],
        "rl_group_size": config.group_size,
        "plugin_useful_group_probability": plugin_useful,
        "posterior_expected_useful_group_probability": expected_useful,
        "posterior_useful_group_probability_ci95": [
            _quantile(useful_samples, 0.025),
            _quantile(useful_samples, 0.975),
        ],
        "posterior_probability_above_useful_threshold": ready_probability,
        "rl_ready": ready_probability >= config.ready_posterior_probability,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no prompt rows")
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _analysis_provenance(source: Path) -> dict[str, Any]:
    """Record the immutable input and analyzer checkout in output summaries."""

    repo_root = Path(__file__).resolve().parents[3]
    commit = "unavailable"
    dirty: bool | None = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return {
        "source_prompt_summary": str(source),
        "source_prompt_summary_sha256": _sha256_file(source),
        "analysis_git_commit": commit,
        "analysis_git_dirty": dirty,
    }


def analyze_rows(
    rows: Sequence[Mapping[str, Any]],
    config: FrontierConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Analyze prompt rows and return prompt-level and aggregate artifacts."""

    config.validate()
    analyzed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        result = analyze_prompt(row, config, seed_offset=index * 1_000_003)
        uid = result["sample_uid"]
        if uid in seen:
            raise ValueError(f"duplicate sample_uid: {uid}")
        seen.add(uid)
        analyzed.append(result)

    posterior_values = [
        float(row["posterior_expected_useful_group_probability"])
        for row in analyzed
    ]
    plugin_values = [float(row["plugin_useful_group_probability"]) for row in analyzed]
    stratum_counts = Counter(str(row["observed_support_stratum"]) for row in analyzed)
    legacy_counts = Counter(str(row["legacy_support_state"]) for row in analyzed)
    summary = {
        "schema_version": "support_frontier_v1",
        "interpretation": (
            "Predicted probability that a fresh binary-reward RL group is mixed; "
            "greedy rollout excluded from the support posterior."
        ),
        "config": asdict(config),
        "num_prompts": len(analyzed),
        "posterior_frontier_mass": sum(posterior_values),
        "mean_posterior_useful_group_probability": sum(posterior_values)
        / len(posterior_values),
        "plugin_frontier_mass": sum(plugin_values),
        "mean_plugin_useful_group_probability": sum(plugin_values)
        / len(plugin_values),
        "rl_ready_count": sum(bool(row["rl_ready"]) for row in analyzed),
        "observed_support_stratum_counts": dict(sorted(stratum_counts.items())),
        "legacy_support_state_counts": dict(sorted(legacy_counts.items())),
    }
    return analyzed, summary


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int,
) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sequence")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(resamples)
    )
    return [_quantile(means, 0.025), _quantile(means, 0.975)]


def compare_analyses(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    config: FrontierConfig,
    *,
    bootstrap_resamples: int = 10_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare matched pre/post support analyses without claiming mediation."""

    if bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")
    before_by_uid = {str(row["sample_uid"]): row for row in before}
    after_by_uid = {str(row["sample_uid"]): row for row in after}
    if len(before_by_uid) != len(before) or len(after_by_uid) != len(after):
        raise ValueError("duplicate sample_uid in before or after analysis")
    before_uids = set(before_by_uid)
    after_uids = set(after_by_uid)
    if before_uids != after_uids:
        missing_after = sorted(before_uids - after_uids)
        missing_before = sorted(after_uids - before_uids)
        raise ValueError(
            "pre/post UID sets differ; "
            f"missing_after={missing_after[:5]}, missing_before={missing_before[:5]}"
        )

    transitions: defaultdict[str, Counter[str]] = defaultdict(Counter)
    compared: list[dict[str, Any]] = []
    for uid in sorted(before_uids):
        pre = before_by_uid[uid]
        post = after_by_uid[uid]
        pre_stratum = str(pre["observed_support_stratum"])
        post_stratum = str(post["observed_support_stratum"])
        transitions[pre_stratum][post_stratum] += 1
        pre_u = float(pre["posterior_expected_useful_group_probability"])
        post_u = float(post["posterior_expected_useful_group_probability"])
        compared.append(
            {
                "sample_uid": uid,
                "before_stratum": pre_stratum,
                "after_stratum": post_stratum,
                "before_correct_count": int(pre["correct_count"]),
                "after_correct_count": int(post["correct_count"]),
                "before_posterior_mean_pass_rate": float(
                    pre["posterior_mean_pass_rate"]
                ),
                "after_posterior_mean_pass_rate": float(
                    post["posterior_mean_pass_rate"]
                ),
                "delta_posterior_mean_pass_rate": float(
                    post["posterior_mean_pass_rate"]
                )
                - float(pre["posterior_mean_pass_rate"]),
                "before_expected_useful_group_probability": pre_u,
                "after_expected_useful_group_probability": post_u,
                "delta_expected_useful_group_probability": post_u - pre_u,
                "before_rl_ready": bool(pre["rl_ready"]),
                "after_rl_ready": bool(post["rl_ready"]),
            }
        )

    deltas = [float(row["delta_expected_useful_group_probability"]) for row in compared]
    before_mass = sum(
        float(row["before_expected_useful_group_probability"]) for row in compared
    )
    after_mass = sum(
        float(row["after_expected_useful_group_probability"]) for row in compared
    )
    summary = {
        "schema_version": "support_frontier_comparison_v1",
        "interpretation": (
            "Matched descriptive support transition. A positive delta predicts "
            "more mixed RL groups but is not by itself causal mediation evidence."
        ),
        "config": asdict(config),
        "num_matched_prompts": len(compared),
        "before_posterior_frontier_mass": before_mass,
        "after_posterior_frontier_mass": after_mass,
        "delta_posterior_frontier_mass": after_mass - before_mass,
        "mean_delta_expected_useful_group_probability": sum(deltas) / len(deltas),
        "mean_delta_ci95_prompt_bootstrap": _bootstrap_mean_ci(
            deltas,
            seed=config.seed + 9_999_991,
            resamples=bootstrap_resamples,
        ),
        "newly_rl_ready_count": sum(
            not row["before_rl_ready"] and row["after_rl_ready"] for row in compared
        ),
        "lost_rl_ready_count": sum(
            row["before_rl_ready"] and not row["after_rl_ready"] for row in compared
        ),
        "newly_observed_success_count": sum(
            row["before_correct_count"] == 0 and row["after_correct_count"] > 0
            for row in compared
        ),
        "became_all_correct_observed_count": sum(
            row["after_stratum"] == "all_correct_observed"
            and row["before_stratum"] != "all_correct_observed"
            for row in compared
        ),
        "transition_matrix": {
            source: dict(sorted(destinations.items()))
            for source, destinations in sorted(transitions.items())
        },
    }
    return compared, summary


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_prompt_summary(run_or_file: str) -> Path:
    path = Path(run_or_file).expanduser().resolve()
    if path.is_dir():
        path = path / "prompt_support_summary.jsonl"
    return path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate and compare posterior mixed-group frontier mass."
    )
    parser.add_argument(
        "run",
        help="Diagnostic run directory or prompt_support_summary.jsonl file.",
    )
    parser.add_argument(
        "--after-run",
        help="Optional matched post-bridge run directory/file for transitions.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Output directory. Defaults to <run>/frontier_analysis for one run "
            "or <after-run>/frontier_comparison for a comparison."
        ),
    )
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--prior-alpha", type=float, default=DEFAULT_PRIOR_ALPHA)
    parser.add_argument("--prior-beta", type=float, default=DEFAULT_PRIOR_BETA)
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--useful-group-threshold",
        type=float,
        default=DEFAULT_USEFUL_GROUP_THRESHOLD,
    )
    parser.add_argument(
        "--ready-posterior-probability",
        type=float,
        default=DEFAULT_READY_POSTERIOR_PROBABILITY,
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = FrontierConfig(
        group_size=args.group_size,
        prior_alpha=args.prior_alpha,
        prior_beta=args.prior_beta,
        mc_samples=args.mc_samples,
        seed=args.seed,
        useful_group_threshold=args.useful_group_threshold,
        ready_posterior_probability=args.ready_posterior_probability,
    )
    before_file = _resolve_prompt_summary(args.run)
    before_rows = _read_jsonl(before_file)
    before_analysis, before_summary = analyze_rows(before_rows, config)
    before_summary["provenance"] = _analysis_provenance(before_file)

    if args.after_run:
        after_file = _resolve_prompt_summary(args.after_run)
        after_rows = _read_jsonl(after_file)
        after_analysis, after_summary = analyze_rows(after_rows, config)
        after_summary["provenance"] = _analysis_provenance(after_file)
        compared, comparison = compare_analyses(
            before_analysis,
            after_analysis,
            config,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        comparison["before_provenance"] = before_summary["provenance"]
        comparison["after_provenance"] = after_summary["provenance"]
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else after_file.parent / "frontier_comparison"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "before_frontier_summary.json", before_summary)
        _write_json(output_dir / "after_frontier_summary.json", after_summary)
        _write_json(output_dir / "frontier_comparison.json", comparison)
        _write_jsonl(output_dir / "frontier_transitions.jsonl", compared)
        _write_csv(output_dir / "frontier_transitions.csv", compared)
        print(json.dumps(comparison, indent=2, sort_keys=True))
    else:
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else before_file.parent / "frontier_analysis"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "frontier_summary.json", before_summary)
        _write_jsonl(output_dir / "frontier_prompts.jsonl", before_analysis)
        _write_csv(output_dir / "frontier_prompts.csv", before_analysis)
        print(json.dumps(before_summary, indent=2, sort_keys=True))
    print(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
