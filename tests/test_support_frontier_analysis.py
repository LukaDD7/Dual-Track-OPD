import json
import math

import pytest

from dual_track_opd.support_aware.frontier_analysis import (
    FrontierConfig,
    analyze_rows,
    compare_analyses,
    main,
    observed_support_stratum,
    posterior_expected_useful_group_probability,
    useful_group_probability,
)


def _row(uid: str, correct_count: int, K: int = 8) -> dict:
    return {
        "sample_uid": uid,
        "K": K,
        "correct_count": correct_count,
        "greedy_correct": False,
        "support_state": "correct_tail" if correct_count else "no_correct_observed",
    }


def test_useful_group_probability_is_mixed_group_probability() -> None:
    assert useful_group_probability(0.0, 8) == 0.0
    assert useful_group_probability(1.0, 8) == 0.0
    assert useful_group_probability(0.5, 2) == pytest.approx(0.5)
    assert useful_group_probability(0.5, 8) == pytest.approx(1.0 - 2.0 / 256.0)


def test_exact_beta_posterior_expectation() -> None:
    # For p ~ Uniform(0, 1) and G=2: E[1-p^2-(1-p)^2] = 1/3.
    value = posterior_expected_useful_group_probability(1.0, 1.0, 2)
    assert value == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize(
    ("correct_count", "expected"),
    [
        (0, "no_correct_observed"),
        (1, "rare_success"),
        (2, "rare_success"),
        (3, "mixed_support"),
        (7, "mixed_support"),
        (8, "all_correct_observed"),
    ],
)
def test_observed_support_strata_are_greedy_independent(
    correct_count: int,
    expected: str,
) -> None:
    assert observed_support_stratum(correct_count, 8) == expected


def test_analyze_rows_uses_only_stochastic_counts() -> None:
    config = FrontierConfig(mc_samples=1_000, seed=7)
    first_rows, first_summary = analyze_rows([_row("p0", 0), _row("p1", 1)], config)
    changed_greedy = [_row("p0", 0), _row("p1", 1)]
    for row in changed_greedy:
        row["greedy_correct"] = True
    second_rows, second_summary = analyze_rows(changed_greedy, config)

    for first, second in zip(first_rows, second_rows, strict=True):
        assert first["posterior_mean_pass_rate"] == second["posterior_mean_pass_rate"]
        assert first["posterior_expected_useful_group_probability"] == second[
            "posterior_expected_useful_group_probability"
        ]
    assert first_summary["posterior_frontier_mass"] == second_summary[
        "posterior_frontier_mass"
    ]


def test_comparison_reports_matched_transitions_and_mass_delta() -> None:
    config = FrontierConfig(mc_samples=1_000, seed=11)
    before, _ = analyze_rows([_row("p0", 0), _row("p1", 1)], config)
    after, _ = analyze_rows([_row("p0", 1), _row("p1", 3)], config)

    rows, summary = compare_analyses(
        before,
        after,
        config,
        bootstrap_resamples=500,
    )

    assert len(rows) == 2
    assert summary["newly_observed_success_count"] == 1
    assert summary["transition_matrix"]["no_correct_observed"]["rare_success"] == 1
    assert summary["transition_matrix"]["rare_success"]["mixed_support"] == 1
    assert math.isfinite(summary["delta_posterior_frontier_mass"])


def test_comparison_rejects_unmatched_uid_sets() -> None:
    config = FrontierConfig(mc_samples=500)
    before, _ = analyze_rows([_row("p0", 0)], config)
    after, _ = analyze_rows([_row("p1", 1)], config)
    with pytest.raises(ValueError, match="UID sets differ"):
        compare_analyses(before, after, config, bootstrap_resamples=100)


def test_cli_writes_single_run_artifacts(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    input_path = run_dir / "prompt_support_summary.jsonl"
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in [_row("p0", 0), _row("p1", 2)]),
        encoding="utf-8",
    )

    assert main([str(run_dir), "--mc-samples", "500"]) == 0
    output_dir = run_dir / "frontier_analysis"
    assert (output_dir / "frontier_summary.json").is_file()
    assert (output_dir / "frontier_prompts.jsonl").is_file()
    summary = json.loads((output_dir / "frontier_summary.json").read_text())
    assert summary["num_prompts"] == 2
    assert summary["config"]["group_size"] == 8
    assert summary["provenance"]["source_prompt_summary"] == str(input_path.resolve())
    assert len(summary["provenance"]["source_prompt_summary_sha256"]) == 64
