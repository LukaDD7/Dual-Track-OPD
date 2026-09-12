"""CPU tests for STP-OPD P1 mechanics-gate statistics."""

from dual_track_opd.support_aware.support_transition_eval import (
    jeffreys_posterior_mean,
    p1_mechanics_gate,
    per_prompt_posterior_means,
    prompt_macro_lift,
    support_transition_buckets,
)


def test_jeffreys_posterior_mean():
    assert abs(jeffreys_posterior_mean(0, 32) - 0.5 / 33) < 1e-9
    assert abs(jeffreys_posterior_mean(32, 32) - 32.5 / 33) < 1e-9
    assert abs(jeffreys_posterior_mean(16, 32) - 16.5 / 33) < 1e-9


def test_per_prompt_posterior_means():
    means = per_prompt_posterior_means({"p1": 1, "p2": 31}, k=32)
    assert abs(means["p1"] - 1.5 / 33) < 1e-9
    assert abs(means["p2"] - 31.5 / 33) < 1e-9


def test_prompt_macro_lift():
    treatment = {"p1": 0.6, "p2": 0.4}
    baseline = {"p1": 0.5, "p2": 0.5}
    assert abs(prompt_macro_lift(treatment, baseline) - 0.0) < 1e-9
    treatment = {"p1": 0.8, "p2": 0.4}
    assert abs(prompt_macro_lift(treatment, baseline) - 0.1) < 1e-9


def test_p1_gate_passes_with_4_of_7_improved_and_lift():
    a0 = {f"p{i}": 0.3 for i in range(7)}
    a3 = {f"p{i}": 0.3 for i in range(7)}
    for i in range(4):
        a3[f"p{i}"] = 0.6
    a3["p6"] = 0.55
    result = p1_mechanics_gate(a3, a0)
    assert result["n_improved"] == 5
    assert result["prompt_macro_lift"] >= 0.10
    assert result["passed"] is True


def test_p1_gate_fails_without_required_lift():
    a0 = {f"p{i}": 0.3 for i in range(7)}
    a3 = {f"p{i}": 0.32 for i in range(7)}
    result = p1_mechanics_gate(a3, a0)
    assert result["n_improved"] == 7
    assert result["passed"] is False
    assert result["prompt_macro_lift"] < 0.10


def test_support_transition_buckets():
    before = {"p1": 0.3, "p2": 0.3, "p3": 0.6}
    after = {"p1": 0.7, "p2": 0.2, "p3": 0.7}
    buckets = support_transition_buckets(before, after)
    assert buckets["rare_to_stable"] == 1
    assert buckets["rare_to_no_correct"] == 1
