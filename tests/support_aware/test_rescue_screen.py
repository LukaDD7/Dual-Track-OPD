"""CPU unit tests for the rescue-screen adaptive state machine."""

from dual_track_opd.support_aware.rescue_screen import decide_stage1, decide_stage2


def _posterior(mean_lift: float, probability: float) -> dict[str, dict[str, float]]:
    return {
        "vs_wrong": {"mean_lift": mean_lift, "probability": probability},
        "vs_unaided": {"mean_lift": mean_lift, "probability": probability},
    }


def test_stage1_h128_promotes_and_h64_not():
    posteriors = {
        64: _posterior(0.10, 0.70),
        128: _posterior(0.30, 0.95),
        256: _posterior(0.05, 0.60),
    }
    decision, candidate, notes = decide_stage1(posteriors, {128: True, 256: True})
    assert decision == "candidate"
    assert candidate == 128


def test_stage1_h128_and_h64_promote():
    posteriors = {
        64: _posterior(0.40, 0.99),
        128: _posterior(0.30, 0.95),
        256: _posterior(0.05, 0.60),
    }
    decision, candidate, _ = decide_stage1(posteriors, {128: True, 256: True})
    assert (decision, candidate) == ("candidate", 64)


def test_stage1_h256_promotes():
    posteriors = {
        128: _posterior(0.05, 0.60),
        256: _posterior(0.35, 0.97),
    }
    decision, candidate, _ = decide_stage1(posteriors, {128: False, 256: True})
    assert (decision, candidate) == ("candidate", 256)


def test_stage1_no_promote_but_teacher_success_tries_512():
    posteriors = {
        128: _posterior(0.10, 0.70),
        256: _posterior(0.08, 0.65),
        512: _posterior(0.40, 0.98),
    }
    decision, candidate, _ = decide_stage1(posteriors, {128: False, 256: True})
    assert (decision, candidate) == ("candidate", 512)


def test_stage1_rescue_negative():
    posteriors = {
        128: _posterior(0.02, 0.55),
        256: _posterior(0.03, 0.55),
    }
    decision, candidate, _ = decide_stage1(posteriors, {128: False, 256: False})
    assert (decision, candidate) == ("rescue_negative", None)


def test_stage1_rescue_negative_despite_teacher_success_without_512():
    posteriors = {
        128: _posterior(0.10, 0.70),
        256: _posterior(0.08, 0.65),
    }
    decision, candidate, _ = decide_stage1(posteriors, {128: False, 256: True})
    assert (decision, candidate) == ("rescue_negative", None)


def test_stage2_strict_rule():
    posterior = _posterior(0.30, 0.95)
    decision, notes = decide_stage2(posterior)
    assert decision == "rescue_positive"


def test_stage2_fails_below_threshold():
    decision, _ = decide_stage2(_posterior(0.10, 0.80))
    assert decision == "stage2_failed"
