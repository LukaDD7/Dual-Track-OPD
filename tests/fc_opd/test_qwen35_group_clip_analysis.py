"""GPU-free tests for the group-level clip analysis aggregator."""

from dual_track_opd.fc_opd.qwen35_group_clip import (
    aggregate_group_metrics,
    clean_response,
    group_key,
)


def _rec(group, length, clip, correct=False, boxed=False):
    return {
        "group": group,
        "length": length,
        "is_clip": clip,
        "is_correct": correct,
        "has_boxed": boxed,
        "cap": 4096,
    }


def test_group_key_splits_rollout_uid():
    assert group_key("abc123_0_0") == "abc123"
    assert group_key("a_b_c_1_2") == "a_b_c"
    assert group_key("flat") == "flat"


def test_aggregate_sequence_and_group_metrics():
    # 2 groups x 4 rollouts; group A: 1 clipped, 1 correct; group B: all EOS, none correct
    records = [
        _rec("A", 300, False, correct=True, boxed=True),
        _rec("A", 4096, True),
        _rec("A", 200, False, boxed=True),
        _rec("A", 150, False, boxed=True),
        _rec("B", 100, False, boxed=True),
        _rec("B", 120, False, boxed=True),
        _rec("B", 90, False, boxed=True),
        _rec("B", 80, False),
    ]
    m = aggregate_group_metrics(records)
    assert m["n"] == 8
    assert m["sequence"]["clip_rate"] == 1 / 8
    assert m["sequence"]["n_correct"] == 1
    assert m["sequence"]["boxed_rate"] == 6 / 8
    assert m["group"]["n_groups"] == 2
    assert m["group"]["any_clip_rate"] == 0.5
    assert m["group"]["n_groups_any_correct"] == 1
    assert m["group"]["all_eos_rate"] == 0.5
    assert m["group"]["mean_clipped_per_group"] == 0.5
    assert m["group"]["rollouts_per_group"] == 4.0


def test_clean_response_strips_assistant_prefix():
    assert clean_response("assistant\nhello") == "hello"
    assert clean_response("hello") == "hello"
