import json
from dataclasses import asdict
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from dual_track_opd.support_aware.prefix_intervention import (
    InterventionConfig,
    _candidate_prefix,
    _extend_prompt_inputs,
    expected_group_utility,
    generate_continuation,
    merge_shards,
    posterior_lift_probability,
    prefix_leakage_reason,
    select_intervention_records,
)


class _Tokenizer:
    eos_token_id = 99

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        del skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(str(value) for value in ids)


class _Processor:
    tokenizer = _Tokenizer()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        del messages, tokenize, add_generation_prompt
        return "prompt"

    def __call__(self, *, text, images, return_tensors):
        del text, images, return_tensors
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }


class _Model:
    generation_config = SimpleNamespace(
        eos_token_id=99, do_sample=None, max_new_tokens=None,
        pad_token_id=None, temperature=None, top_p=None,
    )

    def generate(self, **inputs):
        suffix = torch.tensor([[7, 99]], dtype=inputs["input_ids"].dtype)
        return torch.cat([inputs["input_ids"].cpu(), suffix], dim=1)


def test_expected_group_utility_is_zero_at_known_extremes_and_high_at_mixed():
    # Jeffreys posterior is uncertain rather than exactly zero at finite K.
    low = expected_group_utility(0, 32)
    high = expected_group_utility(32, 32)
    mixed = expected_group_utility(16, 32)
    assert low == pytest.approx(high)
    assert 0 < low < mixed < 1


def test_posterior_lift_probability_is_deterministic_and_directional():
    first = posterior_lift_probability(7, 1, 8, seed=5, draws=5000)
    second = posterior_lift_probability(7, 1, 8, seed=5, draws=5000)
    assert first == second
    assert first[0] > 0.5
    assert first[1] > 0.99


def test_prefix_leakage_and_candidate_length_gate():
    assert prefix_leakage_reason("reasoning only", "5") is None
    assert prefix_leakage_reason("\nAnswer: 5", "5") == "final_answer_marker"
    prefix, reason, _ = _candidate_prefix(
        [1, 2, 3], horizon=4, tokenizer=_Tokenizer(), gold_answer="5"
    )
    assert prefix is None
    assert reason == "source_shorter_than_horizon"


def test_extend_prompt_inputs_appends_prefix_and_masks():
    inputs = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "mm_token_type_ids": torch.tensor([[1, 0]]),
    }
    extended = _extend_prompt_inputs(inputs, [3, 4], "cpu")
    assert extended["input_ids"].tolist() == [[1, 2, 3, 4]]
    assert extended["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert extended["mm_token_type_ids"].tolist() == [[1, 0, 0, 0]]


def test_generate_continuation_preserves_prefix_and_raw_terminal_id():
    result = generate_continuation(
        _Model(), _Processor(), image=None, prompt_text="q", prefix_ids=[3, 4],
        max_continuation_tokens=8, temperature=0.7, top_p=0.95, seed=9, device="cpu",
    )
    assert result["continuation_token_ids"] == (7, 99)
    assert result["generation"].response_token_ids_raw == (3, 4, 7, 99)
    assert result["generation"].finish_reason == "stop"


def test_select_intervention_records_joins_only_retained_with_wrong_control(tmp_path):
    proposal = tmp_path / "proposal"
    k32 = tmp_path / "k32"
    cohort = tmp_path / "cohort"
    proposal.mkdir()
    k32.mkdir()
    cohort.mkdir()
    (proposal / "run_manifest.json").write_text(json.dumps({"status": "completed"}))
    (proposal / "summary.json").write_text(json.dumps({"complete": True}))
    retained = [
        {
            "sample_uid": "p1", "correct": True, "retained_for_fkl": True,
            "reachability_rank": 1, "response_token_ids": [1, 2, 3],
        },
        {
            "sample_uid": "p2", "correct": True, "retained_for_fkl": True,
            "reachability_rank": 1, "response_token_ids": [4, 5, 6],
        },
    ]
    (proposal / "retained_proposals.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in retained)
    )
    (k32 / "k32_validation.json").write_text(json.dumps({"valid": True, "K": 32}))
    rollouts = [
        {
            "sample_uid": "p1", "rollout_id": 1, "is_greedy": False,
            "correct": False, "exact_token_alignment": True,
            "response_token_ids": [8, 9, 10], "response_token_hash": "w1",
        },
        {
            "sample_uid": "p2", "rollout_id": 1, "is_greedy": False,
            "correct": True, "exact_token_alignment": True,
            "response_token_ids": [8, 9, 10], "response_token_hash": "w2",
        },
    ]
    (k32 / "rollouts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rollouts)
    )
    pd.DataFrame({
        "sample_uid": ["p1", "p2"], "question": ["q1", "q2"], "answer": ["1", "2"]
    }).to_parquet(cohort / "cohort.parquet", index=False)
    config = InterventionConfig(
        proposal_dir=str(proposal), k32_run_dir=str(k32), cohort_dir=str(cohort),
        output_dir=str(tmp_path / "out"), student_model_path="student",
    )
    records, provenance = select_intervention_records(config)
    assert [row["sample_uid"] for row in records] == ["p1"]
    assert provenance["all_selected_uids"] == ["p1"]


def test_merge_shards_requires_clean_complete_coverage(tmp_path):
    all_uids = ["p1", "p2"]
    shard_dirs = []
    base = InterventionConfig(
        proposal_dir="proposal", k32_run_dir="k32", cohort_dir="cohort",
        output_dir="unused", student_model_path="student", num_shards=2,
    )
    for shard_index, uid in enumerate(all_uids):
        shard = tmp_path / f"shard-{shard_index}"
        results = shard / "prompt_results"
        results.mkdir(parents=True)
        config = asdict(base)
        config.update(output_dir=str(shard), shard_index=shard_index)
        manifest = {
            "status": "completed", "git_commit": "abc123", "git_dirty": False,
            "config": config,
            "provenance": {"all_selected_uids": all_uids, "expected_uids": [uid]},
        }
        (shard / "run_manifest.json").write_text(json.dumps(manifest))
        unit = {
            "sample_uid": uid, "arm": "unaided", "horizon": 0,
            "skipped_reason": None, "K": 1, "correct_count": 0,
            "pass_rate": 0.0, "expected_U8": expected_group_utility(0, 1),
        }
        prompt = {
            "sample_uid": uid, "intervention_units": [unit], "rollouts": [],
        }
        (results / f"{uid}.json").write_text(json.dumps(prompt))
        shard_dirs.append(shard)
    output = tmp_path / "merged"
    summary = merge_shards(shard_dirs, output)
    assert summary["complete"] is True
    assert summary["completed_prompt_count"] == 2
    assert (output / "run_manifest.json").is_file()
