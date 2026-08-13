from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from PIL import Image

from dual_track_opd.support_aware.causal_dataset import select_probe_inputs
from dual_track_opd.support_aware.causal_image import build_image_conditions
from dual_track_opd.support_aware.causal_probes import (
    classify_actionability,
    detect_reachability_barriers,
)
from dual_track_opd.support_aware.causal_report import write_reports
from dual_track_opd.support_aware.causal_runtime import (
    RuntimeModels,
    direct_answer_estimate,
    fixed_trajectory_visual_statistics,
    response_chunk_logits,
    teacher_relay_estimate,
    transport_estimate,
)
from dual_track_opd.support_aware.causal_schema import (
    CandidateWindow,
    CausalStateRecord,
    continuation_estimate,
    StudentSupport,
)
from dual_track_opd.support_aware.causal_state_probe import (
    RUN_SCHEMA_VERSION,
    _slice_pending_items,
    _work_id,
    _work_slice_items,
    load_config,
    summarize,
)
from dual_track_opd.support_aware.visual_js import (
    CandidateSelectionConfig,
    concatenate_chunk_statistics,
    full_vocab_js_statistics,
    select_visual_candidates,
)


class _ToyTokenizer:
    eos_token_id = 9
    special_tokens_map = {"eos_token": "<eos>"}

    def get_vocab(self):
        return {str(index): index for index in range(10)}

    def get_added_vocab(self):
        return {}

    def convert_ids_to_tokens(self, ids):
        return [f"{value}." if index == 2 else str(value) for index, value in enumerate(ids)]

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        del skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(str(value) for value in ids)


class _ToyProcessor:
    tokenizer = _ToyTokenizer()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        del messages, tokenize, add_generation_prompt
        return "rendered"

    def __call__(self, *, text, images, return_tensors):
        del text, return_tensors
        image = images[0]
        mean = torch.tensor(list(image.resize((1, 1)).getdata())[0], dtype=torch.float32).mean() / 255
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "pixel_values": mean.reshape(1, 1, 1, 1),
            "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }


class _ToyModel:
    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        *,
        use_cache,
        logits_to_keep=0,
    ):
        del attention_mask, image_grid_thw, use_cache
        length = input_ids.shape[1]
        vocab = 10
        position = torch.arange(length, dtype=torch.float32).reshape(1, length, 1)
        vocab_axis = torch.arange(vocab, dtype=torch.float32).reshape(1, 1, vocab)
        logits = position * 0.01 + vocab_axis * pixel_values.reshape(1, 1, 1)
        if logits_to_keep:
            logits = logits[:, -int(logits_to_keep) :, :]
        return SimpleNamespace(logits=logits)

    __call__ = forward


class _GenerationTokenizer(_ToyTokenizer):
    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        del skip_special_tokens, clean_up_tokenization_spaces
        values = list(ids)
        if 7 in values:
            return "Reasoning complete.\nAnswer: 5"
        return "partial reasoning"


class _GenerationProcessor(_ToyProcessor):
    tokenizer = _GenerationTokenizer()


class _GenerationModel(_ToyModel):
    def __init__(self, suffix):
        self.suffix = tuple(suffix)
        self.generation_config = SimpleNamespace(
            eos_token_id=9,
            do_sample=None,
            max_new_tokens=None,
            pad_token_id=None,
            temperature=None,
            top_p=None,
        )

    def generate(self, **inputs):
        suffix = torch.tensor([self.suffix], dtype=inputs["input_ids"].dtype)
        return torch.cat((inputs["input_ids"].cpu(), suffix), dim=1)


def test_full_vocab_js_is_symmetric_bounded_and_zero_on_identity():
    left = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]])
    right = torch.tensor([[0.0, 2.0, -1.0], [0.0, 1.0, 2.0]])
    forward = full_vocab_js_statistics(left, right)
    backward = full_vocab_js_statistics(right, left)
    identity = full_vocab_js_statistics(left, left)
    assert torch.allclose(forward.js, backward.js, atol=1e-7)
    assert identity.js.abs().max().item() < 1e-7
    assert torch.all(forward.js >= 0)
    assert torch.all(forward.js <= torch.log(torch.tensor(2.0)) + 1e-6)
    assert torch.all(forward.entropy_left > 0)


def test_chunked_js_matches_single_pass():
    torch.manual_seed(2)
    left = torch.randn(7, 13)
    right = torch.randn(7, 13)
    expected = full_vocab_js_statistics(left, right)
    actual = concatenate_chunk_statistics([
        (left[:3], right[:3]),
        (left[3:5], right[3:5]),
        (left[5:], right[5:]),
    ])
    assert torch.allclose(actual.js, expected.js.cpu())
    assert torch.allclose(actual.entropy_left, expected.entropy_left.cpu())


def test_response_chunk_uses_logits_to_keep_and_aligns_positions():
    processor = _ToyProcessor()
    prompt = processor(text=["x"], images=[Image.new("RGB", (4, 4), "black")], return_tensors="pt")
    logits, used = response_chunk_logits(
        _ToyModel(), prompt, [4, 5, 6, 7, 8], start=2, end=5, device="cpu"
    )
    assert used is True
    assert logits.shape == (3, 10)


def test_fixed_trajectory_validates_layout_and_emits_finite_scalars():
    conditions = build_image_conditions(
        Image.new("RGB", (8, 8), (200, 10, 10)), degraded_mode="blur_sigma_2"
    )
    result = fixed_trajectory_visual_statistics(
        _ToyModel(),
        _ToyProcessor(),
        prompt_text="q",
        images=conditions,
        response_ids=[4, 5, 6, 7, 8],
        device="cpu",
        chunk_size=2,
    )
    assert len(result.token_signals) == 5
    assert result.layout_metadata["logits_to_keep_used_everywhere"] is True
    assert all(float(row["js_full_null"]) >= 0 for row in result.token_signals)
    assert max(float(row["js_full_null"]) for row in result.token_signals) > 0


def test_transport_relay_and_leakage_smokes_use_verifier_and_exact_prefixes():
    processor = _GenerationProcessor()
    student = _GenerationModel([7, 9])
    teacher = _GenerationModel([6])
    models = RuntimeModels(student, processor, teacher, processor, "same")
    image = Image.new("RGB", (8, 8), "black")
    transport = transport_estimate(
        models=models,
        image=image,
        prompt_text="q",
        teacher_prefix_ids=[4],
        gold_answer="5",
        k=2,
        max_continuation_tokens=8,
        temperature=0.7,
        top_p=0.95,
        seed=3,
        student_device="cpu",
    )
    relay = teacher_relay_estimate(
        models=models,
        image=image,
        prompt_text="q",
        student_prefix_ids=[4],
        gold_answer="5",
        relay_length=2,
        k=2,
        max_student_tokens=8,
        temperature=0.7,
        top_p=0.95,
        seed=3,
        student_device="cpu",
        teacher_device="cpu",
    )
    leakage = direct_answer_estimate(
        student,
        processor,
        image=image,
        prompt_text="q",
        prefix_ids=[4],
        gold_answer="5",
        k=2,
        max_answer_tokens=8,
        temperature=0.7,
        top_p=0.95,
        seed=3,
        device="cpu",
    )
    assert transport.n_correct == relay.n_correct == leakage.n_correct == 2
    leaking_teacher = RuntimeModels(student, processor, _GenerationModel([7, 9]), processor, "same")
    blocked = teacher_relay_estimate(
        models=leaking_teacher,
        image=image,
        prompt_text="q",
        student_prefix_ids=[4],
        gold_answer="5",
        relay_length=2,
        k=2,
        max_student_tokens=8,
        temperature=0.7,
        top_p=0.95,
        seed=3,
        student_device="cpu",
        teacher_device="cpu",
    )
    assert blocked.n_correct == 0
    assert blocked.n_malformed == 2


def test_candidate_selector_returns_transition_zone_controls_and_snap():
    signal = [0.01] * 10 + [0.8] * 12 + [0.02] * 30
    tokens = ["x"] * len(signal)
    tokens[23] = "."
    selected = select_visual_candidates(
        signal,
        decoded_tokens=tokens,
        config=CandidateSelectionConfig(
            windows=(4, 8),
            topk_high=1,
            topk_drop=2,
            fixed_positions=(0.2, 0.5, 0.8),
            min_candidates=3,
            max_candidates=6,
            nms_radius=4,
            snap_radius=4,
            output_half_width=3,
        ),
    )
    assert 3 <= len(selected) <= 6
    assert all(value.start <= value.anchor < value.end for value in selected)
    assert any("visual_dependence_drop" in value.sources for value in selected)
    assert any(value.snapped for value in selected)


def test_image_interventions_are_deterministic_and_shape_preserving():
    source = Image.new("RGB", (17, 11), (20, 40, 80))
    first = build_image_conditions(source, degraded_mode="lowres_50_bilinear_nearest")
    second = build_image_conditions(source, degraded_mode="lowres_50_bilinear_nearest")
    assert first.full.size == first.degraded.size == first.null.size == (17, 11)
    assert first.metadata["sha256"] == second.metadata["sha256"]
    assert first.metadata["sha256"]["full"] != first.metadata["sha256"]["null"]


def test_actionability_taxonomy_and_teacher_path_barriers():
    assert classify_actionability(relay_gain=0.3, transport_gain=0.0) == "on_policy_repairable"
    assert classify_actionability(relay_gain=0.02, transport_gain=0.35) == "transportable_low_reachability"
    assert classify_actionability(relay_gain=0.02, transport_gain=0.01) == "unresolved_under_current_intervention"
    nll = [0.2] * 20 + [5.0] * 12 + [0.3] * 20
    barriers = detect_reachability_barriers(nll, window=8, top_k=2, nms_radius=4)
    assert barriers
    assert any(20 <= value.anchor <= 35 for value in barriers)


def test_unified_schema_and_reports(tmp_path):
    candidate = CandidateWindow(
        candidate_id="c0",
        start=0,
        end=2,
        anchor=1,
        relative_position=0.5,
        sources=("fixed_0.50",),
        student_js_full_degraded=0.1,
        student_js_full_null=0.2,
        state_class="insufficient_evidence",
    )
    record = CausalStateRecord(
        prompt_id="p1",
        dataset="geometry3k",
        question="q",
        image_path=None,
        image_sha256="img",
        ground_truth="1",
        student_support=StudentSupport.from_counts(1, 32),
        trajectory_id="p1:r1",
        trajectory_correct=True,
        trajectory_token_ids=(1, 2, 3),
        trajectory_token_hash="tokens",
        trajectory_text="reasoning",
        teacher_trajectories=(),
        token_signals=(
            {"position": 0, "js_full_degraded": 0.1, "js_full_null": 0.2},
            {"position": 1, "js_full_degraded": 0.2, "js_full_null": 0.3},
        ),
        candidate_windows=(candidate,),
    )
    paths = write_reports(tmp_path, [record.to_dict()])
    assert json.loads((tmp_path / "summary.json").read_text())["candidate_count"] == 1
    assert "<svg" in (tmp_path / "trajectory_overview.svg").read_text()
    assert all(tmp_path in __import__("pathlib").Path(path).parents for path in paths.values())


def test_select_probe_inputs_stratifies_correct_and_wrong(tmp_path):
    k32 = tmp_path / "k32"
    cohort = tmp_path / "cohort"
    proposals = tmp_path / "proposals"
    k32.mkdir()
    cohort.mkdir()
    proposals.mkdir()
    (k32 / "k32_validation.json").write_text(json.dumps({"valid": True, "K": 32}))
    (k32 / "prompt_support_summary.jsonl").write_text(
        json.dumps({"sample_uid": "p1", "K": 32, "correct_count": 1}) + "\n"
    )
    rollouts = [
        {"sample_uid": "p1", "rollout_id": 1, "is_greedy": False, "correct": True,
         "exact_token_alignment": True, "response_token_ids": [1, 2], "response_token_hash": "a",
         "student_mean_logp": -1.0, "student_tokenizer_hash": "tok", "teacher_tokenizer_hash": "tok"},
        {"sample_uid": "p1", "rollout_id": 2, "is_greedy": False, "correct": False,
         "exact_token_alignment": True, "response_token_ids": [3, 4], "response_token_hash": "b",
         "student_mean_logp": -0.5, "student_tokenizer_hash": "tok", "teacher_tokenizer_hash": "tok"},
    ]
    (k32 / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rollouts))
    pd.DataFrame({"sample_uid": ["p1"], "question": ["q"], "answer": ["1"]}).to_parquet(
        cohort / "cohort.parquet", index=False
    )
    (proposals / "retained_proposals.jsonl").write_text(json.dumps({
        "sample_uid": "p1", "correct": True, "retained_for_fkl": True,
        "reachability_rank": 1, "response_token_ids": [7, 8], "response_token_hash": "t",
    }) + "\n")
    (proposals / "prompt_results").mkdir()
    (proposals / "prompt_results" / "p1.json").write_text(json.dumps({"tokenizer_hash": "tok"}))
    selected, provenance = select_probe_inputs(
        k32_run_dir=k32,
        cohort_dir=cohort,
        proposal_dir=proposals,
    )
    assert [row.student_rollout["correct"] for row in selected] == [True, False]
    assert all(len(row.teacher_proposals) == 1 for row in selected)
    assert provenance["work_unit_count"] == 2


def test_config_matches_repository_hpc_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("DTOPD_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("DTOPD_OUTPUT_ROOT", str(tmp_path / "outputs"))
    args = SimpleNamespace(
        k32_run_dir=None, cohort_dir=None, proposal_dir=None, output_dir=None,
        student_device=None, teacher_device=None, max_prompts=1,
        continuation_k=2, relay_k=2, transport_k=2, answer_k=2,
        max_continuation_tokens=128, max_answer_tokens=16, relay_lengths=[32],
        shard_index=0, num_shards=1,
        work_slice_index=0, work_slice_total=1,
    )
    config = load_config("configs/diagnostics/causal_state_probe.yaml", args)
    assert config.student_model_path == str(tmp_path / "models" / "Qwen3-VL-4B-Instruct")
    assert config.teacher_model_path == str(tmp_path / "models" / "Qwen3-VL-32B-Instruct")
    assert config.k32_run_dir.startswith(str(tmp_path / "outputs"))
    assert config.features.visual_js is True
    assert config.continuation_k == config.relay_k == 2


def test_work_slice_items_partitions_pending_without_overlap():
    items = [f"unit-{i}" for i in range(10)]
    slices = [_work_slice_items(items, index=index, total=4) for index in range(4)]
    assert all(slices)  # every slice gets at least one unit
    assert sorted(unit for slice_ in slices for unit in slice_) == items
    assert sum(len(slice_) for slice_ in slices) == len(items)
    # Deterministic modulo partition: index 0 -> items 0,4,8 ; index 3 -> 3,7
    assert slices[0] == ["unit-0", "unit-4", "unit-8"]
    assert slices[3] == ["unit-3", "unit-7"]
    # total == 1 keeps legacy resume behavior
    assert _work_slice_items(items, index=0, total=1) == items


def _slice_test_items(count=12):
    return [
        SimpleNamespace(
            sample_uid=f"geo3k:{index // 3}",
            student_rollout={
                "rollout_id": index % 3,
                "response_token_hash": f"hash-{index}",
            },
        )
        for index in range(count)
    ]


def test_slice_pending_stable_ownership_across_staggered_resume():
    """Handoff B.3.1: ownership comes from the immutable full input set, so
    staggered starts, partial completion, and repeated resumes never move a
    work ID between slices."""

    items = _slice_test_items()
    all_ids = {_work_id(item) for item in items}
    initial = {
        index: {_work_id(item) for item in _slice_pending_items(items, set(), index, 4)}
        for index in range(4)
    }
    # union of slices equals the full set; pairwise intersections empty
    assert set().union(*initial.values()) == all_ids
    for left in range(4):
        for right in range(left + 1, 4):
            assert not (initial[left] & initial[right])
    # staggered launch: slice 0 finishing its work must not change slice 1
    done_by_slice0 = {next(iter(initial[0]))}
    slice1_after = {
        _work_id(item)
        for item in _slice_pending_items(items, done_by_slice0, 1, 4)
    }
    assert slice1_after == initial[1]
    # partial completion of slice 1's own work drops only its own completed IDs
    own = sorted(initial[1])
    completed = {own[0], own[1]}
    slice1_partial = {
        _work_id(item)
        for item in _slice_pending_items(items, completed, 1, 4)
    }
    assert slice1_partial == initial[1] - completed
    # repeated resume is idempotent
    assert {
        _work_id(item)
        for item in _slice_pending_items(items, completed, 1, 4)
    } == slice1_partial


def test_atomic_json_write_survives_concurrent_writers(tmp_path):
    import multiprocessing as mp

    from dual_track_opd.support_aware.causal_state_probe import _write_json_atomic

    target = tmp_path / "run_manifest.json"

    def writer(worker_id):
        for _ in range(10):
            _write_json_atomic(target, {"worker": worker_id, "payload": "x" * 200})

    context = mp.get_context("fork")
    processes = [context.Process(target=writer, args=(i,)) for i in range(8)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(60)
    assert all(process.exitcode == 0 for process in processes)
    assert "worker" in json.loads(target.read_text(encoding="utf-8"))
    assert not list(tmp_path.glob("*.tmp"))


def test_continuation_estimate_reason_split():
    correctness = [True, False, None, None, None, None, None, None]
    reasons = [
        "correct",
        "wrong_format_valid",
        "no_answer_marker",
        "no_answer_marker",
        "truncated",
        "truncated",
        "relay_answer_leakage",
        "generation_error",
    ]
    estimate = continuation_estimate(
        "relay_l32", correctness, reasons=reasons
    )
    assert estimate.n_correct == 1
    assert estimate.n_malformed == 6
    assert estimate.n_student_continuations_generated == 7
    assert estimate.n_wrong_format_valid == 1
    assert estimate.n_no_answer_marker == 2
    assert estimate.n_truncated == 2
    assert estimate.n_relay_answer_leakage == 1
    assert estimate.n_generation_error == 1
    # validate() enforces count identities
    estimate.validate()


def _load_precheck_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "hpc" / "precheck_causal_state_probe.py"
    spec = importlib.util.spec_from_file_location("precheck_mod", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _relay_acceptance_records():
    """8 synthetic candidates whose K=8 relay estimates each have 1 malformed."""

    records = []
    for index in range(8):
        estimate = {
            "condition": "relay_l32",
            "n": 8,
            "n_correct": 3,
            "n_malformed": 1,
            "pass_rate": 3 / 8,
            "seeds": list(range(index * 8, index * 8 + 8)),
            "response_hashes": [f"h{index}-{i}" for i in range(8)],
        }
        candidate = {
            "candidate_id": f"traj-{index}:candidate-0",
            "start": 0,
            "anchor": 5,
            "end": 10,
            "relative_position": 0.5,
            "sources": ["high_visual_dependence"],
            "state_class": "insufficient_evidence",
            "relay_gain_by_length": {"32": 0.1, "64": 0.1, "128": 0.1},
            "relay_probability_by_length": {"32": 0.7, "64": 0.7, "128": 0.7},
            "relay_continuations": {
                length: dict(estimate, condition=f"relay_l{length}")
                for length in ("32", "64", "128")
            },
            "visual_continuations": [{
                "condition": "full",
                "n": 8,
                "n_correct": 4,
                "n_malformed": 0,
                "pass_rate": 0.5,
            }],
            "transport_gain": 0.1,
            "transport_probability": 0.5,
            "transport_vs_wrong_gain": 0.05,
            "transport_vs_wrong_probability": 0.4,
            "visual_fine_gain": 0.0,
            "visual_all_gain": 0.0,
            "answer_leakage": 0.0,
            "student_js_full_degraded": 0.01,
            "student_js_full_null": 0.1,
            "notes": ["relay_l32_answer_leakage_or_malformed:1"],
        }
        records.append({
            "prompt_id": f"geo3k:{index}",
            "dataset": "geometry3k",
            "question": "q",
            "image_path": None,
            "image_sha256": "a",
            "ground_truth": 1,
            "student_support": {"n_rollouts": 32, "n_correct": 16, "pass_rate": 0.5, "posterior_mean": 0.5},
            "trajectory_id": f"traj-{index}",
            "trajectory_correct": bool(index % 2),
            "trajectory_token_ids": list(range(20)),
            "trajectory_token_hash": f"hash-{index}",
            "trajectory_text": "t",
            "teacher_trajectories": [],
            "token_signals": [],
            "candidate_windows": [candidate],
            "metadata": {},
            "schema_version": "causal-state-probe-v1",
        })
    return records


def test_precheck_relay_coverage_acceptance_criterion():
    """Review P0-1: one malformed per K=8 estimate must yield 87.5%
    continuation coverage, 0% all-clean, 0% fully-malformed."""

    precheck = _load_precheck_module()
    stats = precheck.deep_stats(_relay_acceptance_records())
    coverage = stats["relay_coverage"]["32"]
    assert coverage["all_clean_estimate_rate"] == 0.0
    assert coverage["continuation_valid_rate"] == 0.875
    assert coverage["fully_malformed"] == 0
    assert coverage["with_any_malformed"] == 8


def test_strict_shard_summary_requires_clean_exact_coverage(tmp_path):
    input_hashes = {
        "k32_validation_sha256": "a",
        "k32_rollouts_sha256": "b",
        "k32_support_summary_sha256": "c",
        "cohort_sha256": "d",
        "retained_proposals_sha256": "e",
    }

    def make_shard(index, trajectory_id, *, dirty=False):
        shard = tmp_path / f"shard-{index}"
        (shard / "trajectory_results").mkdir(parents=True)
        (shard / "trajectory_results" / f"{trajectory_id}.json").write_text(
            json.dumps({
                "prompt_id": f"p{index}",
                "trajectory_id": trajectory_id,
                "trajectory_correct": bool(index),
                "candidate_windows": [],
            })
        )
        manifest = {
            "schema_version": RUN_SCHEMA_VERSION,
            "status": "completed",
            "git_commit": "abc123",
            "git_dirty": dirty,
            "config": {
                "output_dir": str(shard),
                "num_shards": 2,
                "shard_index": index,
                "continuation_k": 8,
            },
            "provenance": {
                **input_hashes,
                "expected_work_ids": [trajectory_id],
            },
        }
        (shard / "run_manifest.json").write_text(json.dumps(manifest))
        return shard

    shard0 = make_shard(0, "p0:r0")
    shard1 = make_shard(1, "p1:r1")
    summary = summarize([shard0, shard1], tmp_path / "merged")
    assert summary["record_count"] == 2
    assert json.loads((tmp_path / "merged" / "run_manifest.json").read_text())["git_dirty"] is False

    dirty_manifest_path = shard1 / "run_manifest.json"
    dirty_manifest = json.loads(dirty_manifest_path.read_text())
    dirty_manifest["git_dirty"] = True
    dirty_manifest_path.write_text(json.dumps(dirty_manifest))
    dirty_summary = summarize([shard0, shard1], tmp_path / "merged_dirty")
    assert dirty_summary["record_count"] == 2
    merged_manifest = json.loads(
        (tmp_path / "merged_dirty" / "run_manifest.json").read_text()
    )
    assert merged_manifest["git_dirty"] is True
    assert str(shard1) in merged_manifest["dirty_shards"]
