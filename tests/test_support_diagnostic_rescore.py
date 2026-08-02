"""Append-only migration tests using CPU scorer doubles."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from dual_track_opd.fc_opd.dataset_signal_audit import hash_token_ids
from dual_track_opd.fc_opd.teacher_protocol import TeacherMetadata, tokenizer_fingerprint
from dual_track_opd.support_aware.reporter import (
    DiagnosticRunMeta,
    write_prompt_support_summary_jsonl,
    write_resolved_config_yaml,
    write_rollouts_jsonl,
    write_run_manifest,
    write_selected_prompts_jsonl,
    write_summary_json,
)
from dual_track_opd.support_aware.rescore import RescoreConfig, rescore_existing
from dual_track_opd.support_aware.scorer import TeacherScorer


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 9
    special_tokens_map = {"eos_token": "<eos>"}
    chat_template = "fake-template-v1"

    def get_vocab(self):
        return {str(index): index for index in range(12)}

    def get_added_vocab(self):
        return {}

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        del clean_up_tokenization_spaces
        visible = [token_id for token_id in ids if not (skip_special_tokens and token_id == 9)]
        return " ".join(str(token_id) for token_id in visible)


class _Processor:
    tokenizer = _Tokenizer()

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        del messages, tokenize, add_generation_prompt
        return "PROMPT"

    def __call__(self, *, text, images, return_tensors, padding=True):
        del images, return_tensors, padding
        return {
            "input_ids": torch.tensor([[1, 2]] * len(text), dtype=torch.long),
            "attention_mask": torch.ones((len(text), 2), dtype=torch.long),
        }


class _Model:
    generation_config = SimpleNamespace(eos_token_id=9)
    config = SimpleNamespace(_name_or_path="fake-student", _commit_hash="student-rev")

    def __init__(self):
        self._parameter = torch.zeros(1)

    def eval(self):
        return self

    def parameters(self):
        return iter([self._parameter])

    def __call__(self, **kwargs):
        batch, width = kwargs["input_ids"].shape
        logits = torch.zeros((batch, width, 12), dtype=torch.float32)
        for position in range(width):
            logits[:, position, (position + 3) % 12] = 3.0
        return SimpleNamespace(logits=logits)


class _Teacher:
    def __init__(self, tokenizer_hash: str):
        self._metadata = TeacherMetadata(
            model_id="fake-teacher",
            tokenizer_hash=tokenizer_hash,
            vocab_size=12,
            top_k=2,
            dtype="float32",
            git_revision="teacher-rev",
        )

    @property
    def metadata(self):
        return self._metadata

    @property
    def tokenizer_hash(self):
        return self._metadata.tokenizer_hash

    @property
    def model_id(self):
        return self._metadata.model_id

    def health(self):
        return True

    def score_batch(self, *, request_ids, response_token_ids_list, **kwargs):
        del kwargs
        return [
            TeacherScorer.ScoreResult(
                request_id=request_id,
                sampled_token_log_probs=tuple(-0.5 for _ in response_ids),
                mean_logp=-0.5,
                scored_token_ids=tuple(response_ids),
                scored_token_hash=hash_token_ids(response_ids),
                response_mask=tuple(True for _ in response_ids),
            )
            for request_id, response_ids in zip(
                request_ids, response_token_ids_list, strict=True
            )
        ]


def _write_source(source: Path) -> None:
    source.mkdir()
    rollouts = [
        {
            "run_id": "old-run",
            "sample_uid": "p1",
            "source_index": 0,
            "rollout_id": 0,
            "is_greedy": True,
            "response_text": "Answer: 5",
            "response_token_ids": [5, 9],
            "max_new_tokens": 8,
        },
        {
            "run_id": "old-run",
            "sample_uid": "p1",
            "source_index": 0,
            "rollout_id": 1,
            "is_greedy": False,
            "response_text": "Answer: 6",
            "response_token_ids": [6, 9],
            "max_new_tokens": 8,
        },
    ]
    write_rollouts_jsonl(source, rollouts)
    write_prompt_support_summary_jsonl(source, [{
        "sample_uid": "p1",
        "K": 1,
        "support_state": "exposed",
        "correct_count": 0,
    }])
    write_selected_prompts_jsonl(source, [{"sample_uid": "p1", "question": "Find x"}])
    write_summary_json(source, {
        "mode": "smoke",
        "selection_manifest": {"dataset_path": "/fake/dataset.parquet"},
        "gate_config": {},
    })
    write_resolved_config_yaml(source, {
        "num_prompts": "1",
        "seed": "42",
        "response_format": "legacy_answer",
    })
    now = time.time()
    write_run_manifest(source, DiagnosticRunMeta(
        run_id="old-run",
        output_dir=source,
        config={},
        git_commit="old-commit",
        git_dirty=False,
        num_prompts=1,
        rollouts_per_prompt=1,
        seed=42,
        start_time=now,
        end_time=now,
        exit_status="GATE_FAIL",
    ))


def test_rescore_is_append_only_and_records_provenance(tmp_path, monkeypatch):
    import dual_track_opd.support_aware.rescore as module

    source = tmp_path / "source"
    output = tmp_path / "rescored"
    _write_source(source)
    source_before = {
        path.name: path.read_bytes() for path in source.iterdir() if path.is_file()
    }
    dataset_row = {
        "sample_uid": "p1",
        "question": "Find x",
        "answer": "5",
        "images": [b"unused"],
    }
    monkeypatch.setattr(module, "load_and_select_prompts", lambda *args, **kwargs: (
        [dataset_row],
        {
            "dataset_path": "/fake/dataset.parquet",
            "dataset_sha256": "dataset-hash",
            "selection_sha256": "selection-hash",
            "selected_rows": 1,
        },
    ))
    monkeypatch.setattr(module, "extract_image", lambda row: Image.new("RGB", (2, 2)))
    monkeypatch.setattr(module, "save_image_for_teacher", lambda image, directory: str(directory / "image.png"))
    monkeypatch.setattr(module, "_get_git_commit", lambda: "new-commit")
    monkeypatch.setattr(module, "_get_git_dirty", lambda: False)

    processor = _Processor()
    summary = rescore_existing(
        RescoreConfig(
            source_run=str(source),
            output_dir=str(output),
            student_model="fake-student",
            bootstrap_resamples=20,
        ),
        model=_Model(),
        processor=processor,
        teacher=_Teacher(tokenizer_fingerprint(processor.tokenizer)),
    )

    assert {
        path.name: path.read_bytes() for path in source.iterdir() if path.is_file()
    } == source_before
    assert output.is_dir()
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["source_run_path"] == str(source.resolve())
    assert manifest["source_git_commit"] == "old-commit"
    assert manifest["rescore_git_commit"] == "new-commit"
    assert manifest["prompt_token_hash_status"] == "backfilled"
    assert manifest["response_token_hash_status"] == "backfilled"
    assert "student_forced_scores" in manifest["fields_recomputed"]
    rows = [json.loads(line) for line in (output / "rollouts.jsonl").read_text().splitlines()]
    assert [row["response_token_ids"] for row in rows] == [[5, 9], [6, 9]]
    assert all(row["exact_token_alignment"] for row in rows)
    assert summary["exact_token_alignment_rate"] == 1.0
