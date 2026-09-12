import json

import pytest

from dual_track_opd.fc_opd.condition_prompt_dump import (
    PromptDumpConfig,
    detect_prompt_leakage,
    dump_condition_prompts,
)
from dual_track_opd.fc_opd.dataset_adapters import load_normalized_records
from dual_track_opd.fc_opd.dataset_adapters import normalize_prompt_text
from dual_track_opd.fc_opd.dataset_signal_audit import materialize_gaussian_blur


def _write_dataset(path, image_path, *, bbox_image_path=None, answer="red"):
    record = {
        "id": "sample-0001",
        "image_path": str(image_path),
        "bbox_image_path": "" if bbox_image_path is None else str(bbox_image_path),
        "question": "What color is the marker?",
        "answer": answer,
        "free_caption": "A marker is visible.",
        "task_evidence": f"The correct answer is {answer}.",
    }
    path.write_text(json.dumps([record]), encoding="utf-8")
    return record


def test_prompt_dump_does_not_leak_answer_by_default(tmp_path):
    dataset = tmp_path / "candidate.json"
    image = tmp_path / "image.jpg"
    image.write_bytes(b"placeholder")
    _write_dataset(dataset, image)

    result = dump_condition_prompts(
        PromptDumpConfig(
            dataset=dataset,
            dataset_type="vision_opd_json",
            source_dataset="vision-opd-6k",
            output=tmp_path / "prompts.md",
        )
    )

    record = result.records[0]
    task_text = record["rendered_prompts"]["task"]["text"]
    assert "red" not in task_text.lower()
    assert record["leakage_flags"]["task_evidence_contains_answer"] is False
    assert record["condition_metadata"]["task_evidence_mode"] == "none"
    assert result.markdown_path.is_file()
    assert result.jsonl_path.is_file()


def test_prompt_dump_oracle_answer_emits_leakage_and_oracle_warnings(tmp_path):
    dataset = tmp_path / "candidate.json"
    image = tmp_path / "image.jpg"
    image.write_bytes(b"placeholder")
    _write_dataset(dataset, image, answer="A")

    result = dump_condition_prompts(
        PromptDumpConfig(
            dataset=dataset,
            dataset_type="vision_opd_json",
            source_dataset="vision-opd-6k",
            task_evidence_mode="oracle_answer",
            output=tmp_path / "prompts.md",
        )
    )

    flags = result.records[0]["leakage_flags"]
    assert flags["oracle_mode_enabled"] is True
    assert flags["task_evidence_contains_option_letter"] is True
    assert "oracle/upper-bound" in result.records[0]["non_oracle_warning"][0]


def test_vision_opd_adapter_preserves_crop_as_metadata_only(tmp_path):
    dataset = tmp_path / "candidate.json"
    image = tmp_path / "global.jpg"
    crop = tmp_path / "crop.jpg"
    image.write_bytes(b"global")
    crop.write_bytes(b"crop")
    _write_dataset(dataset, image, bbox_image_path=crop)

    records = load_normalized_records(dataset, "vision_opd_json", source_dataset="vision-opd-6k")
    assert records[0]["image_path"].endswith("global.jpg")
    assert records[0]["bbox_image_path"].endswith("crop.jpg")
    assert records[0]["metadata"]["crop_bbox_policy"] == "metadata_only_default_no_crop_condition"

    result = dump_condition_prompts(
        PromptDumpConfig(
            dataset=dataset,
            dataset_type="vision_opd_json",
            source_dataset="vision-opd-6k",
            output=tmp_path / "prompts.md",
        )
    )
    full_images = result.records[0]["rendered_prompts"]["full"]["image_paths"]
    task_images = result.records[0]["rendered_prompts"]["task"]["image_paths"]
    assert str(crop) not in full_images
    assert task_images == ()
    assert result.records[0]["bbox_image_path"].endswith("crop.jpg")


def test_vision_opd_adapter_preserves_bbox_images_plural_as_metadata(tmp_path):
    dataset = tmp_path / "candidate.json"
    image = tmp_path / "global.jpg"
    crop = tmp_path / "crop.jpg"
    image.write_bytes(b"global")
    crop.write_bytes(b"crop")
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "sample-0001",
                    "image_path": str(image),
                    "bbox_images": [str(crop)],
                    "prompt": "<image>\nWhat color is the marker?",
                    "answer": "red",
                }
            ]
        ),
        encoding="utf-8",
    )

    records = load_normalized_records(dataset, "vision_opd_json", source_dataset="vision-opd-6k")

    assert records[0]["bbox_image_path"].endswith("crop.jpg")
    assert records[0]["bbox_image_paths"] == [str(crop)]
    assert records[0]["bbox_image_exists"] is True
    assert records[0]["question"] == "What color is the marker?"


def test_degraded_image_materialization_works(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    source = tmp_path / "source.png"
    target = tmp_path / "blurred" / "source.png"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(source)

    materialize_gaussian_blur(str(source), str(target), 2.0)

    assert target.is_file()


def test_prompt_text_normalization_handles_string_dict_and_messages():
    assert normalize_prompt_text("<image>\nWhat color?") == "What color?"
    assert (
        normalize_prompt_text({"role": "user", "content": "<image>\nWhat color?"})
        == "What color?"
    )
    messages = [
        {"role": "system", "content": "Ignore this."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "x.png"},
                {
                    "type": "text",
                    "text": "<image>\nWhat color?\n\nA. red\nB. blue",
                },
            ],
        },
    ]
    assert normalize_prompt_text(messages) == "What color?\n\nA. red\nB. blue"


def test_alignment_cosine_diagnostic_is_non_ideal_alignment():
    flags = detect_prompt_leakage(
        task_evidence="No answer here.",
        answer="red",
        options=[],
        oracle_mode=False,
    )
    assert flags["oracle_mode_enabled"] is False
