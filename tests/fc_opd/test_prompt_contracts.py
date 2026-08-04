import pytest

from dual_track_opd.fc_opd.prompt_contracts import (
    geometry3k_training_prompt,
    geometry3k_training_prompt_boxed_only,
    get_geometry3k_prompt_builder,
)


def test_geometry3k_training_prompt_uses_native_image_question_contract():
    prompt = geometry3k_training_prompt("Find x.")
    assert prompt == [{
        "role": "user",
        "content": "<image>\nFind x.\n\nPut the final answer in \\boxed{}.",
    }]


def test_geometry3k_training_prompt_does_not_inject_reasoning_markup():
    content = geometry3k_training_prompt("Find x.")[0]["content"]
    assert "<think>" not in content
    assert "internal monologue" not in content


def test_boxed_only_prompt_keeps_contract_and_adds_closing_line():
    prompt = geometry3k_training_prompt_boxed_only("Find x.")
    assert prompt[0]["role"] == "user"
    content = prompt[0]["content"]
    assert content.startswith("<image>\nFind x.")
    assert "\\boxed{<answer>}" in content
    assert "exactly one line" in content
    assert "write nothing else" in content
    assert "<think>" not in content


def test_prompt_builder_registry_versions_and_fail_fast():
    assert get_geometry3k_prompt_builder("v1") is geometry3k_training_prompt
    assert get_geometry3k_prompt_builder("boxed_only") is geometry3k_training_prompt_boxed_only
    with pytest.raises(ValueError, match="unknown geometry3k prompt_version"):
        get_geometry3k_prompt_builder("nope")
