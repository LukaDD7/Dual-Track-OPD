import pytest

from dual_track_opd.fc_opd.prompt_contracts import (
    clean_geometry3k_question_rows,
    geometry3k_training_prompt,
    geometry3k_training_prompt_answer_only,
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


def test_answer_only_prompt_is_an_explicit_single_line_no_reasoning_gate():
    prompt = geometry3k_training_prompt_answer_only("Find x.")
    assert prompt[0]["role"] == "user"
    content = prompt[0]["content"]
    assert content.startswith("<image>\nFind x.")
    assert "\\boxed{<final answer>}" in content
    assert "exactly one line" in content
    assert "Do not show reasoning" in content
    assert "<think>" not in content


def test_prompt_builder_registry_versions_and_fail_fast():
    assert get_geometry3k_prompt_builder("v1") is geometry3k_training_prompt
    assert get_geometry3k_prompt_builder("boxed_only") is geometry3k_training_prompt_boxed_only
    assert get_geometry3k_prompt_builder("answer_only") is geometry3k_training_prompt_answer_only
    with pytest.raises(ValueError, match="unknown geometry3k prompt_version"):
        get_geometry3k_prompt_builder("nope")


def test_clean_geometry3k_question_rows_uses_versioned_builder():
    rows = [{"question": "Find x.", "answer": "3"}, {"question": "  "}]
    out = clean_geometry3k_question_rows(rows, "boxed_only")
    assert out[0]["prompt"][0]["role"] == "user"
    assert out[0]["prompt"][0]["content"].startswith("<image>\nFind x.")
    assert "\\boxed{<answer>}" in out[0]["prompt"][0]["content"]
    assert out[1] == {"question": "  "}  # 空 question 不加 prompt，原行不变
    # 默认 v1 与旧行为一致
    out_v1 = clean_geometry3k_question_rows([{"question": "Find x."}], "v1")
    assert out_v1[0]["prompt"] == geometry3k_training_prompt("Find x.")
