from dual_track_opd.fc_opd.prompt_contracts import geometry3k_training_prompt


def test_geometry3k_training_prompt_uses_native_image_question_contract():
    prompt = geometry3k_training_prompt("Find x.")
    assert prompt == [{"role": "user", "content": "<image>\nFind x."}]


def test_geometry3k_training_prompt_does_not_inject_reasoning_markup():
    content = geometry3k_training_prompt("Find x.")[0]["content"]
    assert "<think>" not in content
    assert "\\boxed" not in content
