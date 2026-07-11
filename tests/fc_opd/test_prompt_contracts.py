from dual_track_opd.fc_opd.prompt_contracts import geometry3k_training_prompt


def test_geometry3k_training_prompt_preserves_image_and_format_contract():
    prompt = geometry3k_training_prompt("Find x.")
    assert prompt[0]["role"] == "user"
    assert prompt[0]["content"].startswith("<image>\nFind x.")
    assert "<think> </think>" in prompt[0]["content"]
    assert "\\boxed{}" in prompt[0]["content"]
