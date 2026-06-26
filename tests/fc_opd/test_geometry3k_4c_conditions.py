from dual_track_opd.fc_opd.conditions import Condition, ConditionInputs, ImageInput
from dual_track_opd.fc_opd.evidence_generation import (
    FREE_CAPTION_PROMPTS,
    TASK_EVIDENCE_PROMPTS,
    detect_evidence_leakage,
)
from dual_track_opd.fc_opd.teacher_prompts import render_teacher_prompt


def test_geometry_4c_prompts_keep_free_caption_question_free():
    free_prompt = FREE_CAPTION_PROMPTS["geometry"]
    task_prompt = TASK_EVIDENCE_PROMPTS["geometry"]

    assert "What is angle ABC?" not in free_prompt
    assert "A. 30" not in free_prompt
    assert "do not solve" in task_prompt.lower()
    assert "gold answer" in task_prompt.lower()


def test_degraded_condition_uses_degraded_image_path():
    inputs = ConditionInputs(
        full_image=ImageInput(path="/tmp/full.png"),
        degraded_image=ImageInput(
            path="/tmp/degraded.png",
            transform={"type": "lowres_nearest", "scale": 0.1},
        ),
        free_caption="diagram observations",
        task_evidence="question-relevant evidence",
    )

    rendered = render_teacher_prompt(Condition.DEGRADED, "What is x?", inputs)

    assert rendered.image_paths == ("/tmp/degraded.png",)
    assert "degraded image" in rendered.messages[0]["content"][1]["text"]


def test_evidence_leakage_warns_on_answer_phrase_but_not_plain_observation():
    clean = detect_evidence_leakage(
        free_caption="A triangle ABC is visible.",
        task_evidence="Angle ABC is marked.",
        answer="45",
    )
    leaky = detect_evidence_leakage(
        free_caption="A triangle ABC is visible.",
        task_evidence="The answer is B.",
        answer="B",
    )

    assert clean == []
    assert "answer_phrase_leakage" in leaky
