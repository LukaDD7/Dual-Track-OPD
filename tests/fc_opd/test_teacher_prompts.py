import pytest

from dual_track_opd.fc_opd.conditions import Condition, ConditionInputs, ImageInput
from dual_track_opd.fc_opd.teacher_prompts import render_teacher_prompt


def _inputs(with_facts: bool = True) -> ConditionInputs:
    return ConditionInputs(
        full_image=ImageInput("/data/full.png"),
        degraded_image=ImageInput(
            "/data/blur.png",
            {"type": "gaussian_blur", "sigma": 2.0},
        ),
        free_caption="A red square is left of a blue circle.",
        task_evidence="Objects: red square, blue circle. Relation: square left of circle.",
        task_visible_evidence="Objects: red square, blue circle. Relation: square left of circle.",
        task_infer_evidence="The square is left of the circle, so left/right relation is usable.",
        task_solve_evidence="The square is on the left, so the answer is left.",
        verified_facts="There are exactly two objects." if with_facts else None,
        verified_facts_source="human annotation" if with_facts else None,
    )


@pytest.mark.parametrize(
    ("condition", "image_paths", "fragment"),
    [
        (Condition.FULL, ("/data/full.png",), "Question:\nWhere is the square?"),
        (Condition.BLUR, ("/data/blur.png",), "Question:\nWhere is the square?"),
        (Condition.FREE, (), "Image description"),
        (Condition.TASK, (), "Evidence"),
        (Condition.TASK_VISIBLE, (), "Evidence"),
        (Condition.TASK_INFER, (), "Context"),
        (Condition.TASK_SOLVE, (), "Context"),
        (Condition.FACT, (), "Facts"),
    ],
)
def test_all_conditions_render(condition, image_paths, fragment):
    rendered = render_teacher_prompt(condition, "Where is the square?", _inputs())
    assert rendered.image_paths == image_paths
    content = rendered.messages[0]["content"]
    text = next(item["text"] for item in content if item["type"] == "text")
    assert fragment in text
    assert "<visible_evidence>" not in text
    assert "<diagram_inference>" not in text
    assert "<answer>" not in text


def test_fact_condition_requires_external_facts():
    with pytest.raises(ValueError, match="requires verified facts"):
        render_teacher_prompt(Condition.FACT, "Question?", _inputs(with_facts=False))


def test_full_and_degraded_teacher_prompts_differ_only_by_image():
    full = render_teacher_prompt(Condition.FULL, "Which angle is marked?", _inputs())
    degraded = render_teacher_prompt(Condition.DEGRADED, "Which angle is marked?", _inputs())

    full_text = next(item["text"] for item in full.messages[0]["content"] if item["type"] == "text")
    degraded_text = next(item["text"] for item in degraded.messages[0]["content"] if item["type"] == "text")

    assert full.image_paths == ("/data/full.png",)
    assert degraded.image_paths == ("/data/blur.png",)
    assert full_text == degraded_text
    assert "<answer>" not in full_text
