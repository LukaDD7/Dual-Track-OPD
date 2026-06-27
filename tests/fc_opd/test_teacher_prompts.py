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
        (Condition.FULL, ("/data/full.png",), "full image"),
        (Condition.BLUR, ("/data/blur.png",), "degraded image"),
        (Condition.FREE, (), "Image description"),
        (Condition.TASK, (), "Question-conditioned visible evidence"),
        (Condition.TASK_VISIBLE, (), "Question-conditioned visible evidence"),
        (Condition.TASK_INFER, (), "Task-conditioned diagram/geometric inference"),
        (Condition.TASK_SOLVE, (), "Teacher-inferred solution context"),
        (Condition.FACT, (), "Externally verified visual facts"),
    ],
)
def test_all_conditions_render(condition, image_paths, fragment):
    rendered = render_teacher_prompt(condition, "Where is the square?", _inputs())
    assert rendered.image_paths == image_paths
    content = rendered.messages[0]["content"]
    text = next(item["text"] for item in content if item["type"] == "text")
    assert fragment in text
    assert "<visible_evidence>" in text
    assert "<diagram_inference>" in text
    assert "<answer>" in text


def test_fact_condition_requires_external_facts():
    with pytest.raises(ValueError, match="requires verified facts"):
        render_teacher_prompt(Condition.FACT, "Question?", _inputs(with_facts=False))
