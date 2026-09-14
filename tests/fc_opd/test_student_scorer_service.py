import torch

from dual_track_opd.fc_opd.conditions import Condition, ConditionInputs, ImageInput
from dual_track_opd.fc_opd.online_batch import OnlineFCOPDSample, OnlineStudentScores
from dual_track_opd.fc_opd.student_scorer_client import StudentScorerClient
from dual_track_opd.fc_opd.student_scorer_service import running_student_scorer_server


def _sample(uid: str = "sample-1") -> OnlineFCOPDSample:
    return OnlineFCOPDSample(
        sample_uid=uid,
        question="Find the angle.",
        condition_inputs=ConditionInputs(
            full_image=ImageInput(path="/tmp/full.png"),
            degraded_image=ImageInput(path="/tmp/degraded.png", transform={"type": "lowres_nearest", "scale": 0.1}),
            free_caption="A geometry diagram.",
            task_evidence="The diagram shows a labeled angle.",
            task_visible_evidence="A labeled angle is visible.",
            task_infer_evidence="The relation can be inferred.",
            task_solve_evidence="Solve using the relation.",
        ),
        rollout_token_ids=(1, 2, 3),
        rollout_text="ABC",
    )


class FakeBatchScorer:
    def __call__(self, sample, conditions):
        samples = list(sample) if isinstance(sample, list) else [sample]
        outputs = []
        for item in samples:
            outputs.append(
                OnlineStudentScores(
                    loss_logits=torch.ones((1, len(item.rollout_token_ids), 8)),
                    condition_log_probs={
                        Condition(condition): torch.full((len(item.rollout_token_ids),), -0.5)
                        for condition in conditions
                    },
                )
            )
        return outputs if isinstance(sample, list) else outputs[0]


def test_student_scorer_http_client_round_trips_batch_scores():
    with running_student_scorer_server(FakeBatchScorer()) as server:
        host, port = server.server_address
        client = StudentScorerClient(f"http://{host}:{port}")
        assert client.health()

        result = client([_sample("a"), _sample("b")], [Condition.FULL, Condition.TASK_SOLVE])

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].loss_logits is None
    assert set(result[0].condition_log_probs) == {Condition.FULL, Condition.TASK_SOLVE}
    assert result[0].condition_log_probs[Condition.FULL].tolist() == [-0.5, -0.5, -0.5]
