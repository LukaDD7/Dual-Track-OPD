"""CPU tests for the STP-OPD trainer core and four-arm region contract."""

from pathlib import Path

import torch

from dual_track_opd.support_aware.support_transition_train import (
    ARM_REGIONS,
    SupportTransitionTrainConfig,
    SupportTransitionTrainer,
    arm_regions,
    train_step_loss,
)


def _inputs(batch=2, seq=8, vocab=16, prefix_length=3):
    return {
        "student_logits": torch.randn(batch, seq, vocab, requires_grad=True),
        "teacher_logits": torch.randn(batch, seq, vocab),
        "prefix_ids": torch.randint(0, vocab, (batch, seq)),
        "sampled_ids": torch.randint(0, vocab, (batch, seq)),
        "advantages": torch.randn(batch, seq),
        "response_length": seq,
        "prefix_length": prefix_length,
        "scaffolded": True,
    }


def test_arm_regions_contract():
    assert ARM_REGIONS["A0"] == {"prefix": False, "distill": True, "task": True}
    assert ARM_REGIONS["A1"] == {"prefix": False, "distill": False, "task": True}
    assert ARM_REGIONS["A2"] == {"prefix": True, "distill": False, "task": True}
    assert ARM_REGIONS["A3"] == {"prefix": True, "distill": True, "task": True}


def _prefix_grad(loss):
    logits = torch.randn(2, 8, 16, requires_grad=True)
    inputs = _inputs()
    inputs["student_logits"] = logits
    loss_value, _ = train_step_loss(
        logits,
        inputs["teacher_logits"],
        prefix_ids=inputs["prefix_ids"],
        sampled_ids=inputs["sampled_ids"],
        advantages=inputs["advantages"],
        arm=loss,
        response_length=8,
        prefix_length=3,
    )
    loss_value.backward()
    return logits.grad


def test_a1_prefix_context_never_enters_loss():
    grad = _prefix_grad("A1")
    assert grad is not None and torch.isfinite(grad).all()
    # prefix positions (0..2) receive no gradient in A1
    assert bool((grad[:, :3, :] == 0).all())
    assert bool((grad[:, 3:, :] != 0).any())


def test_a0_has_no_prefix_region():
    grad = _prefix_grad("A0")
    assert grad is not None
    assert bool((grad[:, :3, :] == 0).all())


def test_a2_has_prefix_but_no_distill():
    grad = _prefix_grad("A2")
    assert grad is not None
    assert bool((grad[:, :3, :] != 0).any())


def test_a3_has_all_regions():
    grad = _prefix_grad("A3")
    assert grad is not None
    assert bool((grad[:, :3, :] != 0).any())
    assert bool((grad[:, 3:, :] != 0).any())


def test_trainer_step_and_manifest_resume(tmp_path):
    config = SupportTransitionTrainConfig(
        arm="A3",
        run_dir=str(tmp_path),
    )

    def score_fn(batch):
        del batch
        return torch.randn(2, 8)

    trainer = SupportTransitionTrainer(config, score_fn=score_fn)
    inputs = _inputs()
    for _ in range(3):
        trainer.step(
            student_logits=inputs["student_logits"],
            teacher_logits=inputs["teacher_logits"],
            prefix_ids=inputs["prefix_ids"],
            sampled_ids=inputs["sampled_ids"],
            prefix_length=inputs["prefix_length"],
            scaffolded=True,
            batch={},
        )
    manifest = trainer.save_manifest()
    assert manifest.exists()
    assert trainer.state.step == 3

    resumed = SupportTransitionTrainer(config, score_fn=score_fn)
    resumed.load_manifest(manifest)
    assert resumed.state.step == 3
    assert resumed.state.scaffolded_prompts == 6


def test_trainer_rejects_arm_mismatch_on_resume(tmp_path):
    config = SupportTransitionTrainConfig(arm="A3", run_dir=str(tmp_path))
    trainer = SupportTransitionTrainer(config, score_fn=lambda batch: torch.randn(2, 8))
    manifest = trainer.save_manifest()
    other = SupportTransitionTrainer(
        SupportTransitionTrainConfig(arm="A0", run_dir=str(tmp_path)),
        score_fn=lambda batch: torch.randn(2, 8),
    )
    try:
        other.load_manifest(manifest)
    except ValueError as error:
        assert "manifest arm" in str(error)
    else:
        raise AssertionError("expected arm-mismatch error")


def test_config_from_yaml_parses_pilot(tmp_path):
    config_path = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "support_transition_prefix_opd_pilot.yaml"
    config = SupportTransitionTrainConfig.from_yaml(config_path, arm="A3", steps=60)
    assert config.arm == "A3"
    assert config.steps == 60
    assert config.lambda_prefix == 1.0
    assert config.schedule == ((1, 20, 0.75), (21, 40, 0.50), (41, 60, 0.25))
