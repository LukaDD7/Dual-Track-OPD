"""STP-OPD training integration (CPU-testable core; verl hook is thin).

Implements the four-arm region contract from handoff
docs/cc_causal_state_to_stp_handoff_20260813.md §5:

| arm | prefix region | suffix region |
|---|---|---|
| A0 | none | teacher RKL/K1 + GRPO |
| A1 | context only; fully masked | GRPO |
| A2 | FKL | GRPO |
| A3 | FKL | teacher RKL/K1 + GRPO |

The trainer core is model-free: it consumes per-step logits/ids/advantages
and emits the regional loss plus a resumable manifest.  The real verl
integration feeds those tensors from rollouts; the online student scorer is an
injected callable so tests can substitute a fake.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from .prefix_scaffold import DEFAULT_SCHEDULE, region_masks
from .support_transition_loss import stp_opd_loss


ARM_REGIONS: dict[str, Mapping[str, bool]] = {
    "A0": {"prefix": False, "distill": True, "task": True},
    "A1": {"prefix": False, "distill": False, "task": True},
    "A2": {"prefix": True, "distill": False, "task": True},
    "A3": {"prefix": True, "distill": True, "task": True},
}


def arm_regions(arm: str) -> Mapping[str, bool]:
    if arm not in ARM_REGIONS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARM_REGIONS)}")
    return ARM_REGIONS[arm]


@dataclass(frozen=True)
class SupportTransitionTrainConfig:
    name: str = "support_transition_prefix_opd_pilot"
    arm: str = "A3"
    steps: int = 60
    seed: int = 42
    schedule: tuple[tuple[int, int, float], ...] = DEFAULT_SCHEDULE
    lambda_prefix: float = 1.0
    lambda_distill: float = 1.0
    lambda_task: float = 1.0
    run_dir: str = "."

    def validate(self) -> None:
        arm_regions(self.arm)
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        for start, end, fraction in self.schedule:
            if fraction < 0.0 or fraction > 1.0:
                raise ValueError(f"scaffold fraction out of range: {fraction}")

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        arm: str | None = None,
        steps: int | None = None,
    ) -> "SupportTransitionTrainConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        experiment = raw.get("experiment") or {}
        loss = raw.get("loss") or {}
        output = raw.get("output") or {}
        config = cls(
            name=str(experiment.get("name") or cls.name),
            arm=str(arm or (experiment.get("arms") or ["A3"])[0]),
            steps=int(steps or experiment.get("steps") or cls.steps),
            seed=int((experiment.get("seed_grid") or [cls.seed])[0]),
            schedule=tuple(
                tuple(value) for value in (experiment.get("scaffold_schedule") or DEFAULT_SCHEDULE)
            ),
            lambda_prefix=float(loss.get("lambda_prefix") or cls.lambda_prefix),
            lambda_distill=float(loss.get("lambda_distill") or cls.lambda_distill),
            lambda_task=float(loss.get("lambda_task") or cls.lambda_task),
            run_dir=str(output.get("run_dir") or cls.run_dir),
        )
        config.validate()
        return config


def train_step_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    prefix_ids: torch.Tensor,
    sampled_ids: torch.Tensor,
    advantages: torch.Tensor,
    arm: str,
    response_length: int,
    prefix_length: int,
    scaffolded: bool = True,
    valid_mask: torch.Tensor | None = None,
    config: SupportTransitionTrainConfig | None = None,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    """One regional training step for the requested arm.

    Disabled regions receive an empty mask, so their term is exactly zero with
    zero gradient (never NaN) and prefix context in A1 never enters the loss.
    """

    regions = arm_regions(arm)
    cfg = config or SupportTransitionTrainConfig(arm=arm)
    prefix_mask_all, suffix_mask_all = region_masks(
        response_length, prefix_length, scaffolded=scaffolded
    )
    batch = student_logits.shape[0]
    prefix_mask_all = prefix_mask_all.unsqueeze(0).expand(batch, -1)
    suffix_mask_all = suffix_mask_all.unsqueeze(0).expand(batch, -1)
    zeros = torch.zeros_like(prefix_mask_all)
    prefix_mask = prefix_mask_all if regions["prefix"] else zeros
    distill_mask = suffix_mask_all if regions["distill"] else zeros
    task_mask = suffix_mask_all if regions["task"] else zeros
    return stp_opd_loss(
        student_logits,
        teacher_logits,
        prefix_ids=prefix_ids,
        sampled_ids=sampled_ids,
        advantages=advantages,
        prefix_mask=prefix_mask,
        suffix_mask=suffix_mask_all,
        distill_mask=distill_mask,
        task_mask=task_mask,
        valid_mask=valid_mask,
        lambda_prefix=cfg.lambda_prefix,
        lambda_distill=cfg.lambda_distill,
        lambda_task=cfg.lambda_task,
    )


@dataclass
class TrainerState:
    step: int = 0
    rng_state: dict[str, Any] = field(default_factory=dict)
    total_loss: float = 0.0
    scaffolded_prompts: int = 0
    unscaffolded_prompts: int = 0

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


class SupportTransitionTrainer:
    """Minimal resumable trainer core (no model I/O).

    ``score_fn`` is the injected online student scorer: ``score_fn(batch) ->
    advantages``.  ``step()`` advances global step, applies the scaffold
    schedule, computes the arm loss, and records resumable state.
    """

    def __init__(
        self,
        config: SupportTransitionTrainConfig,
        *,
        score_fn: Callable[[Mapping[str, Any]], torch.Tensor],
    ):
        config.validate()
        self.config = config
        self.score_fn = score_fn
        self.state = TrainerState(step=0)

    def save_manifest(self, path: str | Path | None = None) -> Path:
        target = Path(path or Path(self.config.run_dir) / "run_manifest.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "support-transition-train-v1",
            "arm": self.config.arm,
            "config": asdict(self.config),
            "state": self.state.to_manifest(),
            "saved_at_unix": time.time(),
        }
        target.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return target

    def load_manifest(self, path: str | Path) -> None:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "support-transition-train-v1":
            raise ValueError(f"unexpected manifest schema: {manifest.get('schema_version')}")
        if manifest.get("arm") != self.config.arm:
            raise ValueError(
                f"manifest arm {manifest.get('arm')} != config arm {self.config.arm}"
            )
        self.state = TrainerState(**manifest["state"])

    def step(
        self,
        *,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        prefix_ids: torch.Tensor,
        sampled_ids: torch.Tensor,
        prefix_length: int,
        scaffolded: bool,
        batch: Mapping[str, Any],
    ) -> Mapping[str, torch.Tensor]:
        step = self.state.step + 1
        advantages = self.score_fn(batch)
        response_length = int(student_logits.shape[1])
        loss, terms = train_step_loss(
            student_logits,
            teacher_logits,
            prefix_ids=prefix_ids,
            sampled_ids=sampled_ids,
            advantages=advantages,
            arm=self.config.arm,
            response_length=response_length,
            prefix_length=prefix_length,
            scaffolded=scaffolded,
            config=self.config,
        )
        self.state.step = step
        self.state.total_loss += float(loss.detach())
        if scaffolded:
            self.state.scaffolded_prompts += int(student_logits.shape[0])
        else:
            self.state.unscaffolded_prompts += int(student_logits.shape[0])
        return {"loss": loss, **dict(terms)}
