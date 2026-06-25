"""Offline FC-OPD loss/backward smoke over recorded teacher scores.

This stage closes the loop between the offline-score dataset
(``build_fc_opd_offline_scores.py``) and the existing FC-OPD loss/router path,
*without* any student model, teacher service, or ``third_party/verl`` import. It
reads recorded teacher top-k scores, rebuilds the tensors and chunk masks,
attaches synthetic student logits with ``requires_grad=True``, and runs the real
``route_condition_weights`` -> ``compute_fc_opd_loss`` -> ``backward`` path so the
gradient plumbing can be verified before touching the trainer backend.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .conditions import Condition
from .loss import FCOPDLossConfig, compute_fc_opd_loss
from .router import RouterConfig, route_condition_weights
from .signal_decomposer import TeacherTopK

CHUNK_NAMES = ("visual_evidence", "reasoning", "answer")
FOUR_CONDITIONS: tuple[Condition, ...] = (
    Condition.FULL,
    Condition.BLUR,
    Condition.FREE,
    Condition.TASK,
)

# Router that consumes all four conditions: the three structured chunks map to
# task/free/full, and the remaining response tokens (tags, whitespace) fall back
# to blur. This deliberately exercises every recorded condition in the loss.
FOUR_CONDITION_ROUTER = RouterConfig(
    mode="chunk",
    chunk_condition={
        "visual_evidence": Condition.TASK,
        "reasoning": Condition.FREE,
        "answer": Condition.FULL,
    },
    invalid_format_condition=Condition.BLUR,
)


def load_offline_score_records(path: str | Path) -> list[dict[str, Any]]:
    """Read offline-score payloads from a JSONL file."""

    path = Path(path)
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if not isinstance(record, Mapping):
            raise ValueError("each offline-score line must be a JSON object")
        records.append(dict(record))
    return records


def _condition_topk(
    block: Mapping[str, Any],
    *,
    device: torch.device,
) -> TeacherTopK:
    token_ids = torch.tensor([block["token_ids"]], dtype=torch.int64, device=device)
    log_probs = torch.tensor([block["log_probs"]], dtype=torch.float32, device=device)
    tail = block.get("tail_log_prob")
    entropy = block.get("entropy")
    scores = TeacherTopK(
        token_ids=token_ids,
        log_probs=log_probs,
        tail_log_prob=(
            None if tail is None else torch.tensor([tail], dtype=torch.float32, device=device)
        ),
        entropy=(
            None if entropy is None else torch.tensor([entropy], dtype=torch.float32, device=device)
        ),
    )
    scores.validate()
    return scores


def _spans_to_mask(spans: Sequence[Sequence[int]], length: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((1, length), dtype=torch.bool, device=device)
    for span in spans:
        start, end = int(span[0]), int(span[1])
        if start < 0 or end > length or start > end:
            raise ValueError(f"chunk span {span} is out of bounds for T={length}")
        mask[0, start:end] = True
    return mask


@dataclass(frozen=True)
class OfflineRecordTensors:
    sample_uid: str
    teacher_scores: dict[Condition, TeacherTopK]
    chunk_masks: dict[str, torch.Tensor]
    response_mask: torch.Tensor
    format_valid: torch.Tensor
    num_response_tokens: int
    seq_len: int
    vocab_floor: int

    def validate_alignment(self) -> None:
        if self.seq_len != self.num_response_tokens:
            raise ValueError(
                f"{self.sample_uid}: teacher seq length {self.seq_len} != "
                f"response token length {self.num_response_tokens}"
            )
        for name, mask in self.chunk_masks.items():
            if mask.shape != (1, self.seq_len):
                raise ValueError(f"{self.sample_uid}: {name} mask is not aligned to T={self.seq_len}")
        for condition, scores in self.teacher_scores.items():
            if scores.token_ids.shape[1] != self.seq_len:
                raise ValueError(f"{self.sample_uid}: {condition.value} scores are not aligned to T")


def offline_record_to_tensors(
    payload: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> OfflineRecordTensors:
    """Rebuild teacher scores, chunk masks, and response mask from one payload."""

    device = torch.device(device)
    score_block = payload["condition_scores"]
    if not isinstance(score_block, Mapping) or not score_block:
        raise ValueError("payload is missing condition_scores")

    teacher_scores: dict[Condition, TeacherTopK] = {}
    seq_len: int | None = None
    vocab_floor = 0
    for condition_name, block in score_block.items():
        scores = _condition_topk(block, device=device)
        condition = Condition(condition_name)
        teacher_scores[condition] = scores
        if seq_len is None:
            seq_len = scores.token_ids.shape[1]
        vocab_floor = max(vocab_floor, int(scores.token_ids.max().item()) + 1)
    assert seq_len is not None

    response_token_ids = payload.get("response_token_ids")
    if not isinstance(response_token_ids, Sequence):
        raise ValueError("payload is missing response_token_ids")
    num_response_tokens = len(response_token_ids)

    chunk_spans = payload.get("chunk_spans")
    if not isinstance(chunk_spans, Mapping):
        raise ValueError("payload is missing chunk_spans")
    chunk_masks = {
        name: _spans_to_mask(chunk_spans.get(name, ()), seq_len, device) for name in CHUNK_NAMES
    }
    format_valid = torch.tensor([bool(chunk_spans.get("format_valid", False))], device=device)
    response_mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)

    tensors = OfflineRecordTensors(
        sample_uid=str(payload.get("sample_uid", "unknown")),
        teacher_scores=teacher_scores,
        chunk_masks=chunk_masks,
        response_mask=response_mask,
        format_valid=format_valid,
        num_response_tokens=num_response_tokens,
        seq_len=seq_len,
        vocab_floor=vocab_floor,
    )
    tensors.validate_alignment()
    return tensors


def make_student_logits(
    seq_len: int,
    vocab_size: int,
    *,
    seed: int = 0,
    scale: float = 0.1,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Create a synthetic student logit tensor [1, T, V] with requires_grad."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(1, seq_len, vocab_size, generator=generator) * scale
    return logits.to(device).detach().requires_grad_(True)


@dataclass
class OfflineLossResult:
    sample_uid: str
    loss_value: float
    loss_is_finite: bool
    grad_exists: bool
    grad_is_finite: bool
    grad_norm: float
    consumed_conditions: set[Condition]
    seq_len: int
    num_response_tokens: int
    format_valid: bool
    metrics: dict[str, float] = field(default_factory=dict)


def run_offline_loss_backward(
    payload: Mapping[str, Any],
    *,
    router_config: RouterConfig = FOUR_CONDITION_ROUTER,
    loss_config: FCOPDLossConfig | None = None,
    student_vocab_size: int | None = None,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> OfflineLossResult:
    """Run the full route -> loss -> backward path for one offline record."""

    tensors = offline_record_to_tensors(payload, device=device)
    available = tuple(tensors.teacher_scores)
    vocab_size = student_vocab_size if student_vocab_size is not None else tensors.vocab_floor
    if vocab_size < tensors.vocab_floor:
        raise ValueError("student_vocab_size is smaller than the largest teacher token id")

    student_logits = make_student_logits(
        tensors.seq_len, vocab_size, seed=seed, device=device
    )

    weights = route_condition_weights(
        signals={},
        chunk_masks=tensors.chunk_masks,
        router_config=router_config,
        response_mask=tensors.response_mask,
        available_conditions=available,
        format_valid=tensors.format_valid,
    )

    loss, metrics = compute_fc_opd_loss(
        student_logits,
        tensors.teacher_scores,
        tensors.chunk_masks,
        weights,
        tensors.response_mask,
        loss_config or FCOPDLossConfig(),
    )

    loss_is_finite = bool(torch.isfinite(loss).all().item())
    loss.backward()

    grad = student_logits.grad
    grad_exists = grad is not None
    grad_is_finite = bool(grad_exists and torch.isfinite(grad).all().item())
    grad_norm = float(grad.norm().item()) if grad_exists else 0.0

    consumed = {
        condition
        for condition in available
        if metrics.get(f"selection/{condition.value}", torch.zeros(())).item() > 0.0
    }

    return OfflineLossResult(
        sample_uid=tensors.sample_uid,
        loss_value=float(loss.detach().item()),
        loss_is_finite=loss_is_finite,
        grad_exists=grad_exists,
        grad_is_finite=grad_is_finite,
        grad_norm=grad_norm,
        consumed_conditions=consumed,
        seq_len=tensors.seq_len,
        num_response_tokens=tensors.num_response_tokens,
        format_valid=bool(tensors.format_valid.item()),
        metrics={key: float(value.item()) for key, value in metrics.items()},
    )


@dataclass
class OfflineLossSmokeReport:
    num_records: int
    all_loss_finite: bool
    all_grads_present: bool
    all_grads_finite: bool
    all_masks_aligned: bool
    consumed_conditions: set[Condition]
    results: list[OfflineLossResult] = field(default_factory=list)

    @property
    def four_conditions_consumed(self) -> bool:
        return set(FOUR_CONDITIONS).issubset(self.consumed_conditions)

    @property
    def passed(self) -> bool:
        return (
            self.num_records > 0
            and self.all_loss_finite
            and self.all_grads_present
            and self.all_grads_finite
            and self.all_masks_aligned
            and self.four_conditions_consumed
        )


def run_offline_loss_smoke(
    records: Iterable[Mapping[str, Any]],
    *,
    router_config: RouterConfig = FOUR_CONDITION_ROUTER,
    loss_config: FCOPDLossConfig | None = None,
    student_vocab_size: int | None = None,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> OfflineLossSmokeReport:
    """Run the loss/backward smoke over a set of offline-score records."""

    results: list[OfflineLossResult] = []
    consumed: set[Condition] = set()
    all_loss_finite = True
    all_grads_present = True
    all_grads_finite = True
    all_masks_aligned = True

    for record in records:
        result = run_offline_loss_backward(
            record,
            router_config=router_config,
            loss_config=loss_config,
            student_vocab_size=student_vocab_size,
            seed=seed,
            device=device,
        )
        results.append(result)
        consumed |= result.consumed_conditions
        all_loss_finite &= result.loss_is_finite
        all_grads_present &= result.grad_exists
        all_grads_finite &= result.grad_is_finite
        all_masks_aligned &= result.seq_len == result.num_response_tokens

    return OfflineLossSmokeReport(
        num_records=len(results),
        all_loss_finite=all_loss_finite,
        all_grads_present=all_grads_present,
        all_grads_finite=all_grads_finite,
        all_masks_aligned=all_masks_aligned,
        consumed_conditions=consumed,
        results=results,
    )


# ---------------------------------------------------------------------------
# Minimal optimizer-update smoke
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MinTrainStep:
    step: int
    loss: float
    loss_is_finite: bool
    grad_norm: float
    grad_is_finite: bool
    logits_delta_norm: float
    consumed_conditions: set[Condition]


@dataclass
class MinTrainReport:
    num_records: int
    num_steps: int
    learning_rate: float
    steps: list[MinTrainStep] = field(default_factory=list)

    @property
    def all_loss_finite(self) -> bool:
        return all(step.loss_is_finite for step in self.steps)

    @property
    def all_grads_finite(self) -> bool:
        return all(step.grad_is_finite for step in self.steps)

    @property
    def every_step_updates_logits(self) -> bool:
        return all(step.logits_delta_norm > 0.0 for step in self.steps)

    @property
    def consumed_conditions(self) -> set[Condition]:
        consumed: set[Condition] = set()
        for step in self.steps:
            consumed |= step.consumed_conditions
        return consumed

    @property
    def four_conditions_consumed(self) -> bool:
        return set(FOUR_CONDITIONS).issubset(self.consumed_conditions)

    @property
    def loss_decreased(self) -> bool:
        return bool(self.steps) and self.steps[-1].loss < self.steps[0].loss

    @property
    def passed(self) -> bool:
        return (
            self.num_records > 0
            and self.num_steps > 0
            and self.all_loss_finite
            and self.all_grads_finite
            and self.every_step_updates_logits
            and self.four_conditions_consumed
        )


def _global_grad_norm(params: Sequence[torch.Tensor]) -> tuple[float, bool]:
    norms = []
    finite = True
    for param in params:
        if param.grad is None:
            continue
        norms.append(param.grad.detach().float().norm())
        finite &= bool(torch.isfinite(param.grad).all().item())
    if not norms:
        return 0.0, finite
    return float(torch.stack(norms).norm().item()), finite


def run_offline_min_train_smoke(
    records: Iterable[Mapping[str, Any]],
    *,
    num_steps: int = 10,
    learning_rate: float = 0.05,
    router_config: RouterConfig = FOUR_CONDITION_ROUTER,
    loss_config: FCOPDLossConfig | None = None,
    student_vocab_size: int | None = None,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> MinTrainReport:
    """Run a minimal Adam optimizer-update smoke over offline-score records.

    Each record gets its own synthetic ``student_logits`` parameter; a single
    Adam optimizer steps them against the recorded teacher scores. Router weights
    are fixed (they do not depend on the student), so the per-token loss is purely
    a function of the student logits and the update should reduce it.
    """

    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    device = torch.device(device)
    loss_config = loss_config or FCOPDLossConfig()

    tensors_list = [offline_record_to_tensors(record, device=device) for record in records]
    if not tensors_list:
        raise ValueError("no offline-score records were provided")

    params: list[torch.Tensor] = []
    weights_list: list[dict[Condition, torch.Tensor]] = []
    for index, tensors in enumerate(tensors_list):
        vocab_size = student_vocab_size if student_vocab_size is not None else tensors.vocab_floor
        if vocab_size < tensors.vocab_floor:
            raise ValueError("student_vocab_size is smaller than the largest teacher token id")
        params.append(make_student_logits(tensors.seq_len, vocab_size, seed=seed + index, device=device))
        weights_list.append(
            route_condition_weights(
                signals={},
                chunk_masks=tensors.chunk_masks,
                router_config=router_config,
                response_mask=tensors.response_mask,
                available_conditions=tuple(tensors.teacher_scores),
                format_valid=tensors.format_valid,
            )
        )

    optimizer = torch.optim.Adam(params, lr=learning_rate)
    report = MinTrainReport(
        num_records=len(tensors_list), num_steps=num_steps, learning_rate=learning_rate
    )

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        consumed: set[Condition] = set()
        for tensors, logits, weights in zip(tensors_list, params, weights_list, strict=True):
            loss, metrics = compute_fc_opd_loss(
                logits,
                tensors.teacher_scores,
                tensors.chunk_masks,
                weights,
                tensors.response_mask,
                loss_config,
            )
            total_loss = total_loss + loss
            consumed |= {
                condition
                for condition in tensors.teacher_scores
                if metrics.get(f"selection/{condition.value}", torch.zeros(())).item() > 0.0
            }

        loss_is_finite = bool(torch.isfinite(total_loss).all().item())
        total_loss.backward()
        grad_norm, grad_is_finite = _global_grad_norm(params)

        snapshot = [param.detach().clone() for param in params]
        optimizer.step()
        delta = torch.stack(
            [(param.detach() - old).float().norm() for param, old in zip(params, snapshot, strict=True)]
        ).norm()

        report.steps.append(
            MinTrainStep(
                step=step,
                loss=float(total_loss.detach().item()),
                loss_is_finite=loss_is_finite,
                grad_norm=grad_norm,
                grad_is_finite=grad_is_finite,
                logits_delta_norm=float(delta.item()),
                consumed_conditions=consumed,
            )
        )

    return report


def min_train_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Minimal Adam optimizer-update smoke for offline FC-OPD loss.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scores", type=Path, required=True, help="offline-score JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N records")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--student-vocab-size", type=int, default=None)
    args = parser.parse_args(argv)

    records = load_offline_score_records(args.scores)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        print(f"FAIL: no records found in {args.scores}", file=sys.stderr)
        return 1

    report = run_offline_min_train_smoke(
        records,
        num_steps=args.steps,
        learning_rate=args.lr,
        student_vocab_size=args.student_vocab_size,
        seed=args.seed,
        device=args.device,
    )

    for step in report.steps:
        print(
            json.dumps(
                {
                    "step": step.step,
                    "loss": step.loss,
                    "grad_norm": step.grad_norm,
                    "logits_delta_norm": step.logits_delta_norm,
                    "consumed_conditions": sorted(c.value for c in step.consumed_conditions),
                }
            )
        )

    summary = {
        "num_records": report.num_records,
        "num_steps": report.num_steps,
        "learning_rate": report.learning_rate,
        "all_loss_finite": report.all_loss_finite,
        "all_grads_finite": report.all_grads_finite,
        "every_step_updates_logits": report.every_step_updates_logits,
        "four_conditions_consumed": report.four_conditions_consumed,
        "consumed_conditions": sorted(c.value for c in report.consumed_conditions),
        "initial_loss": report.steps[0].loss,
        "final_loss": report.steps[-1].loss,
        "loss_decreased": report.loss_decreased,
        "passed": report.passed,
    }
    print(json.dumps(summary, indent=2))
    if not report.passed:
        print("FAIL: offline min-train smoke did not pass", file=sys.stderr)
        return 1
    print("PASS: offline min-train smoke is valid")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, required=True, help="offline-score JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="only score the first N records")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--student-vocab-size",
        type=int,
        default=None,
        help="override the synthetic student vocab (defaults to max teacher id + 1)",
    )
    args = parser.parse_args(argv)

    records = load_offline_score_records(args.scores)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        print(f"FAIL: no records found in {args.scores}", file=sys.stderr)
        return 1

    report = run_offline_loss_smoke(
        records,
        student_vocab_size=args.student_vocab_size,
        seed=args.seed,
        device=args.device,
    )

    print(
        json.dumps(
            {
                "num_records": report.num_records,
                "all_loss_finite": report.all_loss_finite,
                "all_grads_present": report.all_grads_present,
                "all_grads_finite": report.all_grads_finite,
                "all_masks_aligned": report.all_masks_aligned,
                "consumed_conditions": sorted(c.value for c in report.consumed_conditions),
                "four_conditions_consumed": report.four_conditions_consumed,
                "mean_loss": (
                    sum(r.loss_value for r in report.results) / report.num_records
                ),
                "passed": report.passed,
            },
            indent=2,
        )
    )
    if not report.passed:
        print("FAIL: offline loss/backward smoke did not pass", file=sys.stderr)
        return 1
    print("PASS: offline loss/backward smoke is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
