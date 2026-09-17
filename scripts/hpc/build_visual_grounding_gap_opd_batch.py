#!/usr/bin/env python
"""Build a Visual Grounding Gap OPD diagnostic batch from scored rollouts.

This first-stage builder intentionally enriches an existing on-policy/scored
JSONL. It does not generate rollouts or load large models; those steps should
remain explicit in the HPC workflow.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from dual_track_opd.fc_opd.visual_grounding_gap_loss import (
    VisualGroundingGapLossConfig,
    compute_va_raw,
    compute_rollout_va_weights,
    compute_visual_grounding_gap_token_weights,
)
from dual_track_opd.fc_opd.signal_decomposer import TeacherTopK


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--top-q", type=float, default=0.20)
    parser.add_argument("--tau-rollout", type=float, default=1.0)
    parser.add_argument("--tau-chunk", type=float, default=0.15)
    parser.add_argument("--inputs-are-ranked", action="store_true")
    args = parser.parse_args()

    rows = _read_jsonl(args.input_jsonl)
    if not rows:
        raise SystemExit("input JSONL is empty")
    config = VisualGroundingGapLossConfig(
        top_q=args.top_q,
        tau_rollout=args.tau_rollout,
        tau_chunk=args.tau_chunk,
    )
    prompt_ids = [str(row.get("sample_uid", row.get("prompt_uid", index))) for index, row in enumerate(rows)]
    rollout_weights = _compute_rollout_weight_overrides(rows, prompt_ids=prompt_ids, config=config)
    output_rows = []
    for index, row in enumerate(rows):
        output_rows.append(
            _enrich_row(
                row,
                rollout_weight=float(rollout_weights[index]),
                config=config,
                inputs_are_ranked=args.inputs_are_ranked,
            )
        )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(output_rows), "output_jsonl": str(args.output_jsonl)}, indent=2))


def _enrich_row(
    row: Mapping[str, Any],
    *,
    rollout_weight: float,
    config: VisualGroundingGapLossConfig,
    inputs_are_ranked: bool,
) -> dict[str, Any]:
    sampled = torch.tensor([_required_list(row, "response_token_ids")], dtype=torch.long)
    teacher_full = _topk_from_row(row, "teacher_full_topk", "full")
    teacher_degraded = _topk_from_row(row, "teacher_degraded_topk", "degraded")
    va_raw = compute_va_raw(teacher_full, teacher_degraded, sampled)
    teacher_vfs = torch.tensor([_required_list(row, "teacher_visual_focus_scores")], dtype=torch.float32)
    student_vfs = torch.tensor([_required_list(row, "student_visual_focus_scores")], dtype=torch.float32)
    if teacher_vfs.shape != sampled.shape or student_vfs.shape != sampled.shape:
        raise ValueError(f"{row.get('sample_uid', 'unknown')}: visual focus scores must align to response tokens")
    chunks = _chunk_masks(row, length=sampled.shape[1])
    weights = compute_visual_grounding_gap_token_weights(
        va_raw,
        teacher_vfs,
        student_vfs,
        chunks,
        rollout_weights_override=torch.tensor([rollout_weight], dtype=torch.float32),
        verifier_outcomes=[row.get("verifier") or row.get("verifier_outcome")],
        inputs_are_ranked=inputs_are_ranked,
        config=config,
    )
    enriched = dict(row)
    enriched["visual_grounding_gap_opd"] = {
        "schema": "vgg_opd_weights_v1",
        "va_raw": _tolist(weights.va_raw[0]),
        "va_pos": _tolist(weights.va_pos[0]),
        "teacher_vfs_rank": _tolist(weights.teacher_vfs_rank[0]),
        "student_vfs_rank": _tolist(weights.student_vfs_rank[0]),
        "gap_raw": _tolist(weights.gap_raw[0]),
        "gap_pos": _tolist(weights.gap_pos[0]),
        "rollout_weight": float(weights.rollout_weights[0].item()),
        "token_weights": _tolist(weights.token_weights[0]),
        "high_va_mask": [bool(item) for item in weights.high_va_mask[0].tolist()],
        "low_va_mask": [bool(item) for item in weights.low_va_mask[0].tolist()],
        "chunk_gap_gate": {
            name: float(value[0].item()) for name, value in weights.chunk_gap_gates.gate.items()
        },
        "chunk_gap_rank": {
            name: float(value[0].item()) for name, value in weights.chunk_gap_gates.chunk_gap_rank.items()
        },
    }
    return enriched


def _compute_rollout_weight_overrides(
    rows: Sequence[Mapping[str, Any]],
    *,
    prompt_ids: Sequence[str],
    config: VisualGroundingGapLossConfig,
) -> list[float]:
    summaries = []
    for row in rows:
        sampled = torch.tensor([_required_list(row, "response_token_ids")], dtype=torch.long)
        teacher_full = _topk_from_row(row, "teacher_full_topk", "full")
        teacher_degraded = _topk_from_row(row, "teacher_degraded_topk", "degraded")
        va_pos = compute_va_raw(teacher_full, teacher_degraded, sampled).clamp_min(0.0)
        summaries.append(float(va_pos.topk(max(1, math.ceil(config.top_q * va_pos.numel()))).values.mean().item()))
    va = torch.tensor([[value] for value in summaries], dtype=torch.float32)
    weights = compute_rollout_va_weights(
        va,
        prompt_ids=prompt_ids,
        top_q=1.0,
        tau_rollout=config.tau_rollout,
    )
    return [float(item) for item in weights.tolist()]


def _topk_from_row(row: Mapping[str, Any], direct_key: str, condition_name: str) -> TeacherTopK:
    block = row.get(direct_key)
    if block is None:
        scores = row.get("condition_scores")
        if isinstance(scores, Mapping):
            block = scores.get(condition_name)
    if not isinstance(block, Mapping):
        raise ValueError(f"{row.get('sample_uid', 'unknown')}: missing {direct_key} or condition_scores.{condition_name}")
    token_ids = block.get("token_ids", block.get("topk_token_ids"))
    log_probs = block.get("log_probs", block.get("topk_log_probs"))
    if token_ids is None or log_probs is None:
        raise ValueError(f"{direct_key}: expected token_ids/log_probs or topk_token_ids/topk_log_probs")
    tail = block.get("tail_log_prob")
    return TeacherTopK(
        token_ids=torch.tensor([token_ids], dtype=torch.long),
        log_probs=torch.tensor([log_probs], dtype=torch.float32),
        tail_log_prob=None if tail is None else torch.tensor([tail], dtype=torch.float32),
    )


def _chunk_masks(row: Mapping[str, Any], *, length: int) -> dict[str, torch.Tensor]:
    labels = row.get("chunk_labels")
    if labels is None:
        block = row.get("chunk_spans", {})
        labels = ["outside"] * length
        if isinstance(block, Mapping):
            for name, spans in block.items():
                if name == "format_valid":
                    continue
                for span in _spans(spans):
                    start, end = span
                    for index in range(max(0, start), min(length, end)):
                        labels[index] = str(name)
    if len(labels) != length:
        raise ValueError(f"{row.get('sample_uid', 'unknown')}: chunk labels must align to response tokens")
    return {
        name: torch.tensor([[label == name for label in labels]], dtype=torch.bool)
        for name in ("visible_evidence", "diagram_inference", "reasoning", "answer")
    }


def _spans(value: Any) -> list[tuple[int, int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    if len(value) == 2 and all(isinstance(item, int) for item in value):
        return [(int(value[0]), int(value[1]))]
    out = []
    for item in value:
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
            out.append((int(item[0]), int(item[1])))
    return out


def _required_list(row: Mapping[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{row.get('sample_uid', 'unknown')}: missing list field {key}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tolist(tensor: torch.Tensor) -> list[float]:
    return [float(item) for item in tensor.detach().cpu().tolist()]


if __name__ == "__main__":
    main()
