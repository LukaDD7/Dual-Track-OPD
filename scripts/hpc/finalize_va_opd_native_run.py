#!/usr/bin/env python3
"""Validate native OPD/VA-OPD terminal logs and finalize the run manifest."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def values(text: str, key: str) -> list[float]:
    pattern = re.compile(re.escape(key) + r"\s*[:=]\s*(" + FLOAT + r")")
    return [float(match.group(1)) for match in pattern.finditer(text)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--objective", choices=("opd", "va_opd"), required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--requested-steps", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()

    text = args.train_log.read_text(encoding="utf-8", errors="replace")
    steps = values(text, "training/global_step")
    distill_losses = values(text, "distillation/loss") + values(text, "actor/distillation/loss")
    grad_norms = values(text, "actor/grad_norm")
    entropies = values(text, "actor/entropy")
    response_means = values(text, "response_length/mean")
    response_clip_ratios = values(text, "response_length/clip_ratio")
    va_means = values(text, "va_opd/mean")
    group_errors = values(text, "va_opd/group_weight_sum_max_error")

    finite_losses = [value for value in distill_losses if math.isfinite(value)]
    finite_grads = [value for value in grad_norms if math.isfinite(value)]
    finite_entropies = [value for value in entropies if math.isfinite(value)]
    completed_steps = int(max(steps)) if steps else 0
    passed = (
        args.exit_code == 0
        and bool(finite_losses)
        and bool(finite_grads)
        and bool(finite_entropies)
        and completed_steps > 0
    )
    failures: list[str] = []
    if args.requested_steps and completed_steps < args.requested_steps:
        passed = False
        failures.append(f"completed {completed_steps}/{args.requested_steps} requested steps")
    if args.objective == "va_opd":
        if not va_means:
            passed = False
            failures.append("no VA metric was logged")
        if not group_errors or max(group_errors) > 1e-5:
            passed = False
            failures.append("rollout group weights did not pass the sum-to-one gate")
    if args.exit_code:
        failures.append(f"trainer exit code {args.exit_code}")
    if not finite_losses:
        failures.append("no finite distillation loss")
    if not finite_grads:
        failures.append("no finite actor gradient")
    if not finite_entropies:
        failures.append("no finite actor entropy; collapse monitoring is unavailable")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = {
        "passed": passed,
        "failures": failures,
        "exit_code": args.exit_code,
        "completed_steps": completed_steps,
        "requested_steps": args.requested_steps,
        "finite_distillation_loss_count": len(finite_losses),
        "finite_gradient_count": len(finite_grads),
        "finite_entropy_count": len(finite_entropies),
        "last_distillation_loss": finite_losses[-1] if finite_losses else None,
        "last_gradient_norm": finite_grads[-1] if finite_grads else None,
        "last_actor_entropy": finite_entropies[-1] if finite_entropies else None,
        "last_response_length_mean": response_means[-1] if response_means else None,
        "last_response_length_clip_ratio": response_clip_ratios[-1] if response_clip_ratios else None,
        "last_va_mean": va_means[-1] if va_means else None,
        "max_group_weight_sum_error": max(group_errors) if group_errors else None,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
    }
    manifest["result"] = result
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_path = args.manifest.parent / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
