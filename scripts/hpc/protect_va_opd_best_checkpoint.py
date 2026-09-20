#!/usr/bin/env python
"""Protect the best-validation VA-OPD checkpoint without changing training.

The trainer may retain only the most recent checkpoints.  This utility watches
the training log, identifies the best validation score, and hard-links that
checkpoint under ``best_val/`` after validation completes.  Hard links add
negligible disk usage while the source exists, and preserve the files after the
trainer prunes the source directory.

The default is a dry run.  Pass ``--apply`` to write the protected copy.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Dict


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STEP_RE = re.compile(r"(?:^|[\s])step:(\d+) - ")
VAL_RE = re.compile(
    r"val-core/[^ :]+/reward/mean@1:(?:np\.float64\()?([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def parse_validation_scores(log_path: Path) -> Dict[int, float]:
    if not log_path.is_file():
        return {}
    scores: Dict[int, float] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line)
            step_match = STEP_RE.search(line)
            val_match = VAL_RE.search(line)
            if not step_match or not val_match:
                continue
            scores[int(step_match.group(1))] = float(val_match.group(1))
    return scores


def checkpoint_steps(root: Path) -> Dict[int, Path]:
    if not root.is_dir():
        return {}
    result: Dict[int, Path] = {}
    for path in root.iterdir():
        match = re.fullmatch(r"global_step_(\d+)", path.name)
        if match and path.is_dir():
            result[int(match.group(1))] = path
    return result


def best_step(scores: Dict[int, float]) -> int | None:
    if not scores:
        return None
    # Prefer the earliest step on a tie to reduce late-training overfit risk.
    return max(scores, key=lambda step: (scores[step], -step))


def hardlink_tree(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary, copy_function=_hardlink)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)


def _hardlink(source: str, destination: str) -> None:
    try:
        import os

        os.link(source, destination)
    except OSError:
        # A non-linkable filesystem still gets a protected copy, at the cost of
        # actual bytes rather than directory entries.
        shutil.copy2(source, destination)


def protect_once(root: Path, log_path: Path, apply: bool) -> int | None:
    scores = parse_validation_scores(log_path)
    chosen_step = best_step(scores)
    checkpoints = checkpoint_steps(root)
    marker_path = root / "best_val.json"
    protected_root = root / "best_val"

    if chosen_step is None:
        print(f"best checkpoint: no validation score found in {log_path}")
        return None
    if chosen_step not in checkpoints:
        print(
            f"best checkpoint: step {chosen_step} (score={scores[chosen_step]}) "
            "has no source checkpoint yet"
        )
        return None

    source = checkpoints[chosen_step]
    destination = protected_root / f"global_step_{chosen_step}"
    marker = {
        "schema_version": 1,
        "step": chosen_step,
        "validation_score": scores[chosen_step],
        "source": str(source),
        "protected_path": str(destination),
    }
    print(
        f"best checkpoint: step {chosen_step} "
        f"score={scores[chosen_step]} source={source} protected={destination} apply={apply}"
    )
    if not apply:
        return chosen_step

    if not destination.exists():
        hardlink_tree(source, destination)
    elif destination.name != f"global_step_{chosen_step}":
        # Unreachable by construction, retained as a defensive assertion.
        raise RuntimeError("protected destination does not match chosen step")

    for child in protected_root.glob("global_step_*"):
        if child != destination:
            shutil.rmtree(child)

    marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    return chosen_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="write best_val; default is dry-run")
    parser.add_argument("--watch", action="store_true", help="run continuously after training starts")
    parser.add_argument("--poll-interval", type=float, default=300.0)
    args = parser.parse_args()
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not args.checkpoint_root.is_dir():
        raise SystemExit(f"checkpoint root does not exist: {args.checkpoint_root}")
    if args.watch:
        while True:
            protect_once(args.checkpoint_root, args.train_log, args.apply)
            time.sleep(args.poll_interval)
    else:
        protect_once(args.checkpoint_root, args.train_log, args.apply)


if __name__ == "__main__":
    main()
