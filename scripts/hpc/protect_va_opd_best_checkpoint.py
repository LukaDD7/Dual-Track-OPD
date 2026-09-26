#!/usr/bin/env python
"""Protect the best-validation VA-OPD checkpoint without changing training.

The trainer may retain only the most recent checkpoints.  This utility watches
the training log, identifies the best validation score, and copies that
checkpoint under ``protected_runs`` after validation completes.  The protected
copy is independent of rolling-checkpoint cleanup.

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


def load_history(path: Path) -> Dict[int, float]:
    """Load persistent validation history, surviving train.log rewrites."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_scores = payload.get("scores", {}) if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw_scores, dict):
        return {}

    scores: Dict[int, float] = {}
    for raw_step, raw_score in raw_scores.items():
        try:
            scores[int(raw_step)] = float(raw_score)
        except (TypeError, ValueError):
            continue
    return scores


def load_marker_history(path: Path) -> Dict[int, float]:
    """Back-fill history from a legacy best_val marker once."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        step = int(payload["step"])
        score = float(payload["validation_score"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return {step: score}


def merge_validation_scores(score_maps: list[Dict[int, float]]) -> Dict[int, float]:
    """Merge maps so historical best scores survive resume-time revalidation.

    Sources are ordered oldest-to-newest. For a repeated validation step, we
    keep the maximum score: the historical score selected the checkpoint, and
    a lower resume-time score is a different random validation draw rather than
    evidence that the already-written checkpoint became worse.
    """
    merged: Dict[int, float] = {}
    for scores in score_maps:
        for step, score in scores.items():
            if step not in merged or score > merged[step]:
                merged[step] = score
    return merged


def best_step(
    scores: Dict[int, float], checkpoints: Dict[int, Path], protected_root: Path
) -> int | None:
    available = [
        step
        for step in scores
        if _is_complete_checkpoint(checkpoints.get(step, protected_root / f"global_step_{step}"))
    ]
    if not available:
        return None
    # Prefer the latest available checkpoint on a validation-score tie. A
    # protected copy remains available after the trainer prunes its rolling
    # source, so a historical best must not be silently replaced by a newer,
    # lower-scoring checkpoint.
    return max(available, key=lambda step: (scores[step], step))


def _is_complete_checkpoint(path: Path) -> bool:
    """Reject trainer-retention tombstones.

    VERL removes checkpoint shard files but can leave the directory and
    ``data.pt`` behind. A protected best must contain model shards and remain
    independently loadable, not merely have a directory named like a checkpoint.
    """
    actor = path / "actor"
    if not actor.is_dir():
        return False
    model_shards = sorted(actor.glob("model*.pt")) or sorted(actor.glob("model*.safetensors"))
    return bool(model_shards) and all(shard.is_file() for shard in model_shards)


def protected_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    # Hard links do not protect a checkpoint from the trainer's directory
    # replacement logic: replacing the source directory can also remove the
    # links.  Copy the bytes so the protected best is independent of rolling
    # checkpoint cleanup.
    shutil.copytree(source, temporary)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)


def protect_once(
    root: Path, log_path: Path, history_log_path: Path | None, apply: bool
) -> int | None:
    history_path = root / "best_val_history.json"
    marker_path = root / "best_val.json"
    protected_root = root.parent / "protected_runs" / root.name

    # The current train.log is rewritten on every resume. Persistent history
    # and the append-only launcher log keep the global best validation score.
    scores = merge_validation_scores(
        [
            load_history(history_path),
            load_marker_history(marker_path),
            parse_validation_scores(history_log_path) if history_log_path else {},
            parse_validation_scores(log_path),
        ]
    )
    checkpoints = checkpoint_steps(root)
    chosen_step = best_step(scores, checkpoints, protected_root)

    if chosen_step is None:
        print(f"best checkpoint: no validation score found in {log_path}")
        return None

    complete_checkpoints = {
        step: path for step, path in checkpoints.items() if _is_complete_checkpoint(path)
    }
    if chosen_step not in complete_checkpoints:
        protected_candidate = protected_root / f"global_step_{chosen_step}"
        if _is_complete_checkpoint(protected_candidate):
            source = protected_candidate
        else:
            print(
                f"best checkpoint: step {chosen_step} "
                f"(score={scores[chosen_step]}) has no complete source checkpoint"
            )
            return None
    else:
        source = complete_checkpoints[chosen_step]

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

    history_payload = {
        "schema_version": 1,
        "scores": {str(step): scores[step] for step in sorted(scores)},
    }
    history_path.write_text(json.dumps(history_payload, indent=2) + "\n", encoding="utf-8")

    if not destination.exists():
        protected_copy(source, destination)
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
    parser.add_argument(
        "--history-log",
        type=Path,
        default=None,
        help="append-only log with validation history; defaults to launcher.log beside train.log",
    )
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
    history_log = args.history_log
    if history_log is None:
        candidate = args.train_log.parent / "launcher.log"
        if candidate.is_file():
            history_log = candidate
    if args.watch:
        while True:
            protect_once(args.checkpoint_root, args.train_log, history_log, args.apply)
            time.sleep(args.poll_interval)
    else:
        protect_once(args.checkpoint_root, args.train_log, history_log, args.apply)


if __name__ == "__main__":
    main()
