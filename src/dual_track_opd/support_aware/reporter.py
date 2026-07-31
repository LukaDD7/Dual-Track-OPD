"""Output writer for frozen-policy diagnostic runs.

Produces the artifact contract specified in runbook §4.6.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


@dataclass(frozen=True)
class DiagnosticRunMeta:
    run_id: str
    output_dir: Path
    config: Mapping[str, Any]
    git_commit: str
    git_dirty: bool
    num_prompts: int
    rollouts_per_prompt: int
    seed: int
    start_time: float
    end_time: float | None = None
    exit_status: str = "running"


def _default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


def write_rollouts_jsonl(
    output_dir: Path,
    rollouts: Sequence[Mapping[str, Any]],
) -> Path:
    """Write per-rollout records to ``rollouts.jsonl``."""
    path = output_dir / "rollouts.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rollout in rollouts:
            fh.write(json.dumps(rollout, ensure_ascii=False, default=_default))
            fh.write("\n")
    return path


def write_prompt_support_summary_jsonl(
    output_dir: Path,
    summaries: Sequence[Mapping[str, Any]],
) -> Path:
    """Write per-prompt support summaries to ``prompt_support_summary.jsonl``."""
    path = output_dir / "prompt_support_summary.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for summary in summaries:
            fh.write(json.dumps(summary, ensure_ascii=False, default=_default))
            fh.write("\n")
    return path


def write_selected_prompts_jsonl(
    output_dir: Path,
    prompts: Sequence[Mapping[str, Any]],
) -> Path:
    """Write the selected prompt metadata to ``selected_prompts.jsonl``."""
    path = output_dir / "selected_prompts.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for prompt in prompts:
            # Strip large binary fields
            record = dict(prompt)
            record.pop("image_bytes", None)
            record.pop("_image_bytes", None)
            fh.write(json.dumps(record, ensure_ascii=False, default=_default))
            fh.write("\n")
    return path


def write_summary_json(
    output_dir: Path,
    summary: Mapping[str, Any],
) -> Path:
    """Write aggregate summary to ``summary.json``."""
    path = output_dir / "summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def write_run_manifest(
    output_dir: Path,
    meta: DiagnosticRunMeta,
) -> Path:
    """Write run manifest to ``run_manifest.json``."""
    path = output_dir / "run_manifest.json"
    manifest = {
        "run_id": meta.run_id,
        "output_dir": str(meta.output_dir),
        "config": dict(meta.config),
        "git_commit": meta.git_commit,
        "git_dirty": meta.git_dirty,
        "num_prompts": meta.num_prompts,
        "rollouts_per_prompt": meta.rollouts_per_prompt,
        "seed": meta.seed,
        "start_time": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(meta.start_time)
        ),
        "end_time": (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(meta.end_time))
            if meta.end_time
            else None
        ),
        "exit_status": meta.exit_status,
    }
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def write_resolved_config_yaml(
    output_dir: Path,
    config: Mapping[str, Any],
) -> Path:
    """Write resolved config to ``resolved_config.yaml``."""
    path = output_dir / "resolved_config.yaml"
    path.write_text(
        yaml.safe_dump(dict(config), default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path
