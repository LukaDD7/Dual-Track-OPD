"""Generate student wrong-control rollouts for pre-scored expansion prompts.

Phase-5B (brief 2026-08-14, section 8): the 19 expansion prompts with verified
teacher traces and pre-scored proxy tokens but no wrong-student control in the
256-pool get fresh unaided student rollouts so the adaptive rescue rule can be
applied.  Student-only generation; frozen policy; no training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .causal_dataset import read_jsonl
from .diagnostic import build_prompt, extract_image
from .prefix_intervention import InterventionConfig, _load_student, generate_continuation
from .verifier import verify_answer


SCHEMA_VERSION = "wrong-control-pool-v1"


def _selection_key(uid: str) -> str:
    return hashlib.sha256(uid.encode()).hexdigest()


@dataclass(frozen=True)
class WrongControlConfig:
    manifest_path: str
    proposal_dir: str
    rescue_dirs: tuple[str, ...]
    cohort_dir: str
    cohort_parquet_path: str | None
    output_dir: str
    student_model_path: str
    response_format: str = "legacy_answer"
    k: int = 16
    max_continuation_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    seed: int = 20260815
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    shard_index: int = 0
    num_shards: int = 1
    max_prompts: int | None = None

    def validate(self) -> None:
        if self.k <= 0 or self.max_continuation_tokens <= 0:
            raise ValueError("generation counts and token limits must be positive")
        if not 0 <= self.shard_index < self.num_shards or self.num_shards <= 0:
            raise ValueError("invalid shard assignment")
        if self.max_prompts is not None and self.max_prompts <= 0:
            raise ValueError("max_prompts must be positive")


def load_config(path: str | Path, args: argparse.Namespace) -> WrongControlConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = WrongControlConfig(
        manifest_path=str(args.manifest or raw["data"]["manifest"]),
        proposal_dir=str(raw["data"]["proposal_dir"]),
        rescue_dirs=tuple(str(value) for value in raw["data"].get("rescue_dirs", ())),
        cohort_dir=str(raw["data"]["cohort_dir"]),
        cohort_parquet_path=raw["data"].get("cohort_parquet_path"),
        output_dir=str(args.output_dir or raw["output"]["dir"]),
        student_model_path=str(raw["model"]["student"]),
        response_format=str(raw["generation"].get("response_format", "legacy_answer")),
        k=int(args.k or raw["generation"].get("k", 16)),
        max_continuation_tokens=int(
            args.max_continuation_tokens
            or raw["generation"].get("max_continuation_tokens", 2048)
        ),
        temperature=float(raw["generation"].get("temperature", 0.7)),
        top_p=float(raw["generation"].get("top_p", 0.95)),
        seed=int(raw["generation"].get("seed", 20260815)),
        device=str(raw["hardware"].get("device", "cuda:0")),
        dtype=str(raw["hardware"].get("dtype", "bfloat16")),
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
        max_prompts=args.max_prompts,
    )
    def expand(value: Any) -> Any:
        if isinstance(value, str):
            return os.path.expandvars(value)
        if isinstance(value, tuple):
            return tuple(expand(item) for item in value)
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    config = WrongControlConfig(**{key: expand(value) for key, value in asdict(config).items()})
    config.validate()
    return config


def select_targets(config: WrongControlConfig) -> tuple[list[str], dict[str, Any]]:
    manifest = read_jsonl(Path(config.manifest_path))
    manifest_uids = [str(row["sample_uid"]) for row in manifest]
    retained_uids = {
        str(row["sample_uid"])
        for row in read_jsonl(Path(config.proposal_dir) / "retained_proposals.jsonl")
    }
    rescued_uids: set[str] = set()
    for rescue_dir in config.rescue_dirs:
        rescued_uids.update(
            str(row["sample_uid"])
            for row in read_jsonl(Path(rescue_dir) / "rescue_comparisons.jsonl")
        )
    targets = [uid for uid in manifest_uids if uid in retained_uids and uid not in rescued_uids]
    info = {
        "manifest_prompts": len(manifest_uids),
        "retained_traces": len(retained_uids),
        "already_rescued": len(rescued_uids),
        "targets": len(targets),
    }
    return targets, info


def _cohort_frame(config: WrongControlConfig):
    import pandas as pd

    cohort_path = (
        Path(config.cohort_parquet_path).expanduser().resolve()
        if config.cohort_parquet_path
        else Path(config.cohort_dir) / "cohort.parquet"
    )
    frame = pd.read_parquet(cohort_path).copy()
    frame.index = frame["sample_uid"].astype(str)
    if frame.index.duplicated().any():
        raise ValueError("cohort contains duplicate sample_uid")
    return frame


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _aggregate(output_dir: Path, expected_uids: Sequence[str]) -> dict[str, Any]:
    result_dir = output_dir / "prompt_results"
    rollouts: list[dict[str, Any]] = []
    completed: list[str] = []
    for path in sorted(result_dir.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        uid = str(value["sample_uid"])
        if uid not in expected_uids:
            continue
        completed.append(uid)
        rollouts.extend(value["rollouts"])
    wrong_by_uid = {
        uid: sum(1 for row in rollouts if row["sample_uid"] == uid and row["correct"] is False)
        for uid in completed
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "expected_prompt_count": len(expected_uids),
        "completed_prompt_count": len(completed),
        "complete": sorted(completed) == sorted(expected_uids),
        "rollout_count": len(rollouts),
        "wrong_rollout_count": sum(1 for row in rollouts if row["correct"] is False),
        "wrong_by_prompt": wrong_by_uid,
        "prompts_without_wrong": sorted(uid for uid in completed if not wrong_by_uid.get(uid)),
        "completed_uids": completed,
        "missing_uids": [uid for uid in expected_uids if uid not in completed],
    }
    with (output_dir / "rollouts.jsonl").open("w", encoding="utf-8") as handle:
        for row in rollouts:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _write_json(output_dir / "summary.json", summary)
    return summary


def run(config: WrongControlConfig) -> dict[str, Any]:
    config.validate()
    output = Path(config.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_dir = output / "prompt_results"
    result_dir.mkdir(exist_ok=True)
    targets, info = select_targets(config)
    start = len(targets) * config.shard_index // config.num_shards
    end = len(targets) * (config.shard_index + 1) // config.num_shards
    slice_uids = targets[start:end]
    if config.max_prompts is not None:
        slice_uids = slice_uids[: config.max_prompts]
    expected_uids = list(slice_uids)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "config": json.loads(json.dumps(asdict(config))),
        "selection": info,
        "expected_uids": expected_uids,
        "git_commit": _git_state()[0],
        "git_dirty": _git_state()[1],
        "started_at_unix": time.time(),
    }
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config") != manifest["config"] or existing.get("selection") != manifest["selection"]:
            raise ValueError("existing wrong-control manifest differs from requested config")
    _write_json(manifest_path, manifest)

    completed = {
        str(json.loads(path.read_text(encoding="utf-8"))["sample_uid"])
        for path in result_dir.glob("*.json")
    }
    pending = [uid for uid in expected_uids if uid not in completed]
    if pending:
        model, processor, tokenizer_hash = _load_student(
            InterventionConfig(
                proposal_dir=str(output),
                k32_run_dir=str(output),
                cohort_dir=str(output),
                output_dir=str(output),
                student_model_path=config.student_model_path,
                device=config.device,
                dtype=config.dtype,
            )
        )
        frame = _cohort_frame(config)
        for prompt_index, uid in enumerate(pending, start=1):
            cohort_row = frame.loc[uid].to_dict()
            question = str(cohort_row.get("question") or "").strip()
            gold_answer = cohort_row.get("answer")
            if not question:
                raise ValueError(f"{uid}: missing question")
            image = extract_image(cohort_row).convert("RGB")
            prompt_text = build_prompt(question, config.response_format)
            rollouts: list[dict[str, Any]] = []
            for rollout_id in range(1, config.k + 1):
                generation_seed = (
                    config.seed + int(_selection_key(uid)[:8], 16) + rollout_id
                )
                generated = generate_continuation(
                    model,
                    processor,
                    image=image,
                    prompt_text=prompt_text,
                    prefix_ids=(),
                    max_continuation_tokens=config.max_continuation_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    seed=generation_seed,
                    device=config.device,
                )
                generation = generated["generation"]
                verdict = verify_answer(generation.response_text_display, gold_answer)
                rollouts.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "sample_uid": uid,
                        "rollout_id": rollout_id,
                        "generation_seed": generation_seed,
                        "is_greedy": False,
                        "correct": verdict.get("correct"),
                        "exact_token_alignment": True,
                        "response_token_ids": list(generation.response_token_ids_raw),
                        "response_token_count": len(generation.response_token_ids_raw),
                        "response_token_hash": generation.response_token_hash,
                        "prompt_token_hash": generation.prompt_token_hash,
                        "finish_reason": generation.finish_reason,
                        "terminal_token_id": generation.terminal_token_id,
                        "gold_answer": str(gold_answer),
                    }
                )
            prompt_result = {
                "schema_version": SCHEMA_VERSION,
                "sample_uid": uid,
                "student_tokenizer_hash": tokenizer_hash,
                "rollouts": rollouts,
            }
            target = result_dir / f"{hashlib.sha256(uid.encode()).hexdigest()[:12]}.json"
            temporary = target.with_suffix(".json.tmp")
            _write_json(temporary, prompt_result)
            temporary.replace(target)
            image.close()
            print(
                f"[{prompt_index}/{len(pending)}] {uid}: "
                f"wrong={sum(1 for row in rollouts if row['correct'] is False)}/{config.k}",
                flush=True,
            )
    summary = _aggregate(output, expected_uids)
    manifest["status"] = "completed" if summary["complete"] else "partial"
    manifest["summary"] = summary
    manifest["finished_at_unix"] = time.time()
    _write_json(manifest_path, manifest)

    # Subset manifest for the follow-up adaptive rescue: targets with wrong rows.
    subset_rows = [
        {"sample_uid": uid, "stratum": "wrong_control_expansion"}
        for uid in expected_uids
        if summary["wrong_by_prompt"].get(uid)
    ]
    with (output / "intervention_subset_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in subset_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return summary


def _git_state() -> tuple[str, bool]:
    try:
        repo = str(Path(__file__).resolve().parents[3])
        commit = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", repo, "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "", True


def merge_shards(shard_dirs: Sequence[str | Path], output_dir: str | Path) -> dict[str, Any]:
    """Strictly merge completed wrong-control shards into one consumable pool."""

    shards = [Path(path).expanduser().resolve() for path in shard_dirs]
    manifests = [
        json.loads((path / "run_manifest.json").read_text(encoding="utf-8")) for path in shards
    ]
    if any(manifest.get("status") != "completed" for manifest in manifests):
        raise ValueError("all wrong-control shards must be completed")
    commits = {str(manifest.get("git_commit") or "") for manifest in manifests}
    if len(commits) != 1 or "" in commits:
        raise ValueError(f"wrong-control shard commits differ: {sorted(commits)}")
    allow_dirty = os.environ.get("DTOPD_ALLOW_DIRTY_MERGE") == "1"
    if {bool(manifest.get("git_dirty")) for manifest in manifests} != {False} and not allow_dirty:
        raise ValueError("wrong-control shards must be produced from clean worktrees")
    expected_uids: list[str] = []
    seen: set[str] = set()
    for manifest in sorted(manifests, key=lambda item: int(item["config"]["shard_index"])):
        shard_uids = list(manifest["expected_uids"])
        overlap = seen.intersection(shard_uids)
        if overlap:
            raise ValueError(f"wrong-control shards overlap: {sorted(overlap)[:5]}")
        seen.update(shard_uids)
        expected_uids.extend(shard_uids)
    if seen != set(expected_uids):
        raise ValueError("wrong-control shards do not cover their expected UID universe")

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rollouts: list[dict[str, Any]] = []
    wrong_by_prompt: dict[str, int] = {}
    for shard in shards:
        for row in read_jsonl(shard / "rollouts.jsonl"):
            rollouts.append(row)
            if row["correct"] is False:
                wrong_by_prompt[str(row["sample_uid"])] = (
                    wrong_by_prompt.get(str(row["sample_uid"]), 0) + 1
                )
    with (output / "rollouts.jsonl").open("w", encoding="utf-8") as handle:
        for row in rollouts:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    subset = [
        {"sample_uid": uid, "stratum": "wrong_control_expansion"}
        for uid in expected_uids
        if wrong_by_prompt.get(uid)
    ]
    with (output / "intervention_subset_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in subset:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "expected_prompt_count": len(expected_uids),
        "completed_prompt_count": len(expected_uids),
        "complete": True,
        "rollout_count": len(rollouts),
        "wrong_rollout_count": sum(1 for row in rollouts if row["correct"] is False),
        "wrong_by_prompt": wrong_by_prompt,
        "intervention_subset_count": len(subset),
        "completed_uids": expected_uids,
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "run_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "git_commit": next(iter(commits)),
        "git_dirty": False,
        "merged_from": [str(path) for path in shards],
        "config": manifests[0]["config"],
        "expected_uids": expected_uids,
        "summary": summary,
    })
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--manifest")
    run_parser.add_argument("--output-dir")
    run_parser.add_argument("--k", type=int)
    run_parser.add_argument("--max-continuation-tokens", type=int)
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--num-shards", type=int, default=1)
    run_parser.add_argument("--max-prompts", type=int)
    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("--shard-dirs", nargs="+", required=True)
    merge_parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = (
            run(load_config(args.config, args))
            if args.command == "run"
            else merge_shards(args.shard_dirs, args.output_dir)
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
