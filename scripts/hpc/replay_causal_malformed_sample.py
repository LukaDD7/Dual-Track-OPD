#!/usr/bin/env python3
"""Replay high-``n_malformed`` relay candidates with saved seeds (review P0-2).

The historical 20260808 causal-probe records predate the reason-split schema,
so ``n_malformed`` conflates relay answer leakage, missing answer marker,
truncation, and verifier-unparseable outputs.  This script re-runs a small
stratified sample of the worst candidates with the deterministic saved seeds
and reports the reason distribution, which decides whether historical units
need regeneration.

Usage (on a GPU instance with the pinned env):
  python -u scripts/hpc/replay_causal_malformed_sample.py \
    --config configs/diagnostics/causal_state_probe.yaml \
    --records /inspire/.../causal_state_probe_20260808_merged/causal_state_records.jsonl \
    --output-dir /inspire/.../causal_malformed_replay_20260813 \
    --max-samples 20 --min-malformed 4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dual_track_opd.support_aware.causal_dataset import select_probe_inputs  # noqa: E402
from dual_track_opd.support_aware.causal_runtime import (  # noqa: E402
    load_runtime_models,
    teacher_relay_estimate,
)
from dual_track_opd.support_aware.causal_state_probe import (  # noqa: E402
    _seed,
    _work_id,
    load_config,
)
from dual_track_opd.support_aware.diagnostic import build_prompt, extract_image  # noqa: E402


REASON_FIELDS = (
    "n_correct",
    "n_wrong_format_valid",
    "n_no_answer_marker",
    "n_truncated",
    "n_relay_answer_leakage",
    "n_generation_error",
)


def collect_samples(records_path: Path, min_malformed: int) -> list[dict]:
    """Candidate x relay-length samples with n_malformed >= min_malformed."""

    samples = []
    for line in records_path.open(encoding="utf-8"):
        record = json.loads(line)
        trajectory_id = str(record.get("trajectory_id") or "")
        for candidate in record.get("candidate_windows") or ():
            for length, estimate in (candidate.get("relay_continuations") or {}).items():
                malformed = int(estimate.get("n_malformed") or 0)
                if malformed >= min_malformed:
                    samples.append({
                        "trajectory_id": trajectory_id,
                        "candidate_id": candidate.get("candidate_id"),
                        "anchor": int(candidate.get("anchor") or 0),
                        "relay_length": str(length),
                        "original_n": int(estimate.get("n") or 0),
                        "original_n_malformed": malformed,
                    })
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--min-malformed", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    config = load_config(args.config)
    records_path = Path(args.records).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(records_path, args.min_malformed)
    by_length: dict[str, list[dict]] = {length: [] for length in ("32", "64", "128")}
    for sample in samples:
        by_length[sample["relay_length"]].append(sample)
    # deterministic stratified sample across relay lengths
    selected: list[dict] = []
    for length in ("32", "64", "128"):
        pool = by_length[length]
        pool.sort(key=lambda item: (-item["original_n_malformed"], item["trajectory_id"]))
        budget = max(1, args.max_samples // 3)
        selected.extend(pool[:budget])
    selected = selected[: args.max_samples]
    if not selected:
        print(json.dumps({"error": "no samples meet min-malformed threshold"}, indent=2))
        return 1

    inputs, provenance = select_probe_inputs(
        k32_run_dir=config.k32_run_dir,
        cohort_dir=config.cohort_dir,
        proposal_dir=config.proposal_dir,
        trajectory_outcomes=config.trajectory_outcomes,
        max_trajectories_per_outcome=config.max_trajectories_per_outcome,
        num_shards=1,
    )
    items_by_work_id = {_work_id(item): item for item in inputs}

    models = load_runtime_models(
        student_model_path=config.student_model_path,
        student_device=config.student_device,
        dtype=config.dtype,
        teacher_model_path=config.teacher_model_path,
        teacher_device=config.teacher_device,
    )

    rows = []
    reason_counts: Counter[str] = Counter()
    total_malformed = 0
    for sample in selected:
        item = items_by_work_id.get(sample["trajectory_id"])
        if item is None:
            rows.append({**sample, "error": "work_id not found in immutable inputs"})
            continue
        rollout = item.student_rollout
        response_ids = tuple(int(value) for value in rollout["response_token_ids"])
        prefix_ids = response_ids[: sample["anchor"] + 1]
        question = str(item.cohort.get("question") or "").strip()
        gold_answer = item.cohort.get("answer")
        prompt_text = build_prompt(question, config.response_format)
        image = extract_image(item.cohort).convert("RGB")
        seed = _seed(config, sample["trajectory_id"], sample["anchor"], f"relay-{sample['relay_length']}")
        estimate = teacher_relay_estimate(
            models=models,
            image=image,
            prompt_text=prompt_text,
            student_prefix_ids=prefix_ids,
            gold_answer=gold_answer,
            relay_length=int(sample["relay_length"]),
            k=config.relay_k,
            max_student_tokens=config.max_continuation_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            seed=seed,
            student_device=config.student_device,
            teacher_device=config.teacher_device,
        )
        reason_row = {field: int(getattr(estimate, field) or 0) for field in REASON_FIELDS}
        for field, count in reason_row.items():
            reason_counts[field] += count
        total_malformed += sum(
            reason_row[field]
            for field in ("n_relay_answer_leakage", "n_no_answer_marker", "n_truncated", "n_generation_error")
        )
        rows.append({
            **sample,
            "seed": seed,
            "replayed_n": estimate.n,
            "replayed_n_correct": estimate.n_correct,
            "reasons": reason_row,
        })

    summary = {
        "config": {
            "repo": "Dual-Track-OPD",
            "env": "va-opd-qwen35-cu128",
            "tokenizer_hash": models.tokenizer_hash,
        },
        "sampled": len(rows),
        "by_relay_length": {
            length: sum(1 for row in rows if row["relay_length"] == length)
            for length in ("32", "64", "128")
        },
        "reason_counts": dict(reason_counts),
        "total_malformed_replayed": total_malformed,
        "leakage_share_of_malformed": round(
            reason_counts["n_relay_answer_leakage"] / total_malformed, 4
        ) if total_malformed else None,
        "rows": rows,
    }
    (output_dir / "replay_result.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "sampled": len(rows),
        "reason_counts": dict(reason_counts),
        "total_malformed_replayed": total_malformed,
        "leakage_share_of_malformed": summary["leakage_share_of_malformed"],
        "result_path": str(output_dir / "replay_result.json"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
