#!/usr/bin/env python
"""Smoke-test the Vision-OPD-6K adapter without teacher service or training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dual_track_opd.fc_opd.condition_prompt_dump import (  # noqa: E402
    PromptDumpConfig,
    dump_condition_prompts,
)
from dual_track_opd.fc_opd.dataset_adapters import (  # noqa: E402
    discover_vision_opd_train_parquet,
    load_normalized_records,
)
from dual_track_opd.fc_opd.dataset_signal_audit import (  # noqa: E402
    DatasetAuditConfig,
    run_dataset_signal_audit,
)
from dual_track_opd.fc_opd.offline_scoring import ByteTokenizer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--dataset-type", default="vision_opd_parquet")
    parser.add_argument("--source-dataset", default="vision-opd-6k")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("DTOPD_OUTPUT_ROOT", "artifacts"))
        / "fc_opd"
        / "vision_opd_adapter_smoke",
    )
    parser.add_argument("--normalized-limit", type=int, default=8)
    parser.add_argument("--prompt-limit", type=int, default=3)
    parser.add_argument("--audit-limit", type=int, default=8)
    parser.add_argument("--no-materialize-degraded-images", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = args.dataset or discover_vision_opd_train_parquet(args.project_root)
    if dataset is None:
        raise SystemExit("Vision-OPD train.parquet was not found; pass --dataset explicitly")

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized = load_normalized_records(
        dataset,
        args.dataset_type,
        source_dataset=args.source_dataset,
    )[: args.normalized_limit]
    normalized_path = output_dir / "vision_opd_normalized_records.jsonl"
    with normalized_path.open("w", encoding="utf-8") as handle:
        for record in normalized:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    prompt_result = dump_condition_prompts(
        PromptDumpConfig(
            dataset=dataset,
            dataset_type=args.dataset_type,
            source_dataset=args.source_dataset,
            limit=args.prompt_limit,
            output=output_dir / "vision_opd_condition_prompts.md",
            include_images_as_paths=True,
            materialize_degraded_images=not args.no_materialize_degraded_images,
        )
    )

    audit_result = run_dataset_signal_audit(
        DatasetAuditConfig(
            dataset=dataset,
            dataset_type=args.dataset_type,
            source_dataset=args.source_dataset,
            limit=args.audit_limit,
            output_dir=output_dir / "dataset_audit_dryrun",
            tokenizer="byte",
            dry_run=True,
            materialize_degraded_images=not args.no_materialize_degraded_images,
        ),
        tokenizer=ByteTokenizer(),
    )

    print(f"dataset: {dataset}")
    print(f"normalized records: {normalized_path}")
    print(f"prompt dump: {prompt_result.markdown_path}, {prompt_result.jsonl_path}")
    print(f"audit dry-run: {audit_result.summary_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
