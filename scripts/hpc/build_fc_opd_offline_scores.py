#!/usr/bin/env python
"""Build an offline FC-OPD scoring dataset from a Vision-OPD-style file.

This CLI never loads the student model. It tokenises student responses with a
model-free byte tokenizer (or an optional Hugging Face tokenizer), queries a
running teacher service for top-k condition scores, and writes a self-describing
offline-score dataset under ``$DTOPD_OUTPUT_ROOT/fc_opd/offline_scores/``.

A self-contained smoke mode starts an in-process synthetic teacher so the whole
pipeline can be exercised on CPU without any weights or external service.

Examples
--------
Smoke run on the first 16 synthetic VStar samples::

    python scripts/hpc/build_fc_opd_offline_scores.py \
        --self-contained-smoke --limit 16 --source-dataset vstar

Score real student responses against a running teacher::

    python scripts/hpc/build_fc_opd_offline_scores.py \
        --dataset data/vstar_eval.json \
        --mode student --student-responses outputs/vstar_student.jsonl \
        --teacher-url http://127.0.0.1:18080 \
        --source-dataset vstar
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dual_track_opd.fc_opd.conditions import Condition  # noqa: E402
from dual_track_opd.fc_opd.offline_scoring import (  # noqa: E402
    DEFAULT_TEACHER_URL,
    ByteTokenizer,
    OfflineScoringConfig,
    default_output_dir,
    iter_offline_scores,
    load_student_responses,
    load_vision_opd_records,
    make_smoke_dataset,
    write_offline_scores,
)
from dual_track_opd.fc_opd.teacher_client import TeacherClient  # noqa: E402
from dual_track_opd.fc_opd.teacher_protocol import tokenizer_fingerprint  # noqa: E402


def _parse_conditions(value: str) -> tuple[Condition, ...]:
    conditions = tuple(Condition(item.strip()) for item in value.split(",") if item.strip())
    if not conditions:
        raise argparse.ArgumentTypeError("at least one condition is required")
    return conditions


def _build_tokenizer(spec: str):
    if spec == "byte":
        return ByteTokenizer()
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer  # local import keeps byte mode model-free

        return AutoTokenizer.from_pretrained(spec[len("hf:") :])
    raise argparse.ArgumentTypeError("tokenizer must be 'byte' or 'hf:<name-or-path>'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, help="Vision-OPD-style JSON or JSONL dataset")
    parser.add_argument(
        "--mode",
        choices=("student", "protocol_smoke"),
        default="protocol_smoke",
        help="how to obtain student responses",
    )
    parser.add_argument("--student-responses", type=Path, help="student response JSONL (student mode)")
    parser.add_argument("--teacher-url", default=DEFAULT_TEACHER_URL)
    parser.add_argument("--source-dataset", default="vstar")
    parser.add_argument("--conditions", type=_parse_conditions, default="full,blur,free,task")
    parser.add_argument("--blur-sigma", type=float, default=2.0)
    parser.add_argument("--data-root", default=os.environ.get("DTOPD_DATA_ROOT"))
    parser.add_argument("--degraded-dir", default=None)
    parser.add_argument("--require-paths", action="store_true")
    parser.add_argument("--tokenizer", default="byte", help="'byte' or 'hf:<name-or-path>'")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--parquet", action="store_true")
    parser.add_argument(
        "--self-contained-smoke",
        action="store_true",
        help="start an in-process synthetic teacher and ignore --teacher-url",
    )
    parser.add_argument("--smoke-top-k", type=int, default=32)
    parser.add_argument(
        "--verify-top-k",
        type=int,
        default=None,
        help="assert every condition tensor has this top-k width",
    )
    return parser


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    output_root = os.environ.get("DTOPD_OUTPUT_ROOT")
    if output_root is None:
        raise SystemExit("set DTOPD_OUTPUT_ROOT or pass --output-dir")
    return default_output_dir(output_root)


def _load_records(args: argparse.Namespace) -> list[dict]:
    if args.self_contained_smoke and args.dataset is None:
        limit = args.limit if args.limit is not None else 16
        return make_smoke_dataset(limit, dataset_name=args.source_dataset)
    if args.dataset is None:
        raise SystemExit("--dataset is required unless --self-contained-smoke is used")
    return load_vision_opd_records(args.dataset)


def _verify_records(records: Sequence, conditions: Sequence[Condition], top_k: int) -> None:
    for record in records:
        scores = record.payload["condition_scores"]
        missing = [c.value for c in conditions if c.value not in scores]
        if missing:
            raise SystemExit(f"{record.payload['sample_uid']} missing conditions: {missing}")
        for condition in conditions:
            token_ids = scores[condition.value]["token_ids"]
            seq_len = len(token_ids)
            if seq_len == 0 or any(len(row) != top_k for row in token_ids):
                raise SystemExit(
                    f"{record.payload['sample_uid']} condition {condition.value} is not [T,{top_k}]"
                )
            tail = scores[condition.value]["tail_log_prob"]
            entropy = scores[condition.value]["entropy"]
            if tail is not None and len(tail) != seq_len:
                raise SystemExit(f"{record.payload['sample_uid']} tail_log_prob is not [T]")
            if entropy is not None and len(entropy) != seq_len:
                raise SystemExit(f"{record.payload['sample_uid']} entropy is not [T]")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode == "student" and not args.self_contained_smoke and args.student_responses is None:
        raise SystemExit("--mode student requires --student-responses")

    tokenizer = _build_tokenizer(args.tokenizer)
    records = _load_records(args)
    student_responses = (
        load_student_responses(args.student_responses) if args.student_responses else None
    )

    config = OfflineScoringConfig(
        source_dataset=args.source_dataset,
        conditions=args.conditions,
        blur_sigma=args.blur_sigma,
        data_root=args.data_root,
        degraded_dir=args.degraded_dir,
        require_paths=args.require_paths,
    )

    server_context = nullcontext(None)
    teacher_url = args.teacher_url
    if args.self_contained_smoke:
        from dual_track_opd.fc_opd.teacher_scorer import SyntheticTeacherScorer
        from dual_track_opd.fc_opd.teacher_service import running_teacher_server

        scorer = SyntheticTeacherScorer(
            vocab_size=max(args.smoke_top_k + 1, 320),
            top_k=args.smoke_top_k,
            tokenizer_hash=tokenizer_fingerprint(tokenizer),
        )
        server_context = running_teacher_server(scorer)

    with server_context as server:
        if server is not None:
            host, port = server.server_address
            teacher_url = f"http://{host}:{port}"
        teacher_client = TeacherClient(
            teacher_url, expected_tokenizer_hash=tokenizer_fingerprint(tokenizer)
        )
        scored = list(
            iter_offline_scores(
                records,
                config=config,
                tokenizer=tokenizer,
                teacher_client=teacher_client,
                mode=args.mode,
                student_responses=student_responses,
                limit=args.limit,
            )
        )

    output_dir = _resolve_output_dir(args)
    output_name = args.output_name or f"{args.source_dataset}_offline_scores"
    result = write_offline_scores(
        scored, output_dir=output_dir, filename=output_name, write_parquet=args.parquet
    )

    verify_top_k = args.verify_top_k
    if verify_top_k is None and args.self_contained_smoke:
        verify_top_k = args.smoke_top_k
    if verify_top_k is not None:
        _verify_records(result.records, args.conditions, verify_top_k)

    print(
        f"wrote {len(result.records)} offline-score records to {result.jsonl_path}"
        + (f" and {result.parquet_path}" if result.parquet_path else "")
    )
    if verify_top_k is not None:
        print(
            f"verified {len(result.records)} samples: "
            f"{len(args.conditions)} conditions each, top-k width {verify_top_k}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
