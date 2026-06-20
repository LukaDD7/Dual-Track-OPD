"""Conservative deterministic scoring for baseline raw response JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .answer_extractors import (
    extract_choice,
    normalize_numeric_answer,
    normalize_text_answer,
)

PARSER_VERSION = "conservative_v1"

PREDICTION_KEYS = (
    "prediction",
    "pred",
    "response",
    "output",
    "text",
    "model_answer",
    "generated_text",
    "completion",
    "raw_response",
)
GROUND_TRUTH_KEYS = (
    "ground_truth",
    "gt",
    "gt_answer",
    "answer",
    "answers",
    "label",
    "target",
    "gold",
    "correct_answer",
)
ID_KEYS = ("row_id", "sample_id", "id", "question_id", "uid", "index")

SCORE_COLUMNS = [
    "dataset",
    "file",
    "scoring_type",
    "n",
    "scored_n",
    "correct",
    "accuracy",
    "errors",
    "length_rows",
    "unparsed_rows",
    "needs_judge",
    "parser_version",
    "notes",
]
AUDIT_COLUMNS = [
    "dataset",
    "file",
    "row_id",
    "prediction",
    "parsed_prediction",
    "ground_truth",
    "parsed_ground_truth",
    "correct",
    "finish_reason",
    "error",
    "audit_reason",
]


@dataclass(frozen=True)
class ManifestEntry:
    dataset: str
    file: str
    scoring_type: str
    notes: str = ""


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            try:
                entries.append(
                    ManifestEntry(
                        dataset=str(item["dataset"]),
                        file=str(item["file"]),
                        scoring_type=str(item["scoring_type"]),
                        notes=str(item.get("notes", "")),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"manifest line {line_number} missing required field: {exc}") from exc
    return entries


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _first_present(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return None


def _extract_openai_text(item: dict[str, Any]) -> str | None:
    choices = item.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if isinstance(message, dict) and message.get("content"):
        return _stringify(message["content"])
    if first.get("text"):
        return _stringify(first["text"])
    return None


def get_prediction(item: dict[str, Any]) -> str:
    value = _first_present(item, PREDICTION_KEYS)
    if isinstance(value, dict):
        nested = _first_present(value, PREDICTION_KEYS)
        if nested is not None:
            return _stringify(nested)
        if value.get("content"):
            return _stringify(value["content"])
    if value is not None:
        return _stringify(value)
    return _extract_openai_text(item) or ""


def get_ground_truths(item: dict[str, Any]) -> list[str]:
    value = _first_present(item, GROUND_TRUTH_KEYS)
    if value is None:
        return []
    if isinstance(value, list):
        return [_stringify(v) for v in value if _stringify(v)]
    if isinstance(value, dict):
        nested = _first_present(value, GROUND_TRUTH_KEYS)
        if nested is not None:
            return [_stringify(nested)]
    text = _stringify(value)
    return [text] if text else []


def get_finish_reason(item: dict[str, Any]) -> str:
    value = item.get("finish_reason")
    if value is not None:
        return _stringify(value)
    choices = item.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        nested = choices[0].get("finish_reason")
        if nested is not None:
            return _stringify(nested)
    return ""


def get_row_id(item: dict[str, Any], row_number: int) -> str:
    value = _first_present(item, ID_KEYS)
    return _stringify(value) if value is not None else str(row_number)


def _score_mcq(prediction: str, truths: list[str]) -> tuple[str, str, bool | None]:
    parsed_prediction = extract_choice(prediction) or ""
    parsed_truths = [choice for truth in truths if (choice := extract_choice(truth))]
    parsed_truth = "|".join(sorted(set(parsed_truths)))
    if not parsed_prediction or not parsed_truths:
        return parsed_prediction, parsed_truth, None
    return parsed_prediction, parsed_truth, parsed_prediction in parsed_truths


def _score_normalized_exact(prediction: str, truths: list[str]) -> tuple[str, str, bool | None]:
    parsed_prediction = normalize_text_answer(prediction)
    parsed_truths = [normalize_text_answer(truth) for truth in truths]
    parsed_truths = [truth for truth in parsed_truths if truth]
    parsed_truth = "|".join(sorted(set(parsed_truths)))
    if not parsed_prediction or not parsed_truths:
        return parsed_prediction, parsed_truth, None
    return parsed_prediction, parsed_truth, parsed_prediction in parsed_truths


def _score_numeric_exact(prediction: str, truths: list[str]) -> tuple[str, str, bool | None]:
    parsed_prediction = normalize_numeric_answer(prediction) or ""
    parsed_truths = [value for truth in truths if (value := normalize_numeric_answer(truth))]
    parsed_truth = "|".join(sorted(set(parsed_truths)))
    if not parsed_prediction or not parsed_truths:
        return parsed_prediction, parsed_truth, None
    return parsed_prediction, parsed_truth, parsed_prediction in parsed_truths


def _append_audit(
    audit_rows: list[dict[str, object]],
    audit_counts: dict[str, int],
    *,
    entry: ManifestEntry,
    row_id: str,
    prediction: str,
    parsed_prediction: str,
    ground_truth: str,
    parsed_ground_truth: str,
    correct: bool | None,
    finish_reason: str,
    error: str,
    audit_reason: str,
    max_audit_rows_per_file: int,
) -> None:
    count = audit_counts.get(entry.file, 0)
    if count >= max_audit_rows_per_file:
        return
    audit_counts[entry.file] = count + 1
    audit_rows.append(
        {
            "dataset": entry.dataset,
            "file": entry.file,
            "row_id": row_id,
            "prediction": prediction,
            "parsed_prediction": parsed_prediction,
            "ground_truth": ground_truth,
            "parsed_ground_truth": parsed_ground_truth,
            "correct": "" if correct is None else str(correct).lower(),
            "finish_reason": finish_reason,
            "error": error,
            "audit_reason": audit_reason,
        }
    )


def score_raw_responses(
    raw_dir: str | Path,
    manifest: str | Path,
    out_scores: str | Path,
    out_audit: str | Path,
    max_audit_rows_per_file: int = 200,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    entries = load_manifest(manifest)
    raw_dir = Path(raw_dir)
    score_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    audit_counts: dict[str, int] = {}

    for entry in entries:
        raw_path = raw_dir / entry.file
        if not raw_path.exists():
            raise FileNotFoundError(f"missing raw file for {entry.dataset}: {raw_path}")

        n = scored_n = correct_n = errors = length_rows = unparsed_rows = 0
        needs_judge = entry.scoring_type == "needs_judge"

        with raw_path.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors += 1
                    _append_audit(
                        audit_rows,
                        audit_counts,
                        entry=entry,
                        row_id=str(row_number),
                        prediction="",
                        parsed_prediction="",
                        ground_truth="",
                        parsed_ground_truth="",
                        correct=None,
                        finish_reason="",
                        error=str(exc),
                        audit_reason="bad_json",
                        max_audit_rows_per_file=max_audit_rows_per_file,
                    )
                    continue

                row_id = get_row_id(item, row_number)
                prediction = get_prediction(item)
                truths = get_ground_truths(item)
                ground_truth = "|".join(truths)
                finish_reason = get_finish_reason(item)
                error = _stringify(item.get("error"))
                if error:
                    errors += 1
                is_length = finish_reason.lower() == "length"
                if is_length:
                    length_rows += 1

                parsed_prediction = ""
                parsed_ground_truth = ""
                is_correct: bool | None = None
                audit_reason = ""

                if needs_judge:
                    audit_reason = "needs_judge"
                elif error:
                    audit_reason = "error"
                elif is_length:
                    audit_reason = "length"
                else:
                    if entry.scoring_type == "mcq":
                        parsed_prediction, parsed_ground_truth, is_correct = _score_mcq(
                            prediction, truths
                        )
                    elif entry.scoring_type == "normalized_exact":
                        parsed_prediction, parsed_ground_truth, is_correct = _score_normalized_exact(
                            prediction, truths
                        )
                    elif entry.scoring_type == "numeric_exact":
                        parsed_prediction, parsed_ground_truth, is_correct = _score_numeric_exact(
                            prediction, truths
                        )
                    else:
                        raise ValueError(
                            f"unsupported scoring_type for {entry.dataset}: {entry.scoring_type}"
                        )

                    if is_correct is None:
                        unparsed_rows += 1
                        audit_reason = "unparsed"
                    else:
                        scored_n += 1
                        correct_n += int(is_correct)
                        if not is_correct:
                            audit_reason = "incorrect"

                if audit_reason:
                    _append_audit(
                        audit_rows,
                        audit_counts,
                        entry=entry,
                        row_id=row_id,
                        prediction=prediction,
                        parsed_prediction=parsed_prediction,
                        ground_truth=ground_truth,
                        parsed_ground_truth=parsed_ground_truth,
                        correct=is_correct,
                        finish_reason=finish_reason,
                        error=error,
                        audit_reason=audit_reason,
                        max_audit_rows_per_file=max_audit_rows_per_file,
                    )

        accuracy = "" if scored_n == 0 else f"{correct_n / scored_n:.6f}"
        score_rows.append(
            {
                "dataset": entry.dataset,
                "file": entry.file,
                "scoring_type": entry.scoring_type,
                "n": n,
                "scored_n": scored_n,
                "correct": correct_n,
                "accuracy": accuracy,
                "errors": errors,
                "length_rows": length_rows,
                "unparsed_rows": unparsed_rows,
                "needs_judge": str(needs_judge).lower(),
                "parser_version": PARSER_VERSION,
                "notes": entry.notes,
            }
        )

    write_csv(out_scores, SCORE_COLUMNS, score_rows)
    write_csv(out_audit, AUDIT_COLUMNS, audit_rows)
    return score_rows, audit_rows


def write_csv(path: str | Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help="Directory containing raw JSONL files.")
    parser.add_argument("--manifest", required=True, help="Preferred raw-file manifest JSONL.")
    parser.add_argument("--out-scores", required=True, help="Output dataset-level score CSV.")
    parser.add_argument("--out-audit", required=True, help="Output bounded failure audit CSV.")
    parser.add_argument(
        "--max-audit-rows-per-file",
        type=int,
        default=200,
        help="Maximum audit rows to retain per raw file.",
    )
    args = parser.parse_args()

    score_raw_responses(
        raw_dir=args.raw_dir,
        manifest=args.manifest,
        out_scores=args.out_scores,
        out_audit=args.out_audit,
        max_audit_rows_per_file=args.max_audit_rows_per_file,
    )


if __name__ == "__main__":
    main()
