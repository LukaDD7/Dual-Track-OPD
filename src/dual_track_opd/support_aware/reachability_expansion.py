"""Phase-5 causal-gold expansion: freeze the 256-pool prompt manifest.

CPU-only.  Freezes the preregistered prompt manifest for the reachability
proxy study's causal-rescue expansion (brief 2026-08-14, section 8):

    all 37 ``rare_success`` + 35 of 85 ``no_correct_observed`` +
    20 of 54 ``mixed_support`` controls, drawn deterministically from the
    completed 256-prompt K=8 pool, excluding the 12 already-rescued prompts.

The manifest (JSONL + spec) is written before any teacher/student generation,
so the prompt set, strata, and hashes are frozen up front.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .causal_dataset import read_jsonl, sha256_file


MANIFEST_VERSION = "reachability-expansion-manifest-v1"
DEFAULT_NO_CORRECT_COUNT = 40
DEFAULT_MIXED_CONTROL_COUNT = 20
STRATUM_ORDER = ("rare_success", "no_correct_observed", "mixed_support")


def _selection_key(uid: str) -> str:
    return hashlib.sha256(uid.encode()).hexdigest()


@dataclass(frozen=True)
class ManifestSpec:
    pool256_dir: str
    rescue_dir: str
    output_dir: str
    seed: int = 20260815
    no_correct_count: int = DEFAULT_NO_CORRECT_COUNT
    mixed_control_count: int = DEFAULT_MIXED_CONTROL_COUNT

    def validate(self) -> None:
        if not 0 <= self.no_correct_count <= 85:
            raise ValueError("no_correct_count must be within the 85-prompt stratum")
        if not 0 <= self.mixed_control_count <= 54:
            raise ValueError("mixed_control_count must be within the 54-prompt stratum")


def freeze_manifest(spec: ManifestSpec) -> dict[str, Any]:
    """Deterministically select the frozen expansion prompts."""

    spec.validate()
    pool256 = Path(spec.pool256_dir).expanduser().resolve()
    frontier_path = pool256 / "frontier_analysis" / "frontier_prompts.jsonl"
    if not frontier_path.is_file():
        raise FileNotFoundError(frontier_path)
    frontier = read_jsonl(frontier_path)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frontier:
        stratum = str(row.get("observed_support_stratum") or "")
        if stratum in STRATUM_ORDER:
            by_stratum[stratum].append(row)

    rescue_uids = set()
    rescue_path = Path(spec.rescue_dir).expanduser().resolve() / "rescue_comparisons.jsonl"
    if rescue_path.is_file():
        rescue_uids = {str(row["sample_uid"]) for row in read_jsonl(rescue_path)}

    quotas = {
        "rare_success": None,
        "no_correct_observed": spec.no_correct_count,
        "mixed_support": spec.mixed_control_count,
    }
    selected: list[dict[str, Any]] = []
    per_stratum: dict[str, dict[str, Any]] = {}
    for stratum in STRATUM_ORDER:
        rows = by_stratum.get(stratum, [])
        rows = [row for row in rows if str(row["sample_uid"]) not in rescue_uids]
        rows.sort(key=lambda row: _selection_key(str(row["sample_uid"])))
        quota = quotas[stratum]
        chosen = rows if quota is None else rows[:quota]
        per_stratum[stratum] = {
            "available": len(rows),
            "quota": quota if quota is not None else len(rows),
            "selected": len(chosen),
            "excluded_rescue_prompts": sum(
                1 for row in by_stratum.get(stratum, []) if str(row["sample_uid"]) in rescue_uids
            ),
        }
        for row in chosen:
            selected.append(
                {
                    "sample_uid": str(row["sample_uid"]),
                    "stratum": stratum,
                    "selection_index": len(selected),
                }
            )

    output = Path(spec.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_name = f"expansion_manifest_{spec.seed}.jsonl"
    manifest_path = output / manifest_name
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    spec_report: dict[str, Any] = {
        "schema_version": MANIFEST_VERSION,
        "seed": spec.seed,
        "pool256_dir": str(pool256),
        "rescue_dir": str(Path(spec.rescue_dir).expanduser().resolve()),
        "pool256_frontier_sha256": sha256_file(frontier_path),
        "stratum_counts": dict(Counter(row["stratum"] for row in selected)),
        "per_stratum": per_stratum,
        "total_prompts": len(selected),
        "excluded_rescue_prompt_count": len(rescue_uids),
        "manifest": manifest_name,
        "manifest_sha256": sha256_file(manifest_path),
        "generated_at": "2026-08-15",
    }
    (output / f"expansion_manifest_{spec.seed}.json").write_text(
        json.dumps(spec_report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return spec_report


def verify_manifest(
    manifest_path: str | Path,
    pool256_dir: str | Path,
) -> dict[str, Any]:
    """Confirm every manifest uid exists in the frozen pool with matching stratum."""

    manifest = read_jsonl(Path(manifest_path))
    frontier = read_jsonl(Path(pool256_dir) / "frontier_analysis" / "frontier_prompts.jsonl")
    by_uid = {str(row["sample_uid"]): str(row.get("observed_support_stratum") or "") for row in frontier}
    duplicate_keys = [
        key for key, count in Counter(row["sample_uid"] for row in manifest).items() if count > 1
    ]
    if duplicate_keys:
        raise ValueError(f"manifest contains duplicate sample_uids: {duplicate_keys}")
    missing = [row["sample_uid"] for row in manifest if row["sample_uid"] not in by_uid]
    mismatched = [
        row["sample_uid"]
        for row in manifest
        if row["sample_uid"] in by_uid and by_uid[row["sample_uid"]] != row["stratum"]
    ]
    if missing:
        raise ValueError(f"manifest uids missing from pool: {missing[:10]}")
    if mismatched:
        raise ValueError(f"manifest stratum mismatches: {mismatched[:10]}")
    return {
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "rows": len(manifest),
        "unique_prompts": len({row["sample_uid"] for row in manifest}),
        "missing": missing,
        "stratum_mismatch": mismatched,
        "verified": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--pool256-dir", required=True)
    freeze.add_argument("--rescue-dir", required=True)
    freeze.add_argument("--output-dir", required=True)
    freeze.add_argument("--seed", type=int, default=20260815)
    freeze.add_argument("--no-correct-count", type=int, default=DEFAULT_NO_CORRECT_COUNT)
    freeze.add_argument("--mixed-control-count", type=int, default=DEFAULT_MIXED_CONTROL_COUNT)
    verify = sub.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--pool256-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "freeze":
        report = freeze_manifest(
            ManifestSpec(
                pool256_dir=args.pool256_dir,
                rescue_dir=args.rescue_dir,
                output_dir=args.output_dir,
                seed=args.seed,
                no_correct_count=args.no_correct_count,
                mixed_control_count=args.mixed_control_count,
            )
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.phase == "verify":
        print(json.dumps(verify_manifest(args.manifest, args.pool256_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
