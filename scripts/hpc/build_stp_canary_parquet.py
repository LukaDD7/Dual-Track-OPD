#!/usr/bin/env python3
"""Build the STP-OPD canary training parquet (4 rescue-positive prompts).

Combines the rescue cohort rows (question/answer/image), the verified
answer-free teacher prefixes (stp_prefixes.json), and the scaffold flag into
the verl training parquet layout used by the FC-OPD smoke, so
``STPTransitionDataset`` can feed the verl pipeline unchanged.

Usage:
  python scripts/hpc/build_stp_canary_parquet.py \
    --cohort-parquet .../rescue_screen_cohort_20260806.parquet \
    --prefix-manifest .../stp_prefixes.json \
    --output .../support_transition_canary/train.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dual_track_opd.fc_opd.prompt_contracts import get_geometry3k_prompt_builder  # noqa: E402
from dual_track_opd.support_aware.support_transition_dataset import load_prefixes  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-parquet", required=True)
    parser.add_argument("--prefix-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--scaffold-all", action="store_true", default=True)
    args = parser.parse_args()

    cohort = pd.read_parquet(args.cohort_parquet)
    prefixes = load_prefixes(args.prefix_manifest)
    builder = get_geometry3k_prompt_builder(args.prompt_version)
    wanted = sorted(prefixes)
    by_uid = {str(row["sample_uid"]): row for _, row in cohort.iterrows()}
    missing = sorted(set(wanted) - set(by_uid))
    if missing:
        print(f"FATAL: cohort missing prompt rows: {missing}", file=sys.stderr)
        return 1

    rows = []
    for prompt_uid in wanted:
        row = by_uid[prompt_uid]
        question = str(row["question"]).strip()
        answer = str(row["answer"])
        images_value = row["images"]
        if hasattr(images_value, "item"):
            images_value = images_value.item()
        image_dict = (
            images_value[0]
            if isinstance(images_value, (list, tuple))
            else images_value
        )
        rows.append({
            "data_source": "geometry3k",
            "prompt": builder(question),
            "images": [image_dict],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": "canary",
                "index": len(rows),
                "answer": answer,
                "question": question,
                "sample_uid": prompt_uid,
                "condition_inputs": (
                    row["condition_inputs"].item()
                    if hasattr(row["condition_inputs"], "item")
                    else row["condition_inputs"]
                ),
            },
            "question": question,
            "answer": answer,
            "answer_metadata": {"answer": answer},
            "sample_uid": prompt_uid,
            "stp_prefix_text": "",  # filled below after decode is impractical here;
            # STPTransitionDataset decodes stp_prefix_token_ids at load time.
            "stp_scaffolded": bool(args.scaffold_all),
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output_path)
    print(json.dumps({
        "prompts": wanted,
        "rows": len(rows),
        "scaffold_all": bool(args.scaffold_all),
        "output": str(output_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
