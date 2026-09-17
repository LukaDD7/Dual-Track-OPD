#!/usr/bin/env python3
"""Build a GRPO-format val parquet from the official geometry3k test split.

Mirrors the columns of fc-opd-storage/outputs/fc_opd/geometry3k_full/train.parquet
so verl's default RLHFDataset (prompt_key=prompt, image_key=images) can consume
it, with train/val fully disjoint by construction.
"""

from __future__ import annotations

import argparse

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="official test parquet (images/problem/answer)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=-1, help="cap rows (smoke)")
    args = parser.parse_args()

    rows = pq.read_table(args.input).to_pylist()
    if args.n > 0:
        rows = rows[: args.n]
    out_rows = []
    for i, r in enumerate(rows):
        question = r["problem"]
        out_rows.append(
            {
                "data_source": "geometry3k_official_test",
                "prompt": [{"role": "user", "content": question}],
                "images": r["images"],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(r["answer"])},
                "extra_info": {"question": question, "choices": [], "answer": str(r["answer"])},
                "question": question,
                "condition_inputs": None,
                "choices": [],
                "answer": str(r["answer"]),
                "sample_uid": f"geo3k_test:{i}",
            }
        )
    pq.write_table(pa.Table.from_pylist(out_rows), args.output)
    print(f"wrote {args.output}: {len(out_rows)} rows")


if __name__ == "__main__":
    main()
