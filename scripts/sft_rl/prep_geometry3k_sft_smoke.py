#!/usr/bin/env python3
"""Build verl SFT-format smoke parquet from the official geometry3k HF dataset.

Input  : dataset/geometry3k/data/train-00000-of-00001.parquet
         (columns: images, problem, answer)
Output : fc-opd-storage/outputs/fc_opd/sft_rl/geo3k_sft_smoke{N}.parquet
         (columns: messages, images)  -- the default MultiTurnSFTDataset schema.

This is a pipeline smoke artifact only (text-only answers, no real solutions).
"""

from __future__ import annotations

import argparse

import pyarrow as pa
import pyarrow.parquet as pq


def build_messages(problem: str, answer: str) -> list[dict]:
    return [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": f"The answer is {answer}."},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=84)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    table = pq.read_table(args.input).to_pylist()
    rng = __import__("random").Random(args.seed)
    rows = rng.sample(table, min(args.n, len(table)))

    messages = [build_messages(r["problem"], r["answer"]) for r in rows]
    images = [r.get("images", []) for r in rows]

    out = pa.table(
        {
            "messages": pa.array(
                messages,
                type=pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())])),
            ),
            "images": pa.array(
                images,
                type=pa.list_(
                    pa.struct([("bytes", pa.binary()), ("path", pa.string())])
                ),
            ),
        }
    )
    pq.write_table(out, args.output)
    print(f"wrote {args.output}: {out.num_rows} rows")


if __name__ == "__main__":
    main()
