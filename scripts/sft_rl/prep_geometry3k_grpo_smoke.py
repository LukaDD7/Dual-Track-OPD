#!/usr/bin/env python3
"""Take the first N rows of the GRPO geometry3k train parquet as a smoke set."""

from __future__ import annotations

import argparse

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=64)
    args = parser.parse_args()

    table = pq.read_table(args.input)
    out = table.slice(0, args.n)
    pq.write_table(out, args.output)
    print(f"wrote {args.output}: {out.num_rows} rows (from {table.num_rows})")


if __name__ == "__main__":
    main()
