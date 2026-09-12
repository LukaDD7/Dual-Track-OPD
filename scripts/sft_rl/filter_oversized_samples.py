#!/usr/bin/env python3
"""Drop oversized rows from warmup shards (output of scan_oversized_samples.py).

Usage: filter_oversized_samples.py --warmup-dir DIR --bad-json /tmp/oversized.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-dir", required=True)
    parser.add_argument("--bad-json", required=True)
    parser.add_argument("--mismatch-only", action="store_true",
                        help="drop only rows where image tokens != features "
                             "(16 docvqa rows); keep long-CoT rows that were "
                             "truncated without touching image tokens")
    args = parser.parse_args()

    bad = json.load(open(args.bad_json))
    if args.mismatch_only:
        bad = [b for b in bad if b["n_tok"] != b["n_feat"]]
    by_shard: dict[str, set[int]] = {}
    src_counts: Counter = Counter()
    for b in bad:
        by_shard.setdefault(b["shard"], set()).add(b["idx"])
        src_counts[b.get("source", "?")] += 1

    total_dropped = 0
    for shard_name, idxs in sorted(by_shard.items()):
        path = Path(args.warmup_dir) / shard_name
        if not path.exists():
            print(f"WARN missing {path}")
            continue
        t = pq.read_table(path)
        rows = t.to_pylist()
        keep = [r for i, r in enumerate(rows) if i not in idxs]
        dropped = len(rows) - len(keep)
        if dropped:
            pq.write_table(pa.Table.from_pylist(keep, schema=t.schema), path)
        total_dropped += dropped
        print(f"[filtered] {shard_name}: {len(rows)} -> {len(keep)} (-{dropped})")

    print(f"total dropped: {total_dropped}")
    print(f"by source: {dict(src_counts)}")


if __name__ == "__main__":
    main()
