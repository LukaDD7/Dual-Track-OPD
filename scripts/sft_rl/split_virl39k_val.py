#!/usr/bin/env python3
"""Hold out a val shard from the converted ViRL39K RL train shards.

Why: convert_virl39k.py writes train-only (26 shards, 38,348 rows), but the
verl launch scripts FATAL without a val file (`[ -f "${VAL_FILE}" ]`). This
splits ~500 rows (default) out of the train set — stratified by source so the
val still covers the pool's diversity — writes them as
``virl39k_rl_val__part_0000.parquet``, and rewrites the reduced train set so
hint generation (which skips ``*val*`` files but otherwise reads every shard)
never sees val rows. Keeps the exact convert_virl39k RL_SCHEMA (including the
``extra_info.gt_type`` field).

Usage:
  python3 scripts/sft_rl/split_virl39k_val.py \
    --input-dir fc-opd-storage/outputs/fc_opd/sft_rl/virl39k \
    --out-dir   fc-opd-storage/outputs/fc_opd/sft_rl/virl39k_train_split
  (defaults: --val-rows 500 --seed 42 --shard-rows 1500)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Reuse the exact RL schema so hint shards and val shards stay drop-in
# interchangeable with convert_virl39k output (train/val hint pipeline contract).
from convert_virl39k import RL_SCHEMA  # noqa: E402  (same-dir import)


def _source_of(row: dict) -> str:
    return str((row.get("extra_info") or {}).get("source", "unknown"))


def stratified_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Deterministically sample `n` rows proportional to each source's share.

    Largest-remainder quota per source, capped at availability, residual reflowed
    into the sources with the most headroom (mirrors
    prep_mixed_rl_pools.select_stratified_by_share so val keeps source diversity).
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[_source_of(r)].append(r)
    srcs = {s: g for s, g in groups.items() if g}
    total = sum(len(g) for g in srcs.values())
    target = min(n, total)
    if target <= 0:
        return []

    quota = {s: int(len(g) * target / total) for s, g in srcs.items()}
    rem = target - sum(quota.values())
    order = sorted(srcs, key=lambda s: (len(srcs[s]) * target / total) - quota[s],
                   reverse=True)
    for s in order[:rem]:
        quota[s] += 1

    picked: dict[str, set[int]] = {}
    residual = 0
    for s in sorted(srcs):
        g = srcs[s]
        q = min(quota[s], len(g))
        idx = set(rng.sample(range(len(g)), q)) if q < len(g) else set(range(len(g)))
        picked[s] = idx
        residual += quota[s] - q
    if residual:
        for s in sorted(srcs, key=lambda s: len(srcs[s]) - len(picked[s]), reverse=True):
            g = srcs[s]
            headroom = len(g) - len(picked[s])
            if headroom <= 0:
                continue
            add = min(residual, headroom)
            pool = [i for i in range(len(g)) if i not in picked[s]]
            picked[s].update(rng.sample(pool, add))
            residual -= add
            if residual == 0:
                break

    selected: list[dict] = []
    for s in sorted(srcs):
        selected.extend(srcs[s][i] for i in sorted(picked[s]))
    return selected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    DTOPD = Path("/inspire/hdd/global_user/mengweicheng-240108120092/lzy")
    base = DTOPD / "fc-opd-storage/outputs/fc_opd/sft_rl"
    ap.add_argument("--input-dir", type=Path, default=base / "virl39k")
    ap.add_argument("--out-dir", type=Path, default=base / "virl39k_train_split")
    ap.add_argument("--val-rows", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard-rows", type=int, default=1500)
    args = ap.parse_args()

    shards = sorted(Path(args.input_dir).glob("virl39k_rl_train__part_*.parquet"))
    if not shards:
        raise SystemExit(f"no virl39k train shards under {args.input_dir}")

    rows: list[dict] = []
    for p in shards:
        rows.extend(pq.read_table(str(p)).to_pylist())

    rng = random.Random(args.seed)
    val_rows = stratified_sample(rows, args.val_rows, rng)
    val_uids = {r["sample_uid"] for r in val_rows}
    train_rows = [r for r in rows if r["sample_uid"] not in val_uids]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_path = out_dir / "virl39k_rl_val__part_0000.parquet"
    pq.write_table(pa.Table.from_pylist(val_rows, schema=RL_SCHEMA), val_path)

    n_shards = 0
    paths: list[str] = []
    for i in range(0, len(train_rows), args.shard_rows):
        p = out_dir / f"virl39k_rl_train__part_{n_shards:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(train_rows[i:i + args.shard_rows],
                                            schema=RL_SCHEMA), p)
        paths.append(str(p))
        n_shards += 1

    val_by_source = defaultdict(int)
    for r in val_rows:
        val_by_source[_source_of(r)] += 1

    stats = {
        "input_dir": str(args.input_dir),
        "input_tallshards": len(shards),
        "input_rows": len(rows),
        "val_rows": len(val_rows),
        "train_rows": len(train_rows),
        "seed": args.seed,
        "val_by_source": dict(sorted(val_by_source.items())),
        "val_file": str(val_path),
        "train_shards": n_shards,
    }
    (out_dir / "virl39k_train_split_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2)
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"wrote val ({len(val_rows)}) + train ({len(train_rows)} rows / "
          f"{n_shards} shards) under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())