#!/usr/bin/env python3
"""Sample a small, rule-verifiable MMFineReason RL set for GRPO (SFT-then-RL / RL-only arms).

Why a subset (runbook 7.9): the full RL pool has 117,688 rows / 5.9 GB; verl GRPO
must not use data.train_max_samples on a big file (pyarrow offset overflow on
`take`), and the big combined file's nested list<struct> columns cannot even be
converted back by pyarrow to_pylist (single 117K-row chunk). So we rebuild the
sample directly from the original flat-shard parquet files.

What it does:
  1. Reads original MMF shards (train-*.parquet, flat `image` struct), keeps rows
     with non-empty `answer`.
  2. Drops rows whose GT form is free text without digits ("other", ~19% of pool) --
     mmf_reward.py covers MCQ letter / yes-no / numeric / LaTeX+mathruler; keeping
     "other" would inject ~0-reward noise into the first GRPO run.
  3. Optional warmup disjointness (image bytes hash + user text) via --warmup-dir;
     default OFF -- RL may reuse the same-domain tasks the model saw in SFT warmup
     to keep improving on them (project decision: same-domain RL is beneficial,
     not leakage). Providing --warmup-dir still excludes those rows if needed.
  4. Stratified by source (largest remainder, fixed seed) -> train + val.

Output (verl GRPO columns, explicit schema, shards <=1500 rows like the warmup):
  <out-dir>/mmf_rl_train__part_*.parquet
  <out-dir>/mmf_rl_val__part_0000.parquet
  <out-dir>/mmf_rl_stats.json

Usage:
  python3 scripts/sft_rl/sample_mmf_rl.py \
    --input-dir dataset/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking/data \
    --out-dir fc-opd-storage/outputs/fc_opd/sft_rl/mmf_rl_20k \
    --n-train 20000 --n-val 500 --seed 42
  # 可选: --warmup-dir <dir> 才做 warmup 去重（默认不去重）
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def image_hash(imgs: list[dict]) -> str:
    """Same hash as convert_mmfinereason_sft.py: comma-joined sha256(bytes)[:16]."""
    return ",".join(hashlib.sha256(i.get("bytes", b"")).hexdigest()[:16] for i in imgs)


def row_key(question: str, imgs: list[dict]) -> str:
    return hashlib.sha256(f"{image_hash(imgs)}::{question}".encode()).hexdigest()


def read_nested_safe(path: str, columns: list[str] | None = None) -> list[dict]:
    try:
        t = pq.read_table(path, columns=columns)
        return t.to_pylist()
    except Exception:
        pf = pq.ParquetFile(path)
        rows: list[dict] = []
        for i in range(pf.metadata.num_row_groups):
            rows.extend(pf.read_row_group(i, columns=columns).to_pylist())
        return rows


def load_warmup_keys(warmup_dir: str) -> set[str]:
    files = sorted(glob.glob(f"{warmup_dir}/sft_warmup_*__part_*.parquet"))
    if not files:
        raise SystemExit(f"no warmup shards under {warmup_dir}")
    keys: set[str] = set()
    for f in files:
        for r in read_nested_safe(f):
            src = str(r.get("source", ""))
            # MMF sources: the cauldron subsets are all lowercase short names like
            # vqav2/chartqa; MMF sources contain uppercase/'-' (MMR1, GameQA-140K, ...).
            if not src or src == src.lower():
                continue
            text = "".join(m.get("content", "") for m in r["messages"] if m.get("role") == "user")
            keys.add(row_key(text, r.get("images", [])))
    print(f"warmup MMF keys: {len(keys)}")
    return keys


def gt_type(gt: str) -> str:
    a = gt.strip()
    if re.fullmatch(r"\(?[A-Ea-e]\)?\.?", a):
        return "letter"
    if re.fullmatch(r"(yes|no|true|false)", a.lower()):
        return "yesno"
    if re.fullmatch(r"-?\d+(\.\d+)?", a):
        return "pure_number"
    if re.search(r"\d", a):
        return "has_number"
    return "other"


def stratify(rows_by_src: dict[str, list[dict]], n: int, rng: random.Random) -> list[dict]:
    """Largest-remainder per-source allocation, then sample without replacement."""
    total = sum(len(v) for v in rows_by_src.values())
    if n >= total:
        return [r for v in rows_by_src.values() for r in v]
    targets = {s: int(n * len(v) / total) for s, v in rows_by_src.items()}
    leftover = n - sum(targets.values())
    rem = sorted(rows_by_src, key=lambda s: (n * len(rows_by_src[s]) / total) % 1.0, reverse=True)
    for s in rem[:leftover]:
        targets[s] += 1
    out: list[dict] = []
    for s, k in sorted(targets.items()):
        rs = rows_by_src[s]
        if k >= len(rs):
            out.extend(rs)
        else:
            out.extend(rs[i] for i in sorted(rng.sample(range(len(rs)), k)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True,
                        help="dataset/MMFineReason-SFT-123K-.../data (train-*.parquet)")
    parser.add_argument("--warmup-dir", default=None,
                        help="optional warmup shard dir to exclude from RL (default: no overlap filter)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-other", action="store_true",
                        help="keep free-text 'other' GT rows (default: drop, reward is rule-based)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warmup_keys = load_warmup_keys(args.warmup_dir) if args.warmup_dir else set()

    by_src: dict[str, list[dict]] = {}
    gt_stats = {"letter": 0, "yesno": 0, "pure_number": 0, "has_number": 0, "other": 0}
    dropped = {"warmup_overlap": 0, "gt_other": 0, "placeholder_mismatch": 0}
    files = sorted(glob.glob(f"{args.input_dir}/train-*.parquet"))
    if not files:
        raise SystemExit(f"no train-*.parquet under {args.input_dir}")
    stats = {"total_rows": 0, "answer_null": 0}
    for f in files:
        for r in pq.read_table(f).to_pylist():
            stats["total_rows"] += 1
            q = str(r.get("question", ""))
            answer = r.get("answer")
            answer = str(answer).strip() if answer is not None else ""
            if not answer:
                stats["answer_null"] += 1
                continue
            imgs = r.get("image")
            if isinstance(imgs, dict):
                imgs = [imgs]
            imgs = imgs or []
            imgs = [{"bytes": i.get("bytes"),
                     "path": i.get("path") or f"{r.get('source','mmf')}_{r.get('id','')}.png"}
                    for i in imgs]
            if q.count("<image>") != len(imgs):
                dropped["placeholder_mismatch"] += 1
                continue
            t = gt_type(answer)
            gt_stats[t] += 1
            if t == "other" and not args.keep_other:
                dropped["gt_other"] += 1
                continue
            if row_key(q, imgs) in warmup_keys:
                dropped["warmup_overlap"] += 1
                continue
            source = str(r.get("source", ""))
            original = r.get("original_answer")
            uid = f"{source}:{r.get('id', stats['total_rows'])}"
            by_src.setdefault(source, []).append({
                "data_source": source,
                "prompt": [{"role": "user", "content": q}],
                "images": imgs,
                "ability": "reasoning",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "question": q,
                    "source": source,
                    "gt_source": "answer",
                    "original_answer": str(original) if original is not None else "",
                },
                "question": q,
                "sample_uid": uid,
            })
    if not by_src:
        raise SystemExit("no rows produced")

    rng = random.Random(args.seed)
    train = stratify(by_src, args.n_train, rng)
    # val from the remainder (remove train uids to keep disjoint)
    train_uids = {r["sample_uid"] for r in train}
    remain: dict[str, list[dict]] = {}
    for s, rs in by_src.items():
        remain[s] = [r for r in rs if r["sample_uid"] not in train_uids]
    val = stratify(remain, args.n_val, rng)

    def write(rows: list[dict], name: str) -> list[str]:
        paths = []
        schema = pa.schema([
            pa.field("data_source", pa.string()),
            pa.field("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            pa.field("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
            pa.field("ability", pa.string()),
            pa.field("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
            pa.field("extra_info", pa.struct([
                ("question", pa.string()), ("source", pa.string()),
                ("gt_source", pa.string()), ("original_answer", pa.string()),
            ])),
            pa.field("question", pa.string()),
            pa.field("sample_uid", pa.string()),
        ])
        for i in range(0, len(rows), 1500):
            p = out_dir / f"{name}__part_{i // 1500:04d}.parquet"
            pq.write_table(pa.Table.from_pylist(rows[i : i + 1500], schema=schema), p)
            paths.append(str(p))
        return paths

    train_paths = write(train, "mmf_rl_train")
    val_paths = write(val, "mmf_rl_val")
    stats = {
        "seed": args.seed,
        "source_rows": stats,
        "pool_after_filter": sum(len(v) for v in by_src.values()),
        "gt_stats": gt_stats,
        "dropped": dropped,
        "train": {"rows": len(train), "paths": train_paths},
        "val": {"rows": len(val), "paths": val_paths},
        "train_by_source": {s: sum(1 for r in train if r["data_source"] == s) for s in sorted({r['data_source'] for r in train})},
    }
    (out_dir / "mmf_rl_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"train={len(train)} val={len(val)}  (pool={stats['pool_after_filter']})")
    print(f"wrote {len(train_paths) + len(val_paths)} shards under {out_dir}")


if __name__ == "__main__":
    main()
