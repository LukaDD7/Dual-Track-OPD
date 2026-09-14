#!/usr/bin/env python3
"""Build MMFineReason-586K rule-verifiable RL pool (verl GRPO schema).

Applies the SAME 4-class GT filter as the repo's MMFineReason-123K RL path
(sample_mmf_rl.py gt_type): keep letter / yesno / pure_number / has_number,
drop free-text "other" answers (reward is rule-based, mmf_reward.py).
Also drops answer=null rows (answer_only policy) and rows whose teacher
response is missing (no oracle to verify against).

Output (verl GRPO columns, shards <=1500 rows):
  <out-dir>/mmf586k_rl_train__part_*.parquet
  <out-dir>/mmf586k_rl_stats.json

Usage:
  python3 scripts/sft_rl/prep_mmfinereason_586k_rl.py \
    --input-dir dataset/MMFineReason-SFT-586K-Qwen3-VL-235B-Thinking/data \
    --out-dir fc-opd-storage/outputs/fc_opd/sft_rl/mmf586k_rl
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def gt_type(gt: str) -> str:
    """Identical to sample_mmf_rl.py / prep_hint_pools.py (repo GT policy)."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-rows", type=int, default=1500)
    parser.add_argument("--keep-other", action="store_true",
                        help="keep free-text 'other' GT rows (default: drop)")
    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.input_dir}/train-*.parquet"))
    if not files:
        raise SystemExit(f"no train-*.parquet under {args.input_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([
        pa.field("data_source", pa.string()),
        pa.field("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        pa.field("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
        pa.field("ability", pa.string()),
        pa.field("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
        pa.field("extra_info", pa.struct([
            ("question", pa.string()), ("source", pa.string()),
            ("gt_source", pa.string()), ("gt_type", pa.string()),
            ("original_answer", pa.string()),
        ])),
        pa.field("question", pa.string()),
        pa.field("sample_uid", pa.string()),
    ])

    rows: list[dict] = []
    stats = {"total": 0, "answer_null": 0, "no_teacher_response": 0,
             "placeholder_mismatch": 0}
    gt_stats: Counter = Counter()
    src_stats: Counter = Counter()
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.metadata.num_row_groups):
            for r in pf.read_row_group(rg).to_pylist():
                stats["total"] += 1
                answer = r.get("answer")
                answer = str(answer).strip() if answer is not None else ""
                if not answer:
                    stats["answer_null"] += 1
                    continue
                if not (r.get("qwen3vl_235b_thinking_response") or ""):
                    stats["no_teacher_response"] += 1
                    continue
                t = gt_type(answer)
                gt_stats[t] += 1
                if t == "other" and not args.keep_other:
                    continue
                question = str(r.get("question", ""))
                imgs = r.get("image")
                if isinstance(imgs, dict):
                    imgs = [imgs]
                imgs = imgs or []
                if question.count("<image>") != len(imgs):
                    stats["placeholder_mismatch"] += 1
                    continue
                source = str(r.get("source", ""))
                uid = f"{source}:{r.get('id', stats['total'])}"
                src_stats[source] += 1
                rows.append({
                    "data_source": source,
                    "prompt": [{"role": "user", "content": question}],
                    "images": [{"bytes": i.get("bytes"), "path": i.get("path") or ""} for i in imgs],
                    "ability": "reasoning",
                    "reward_model": {"style": "rule", "ground_truth": answer},
                    "extra_info": {
                        "question": question,
                        "source": source,
                        "gt_source": "answer",
                        "gt_type": t,
                        "original_answer": str(r.get("original_answer") or ""),
                    },
                    "question": question,
                    "sample_uid": uid,
                })
        print(f"[scan] {f}: pool={len(rows)}", flush=True)

    n_shards = 0
    paths = []
    for i in range(0, len(rows), args.shard_rows):
        p = out_dir / f"mmf586k_rl_train__part_{n_shards:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows[i : i + args.shard_rows], schema=schema), p)
        paths.append(str(p))
        n_shards += 1

    stats_out = {
        "total_rows": stats["total"],
        "answer_null": stats["answer_null"],
        "no_teacher_response": stats["no_teacher_response"],
        "placeholder_mismatch": stats["placeholder_mismatch"],
        "gt_stats": dict(gt_stats),
        "kept_rows": len(rows),
        "shards": paths,
        "by_source": dict(src_stats.most_common()),
    }
    (out_dir / "mmf586k_rl_stats.json").write_text(
        json.dumps(stats_out, ensure_ascii=False, indent=2)
    )
    print(f"kept {len(rows)} rows -> {n_shards} shards under {out_dir}")
    print(f"GT stats: {dict(gt_stats)}")


if __name__ == "__main__":
    main()
