#!/usr/bin/env python3
"""Build MMFineReason RL parquet for verl GRPO (rule reward via custom reward fn).

Input : dataset/MMFineReason-SFT-123K-Qwen3-VL-235B-Thinking/data/train-*.parquet
Output: fc-opd-storage/outputs/fc_opd/sft_rl/mmfinereason/mmfinereason_rl.parquet
        columns: data_source, prompt(list<role,content>), images, ability,
                 reward_model(dict{style,ground_truth}), extra_info, question, sample_uid

GT policy (verified 2026-08-19): 122,603 rows total; 4,915 have answer=null
(mostly FineVision-visualwebinstruct, 19%). original_answer is never null.
  --gt-policy answer_only            : keep rows with non-empty `answer` (117,688)
  --gt-policy answer_then_original   : also keep answer-null rows using
                                       original_answer (4,915), gt_source marks it
Rows with neither are dropped. For rule-based rewards prefer answer_only;
original_answer rows are long prose and need a judge-style reward.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gt-policy", choices=["answer_only", "answer_then_original"],
                        default="answer_only")
    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.input_dir}/train-*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.input_dir}")

    out_rows = []
    stats = {"total": 0, "answer_usable": 0, "answer_null_orig_fallback": 0,
             "dropped": 0, "image_placeholder_mismatch": 0}

    for f in files:
        t = pq.read_table(f)
        for row in t.to_pylist():
            stats["total"] += 1
            answer = row.get("answer")
            original = row.get("original_answer")
            answer = str(answer).strip() if answer is not None else ""
            if answer:
                gt, gt_source = answer, "answer"
                stats["answer_usable"] += 1
            elif args.gt_policy == "answer_then_original" and original is not None and str(original).strip():
                gt, gt_source = str(original).strip(), "original_answer"
                stats["answer_null_orig_fallback"] += 1
            else:
                stats["dropped"] += 1
                continue

            question = str(row.get("question", ""))
            imgs = row.get("image")
            if isinstance(imgs, dict):
                imgs = [imgs]
            imgs = imgs or []
            n_ph = question.count("<image>")
            if n_ph == 0 and imgs:
                question = "<image>" + question
                n_ph = 1
            if n_ph != len(imgs):
                stats["image_placeholder_mismatch"] += 1

            source = str(row.get("source", ""))
            uid = f"{source}:{row.get('id', stats['total'])}"
            out_rows.append({
                "data_source": source,
                "prompt": [{"role": "user", "content": question}],
                "images": imgs,
                "ability": "reasoning",
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": {
                    "question": question,
                    "source": source,
                    "gt_source": gt_source,
                    "original_answer": str(original) if original is not None else "",
                },
                "question": question,
                "sample_uid": uid,
            })

    if not out_rows:
        raise SystemExit("no rows produced")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(out_rows)
    pq.write_table(table, out_path)
    print(f"wrote {out_path}: {len(out_rows)} rows")
    print("stats:", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
