#!/usr/bin/env python3
"""Convert Vision-Flan (vision-flan_191-task-1k) to verl SFT + RL/GRPO parquet.

Input : dataset/Vision-Flan-191-task-1k/annotation_191-task_1k.json
          list of {id, image (basename), task_name, conversations:
          [{from: human, value}, {from: gpt, value}]} — 186,103 rows, 191 tasks,
          1,000 examples each, single human turn with exactly one <image>
          placeholder, image resolved under images_191task_1k/<basename>.
Output: fc-opd-storage/outputs/fc_opd/sft_rl/visionflan/
          visionflan_sft__part_*.parquet     (SFT: full Q -> A, all rows)
          visionflan_rl_train__part_*.parquet (RL: repo 4-class GT only)
          visionflan_stats.json

Policies
  - SFT keeps every row with a non-empty assistant answer; messages are the
    verbatim human/gpt turns (role-mapped). source records the task_name.
  - image bytes resolved by basename: the image field is a bare filename and
    the archive lays images out flat under images_191task_1k/ (basenames are
    globally unique across tasks; 762 names are reused by multiple questions
    within the same task, which is fine — each row keeps its own copy).
  - <image> placeholder injected at the front of the first human turn iff it
    is missing (count must equal len(images) == 1 for verl's assertion;
    source rows already carry exactly one).
  - RL path: repo 4-class GT filter (letter/yesno/pure_number/has_number) on
    the gpt turn; "other" free/word answers are dropped for RL (teacher
    answers are often word-counts "two" or free text — this is the same
    honest GT policy as the other candidate converters).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SFT_SCHEMA = pa.schema([
    pa.field("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    pa.field("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
    pa.field("source", pa.string()),
    pa.field("pass_rate", pa.float64()),
    pa.field("image_hash", pa.string()),
])

RL_SCHEMA = pa.schema([
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

ROLE_MAP = {"human": "user", "gpt": "assistant"}


def gt_type(gt: str) -> str:
    """Identical to sample_mmf_rl.py (repo GT policy)."""
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


def sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def decode_ufname(name: str) -> str:
    """Vision-Flan stores non-ASCII archive filenames in Python-style
    ``#Uxxxx`` escapes (e.g. ``#U91d1`` -> U+91d1), while the annotation JSON
    carries the raw unicode. Decode so annotation ``image`` names match the
    extracted files on disk."""
    return re.sub(r"#U([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--image-root", required=True,
                        help="extracted images dir; walked once to index basename->path")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard-rows", type=int, default=1500)
    parser.add_argument("--skip-rl", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root)

    # one walk -> basename -> (relative posix path, absolute path). Archive
    # non-ASCII names are stored as "#Uxxxx" escapes on disk; key by the
    # decoded unicode so annotation `image` names resolve directly.
    index: dict[str, tuple[str, Path]] = {}
    for fp in image_root.rglob("*"):
        if fp.is_file():
            index.setdefault(decode_ufname(fp.name), (str(fp.relative_to(image_root)), fp))
    print(f"indexed {len(index)} image files under {image_root}", flush=True)

    with open(args.annotation_json) as f:
        data = json.load(f)

    sft_rows: list[dict] = []
    rl_rows: list[dict] = []
    stats = {"total": 0, "sft_kept": 0, "rl_kept": 0, "missing_image": 0,
             "empty_answer": 0, "bad_conversation": 0, "placeholder_injected": 0}
    gt_stats: Counter = Counter()
    task_stats: Counter = Counter()

    for r in data:
        stats["total"] += 1
        convs = r.get("conversations") or []
        if len(convs) < 2 or convs[0].get("from") != "human":
            stats["bad_conversation"] += 1
            continue
        answer = str(convs[1].get("value") or "").strip()
        if not answer:
            stats["empty_answer"] += 1
            continue
        base = str(r.get("image") or "")
        hit = index.get(base)
        if not hit:
            stats["missing_image"] += 1
            continue
        rel, fp = hit
        messages = []
        n_ph = 0
        ok = True
        for c in convs:
            role = ROLE_MAP.get(c.get("from"))
            if role is None:
                ok = False
                break
            content = str(c.get("value") or "")
            n_ph += content.count("<image>")
            messages.append({"role": role, "content": content})
        if not ok:
            stats["bad_conversation"] += 1
            continue
        if n_ph == 0:
            messages[0]["content"] = "<image>" + messages[0]["content"]
            stats["placeholder_injected"] += 1
        elif n_ph > 1:
            # single image, multiple placeholders: normalize first turn to one
            first = messages[0]["content"].replace("<image>", "")
            messages[0]["content"] = "<image>" + first.strip()
            stats["placeholder_injected"] += 1
        bytes_ = fp.read_bytes()
        task = str(r.get("task_name") or "unknown")
        qid = str(r.get("id") or stats["total"])
        task_stats[task] += 1
        img = {"bytes": bytes_, "path": rel}
        sft_rows.append({
            "messages": messages,
            "images": [img],
            "source": f"Vision-Flan/{task}",
            "pass_rate": -1.0,
            "image_hash": sha16(bytes_),
        })
        stats["sft_kept"] += 1
        t = gt_type(answer)
        gt_stats[t] += 1
        if t == "other":
            continue
        question = messages[0]["content"]
        rl_rows.append({
            "data_source": "Vision-Flan",
            "prompt": [{"role": "user", "content": question}],
            "images": [img],
            "ability": "reasoning",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "question": question,
                "source": f"Vision-Flan/{task}",
                "gt_source": "gpt_answer",
                "gt_type": t,
                "original_answer": answer,
            },
            "question": question,
            "sample_uid": f"VF:{qid}",
        })
        stats["rl_kept"] += 1

    # write SFT shards
    n_sft = 0
    sft_paths = []
    for i in range(0, len(sft_rows), args.shard_rows):
        p = out_dir / f"visionflan_sft__part_{n_sft:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(sft_rows[i:i + args.shard_rows], schema=SFT_SCHEMA), p)
        sft_paths.append(str(p))
        n_sft += 1

    # write RL shards
    n_rl = 0
    rl_paths = []
    if not args.skip_rl:
        for i in range(0, len(rl_rows), args.shard_rows):
            p = out_dir / f"visionflan_rl_train__part_{n_rl:04d}.parquet"
            pq.write_table(pa.Table.from_pylist(rl_rows[i:i + args.shard_rows], schema=RL_SCHEMA), p)
            rl_paths.append(str(p))
            n_rl += 1

    stats_out = {
        "total_rows": stats["total"],
        "sft_kept": stats["sft_kept"],
        "rl_kept": stats["rl_kept"],
        "missing_image": stats["missing_image"],
        "empty_answer": stats["empty_answer"],
        "bad_conversation": stats["bad_conversation"],
        "placeholder_injected": stats["placeholder_injected"],
        "gt_stats": dict(gt_stats),
        "n_tasks": len(task_stats),
        "sft_shards": sft_paths,
        "rl_shards": rl_paths,
    }
    (out_dir / "visionflan_stats.json").write_text(
        json.dumps(stats_out, ensure_ascii=False, indent=2)
    )
    print(f"SFT: {stats['sft_kept']} rows -> {n_sft} shards; RL: {stats['rl_kept']} rows -> {n_rl} shards")
    print(f"GT stats: {dict(gt_stats)}")


if __name__ == "__main__":
    main()