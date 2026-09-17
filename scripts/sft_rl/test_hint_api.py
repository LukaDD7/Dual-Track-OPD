#!/usr/bin/env python3
"""One-shot debug for the hint-gen server: send the exact hint request for one
MMF row and print the RAW response (content / reasoning_content / finish_reason
/ usage). Use --text-only to isolate image handling.

Usage (GPU node, server on :8010):
  python3 scripts/sft_rl/test_hint_api.py [--row 0] [--text-only]
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from openai import OpenAI

ROOT = Path("/inspire/hdd/global_user/mengweicheng-240108120092/lzy")
sys.path.insert(0, str(ROOT / "projects/Dual-Track-OPD/scripts/sft_rl"))
from build_mmf_hints import DEFAULT_SYSTEM_PROMPT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parquet",
        default=str(
            ROOT
            / "fc-opd-storage/outputs/fc_opd/sft_rl/mmf_rl_3k/mmf_rl_train__part_0000.parquet"
        ),
    )
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--api-base", default="http://127.0.0.1:8010/v1")
    ap.add_argument("--model", default="Qwen3-VL-235B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--text-only", action="store_true")
    args = ap.parse_args()

    row = pq.read_table(args.parquet).slice(args.row, 1).to_pylist()[0]
    question = str(row["question"])
    answer = str((row.get("reward_model") or {}).get("ground_truth", ""))

    user_parts: list[dict] = []
    if not args.text_only:
        for img in row.get("images") or []:
            b = img.get("bytes")
            if b is None:
                continue
            user_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(b).decode()}"
                    },
                }
            )
    user_parts.append(
        {
            "type": "text",
            "text": (
                f"{question}\n\n"
                f"Ground-truth answer (PRIVATE — for hint construction only, "
                f"never reveal it): {answer}"
            ),
        }
    )

    client = OpenAI(base_url=args.api_base, api_key="EMPTY", timeout=180)
    print(f"== row {args.row} text_only={args.text_only} model={args.model} ==")
    print("system:", DEFAULT_SYSTEM_PROMPT[:120].replace("\n", " "), "...")
    print("question:", question[:120].replace("\n", " "))
    print("images:", len(user_parts) - 1)
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_parts},
        ],
        max_tokens=args.max_tokens,
        temperature=0.7,
    )
    msg = resp.choices[0].message
    out = {
        "model": resp.model,
        "content": msg.content,
        "reasoning_content": getattr(msg, "reasoning_content", None),
        "finish_reason": resp.choices[0].finish_reason,
        "usage": resp.usage.model_dump() if resp.usage else None,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
