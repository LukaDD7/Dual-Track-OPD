#!/usr/bin/env python3
"""Per-step sequence- and prompt-group-level clip/EOS/correct analysis.

verl's v1 trainer dumps per-step training rollouts into
``trainer.rollout_data_dir`` (env ``TRAIN_ROLLOUT_DATA_DIR``) as
``{global_step}.jsonl``.  Each row has ``uid`` of the form
``{sample_uid}_{rollout}_{output}``, so the prompt-group key is the uid prefix
before the last two underscores.  This tool reports the n=4 group contract
directly instead of inferring it from n=1 validation.

Example:
  python3 scripts/qwen35_group_clip_analysis.py \
    --rollout-dir /inspire/.../qwen35_runs/<exp>/train_rollouts \
    --max-response-length 4096 \
    --output /tmp/group_clip.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from dual_track_opd.fc_opd.qwen35_group_clip import (  # noqa: E402
    aggregate_group_metrics,
    clean_response,
    group_key,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--max-response-length", type=int, default=4096)
    parser.add_argument("--tokenizer", default="/inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3.5-4B")
    parser.add_argument("--steps", type=str, default="", help="comma list to filter, e.g. 1,5,10")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    step_filter = {int(s) for s in args.steps.split(",") if s.strip()} if args.steps else None

    results = {}
    for path in sorted(args.rollout_dir.glob("*.jsonl")):
        step = int(path.stem)
        if step_filter is not None and step not in step_filter:
            continue
        records = []
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            output = clean_response(str(row.get("output", "")))
            length = len(tokenizer(output, add_special_tokens=False)["input_ids"])
            records.append({
                "group": group_key(str(row.get("uid", ""))),
                "length": length,
                "is_clip": length >= args.max_response_length,
                "is_correct": float(row.get("score", 0.0)) > 0,
                "has_boxed": "\\boxed{" in output,
                "cap": args.max_response_length,
            })
        results[step] = aggregate_group_metrics(records)

    summary = {
        "schema_version": 1,
        "max_response_length": args.max_response_length,
        "rollout_dir": str(args.rollout_dir.resolve()),
        "steps": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
