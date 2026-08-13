#!/usr/bin/env python3
"""Build the fixed verified answer-free teacher prefixes for the STP-OPD pilot.

The pilot fixes one answer-free horizon per rescue-positive prompt (recorded in
`rescue_screen_20260806/prompt_results/*.json` as `minimal_horizon`).  The
historical teacher proposals were not cached, so this script regenerates a
correct teacher response per prompt and takes the first `minimal_horizon`
tokens as the prefix, re-verifying that the decoded prefix is answer-free.

GPU required (teacher model).  Writes:
  {output_dir}/stp_prefixes.json   -> {sample_uid: [token_ids]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dual_track_opd.support_aware.diagnostic import build_prompt  # noqa: E402
from dual_track_opd.support_aware.prefix_intervention import (  # noqa: E402
    generate_continuation,
    prefix_leakage_reason,
)
from dual_track_opd.support_aware.verifier import verify_answer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-parquet", required=True)
    parser.add_argument("--rescue-results-dir", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--response-format", default="legacy_answer")
    parser.add_argument("--max-proposal-tokens", type=int, default=4096)
    parser.add_argument("--max-proposal-seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    # 1. rescue-positive prompts + verified minimal horizons
    rescue_rows = {}
    for path in Path(args.rescue_results_dir).glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") == "rescue_positive" and row.get("minimal_horizon"):
            rescue_rows[str(row["sample_uid"])] = int(row["minimal_horizon"])
    if not rescue_rows:
        print("FATAL: no rescue-positive prompts with minimal_horizon found", file=sys.stderr)
        return 1

    # 2. cohort rows for those prompts
    cohort = pd.read_parquet(args.cohort_parquet)
    by_uid = {str(row["sample_uid"]): row for _, row in cohort.iterrows()}
    missing = sorted(set(rescue_rows) - set(by_uid))
    if missing:
        print(f"FATAL: cohort missing rescue-positive prompts: {missing}", file=sys.stderr)
        return 1

    # 3. load teacher
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.teacher_model,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.eval()

    prefixes: dict[str, list[int]] = {}
    failures: dict[str, str] = {}
    proposal_seeds = [args.seed + offset for offset in range(args.max_proposal_seeds)]
    for prompt_uid, horizon in sorted(rescue_rows.items()):
        row = by_uid[prompt_uid]
        question = str(row["question"]).strip()
        gold = row["answer"]
        prompt_text = build_prompt(question, args.response_format)
        images_value = row["images"]
        if hasattr(images_value, "item"):
            images_value = images_value.item()
        image_bytes = (
            images_value[0]["bytes"]
            if isinstance(images_value, (list, tuple))
            else images_value["bytes"]
        )
        from PIL import Image
        from io import BytesIO

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        built = False
        correct_count = 0
        leak_count = 0
        for proposal_seed in proposal_seeds:
            generated = generate_continuation(
                model,
                processor,
                image=image,
                prompt_text=prompt_text,
                prefix_ids=(),
                max_continuation_tokens=args.max_proposal_tokens,
                temperature=0.7,
                top_p=0.95,
                seed=proposal_seed,
                device="cuda:0",
            )["generation"]
            verdict = verify_answer(generated.response_text_display, gold)
            if verdict.get("correct") is not True:
                continue
            correct_count += 1
            response_ids = list(generated.response_token_ids_raw)
            prefix_ids = response_ids[:horizon]
            prefix_text = processor.tokenizer.decode(
                prefix_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if prefix_leakage_reason(prefix_text, gold) is not None:
                leak_count += 1
                continue
            prefixes[prompt_uid] = prefix_ids
            print(
                f"  {prompt_uid}: horizon={horizon} prefix_tokens={len(prefix_ids)} "
                f"answer-free=OK (seed {proposal_seed})",
                flush=True,
            )
            built = True
            break
        if not built:
            failures[prompt_uid] = (
                f"no correct answer-free proposal across {len(proposal_seeds)} seeds "
                f"(correct={correct_count}, prefix_leak={leak_count})"
            )
            print(
                f"  {prompt_uid}: SKIP (no correct answer-free proposal; "
                f"correct={correct_count}, prefix_leak={leak_count})",
                flush=True,
            )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stp_prefixes.json").write_text(
        json.dumps(prefixes), encoding="utf-8"
    )
    (output_dir / "stp_prefixes_report.json").write_text(
        json.dumps({
            "horizons": rescue_rows,
            "prefixes_built": {k: len(v) for k, v in prefixes.items()},
            "failures": failures,
            "seed": args.seed,
        }, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "prefixes_built": {k: len(v) for k, v in prefixes.items()},
        "failures": failures,
        "output": str(output_dir / "stp_prefixes.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
