"""Fixed offline eval for Qwen3.5 Track B checkpoints (one condition per call).

Loads a HuggingFace-format model (base Qwen3.5-4B or a merged verl checkpoint),
generates N=8 responses per Geometry3K val prompt with a fixed per-sample seed
grid, and scores each response with the shared conservative boxed-answer
extractor.  The dump is a JSONL of per-(prompt, seed) rows so the summary
script can compute prompt-paired stats and bootstrap CIs.

This is deliberately not the verl trainer: no optimizer, no teacher, no loss.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

from dual_track_opd.fc_opd.answer_extraction import extract_final_answer_candidate
from dual_track_opd.fc_opd.prompt_contracts import geometry3k_training_prompt_boxed_only


BOXED_RE = re.compile(r"\\boxed\{[^}]*\}", re.IGNORECASE)


def _normalize_answer(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip("*_`$ ").rstrip(".")
    return text.lower() if text else None


def _score_response(response: str, gold: str) -> dict[str, bool]:
    extracted = extract_final_answer_candidate(response)
    gold_normalized = _normalize_answer(str(gold))
    return {
        "correct": (
            gold_normalized is not None
            and _normalize_answer(extracted) == gold_normalized
        ),
        "boxed": BOXED_RE.search(response) is not None,
    }


def build_prompts(parquet_path: Path):
    frame = pd.read_parquet(parquet_path)
    required = {"sample_uid", "question", "answer"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"val parquet missing columns: {sorted(missing)}")
    rows = []
    for record in frame.to_dict("records"):
        question = str(record["question"]).strip()
        if not question:
            continue
        rows.append({
            "sample_uid": str(record["sample_uid"]),
            "question": question,
            "gold": str(record["answer"]).strip(),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="HF-format model dir")
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed-grid", default=None,
                        help="comma-separated seeds, default 0..(n-1)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-response-length", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-prompts", type=int, default=None)
    args = parser.parse_args()

    seeds = (
        [int(value) for value in args.seed_grid.split(",") if value.strip()]
        if args.seed_grid
        else list(range(args.n))
    )
    if len(seeds) != args.n:
        raise ValueError("seed-grid length must equal --n")

    import torch
    from vllm import LLM, SamplingParams

    if args.device != "cuda:0":
        torch.cuda.set_device(int(args.device.rsplit(":", 1)[-1]))
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    rows = build_prompts(Path(args.val_parquet))
    if args.max_prompts is not None:
        rows = rows[: args.max_prompts]

    prompts = []
    request_rows = []
    for row in rows:
        messages = geometry3k_training_prompt_boxed_only(row["question"])
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for seed in seeds:
            prompts.append(text)
            request_rows.append((row, seed))

    # vLLM 0.23 applies per-request seed through the sampling params of each
    # generation call; issue one call per seed to keep the seed grid explicit.
    outputs_by_seed: dict[int, list] = {}
    for seed in seeds:
        seed_sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=-1,
            max_tokens=args.max_response_length,
            seed=seed,
        )
        outputs_by_seed[seed] = llm.generate(prompts[:: len(seeds)], seed_sampling)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            for seed_index, seed in enumerate(seeds):
                response = outputs_by_seed[seed][index].outputs[0]
                text = response.text
                score = _score_response(text, row["gold"])
                record = {
                    "condition": args.condition,
                    "sample_uid": row["sample_uid"],
                    "question": row["question"],
                    "gold": row["gold"],
                    "seed": seed,
                    "response": text,
                    "length": len(response.token_ids),
                    "finish_reason": response.finish_reason,
                    "clipped": response.finish_reason == "length",
                    "eos": response.finish_reason == "stop",
                    "correct": score["correct"],
                    "boxed": score["boxed"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    print(json.dumps({
        "condition": args.condition,
        "prompt_count": len(rows),
        "sample_count": written,
        "seeds": seeds,
        "output": str(out_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
