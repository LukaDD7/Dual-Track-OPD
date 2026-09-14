#!/usr/bin/env python3
"""Offline geometry3k evaluation for a Qwen3-VL checkpoint (rule-based reward).

Level-0 gate for SFT warmup quality: vLLM chat inference over the official
geometry3k test set (601 rows, geo3k_grpo_val_test.parquet), scored with
scripts/sft_rl/geo3k_reward.py. Prints exact-match accuracy; optional pass@k
via --rollouts (temperature sampling, k samples per question).

Usage (GPU node, env with vLLM, e.g. qwen3vl-cu128-vllm):
  CUDA_VISIBLE_DEVICES=4 python3 scripts/sft_rl/eval_geo3k.py \
      --model <sft-ckpt>/global_step_1086/huggingface \
      --data fc-opd-storage/outputs/fc_opd/sft_rl/geo3k_grpo_val_test.parquet \
      --out fc-opd-storage/outputs/fc_opd/sft_rl/eval/geo3k_sft_warmup.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def build_messages(row: dict) -> list[dict]:
    """Convert a verl RL parquet row to vLLM chat messages with image."""
    prompt = row["prompt"]
    imgs = row.get("images") or []
    text = prompt[0]["content"]
    text = text.replace("<image>", "").strip() if imgs else text
    content: list[dict] = []
    for img in imgs:
        data = img.get("bytes")
        if data:
            # vLLM 0.27.x chat content part type is "image_pil" (not "image")
            content.append({"type": "image_pil", "image_pil": Image.open(io.BytesIO(bytes(data)))})
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF dir (SFT ckpt huggingface)")
    parser.add_argument("--data", required=True, help="geo3k_grpo_val_test.parquet")
    parser.add_argument("--out", required=True, help="jsonl results path")
    parser.add_argument("--max-response-length", type=int, default=1024)
    parser.add_argument("--max-rows", type=int, default=0, help="0 = all")
    parser.add_argument("--rollouts", type=int, default=1, help="k samples/question for pass@k")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--reward", choices=["geo3k", "mmf"], default="geo3k",
                        help="scorer module: geo3k_reward (geo3k) or mmf_reward (MMF mixed formats)")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-mem", type=float, default=0.45)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    if args.reward == "mmf":
        from mmf_reward import compute_score
    else:
        from geo3k_reward import compute_score

    from vllm import LLM, SamplingParams

    t = pq.read_table(args.data)
    rows = t.to_pylist()
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    print(f"eval rows: {len(rows)}", flush=True)

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 2},
    )
    temp = 0.0 if args.rollouts == 1 else 0.7
    params = SamplingParams(max_tokens=args.max_response_length, temperature=temp)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    n_correct = 0
    n_total = 0

    for start in range(0, len(rows), args.batch):
        batch = rows[start : start + args.batch]
        # repeat each row `rollouts` times for pass@k
        msgs, metas = [], []
        for r in batch:
            gt = (r.get("reward_model") or {}).get("ground_truth", "")
            for _ in range(args.rollouts):
                msgs.append(build_messages(r))
                metas.append({"uid": r.get("sample_uid"), "gt": gt, "src": r.get("data_source")})
        outputs = llm.chat(msgs, sampling_params=params)
        for meta, out in zip(metas, outputs):
            sol = out.outputs[0].text.strip()
            score = compute_score(meta["src"], sol, meta["gt"])["score"]
            results.append({**meta, "solution": sol, "score": float(score)})
            if score == 1.0:
                n_correct += 1
            n_total += 1
        print(f"batch {start // args.batch + 1}: running acc {n_correct / n_total:.4f} "
              f"({n_correct}/{n_total})", flush=True)

    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"pass@1 accuracy: {n_correct / n_total:.4f} ({n_correct}/{n_total})")
    if args.rollouts > 1:
        # group by uid for pass@k
        from collections import defaultdict

        by_uid = defaultdict(list)
        for r in results:
            by_uid[r["uid"]].append(r["score"])
        passed = sum(1 for v in by_uid.values() if any(s == 1.0 for s in v))
        print(f"pass@{args.rollouts}: {passed / len(by_uid):.4f} ({passed}/{len(by_uid)})")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
